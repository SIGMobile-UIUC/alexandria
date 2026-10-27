"""QLoRA fine-tuning of the Hindi sentence generator.

Reads ``train.jsonl`` / ``val.jsonl`` from ``prepare_data`` and trains LoRA adapters on a
4-bit quantized base model (fits a free Colab T4). Outputs, under ``--output_dir``:

  adapter/                 LoRA adapter + tokenizer (small; needs the base model to run)
  merged/                  base model with the adapter merged in, fp16 (what the backend loads)
  generator_config.json    prompt format, number of levels, base model id
  checkpoints/             intermediate Trainer checkpoints (resume with --resume)

Example (Colab)::

    python -m alexandria_models.generator.train \
        --data_dir /content/drive/MyDrive/alexandria/data/generator \
        --output_dir /content/drive/MyDrive/alexandria/artifacts/generator
"""

from __future__ import annotations

import argparse
import gc
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from ..common.io import get_logger, read_json, read_jsonl, set_seed, write_json
from .presets import DEFAULT_PRESET, PRESETS, resolve_base_model
from .prompts import PROMPT_FORMAT_VERSION, bos_prefix, build_completion, build_prompt

log = get_logger("generator.train")
CONFIG_FILE = "generator_config.json"


def tokenize_example(tokenizer, row: dict, num_levels: int, max_len: int) -> dict:
    """Prompt tokens are masked out of the loss (label -100); completion + EOS are trained."""
    prompt = build_prompt(row["level"], num_levels, row.get("word"), row.get("grammar"))
    completion = build_completion(row["hi"], row["en"])
    prompt_ids = bos_prefix(tokenizer) + tokenizer(prompt, add_special_tokens=False)["input_ids"]
    comp_ids = tokenizer(completion, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    input_ids = (prompt_ids + comp_ids)[:max_len]
    labels = ([-100] * len(prompt_ids) + comp_ids)[:max_len]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


@dataclass
class CompletionCollator:
    pad_token_id: int
    pad_to_multiple_of: int = 8

    def __call__(self, features: list[dict]) -> dict:
        import torch

        longest = max(len(f["input_ids"]) for f in features)
        m = self.pad_to_multiple_of
        longest = int(math.ceil(longest / m) * m) if m else longest
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad = longest - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_token_id] * pad)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad)
            batch["labels"].append(f["labels"] + [-100] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def load_tokenizer(name_or_path: str, token: str | None = None):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name_or_path, token=token)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def _nan_guard_callback():
    from transformers import TrainerCallback

    class NaNGuard(TrainerCallback):
        """Stop early with a clear message if the loss becomes NaN/inf (fp16 overflow)."""

        def __init__(self, patience: int = 3):
            self.bad = 0
            self.patience = patience

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            vals = [logs[k] for k in ("loss", "grad_norm", "eval_loss") if isinstance(logs.get(k), (int, float))]
            if not vals:
                return
            loss = next((v for v in vals if math.isnan(v) or math.isinf(v)), None)
            if loss is not None:
                self.bad += 1
                if self.bad >= self.patience:
                    log.error(
                        "Loss is %s for %d logging steps: the base model is overflowing in float16. "
                        "Re-run with --fp32 (slower) or choose another --preset (see README).",
                        loss, self.bad,
                    )
                    control.should_training_stop = True
            else:
                self.bad = 0

    return NaNGuard()


def train(args: argparse.Namespace) -> Path:
    if args.merge_only:  # adapter already trained; just (re)build merged/
        out = Path(args.output_dir)
        cfg = read_json(out / CONFIG_FILE)
        merge_adapter(out, args.base_model or cfg["base_model"], args.hf_token)
        return out
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    meta = read_json(data_dir / "data_config.json")
    num_levels = int(meta["num_levels"])
    base_model = resolve_base_model(args.preset, args.base_model)
    preset = PRESETS.get(args.preset) if not args.base_model else None
    batch_size = args.batch_size or (preset.batch_size if preset else 8)
    grad_accum = args.grad_accum or (preset.grad_accum if preset else 4)
    max_len = args.max_seq_len or (preset.max_seq_len if preset else 192)
    # The backend serves real sentences from the bank, so ship it next to the model.
    if (data_dir / "sentence_bank.jsonl").exists():
        shutil.copy(data_dir / "sentence_bank.jsonl", out / "sentence_bank.jsonl")
    cuda = torch.cuda.is_available()
    use_4bit = cuda and not args.no_4bit
    # L4 / A100 support bfloat16, which cannot overflow like float16 on a T4.
    use_bf16 = cuda and not args.fp32 and torch.cuda.is_bf16_supported()
    use_fp16 = cuda and not args.fp32 and not use_bf16
    log.info("base model %s | cuda=%s 4bit=%s fp16=%s bf16=%s batch=%dx%d max_len=%d",
             base_model, cuda, use_4bit, use_fp16, use_bf16, batch_size, grad_accum, max_len)

    tokenizer = load_tokenizer(base_model, args.hf_token)

    def load_split(name: str, limit: int | None) -> Dataset:
        rows = list(read_jsonl(data_dir / f"{name}.jsonl"))
        if limit:
            random.Random(args.seed).shuffle(rows)
            rows = rows[:limit]
        toks = [tokenize_example(tokenizer, r, num_levels, max_len + 1) for r in rows]
        # Drop (rather than truncate) over-long examples: a truncated example has no EOS,
        # which would teach the model to never stop.
        kept = [t for t in toks if len(t["input_ids"]) <= max_len]
        if len(kept) < len(toks):
            log.warning("%s: dropped %d examples longer than %d tokens", name, len(toks) - len(kept), max_len)
        return Dataset.from_list(kept)

    train_ds = load_split("train", args.max_train_examples)
    val_ds = load_split("val", args.max_eval_examples)
    if len(val_ds) == 0:
        log.warning("validation set is empty; evaluating on a slice of the training data instead")
        val_ds = train_ds.select(range(min(len(train_ds), 200)))
    log.info("train=%d val=%d examples", len(train_ds), len(val_ds))

    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32
    load_kwargs: dict = {"token": args.hf_token}
    if cuda:
        load_kwargs["attn_implementation"] = "sdpa"
    if use_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["device_map"] = {"": 0}
        load_kwargs["dtype"] = compute_dtype
    else:
        load_kwargs["dtype"] = compute_dtype if cuda else torch.float32
        if cuda:
            load_kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
    model.config.use_cache = False
    if use_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    elif cuda:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules.split(",") if args.target_modules != "all-linear" else "all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    if args.init_adapter:
        # Continue from an earlier run's adapter (same base model) instead of a fresh LoRA.
        prev = read_json(Path(args.init_adapter) / "adapter_config.json").get("base_model_name_or_path")
        if prev and prev != base_model:
            raise SystemExit(f"--init_adapter was trained on {prev}, but the base model is {base_model}; "
                             "use the same --preset/--base_model or drop --init_adapter")
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        log.info("continuing from adapter %s", args.init_adapter)
    else:
        model = get_peft_model(model, lora)
    # fp16 AMP needs fp32 master weights for the trainable (LoRA) parameters.
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    model.print_trainable_parameters()

    steps = args.max_steps if args.max_steps > 0 else math.ceil(
        len(train_ds) / (batch_size * grad_accum) * args.epochs
    )
    warmup = int(args.warmup * steps) if args.warmup < 1 else int(args.warmup)
    targs = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_steps=warmup,
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        max_grad_norm=0.5,
        logging_nan_inf_filter=False,  # let NaNGuard see overflowing losses
        train_sampling_strategy="group_by_length",
        fp16=use_fp16,
        bf16=use_bf16,
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=args.report_to,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=args.seed,
        use_cpu=not cuda,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=CompletionCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
        callbacks=[_nan_guard_callback()],
    )
    resume = args.resume and any((out / "checkpoints").glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=True if resume else None)
    metrics = trainer.evaluate()
    log.info("final eval: %s", metrics)

    adapter_dir = out / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    write_json(
        out / CONFIG_FILE,
        {
            "prompt_format_version": PROMPT_FORMAT_VERSION,
            "num_levels": num_levels,
            "max_seq_len": max_len,
            "base_model": base_model,
            "grammar_controls": meta.get("grammar_controls", []),
            "eval_loss": metrics.get("eval_loss"),
            "merged": False,
        },
    )

    del trainer, model
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    if not args.no_merge:
        merge_adapter(out, base_model, args.hf_token)
    return out


def merge_adapter(out_dir: str | Path, base_model: str, hf_token: str | None = None) -> Path:
    """Reload the base model unquantized in fp16 and merge the LoRA weights into it."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    out = Path(out_dir)
    cuda = torch.cuda.is_available()
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.float16 if cuda else torch.float32,
        device_map={"": 0} if cuda else None,
        token=hf_token,
    )
    merged = PeftModel.from_pretrained(base, out / "adapter").merge_and_unload()
    merged_dir = out / "merged"
    merged.save_pretrained(merged_dir, max_shard_size="1GB")
    load_tokenizer(str(out / "adapter")).save_pretrained(merged_dir)
    cfg = read_json(out / CONFIG_FILE)
    cfg["merged"] = True
    write_json(out / CONFIG_FILE, cfg)
    log.info("merged model saved to %s", merged_dir)
    return merged_dir


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True, help="output of generator.prepare_data")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--preset", default=DEFAULT_PRESET, choices=sorted(PRESETS), help="named base model")
    p.add_argument("--base_model", default=None, help="any HF causal LM id or local path (overrides --preset)")
    p.add_argument("--hf_token", default=None, help="needed only for gated models (or set HF_TOKEN)")
    p.add_argument("--max_seq_len", type=int, default=None, help="default: the preset's value")
    p.add_argument("--batch_size", type=int, default=None, help="per-device; default: the preset's value")
    p.add_argument("--grad_accum", type=int, default=None, help="default: the preset's value")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=-1, help="overrides --epochs when > 0")
    p.add_argument("--warmup", type=float, default=0.03, help="warmup steps, or a ratio if < 1")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--init_adapter", default=None, help="adapter/ dir of an earlier run to continue from")
    p.add_argument("--target_modules", default="all-linear")
    p.add_argument("--logging_steps", type=int, default=25)
    p.add_argument("--eval_steps", type=int, default=300, help="evaluate + checkpoint every N steps")
    p.add_argument("--max_train_examples", type=int, default=None)
    p.add_argument("--max_eval_examples", type=int, default=1000)
    p.add_argument("--report_to", default="none")
    p.add_argument("--no_4bit", action="store_true", help="train in fp16 without quantization")
    p.add_argument("--fp32", action="store_true", help="disable fp16 mixed precision (use if loss becomes NaN)")
    p.add_argument("--no_merge", action="store_true", help="skip writing the merged model")
    p.add_argument("--merge_only", action="store_true", help="skip training; merge the saved adapter into the base model")
    p.add_argument("--resume", action="store_true", help="resume from the latest checkpoint in output_dir")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
