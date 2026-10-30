"""Train the naturalness model: a MuRIL encoder reading one Hindi sentence and giving
0..1 (1 = natural everyday sentence worth teaching). GPU recommended (~10-20 min on a T4).

Training rows are the LLM-judged sentences (the reliable labels, repeated
``--judge_repeat`` times) plus the weakly labelled candidates; a sentence the judge saw
uses the judge's label. Validation uses held-out judged sentences when there are enough.

Output (``--output_dir``): model/, naturalness_config.json.

    python -m alexandria_models.naturalness.train \\
        --data_dir data_v2/naturalness --output_dir artifacts_v2/naturalness
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np

from ..common.io import get_logger, read_jsonl, set_seed, write_json
from ..evaluator.tokenization import load_encoder_tokenizer
from ..evaluator.train import DEFAULT_BASE, PairCollator, _make_trainer_cls, correlation_metrics

log = get_logger("naturalness.train")
CONFIG_FILE = "naturalness_config.json"


def build_rows(data_dir: Path, judge_repeat: int, max_weak: int, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    cands = list(read_jsonl(data_dir / "candidates.jsonl"))
    judged_path = data_dir / "judged.jsonl"
    judged = {r["id"]: r for r in read_jsonl(judged_path)} if judged_path.exists() else {}
    log.info("%d candidates, %d judged", len(cands), len(judged))

    judged_rows = [{"hi": r["hi"], "label": float(r["label"]), "kind": "judged"} for r in judged.values()]
    rng.shuffle(judged_rows)
    if len(judged_rows) >= 200:
        n_val = max(100, len(judged_rows) // 10)
        val, judged_rows = judged_rows[:n_val], judged_rows[n_val:]
        held = {r["hi"] for r in val}
    else:
        val, held = [], set()
    weak = [{"hi": r["hi"], "label": float(r["label"]), "kind": "weak"}
            for r in cands if r["id"] not in judged and r["hi"] not in held]
    rng.shuffle(weak)
    weak = weak[:max_weak]
    if not val:  # no judge labels: validate on weak labels
        n_val = max(1, len(weak) // 20)
        val, weak = weak[:n_val], weak[n_val:]
    train = judged_rows * judge_repeat + weak
    rng.shuffle(train)
    return train, val


def train(args: argparse.Namespace) -> Path:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification, TrainingArguments

    set_seed(args.seed)
    out = Path(args.output_dir)
    cuda = torch.cuda.is_available()
    tokenizer = load_encoder_tokenizer(args.base_model)
    train_rows, val_rows = build_rows(Path(args.data_dir), args.judge_repeat, args.max_weak, args.seed)
    if not train_rows or not val_rows:
        raise SystemExit("no training data; run naturalness.prepare_data first")

    def to_ds(rows: list[dict]) -> Dataset:
        enc = tokenizer([r["hi"] for r in rows], truncation=True, max_length=args.max_len)
        enc["labels"] = [r["label"] for r in rows]
        return Dataset.from_dict(dict(enc))

    train_ds, val_ds = to_ds(train_rows), to_ds(val_rows)
    log.info("train=%d val=%d (%s)", len(train_ds), len(val_ds), val_rows[0]["kind"])
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=1)

    def compute_metrics(p):
        preds = 1 / (1 + np.exp(-np.asarray(p.predictions, float).ravel()))
        return correlation_metrics(preds, p.label_ids)

    steps = args.max_steps if args.max_steps > 0 else math.ceil(len(train_ds) / args.batch_size * args.epochs)
    targs = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_steps=int(0.06 * steps),
        weight_decay=0.01,
        fp16=cuda,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_spearman",
        greater_is_better=True,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
        use_cpu=not cuda,
    )
    trainer = _make_trainer_cls()(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=PairCollator(tokenizer.pad_token_id), processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    resume = args.resume and any((out / "checkpoints").glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=True if resume else None)
    metrics = trainer.evaluate()
    log.info("final eval: %s", metrics)
    model_dir = out / "model"
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    write_json(out / CONFIG_FILE, {"base_model": args.base_model, "max_len": args.max_len,
                                   "validated_on": val_rows[0]["kind"], "metrics": metrics})
    return model_dir


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--base_model", default=DEFAULT_BASE)
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--judge_repeat", type=int, default=2)
    p.add_argument("--max_weak", type=int, default=150_000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
