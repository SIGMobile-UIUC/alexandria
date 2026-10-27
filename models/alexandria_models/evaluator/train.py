"""Train the translation-accuracy evaluator: a cross-encoder that reads
``[CLS] source [SEP] translation [SEP]`` and outputs one logit; sigmoid(logit) is the
0..1 accuracy score. Works in both directions (Hindi->English and English->Hindi)
because the training data contains both.

Inputs come from ``evaluator.prepare_data`` (``train.jsonl``, ``val_human.jsonl``,
``val_synthetic.jsonl``). Output directory::

    model/                    transformers model + tokenizer (what the backend loads)
    evaluator_config.json     max length, base model, validation metrics
    checkpoints/              Trainer checkpoints (resume with --resume)

Example::

    python -m alexandria_models.evaluator.train \
        --data_dir /content/drive/MyDrive/alexandria/data/evaluator \
        --output_dir /content/drive/MyDrive/alexandria/artifacts/evaluator
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..common.io import get_logger, read_jsonl, set_seed, write_json
from .tokenization import load_encoder_tokenizer

log = get_logger("evaluator.train")
DEFAULT_BASE = "google/muril-base-cased"
CONFIG_FILE = "evaluator_config.json"


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    # average ranks over ties
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    return (sums / counts)[inv]


def correlation_metrics(preds: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    preds, labels = np.asarray(preds, float).ravel(), np.asarray(labels, float).ravel()
    if len(preds) < 2 or preds.std() == 0 or labels.std() == 0:
        pearson = spearman = 0.0
    else:
        pearson = float(np.corrcoef(preds, labels)[0, 1])
        spearman = float(np.corrcoef(_rank(preds), _rank(labels))[0, 1])
    return {
        "pearson": pearson,
        "spearman": spearman,
        "mae": float(np.abs(preds - labels).mean()),
        # Does the model put good (>=0.7) and bad (<=0.3) answers on the right side of 0.5?
        "pass_fail_acc": float(
            ((preds >= 0.5) == (labels >= 0.5))[(labels >= 0.7) | (labels <= 0.3)].mean()
            if ((labels >= 0.7) | (labels <= 0.3)).any()
            else 0.0
        ),
    }


@dataclass
class PairCollator:
    pad_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        import torch

        longest = max(len(f["input_ids"]) for f in features)
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        has_types = "token_type_ids" in features[0]
        if has_types:
            out["token_type_ids"] = []
        for f in features:
            pad = longest - len(f["input_ids"])
            out["input_ids"].append(f["input_ids"] + [self.pad_token_id] * pad)
            out["attention_mask"].append(f["attention_mask"] + [0] * pad)
            if has_types:
                out["token_type_ids"].append(f["token_type_ids"] + [0] * pad)
            out["labels"].append(float(f["labels"]))
        batch = {k: torch.tensor(v) for k, v in out.items()}
        batch["labels"] = batch["labels"].float()
        return batch


def encode_pairs(tokenizer, sources: list[str], translations: list[str], max_len: int) -> dict:
    return tokenizer(sources, translations, truncation=True, max_length=max_len)


def _make_trainer_cls():
    import torch
    from transformers import Trainer

    class SoftLabelTrainer(Trainer):
        """Binary cross-entropy against soft 0..1 targets (better calibrated than MSE on a sigmoid)."""

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs["labels"]
            outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
            logits = outputs.logits.squeeze(-1).float()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
            return (loss, outputs) if return_outputs else loss

    return SoftLabelTrainer


def train(args: argparse.Namespace) -> Path:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification, TrainingArguments

    set_seed(args.seed)
    out = Path(args.output_dir)
    data_dir = Path(args.data_dir)
    cuda = torch.cuda.is_available()

    tokenizer = load_encoder_tokenizer(args.base_model)

    def load(name: str, limit: int | None = None) -> Dataset | None:
        path = data_dir / f"{name}.jsonl"
        if not path.exists():
            return None
        rows = list(read_jsonl(path))
        if limit:
            rows = rows[:limit]
        if not rows:
            return None
        enc = encode_pairs(tokenizer, [r["source"] for r in rows], [r["translation"] for r in rows], args.max_len)
        enc["labels"] = [float(r["label"]) for r in rows]
        return Dataset.from_dict(dict(enc))

    train_ds = load("train", args.max_train_examples)
    evals = {k: v for k, v in {"human": load("val_human"), "synthetic": load("val_synthetic", 5000)}.items() if v}
    if train_ds is None or not evals:
        raise SystemExit(f"missing train/val files in {data_dir}; run evaluator.prepare_data first")
    log.info("train=%d %s", len(train_ds), {k: len(v) for k, v in evals.items()})
    best_metric = "eval_human_spearman" if "human" in evals else "eval_synthetic_spearman"

    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=1)

    def compute_metrics(p):
        preds = 1 / (1 + np.exp(-np.asarray(p.predictions, float).ravel()))
        return correlation_metrics(preds, p.label_ids)

    steps = args.max_steps if args.max_steps > 0 else math.ceil(len(train_ds) / args.batch_size * args.epochs)
    warmup = int(args.warmup * steps) if args.warmup < 1 else int(args.warmup)
    targs = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_steps=warmup,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        fp16=cuda and not args.fp32,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=best_metric,
        greater_is_better=True,
        report_to=args.report_to,
        remove_unused_columns=False,
        seed=args.seed,
        use_cpu=not cuda,
    )
    trainer = _make_trainer_cls()(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=evals,
        data_collator=PairCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    resume = args.resume and any((out / "checkpoints").glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=True if resume else None)
    metrics = trainer.evaluate()
    log.info("final eval: %s", metrics)

    model_dir = out / "model"
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    test_metrics = {}
    test = load("test_human")
    if test is not None:
        test_metrics = trainer.evaluate(test, metric_key_prefix="test_human")
        log.info("held-out human test: %s", test_metrics)
    learner = load("test_learner")  # your team's scored answers (evaluator.label_sheet --export)
    if learner is not None:
        learner_metrics = trainer.evaluate(learner, metric_key_prefix="test_learner")
        test_metrics.update(learner_metrics)
        log.info("learner answers scored by your team: %s", learner_metrics)
    write_json(
        out / CONFIG_FILE,
        {"base_model": args.base_model, "max_len": args.max_len, "metrics": {**metrics, **test_metrics}},
    )
    return model_dir


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--base_model", default=DEFAULT_BASE)
    p.add_argument("--max_len", type=int, default=160)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--warmup", type=float, default=0.06)
    p.add_argument("--eval_steps", type=int, default=1000)
    p.add_argument("--max_train_examples", type=int, default=None)
    p.add_argument("--report_to", default="none")
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
