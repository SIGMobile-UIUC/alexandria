"""Load the trained naturalness model and score Hindi sentences (0..1, higher = better
for learners). Used to filter generator training data, generated sentences and the
sentence bank."""

from __future__ import annotations

import threading
from pathlib import Path

from ..common.io import read_json, resolve_device
from ..common.text import normalize

CONFIG_FILE = "naturalness_config.json"


class NaturalnessModel:
    def __init__(self, model, tokenizer, max_len: int = 64, device: str = "cpu", batch_size: int = 64):
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device
        self.batch_size = batch_size
        self._lock = threading.Lock()

    @classmethod
    def load(cls, naturalness_dir: str | Path, device: str = "auto") -> "NaturalnessModel":
        import torch
        from transformers import AutoModelForSequenceClassification

        from ..evaluator.tokenization import load_encoder_tokenizer

        d = Path(naturalness_dir)
        cfg = read_json(d / CONFIG_FILE) if (d / CONFIG_FILE).exists() else {}
        model_dir = d / "model" if (d / "model").exists() else d
        device = resolve_device(device)
        tok = load_encoder_tokenizer(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, dtype=torch.float16 if device == "cuda" else torch.float32
        )
        model.to(device).eval()
        return cls(model, tok, int(cfg.get("max_len", 64)), device, batch_size=256 if device == "cuda" else 64)

    def score_batch(self, sentences: list[str]) -> list[float]:
        import torch

        scores: list[float] = []
        for i in range(0, len(sentences), self.batch_size):
            chunk = [normalize(s) for s in sentences[i : i + self.batch_size]]
            enc = self.tokenizer(chunk, truncation=True, max_length=self.max_len, padding=True, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with self._lock, torch.inference_mode():
                logits = self.model(**enc).logits.squeeze(-1).float()
            scores.extend(torch.sigmoid(logits).reshape(-1).tolist())
        return [round(float(s), 4) for s in scores]

    def score(self, sentence: str) -> float:
        return self.score_batch([sentence])[0]
