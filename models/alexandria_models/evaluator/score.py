"""Score a learner's translation (0..1) with the trained evaluator.

``TranslationEvaluator.evaluate`` is what the backend calls. It works out the direction
from the scripts used, and when a learner types Hindi in Latin letters ("main ghar ja
raha hoon") it asks the transliterator for the top few Devanagari readings and keeps the
best-scoring one, so a correct answer is not penalised for a transliteration mistake.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..common.io import read_json, resolve_device
from ..common.text import is_devanagari, is_latin, normalize

if TYPE_CHECKING:
    from ..transliterator.transliterate import Transliterator

CONFIG_FILE = "evaluator_config.json"


class TranslationEvaluator:
    def __init__(self, model, tokenizer, max_len: int = 160, device: str = "cpu", batch_size: int = 32):
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device
        self.batch_size = batch_size
        self._lock = threading.Lock()

    @classmethod
    def load(cls, evaluator_dir: str | Path, device: str = "auto") -> "TranslationEvaluator":
        import torch
        from transformers import AutoModelForSequenceClassification

        from .tokenization import load_encoder_tokenizer

        d = Path(evaluator_dir)
        cfg = read_json(d / CONFIG_FILE) if (d / CONFIG_FILE).exists() else {}
        model_dir = d / "model" if (d / "model").exists() else d
        device = resolve_device(device)
        tok = load_encoder_tokenizer(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, dtype=torch.float16 if device == "cuda" else torch.float32
        )
        model.to(device).eval()
        return cls(model, tok, int(cfg.get("max_len", 160)), device)

    def score_batch(self, sources: list[str], translations: list[str]) -> list[float]:
        import torch

        scores: list[float] = []
        for i in range(0, len(sources), self.batch_size):
            src = [normalize(s) for s in sources[i : i + self.batch_size]]
            tgt = [normalize(t) for t in translations[i : i + self.batch_size]]
            enc = self.tokenizer(src, tgt, truncation=True, max_length=self.max_len, padding=True, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with self._lock, torch.inference_mode():
                logits = self.model(**enc).logits.squeeze(-1).float()
            scores.extend(torch.sigmoid(logits).tolist())
        return [round(float(s), 4) for s in scores]

    def score(self, source: str, translation: str) -> float:
        return self.score_batch([source], [translation])[0]

    def evaluate(
        self,
        source: str,
        answer: str,
        source_lang: str | None = None,
        transliterator: "Transliterator | None" = None,
        num_readings: int = 3,
    ) -> dict[str, Any]:
        """Score a learner ``answer`` to translating ``source``.

        source_lang: 'hi' or 'en'; inferred from the script of ``source`` when omitted.
        Returns the score plus how the answer was interpreted (useful feedback in the app).
        """
        if source_lang is None:
            source_lang = "hi" if is_devanagari(source) else "en"
        if source_lang not in ("hi", "en"):
            raise ValueError("source_lang must be 'hi' or 'en'")
        answer = answer.strip()
        result: dict[str, Any] = {"source_lang": source_lang, "target_lang": "en" if source_lang == "hi" else "hi"}
        if not answer:
            return {**result, "score": 0.0, "interpreted_as": "", "romanized": False, "readings": []}

        readings = [answer]
        romanized = source_lang == "en" and is_latin(answer, threshold=0.5)
        if romanized and transliterator is not None:
            readings = transliterator.transliterate(answer, k=num_readings) or [answer]
        scores = self.score_batch([source] * len(readings), readings)
        best = max(range(len(readings)), key=lambda i: scores[i])
        return {
            **result,
            "score": scores[best],
            "interpreted_as": readings[best],
            "romanized": romanized,
            "readings": [{"text": r, "score": s} for r, s in zip(readings, scores)],
        }
