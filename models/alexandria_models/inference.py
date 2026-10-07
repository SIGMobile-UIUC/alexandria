"""One object that loads every trained model from an artifacts directory.

This is the only module the backend needs::

    from alexandria_models.inference import AlexandriaModels

    models = AlexandriaModels.load("artifacts")          # once, at startup
    models.sentences(level=5, n=3)                        # leveled Hindi + English
    models.evaluate("मैं घर जा रहा हूँ।", "I am going home")  # -> {"score": 0.97, ...}
    models.evaluate("I am going home", "main ghar ja raha hoon")  # romanized Hindi works
    models.difficulty("यह पत्र मेरे दादाजी द्वारा लिखा गया था।")
    models.transliterate("kya haal hai", k=3)

Expected layout (each part is optional; missing parts are skipped)::

    artifacts/
      difficulty/       difficulty.build
      generator/        generator.train (merged/ + sentence_bank.jsonl)
      evaluator/        evaluator.train
      transliterator/   transliterator.train
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .common.io import get_logger, read_jsonl
from .common.text import normalize, tokenize

log = get_logger("inference")
COMPONENTS = ("difficulty", "generator", "evaluator", "transliterator")


class ModelNotLoaded(RuntimeError):
    """A feature was requested whose model is not loaded (the API maps this to 503)."""


class SentenceBank:
    """Real corpus sentences grouped by level; the fast, always-grammatical source."""

    def __init__(self, rows: Iterable[dict]):
        self.by_level: dict[int, list[dict]] = defaultdict(list)
        for r in rows:
            hi, en = r.get("hi") or r.get("hindi"), r.get("en") or r.get("english")
            if not hi or not en or r.get("level") is None:
                continue
            r = {**r, "hi": hi, "en": en, "_tokens": frozenset(tokenize(hi))}
            self.by_level[int(r["level"])].append(r)

    @classmethod
    def load(cls, *paths: str | Path) -> "SentenceBank":
        rows: list[dict] = []
        for p in paths:
            if Path(p).exists():
                rows.extend(read_jsonl(p))
        return cls(rows)

    def __len__(self) -> int:
        return sum(len(v) for v in self.by_level.values())

    def sample(self, level: int, n: int, word: str | None = None, grammar: str | None = None,
               exclude: set[str] | None = None, rng: random.Random | None = None) -> list[dict]:
        rng = rng or random
        pool = [
            r for r in self.by_level.get(level, [])
            if (not word or word in r["_tokens"])
            and (not grammar or grammar in r.get("constructions", []))
            and (not exclude or r["hi"] not in exclude)
        ]
        picked = rng.sample(pool, min(n, len(pool)))
        return [
            {"hindi": r["hi"], "english": r["en"], "level": r["level"], "requested_level": level,
             "constructions": r.get("constructions", []), "source": r.get("source", "corpus")}
            for r in picked
        ]


class AlexandriaModels:
    def __init__(self, scorer=None, generator=None, evaluator=None, transliterator=None, bank: SentenceBank | None = None):
        self.scorer = scorer
        self.generator = generator
        self.evaluator = evaluator
        self.transliterator = transliterator
        self.bank = bank or SentenceBank([])

    @classmethod
    def load(cls, artifacts_dir: str | Path, components: Iterable[str] | None = None, device: str = "auto") -> "AlexandriaModels":
        """Load every available component. By default the generator is loaded only when a
        GPU is used: on CPU it is slow and needs ~7 GB RAM, so serve the pre-generated
        bank instead (pass components=COMPONENTS to force it)."""
        from .common.io import resolve_device

        root = Path(artifacts_dir)
        if components is None:
            gpu = resolve_device(device) == "cuda"
            components = [c for c in COMPONENTS if gpu or c != "generator"]
        components = set(components)
        unknown = components - set(COMPONENTS)
        if unknown:
            raise ValueError(f"unknown components {sorted(unknown)}; choose from {COMPONENTS}")
        scorer = generator = evaluator = translit = None
        if "difficulty" in components and (root / "difficulty" / "scorer.json").exists():
            from .difficulty.scorer import DifficultyScorer

            scorer = DifficultyScorer.load(root / "difficulty")
            log.info("loaded difficulty scorer (%d levels)", scorer.config.num_levels)
        if "evaluator" in components and (root / "evaluator" / "model" / "config.json").exists():
            from .evaluator.score import TranslationEvaluator

            evaluator = TranslationEvaluator.load(root / "evaluator", device=device)
            log.info("loaded evaluator")
        if "transliterator" in components and (root / "transliterator" / "model.pt").exists():
            from .transliterator.transliterate import Transliterator

            translit = Transliterator.load(root / "transliterator", difficulty_dir=root / "difficulty", device=device)
            log.info("loaded transliterator")
        if "generator" in components and (root / "generator" / "generator_config.json").exists():
            from .generator.generate import SentenceGenerator

            generator = SentenceGenerator.load(root / "generator", scorer, evaluator, device=device)
            log.info("loaded generator")
        bank = SentenceBank.load(root / "generator" / "sentence_bank.jsonl", root / "generator" / "generated_bank.jsonl")
        log.info("sentence bank: %d sentences", len(bank))
        return cls(scorer, generator, evaluator, translit, bank)

    # ------------------------------------------------------------------ info
    @property
    def num_levels(self) -> int:
        if self.scorer is not None:
            return self.scorer.config.num_levels
        if self.generator is not None:
            return self.generator.num_levels
        return max(self.bank.by_level, default=0)

    def levels(self) -> list[dict[str, Any]]:
        if self.scorer is not None and self.scorer.level_summaries:
            return self.scorer.level_summaries
        return [{"level": lv, "count": len(rows)} for lv, rows in sorted(self.bank.by_level.items())]

    def status(self) -> dict[str, Any]:
        return {
            "difficulty": self.scorer is not None,
            "generator": self.generator is not None,
            "evaluator": self.evaluator is not None,
            "transliterator": self.transliterator is not None,
            "sentence_bank": len(self.bank),
            "num_levels": self.num_levels,
        }

    # ------------------------------------------------------------------ features
    def sentences(self, level: int, n: int = 5, word: str | None = None, grammar: str | None = None,
                  mode: str = "auto", exclude: Iterable[str] | None = None) -> list[dict[str, Any]]:
        """Hindi sentences (with English) at ``level``.

        mode: 'bank' = real corpus sentences only (fast), 'generate' = model only,
        'auto' = bank first, topped up with generated sentences if the bank runs short.
        """
        if not 1 <= level <= max(1, self.num_levels):
            raise ValueError(f"level must be between 1 and {self.num_levels}")
        if mode not in ("auto", "bank", "generate"):
            raise ValueError("mode must be 'auto', 'bank' or 'generate'")
        word = normalize(word) if word else None
        excluded = set(exclude or ())
        out: list[dict[str, Any]] = []
        if mode in ("auto", "bank"):
            out = self.bank.sample(level, n, word, grammar, excluded)
        if len(out) < n and mode in ("auto", "generate"):
            if self.generator is None:
                if mode == "generate":
                    raise ModelNotLoaded("generator model is not loaded")
            else:
                excluded |= {s["hindi"] for s in out}
                out += self.generator.generate(level, min(n - len(out), 10), word=word, grammar=grammar, exclude=excluded)
        return out

    def difficulty(self, sentence: str) -> dict[str, Any]:
        if self.scorer is None:
            raise ModelNotLoaded("difficulty scorer is not loaded")
        return self.scorer.analyze(sentence)

    def evaluate(self, source: str, answer: str, source_lang: str | None = None) -> dict[str, Any]:
        if self.evaluator is None:
            raise ModelNotLoaded("evaluator model is not loaded")
        from .common.text import is_devanagari, is_latin

        if self.transliterator is None and source_lang != "hi" and not is_devanagari(source) and is_latin(answer, 0.5):
            raise ModelNotLoaded("transliterator is not loaded, so romanized Hindi answers cannot be scored")
        return self.evaluator.evaluate(source, answer, source_lang, self.transliterator)

    def transliterate(self, text: str, k: int = 3) -> list[str]:
        if self.transliterator is None:
            raise ModelNotLoaded("transliterator model is not loaded")
        return self.transliterator.transliterate(text, k=k)
