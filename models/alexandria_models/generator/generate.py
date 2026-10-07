"""Generate leveled Hindi sentences (with English translations) from a trained generator.

The backend uses ``SentenceGenerator``; the CLI pre-generates a sentence bank, which is
the recommended way to serve sentences from a CPU-only server (generation with a
1-4B model on CPU takes seconds per sentence)::

    python -m alexandria_models.generator.generate \
        --generator_dir artifacts/generator --difficulty_dir artifacts/difficulty \
        --per_level 100 --out artifacts/generator/generated_bank.jsonl
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..common.io import get_logger, read_json, resolve_device, write_jsonl
from ..common.text import clean_sentence_ok, normalize
from ..common.text import tokenize
from .prompts import bos_prefix, build_prompt, parse_completion

if TYPE_CHECKING:
    from ..difficulty.scorer import DifficultyScorer
    from ..evaluator.score import TranslationEvaluator

log = get_logger("generator.generate")
CONFIG_FILE = "generator_config.json"


class SentenceGenerator:
    def __init__(
        self,
        model,
        tokenizer,
        num_levels: int,
        scorer: "DifficultyScorer | None" = None,
        evaluator: "TranslationEvaluator | None" = None,
        device: str = "cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.num_levels = num_levels
        self.scorer = scorer
        self.evaluator = evaluator
        self.device = device
        self._lock = threading.Lock()
        self.max_seq_len = 192

    @classmethod
    def load(
        cls,
        generator_dir: str | Path,
        scorer: "DifficultyScorer | None" = None,
        evaluator: "TranslationEvaluator | None" = None,
        device: str = "auto",
        dtype: str = "auto",
    ) -> "SentenceGenerator":
        """Load ``generator_dir/merged`` (or base model + ``adapter`` if not merged)."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        d = Path(generator_dir)
        cfg = read_json(d / CONFIG_FILE)
        device = resolve_device(device)
        if dtype == "auto":  # bf16 halves CPU RAM (a 3B model is ~13 GB in fp32)
            torch_dtype = torch.float16 if device == "cuda" else torch.bfloat16
        else:
            torch_dtype = getattr(torch, dtype)
        if cfg.get("merged") and (d / "merged" / "config.json").exists():
            tok = AutoTokenizer.from_pretrained(d / "merged")
            model = AutoModelForCausalLM.from_pretrained(d / "merged", dtype=torch_dtype)
        else:
            from peft import PeftModel

            tok = AutoTokenizer.from_pretrained(d / "adapter")
            base = AutoModelForCausalLM.from_pretrained(cfg["base_model"], dtype=torch_dtype)
            model = PeftModel.from_pretrained(base, d / "adapter").merge_and_unload()
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"  # decoder-only batched generation pads on the left
        model.to(device).eval()
        gen = cls(model, tok, int(cfg["num_levels"]), scorer, evaluator, device)
        gen.max_seq_len = int(cfg.get("max_seq_len", 192))
        return gen

    def _sample(self, prompts: list[str], temperature: float, top_p: float, max_new_tokens: int) -> list[str]:
        import torch

        tok = self.tokenizer
        bos = bos_prefix(tok)
        enc = [bos + tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
        longest = max(len(e) for e in enc)
        input_ids = torch.tensor([[tok.pad_token_id] * (longest - len(e)) + e for e in enc], device=self.device)
        attention = torch.tensor([[0] * (longest - len(e)) + [1] * len(e) for e in enc], device=self.device)
        with self._lock, torch.inference_mode():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        new = out[:, longest:]
        # Drop outputs cut off by max_new_tokens: their translation would be truncated.
        finished = (new == tok.eos_token_id).any(dim=1).tolist()
        texts = tok.batch_decode(new, skip_special_tokens=True)
        return [t if done else "" for t, done in zip(texts, finished)]

    def generate(
        self,
        level: int,
        n: int = 5,
        word: str | None = None,
        grammar: str | None = None,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_new_tokens: int | None = None,
        oversample: int = 3,
        max_rounds: int = 3,
        level_tolerance: int = 2,
        min_translation_score: float = 0.5,
        exclude: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return up to ``n`` distinct, validated sentences near ``level``.

        Candidates are sampled ``n * oversample`` at a time and kept only if they parse,
        look like clean Hindi / English, contain ``word`` (if requested), measure within
        ``level_tolerance`` of the requested level (when a difficulty scorer is loaded)
        and the English matches the Hindi (when an evaluator is loaded).
        """
        if not 1 <= level <= self.num_levels:
            raise ValueError(f"level must be in 1..{self.num_levels}")
        word = normalize(word) if word else None
        max_new_tokens = max_new_tokens or self.max_seq_len
        prompt = build_prompt(level, self.num_levels, word, grammar)
        seen = set(exclude or ())
        results: list[dict[str, Any]] = []
        for _ in range(max_rounds):
            texts = []
            total = n * oversample
            for start in range(0, total, 16):  # chunks, so the lock is released between them
                texts += self._sample([prompt] * min(16, total - start), temperature, top_p, max_new_tokens)
            cands = []
            for t in texts:
                parsed = parse_completion(t)
                if not parsed:
                    continue
                hi, en = normalize(parsed[0]), normalize(parsed[1])
                if hi in seen or not clean_sentence_ok(hi, lang="hi") or not clean_sentence_ok(en, lang="en"):
                    continue
                if word and word not in tokenize(hi):
                    continue
                seen.add(hi)
                item: dict[str, Any] = {"hindi": hi, "english": en, "requested_level": level}
                if self.scorer is not None:
                    analysis = self.scorer.analyze(hi)
                    if abs(analysis["level"] - level) > level_tolerance:
                        continue
                    item.update(level=analysis["level"], difficulty=analysis)
                cands.append(item)
            if self.evaluator is not None and cands:
                scores = self.evaluator.score_batch([c["hindi"] for c in cands], [c["english"] for c in cands])
                for c, s in zip(cands, scores):
                    c["translation_score"] = s
                cands = [c for c in cands if c["translation_score"] >= min_translation_score]
            results.extend(cands)
            if len(results) >= n:
                break
        return results[:n]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generator_dir", required=True)
    p.add_argument("--difficulty_dir", default=None)
    p.add_argument("--evaluator_dir", default=None)
    p.add_argument("--per_level", type=int, default=50)
    p.add_argument("--levels", default=None, help="comma-separated subset, default all")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    scorer = evaluator = None
    if args.difficulty_dir:
        from ..difficulty.scorer import DifficultyScorer

        scorer = DifficultyScorer.load(args.difficulty_dir)
    if args.evaluator_dir:
        from ..evaluator.score import TranslationEvaluator

        evaluator = TranslationEvaluator.load(args.evaluator_dir, device=args.device)
    gen = SentenceGenerator.load(args.generator_dir, scorer, evaluator, device=args.device)
    levels = [int(x) for x in args.levels.split(",")] if args.levels else range(1, gen.num_levels + 1)
    rows = []
    for level in levels:
        got = gen.generate(level, n=args.per_level, max_rounds=6)
        log.info("level %d: %d sentences", level, len(got))
        # Same schema as sentence_bank.jsonl so SentenceBank can serve it.
        rows.extend(
            {"hi": r["hindi"], "en": r["english"], "level": r.get("level", r["requested_level"]),
             "score": r.get("difficulty", {}).get("score"),
             "constructions": r.get("difficulty", {}).get("constructions", []), "source": "generated"}
            for r in got
        )
    write_jsonl(args.out, rows)


if __name__ == "__main__":
    main()
