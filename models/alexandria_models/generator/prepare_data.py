"""Build the generator's training set from English-Hindi parallel corpora (CPU only).

Every pair is cleaned, scored with the difficulty scorer, and assigned a level. A
weighted reservoir sample then keeps up to ``--per_level`` pairs for each level,
preferring cleaner conversational sources (Tatoeba) over large mined corpora, so all
levels are equally represented. A random share of examples gets a ``Word`` and/or
``Grammar`` control so the trained model can be asked to practise a specific word or
construction.

Version 2: pairs must pass the learner rules (``common/quality.py``: complete sentences,
no headlines, numbers, scripture...), near-duplicates are removed, and with
``--naturalness_dir`` every pair is scored by the naturalness model; pairs below
``--min_naturalness`` are dropped and the rest are sampled in proportion to their score,
so everyday sentences dominate both the training data and the sentence bank.

Output (``--output_dir``): train.jsonl, val.jsonl, sentence_bank.jsonl (every kept pair
with its level, for the backend to serve real sentences), data_config.json.

    python -m alexandria_models.generator.prepare_data \
        --difficulty_dir artifacts/difficulty --output_dir data/generator
"""

from __future__ import annotations

import argparse
import heapq
import random
from collections import Counter
from pathlib import Path

from ..common.io import get_logger, write_json, write_jsonl
from ..common.quality import dedupe_key, learner_issues, tidy_hindi
from ..common.sources import LEARNER_PARALLEL, good_pair, iter_parallel, source_weight
from ..common.text import hindi_words
from ..difficulty.features import CONSTRUCTIONS
from ..difficulty.scorer import DifficultyScorer

log = get_logger("generator.prepare_data")
_STEP = {c.name: c.step for c in CONSTRUCTIONS}
# Grammar controls are only offered for constructions beyond the very basics.
MIN_CONTROL_STEP = 6


def pick_word(tokens: list[str], scorer: DifficultyScorer, rng: random.Random, min_rank: int = 150) -> str | None:
    """A content word worth practising: not a function word, seen in the corpus."""
    cands = [t for t in tokens if len(t) >= 2 and (scorer.freq.rank(t) or 0) > min_rank]
    return rng.choice(cands) if cands else None


def pick_grammar(constructions: list[str], rng: random.Random) -> str | None:
    cands = [c for c in constructions if _STEP.get(c, 0) >= MIN_CONTROL_STEP]
    if not cands:
        return None
    # Prefer the hardest construction: it is what makes the sentence its level.
    cands.sort(key=lambda c: -_STEP[c])
    return cands[0] if rng.random() < 0.7 else rng.choice(cands)


def prepare(args: argparse.Namespace) -> Path:
    rng = random.Random(args.seed)
    out = Path(args.output_dir)
    scorer = DifficultyScorer.load(args.difficulty_dir)
    L = scorer.config.num_levels
    keep = int(args.per_level * (1 + args.val_fraction))
    heaps: list[list] = [[] for _ in range(L)]  # min-heaps of (key, tiebreak, row)
    seen: set[str] = set()
    stats: Counter = Counter()

    sources = [s for s in args.sources.split(",") if s]
    cands: list[dict] = []
    for src, hi, en in iter_parallel(sources, args.max_pairs_per_source, args.hf_token):
        stats["read"] += 1
        hi = tidy_hindi(hi)
        key = dedupe_key(hi)
        if key in seen or not good_pair(hi, en, max_words=args.max_words):
            stats["filtered"] += 1
            continue
        seen.add(key)
        if not args.no_learner_filter:
            issues = learner_issues(hi, en, max_words=args.max_words, stems=scorer.verb_stems)
            if issues:
                stats.update(f"rule:{x}" for x in issues)
                stats["rule_rejected"] += 1
                continue
        cands.append({"hi": hi, "en": en, "source": src})
        if len(cands) % 200_000 == 0:
            log.info("%d pairs read, %d candidates", stats["read"], len(cands))
    log.info("%d candidates after filtering (%s)", len(cands), dict(stats))

    if args.naturalness_dir:
        from ..naturalness.score import NaturalnessModel

        nat = NaturalnessModel.load(args.naturalness_dir, device=args.device)
        scores = nat.score_batch([c["hi"] for c in cands])
        for c, s in zip(cands, scores):
            c["naturalness"] = s
        before = len(cands)
        cands = [c for c in cands if c["naturalness"] >= args.min_naturalness]
        stats["naturalness_rejected"] = before - len(cands)
        log.info("naturalness: kept %d of %d (threshold %.2f)", len(cands), before, args.min_naturalness)

    for i, c in enumerate(cands):
        a = scorer.analyze(c["hi"])
        if a["n_words"] == 0:
            continue
        row = {**c, "level": a["level"], "score": a["score"], "constructions": a["constructions"]}
        # Weighted reservoir sampling (Efraimidis-Spirakis): key = u^(1/w).
        weight = source_weight(c["source"]) * (0.25 + c.get("naturalness", 0.75))
        key = rng.random() ** (1.0 / weight)
        h = heaps[a["level"] - 1]
        if len(h) < keep:
            heapq.heappush(h, (key, i, row))
        elif key > h[0][0]:
            heapq.heapreplace(h, (key, i, row))

    train_rows, val_rows, bank = [], [], []
    for level, h in enumerate(heaps, start=1):
        rows = [r for _, _, r in h]
        rng.shuffle(rows)
        n_val = max(1, round(len(rows) * args.val_fraction / (1 + args.val_fraction))) if len(rows) >= 3 else 0
        for j, r in enumerate(rows):
            bank.append({k: r[k] for k in ("hi", "en", "level", "score", "constructions", "source", "naturalness")
                         if k in r})
            ex = {"hi": r["hi"], "en": r["en"], "level": r["level"], "source": r["source"]}
            if rng.random() < args.word_control_prob:
                w = pick_word(hindi_words(r["hi"]), scorer, rng)
                if w:
                    ex["word"] = w
            if rng.random() < args.grammar_control_prob:
                g = pick_grammar(r["constructions"], rng)
                if g:
                    ex["grammar"] = g
            (val_rows if j < n_val else train_rows).append(ex)
        log.info("level %2d: %d pairs (%s)", level, len(rows), dict(Counter(r["source"] for r in rows)))
        if len(rows) < args.per_level // 4 and level > 3:
            log.warning("level %d has only %d pairs; add more parallel data or lower --per_level", level, len(rows))

    rng.shuffle(train_rows)
    write_jsonl(out / "train.jsonl", train_rows)
    write_jsonl(out / "val.jsonl", val_rows)
    write_jsonl(out / "sentence_bank.jsonl", sorted(bank, key=lambda r: (r["level"], r["score"])))
    write_json(
        out / "data_config.json",
        {
            "num_levels": L,
            "train_examples": len(train_rows),
            "val_examples": len(val_rows),
            "per_level": [len(h) for h in heaps],
            "stats": dict(stats),
            "sources": sources,
            "naturalness_dir": args.naturalness_dir,
            "min_naturalness": args.min_naturalness if args.naturalness_dir else None,
            "grammar_controls": [c.name for c in CONSTRUCTIONS if c.step >= MIN_CONTROL_STEP],
        },
    )
    log.info("wrote %d train / %d val examples to %s", len(train_rows), len(val_rows), out)
    return out


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--difficulty_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--sources", default=",".join(LEARNER_PARALLEL), help="registry names and/or local files")
    p.add_argument("--naturalness_dir", default=None, help="trained naturalness model (artifacts_v2/naturalness)")
    p.add_argument("--min_naturalness", type=float, default=0.5)
    p.add_argument("--no_learner_filter", action="store_true", help="keep pairs that fail the learner rules (v1)")
    p.add_argument("--device", default="auto")
    p.add_argument("--max_pairs_per_source", type=int, default=400_000)
    p.add_argument("--per_level", type=int, default=2000)
    p.add_argument("--val_fraction", type=float, default=0.03)
    p.add_argument("--max_words", type=int, default=20)
    p.add_argument("--word_control_prob", type=float, default=0.3)
    p.add_argument("--grammar_control_prob", type=float, default=0.3)
    p.add_argument("--hf_token", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> None:
    prepare(parse_args(argv))


if __name__ == "__main__":
    main()
