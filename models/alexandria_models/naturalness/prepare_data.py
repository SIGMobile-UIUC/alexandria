"""Build training data for the naturalness model (CPU only).

The naturalness model answers "would a teacher hand this sentence to a learner?": a
complete, natural, everyday Hindi sentence scores high; headlines, fragments, scripture,
voice-assistant commands, translation-ese and stiff news prose score low.

Labels come from two places:

1. Weak labels from where a sentence comes from (no cost): sentences from conversational
   corpora (Tatoeba, BPCC-Daily) that pass the learner rules -> 0.85; other corpora that
   pass -> 0.4; anything failing the rules (fragments, headlines, digits...) -> 0.05.
   These teach "everyday vs. news" but are noisy: plenty of news sentences are fine.
2. A large open LLM rating a sample of the same sentences 1-5 (``label_llm.py``, GPU).
   ``to_judge.jsonl`` written here is that sample, stratified by source.

Output (``--output_dir``): candidates.jsonl (weak labels), to_judge.jsonl, data_info.json.

    python -m alexandria_models.naturalness.prepare_data --output_dir data_v2/naturalness
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

from ..common.io import get_logger, write_json, write_jsonl
from ..common.quality import dedupe_key, learner_issues, tidy_hindi
from ..common.sources import LEARNER_PARALLEL, PARALLEL_SOURCES, iter_parallel
from ..common.text import clean_sentence_ok, normalize

log = get_logger("naturalness.prepare_data")

WEAK_CONVERSATIONAL = 0.85
WEAK_OTHER = 0.4
WEAK_REJECTED = 0.05
DEFAULT_SOURCES = LEARNER_PARALLEL + ["opus100", "bpcc_massive"]  # the last two mostly as negatives


def weak_label(source: str, issues: list[str]) -> float:
    if issues:
        return WEAK_REJECTED
    src = PARALLEL_SOURCES.get(source)
    return WEAK_CONVERSATIONAL if src and "conversational" in src.tags else WEAK_OTHER


def prepare(args: argparse.Namespace) -> Path:
    rng = random.Random(args.seed)
    out = Path(args.output_dir)
    rows: list[dict] = []
    seen: set[str] = set()
    stats: Counter = Counter()
    per_source: dict[str, int] = Counter()
    sources = [s for s in args.sources.split(",") if s]
    for src, hi, en in iter_parallel(sources, args.max_pairs_per_source, args.hf_token):
        if not clean_sentence_ok(hi, lang="hi", max_words=40):
            continue
        hi = tidy_hindi(hi)
        key = dedupe_key(hi)
        if key in seen:
            continue
        seen.add(key)
        issues = learner_issues(hi, en)
        # Rule failures are easy negatives; keep only a share so they don't dominate.
        if issues and rng.random() > args.rejected_keep:
            stats["rejected_dropped"] += 1
            continue
        stats["rejected_kept" if issues else "passed"] += 1
        per_source[src] += 1
        rows.append({"id": len(rows), "hi": hi, "en": normalize(en), "source": src,
                     "label": weak_label(src, issues), "issues": issues})

    rng.shuffle(rows)
    # Stratified sample for the LLM judge: equal share per source, mostly rule-passing sentences.
    by_src: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if not r["issues"] or rng.random() < 0.1:
            by_src[r["source"]].append(r)
    share = max(1, args.judge_sample // max(1, len(by_src)))
    judge = [r for pool in by_src.values() for r in pool[:share]]
    if len(judge) < args.judge_sample:  # small sources leave room: top up from the rest
        picked = {r["id"] for r in judge}
        judge += [r for r in rows if r["id"] not in picked][: args.judge_sample - len(judge)]
    rng.shuffle(judge)

    write_jsonl(out / "candidates.jsonl", rows)
    write_jsonl(out / "to_judge.jsonl", [{k: r[k] for k in ("id", "hi", "en", "source")} for r in judge])
    info = {"candidates": len(rows), "to_judge": len(judge), "per_source": dict(per_source), **stats}
    write_json(out / "data_info.json", info)
    log.info("naturalness data: %s", info)
    return out


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--sources", default=",".join(DEFAULT_SOURCES))
    p.add_argument("--max_pairs_per_source", type=int, default=60_000)
    p.add_argument("--rejected_keep", type=float, default=0.15, help="share of rule-failing sentences to keep")
    p.add_argument("--judge_sample", type=int, default=20_000)
    p.add_argument("--hf_token", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> None:
    prepare(parse_args(argv))


if __name__ == "__main__":
    main()
