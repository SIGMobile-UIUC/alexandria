"""A spreadsheet of learner-style answers for your team to score, and scoring the
evaluator against it. This is the test that matters for the app: WMT measures machine
translation of news, not learners translating everyday sentences.

1. Make the sheet (200 answers by default)::

       python -m alexandria_models.evaluator.label_sheet make \\
           --pairs artifacts_v2/generator/sentence_bank.jsonl --out label_sheet.csv

   Open it in Google Sheets / Excel and fill the ``score`` column 0-4 using RUBRIC
   below (also written next to the CSV). You may also replace an ``answer`` with a
   translation a learner really wrote (romanized Hindi is fine). Leave ``id`` alone.

2. Score the evaluator against your labels::

       python -m alexandria_models.evaluator.label_sheet score --sheet label_sheet.csv \\
           --evaluator_dir artifacts_v2/evaluator --transliterator_dir artifacts/transliterator \\
           --export data_v2/evaluator/test_learner.jsonl

   ``--export`` saves the labelled rows so ``evaluator.train`` reports on them too.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..common.io import get_logger, read_jsonl, write_json, write_jsonl
from ..common.quality import is_learner_sentence
from .diagnose import load_pairs
from .perturb import perturb

log = get_logger("evaluator.label_sheet")

RUBRIC = """Score each answer 0-4 for how well it translates the source sentence:

4  Correct and natural. Small spelling slips (हूं/हूँ, missing nukta) and missing
   punctuation or capitals are fine. A different but correct wording is also a 4.
3  Meaning fully correct, with a minor grammar slip: gender or number agreement,
   a missing article, slightly odd word order.
2  Mostly understandable, but an error changes part of the meaning: wrong tense,
   wrong postposition, a missing word that matters.
1  Major meaning error: wrong person (I/you/he), half the sentence missing, words
   scrambled so the meaning is unclear.
0  Wrong or opposite meaning (negation flipped), unrelated sentence, or not a
   translation at all (e.g. the source copied).

Score the meaning a learner conveyed, not whether the wording matches a reference.
"""
COLUMNS = ["id", "direction", "source", "answer", "score", "notes"]


def make(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    pool = load_pairs(args.pairs, args.n * 20, args.seed, args.hf_token)
    pool = [(hi, en) for hi, en in pool if is_learner_sentence(hi, en)]
    pairs = pool[: args.n // 2]
    if len(pairs) < args.n // 2:
        log.warning("only %d usable sentence pairs; the sheet will have %d rows", len(pairs), 2 * len(pairs))
    rows, key = [], []
    for i, (hi, en) in enumerate(pairs):
        for k in range(2):
            src_lang = rng.choice(["hi", "en"])
            src, ref, ans_lang = (hi, en, "en") if src_lang == "hi" else (en, hi, "hi")
            r = rng.random()
            if r < 0.25:
                answer, kind, expected = ref, "reference", 1.0
            elif r < 0.9:
                p = perturb(ref, ans_lang, rng)
                answer, kind, expected = (p[0], p[2], p[1]) if p else (ref, "reference", 1.0)
            else:
                j = rng.randrange(len(pairs))
                answer = pairs[j][1] if ans_lang == "en" else pairs[j][0]
                kind, expected = "unrelated", 0.0
            rid = f"{i:03d}{'ab'[k]}"
            rows.append({"id": rid, "direction": "Hindi -> English" if src_lang == "hi" else "English -> Hindi",
                         "source": src, "answer": answer, "score": "", "notes": ""})
            key.append({"id": rid, "source_lang": src_lang, "kind": kind, "expected": expected, "original_answer": answer})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:  # BOM so Excel reads Devanagari
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    write_jsonl(out.with_suffix(".key.jsonl"), key)
    out.with_name("RUBRIC.txt").write_text(RUBRIC, encoding="utf-8")
    log.info("wrote %d rows to %s (rubric in %s)", len(rows), out, out.with_name("RUBRIC.txt"))
    print(RUBRIC)


def read_sheet(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = []
        for r in csv.DictReader(f):
            try:
                s = float(str(r.get("score", "")).strip())
            except ValueError:
                continue
            if 0 <= s <= 4 and r.get("source") and r.get("answer") is not None:
                rows.append({**r, "label": s / 4})
    return rows


def score(args: argparse.Namespace) -> dict:
    from .score import TranslationEvaluator
    from .train import correlation_metrics

    rows = read_sheet(args.sheet)
    if not rows:
        raise SystemExit(f"no scored rows in {args.sheet} (fill the 'score' column with 0-4)")
    key_path = Path(args.sheet).with_suffix(".key.jsonl")
    key = {k["id"]: k for k in read_jsonl(key_path)} if key_path.exists() else {}
    translit = None
    if args.transliterator_dir:
        from ..transliterator.transliterate import Transliterator

        translit = Transliterator.load(args.transliterator_dir, difficulty_dir=args.difficulty_dir, device=args.device)
    ev = TranslationEvaluator.load(args.evaluator_dir, device=args.device)
    preds = []
    for r in rows:
        lang = "hi" if r["direction"].lower().startswith("hindi") else "en"
        preds.append(ev.evaluate(r["source"], r["answer"], lang, translit)["score"])
        r["source_lang"] = lang
        k = key.get(r["id"], {})
        edited = k and k.get("original_answer") != r["answer"]
        r["kind"] = "learner_written" if edited or not k else k["kind"]
    preds_a, labels = np.array(preds), np.array([r["label"] for r in rows])
    overall = correlation_metrics(preds_a, labels)
    by_kind = defaultdict(list)
    for r, p in zip(rows, preds):
        by_kind[r["kind"]].append((p, r["label"]))
    per_kind = {k: {"n": len(v), "model_mean": round(float(np.mean([p for p, _ in v])), 3),
                    "human_mean": round(float(np.mean([h for _, h in v])), 3)} for k, v in sorted(by_kind.items())}
    report = {"n": len(rows), "overall": overall, "by_kind": per_kind}
    print(f"\n{len(rows)} human-scored answers: " + ", ".join(f"{k}={v:.3f}" for k, v in overall.items()))
    print(f"{'kind':22} {'n':>4} {'model':>6} {'human':>6}")
    for k, m in per_kind.items():
        print(f"{k:22} {m['n']:>4} {m['model_mean']:>6.2f} {m['human_mean']:>6.2f}")
    if args.out:
        write_json(args.out, report)
    if args.export:
        write_jsonl(args.export, [{"source": r["source"], "translation": r["answer"], "label": r["label"],
                                   "source_lang": r["source_lang"], "origin": f"learner_sheet_{r['kind']}"} for r in rows])
        log.info("exported %d labelled rows to %s", len(rows), args.export)
    return report


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--pairs", required=True, help="sentence_bank.jsonl or comma-separated parallel sources")
    m.add_argument("--out", required=True)
    m.add_argument("--n", type=int, default=200)
    m.add_argument("--hf_token", default=None)
    m.add_argument("--seed", type=int, default=7)
    s = sub.add_parser("score")
    s.add_argument("--sheet", required=True)
    s.add_argument("--evaluator_dir", required=True)
    s.add_argument("--transliterator_dir", default=None, help="needed to score romanized Hindi answers")
    s.add_argument("--difficulty_dir", default=None)
    s.add_argument("--out", default=None)
    s.add_argument("--export", default=None, help="write the labelled rows as evaluator test data (jsonl)")
    s.add_argument("--device", default="auto")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    make(args) if args.cmd == "make" else score(args)


if __name__ == "__main__":
    main()
