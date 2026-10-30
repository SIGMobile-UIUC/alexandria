"""Where does the evaluator go wrong? Score learner-style answers by error type.

For each sentence pair and both directions it scores the reference translation, one
example of every perturbation in ``perturb.py`` (wrong tense, flipped negation, dropped
words, harmless contractions...) and an unrelated sentence, then reports per error type:

  mean        average score the model gives
  expected    the score that error type should get
  mae         mean absolute difference from expected
  below_ref   how often the answer scores lower than the correct reference (should be
              ~100% for real mistakes, ~50% for harmless variants)

A model can have a fine correlation overall and still, say, barely punish a flipped
negation; this shows that. Runs on CPU in a few minutes.

    python -m alexandria_models.evaluator.diagnose --evaluator_dir artifacts/evaluator \\
        --pairs data_v2/generator/sentence_bank.jsonl --out reports/evaluator_v1.json
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..common.io import get_logger, read_jsonl, write_json
from ..common.sources import LEARNER_PARALLEL, iter_parallel
from .perturb import ENGLISH_PERTURBATIONS, HINDI_PERTURBATIONS
from .score import TranslationEvaluator

log = get_logger("evaluator.diagnose")


def load_pairs(spec: str, n: int, seed: int, token: str | None = None) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    if spec.endswith(".jsonl") and Path(spec).exists():
        rows = [(r["hi"], r["en"]) for r in read_jsonl(spec)]
    else:
        rows = [(hi, en) for _, hi, en in iter_parallel([s for s in spec.split(",") if s], n * 5, token)]
    rng.shuffle(rows)
    return rows[:n]


def build_cases(pairs: list[tuple[str, str]], rng: random.Random) -> list[dict]:
    cases = []
    for i, (hi, en) in enumerate(pairs):
        for src_lang, src, ref, table in (("hi", hi, en, ENGLISH_PERTURBATIONS), ("en", en, hi, HINDI_PERTURBATIONS)):
            group = f"{i}-{src_lang}"
            cases.append({"group": group, "dir": src_lang, "kind": "reference", "source": src, "answer": ref, "expected": 1.0})
            for name, fn in table:
                out = fn(ref, rng)
                if out and out[0].strip() and out[0] != ref:
                    cases.append({"group": group, "dir": src_lang, "kind": name, "source": src,
                                  "answer": out[0], "expected": out[1]})
            j = rng.randrange(len(pairs))
            if j != i:
                other = pairs[j][1] if src_lang == "hi" else pairs[j][0]
                cases.append({"group": group, "dir": src_lang, "kind": "unrelated", "source": src,
                              "answer": other, "expected": 0.0})
    return cases


def summarize(cases: list[dict]) -> dict:
    ref = {c["group"]: c["score"] for c in cases if c["kind"] == "reference"}
    by: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for c in cases:
        by[(c["dir"], c["kind"])].append(c)
        by[("both", c["kind"])].append(c)
    report = {}
    for (d, kind), cs in sorted(by.items()):
        s = np.array([c["score"] for c in cs])
        e = np.array([c["expected"] for c in cs])
        below = [c["score"] < ref[c["group"]] for c in cs if c["kind"] != "reference" and c["group"] in ref]
        report.setdefault(d, {})[kind] = {
            "n": len(cs), "mean": round(float(s.mean()), 3), "expected": round(float(e.mean()), 3),
            "mae": round(float(np.abs(s - e).mean()), 3),
            "below_ref": round(float(np.mean(below)), 3) if below else None,
        }
    return report


def print_report(report: dict) -> None:
    for d, title in (("hi", "source Hindi -> English answer"), ("en", "source English -> Hindi answer")):
        if d not in report:
            continue
        print(f"\n{title}\n{'error type':22} {'n':>5} {'mean':>6} {'expect':>7} {'mae':>6} {'below_ref':>9}")
        for kind, m in sorted(report[d].items(), key=lambda kv: -kv[1]["expected"]):
            b = "" if m["below_ref"] is None else f"{m['below_ref']:.0%}"
            flag = "  <-- check" if m["mae"] > 0.25 else ""
            print(f"{kind:22} {m['n']:>5} {m['mean']:>6.2f} {m['expected']:>7.2f} {m['mae']:>6.2f} {b:>9}{flag}")


def diagnose(args: argparse.Namespace) -> dict:
    rng = random.Random(args.seed)
    ev = TranslationEvaluator.load(args.evaluator_dir, device=args.device)
    pairs = load_pairs(args.pairs, args.n_pairs, args.seed, args.hf_token)
    if not pairs:
        raise SystemExit(f"no sentence pairs from {args.pairs}")
    cases = build_cases(pairs, rng)
    scores = ev.score_batch([c["source"] for c in cases], [c["answer"] for c in cases])
    for c, s in zip(cases, scores):
        c["score"] = s
    report = {"evaluator_dir": str(args.evaluator_dir), "pairs": len(pairs), "by_error_type": summarize(cases)}
    print_report(report["by_error_type"])
    if args.out:
        write_json(args.out, report)
        worst = sorted((c for c in cases if c["kind"] != "reference"), key=lambda c: -abs(c["score"] - c["expected"]))
        write_json(Path(args.out).with_suffix(".worst.json"), worst[:100])
        log.info("report written to %s (and the 100 worst cases next to it)", args.out)
    return report


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--evaluator_dir", required=True)
    p.add_argument("--pairs", default=",".join(LEARNER_PARALLEL),
                   help="sentence_bank.jsonl (hi/en rows) or comma-separated parallel sources")
    p.add_argument("--n_pairs", type=int, default=300)
    p.add_argument("--out", default=None, help="write the report here (JSON)")
    p.add_argument("--device", default="auto")
    p.add_argument("--hf_token", default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> None:
    diagnose(parse_args(argv))


if __name__ == "__main__":
    main()
