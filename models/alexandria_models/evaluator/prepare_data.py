"""Build the evaluator's training data (CPU only).

Two ingredients:

1. Human scores: WMT23 Quality Estimation, English->Hindi (7000 train / 1000 dev / 1074
   test segments, each rated 0-100 by professional translators). The label is the raw
   mean / 100, which is an absolute "meaning preserved" scale. Pairs are also added
   with source and translation swapped, so the model scores Hindi->English too
   (adequacy is roughly symmetric; WMT has no human hi->en data).
2. Synthetic examples from parallel corpora: correct translations (label ~1), learner-
   style mistakes from ``perturb.py`` (graded labels) and mismatched sentences (~0).
   Human DA labels sit almost entirely between 0.5 and 0.95, so without these the
   model could not tell a wrong answer from a mediocre one.

Output (``--output_dir``): train.jsonl, val_human.jsonl, val_synthetic.jsonl,
test_human.jsonl, data_info.json. Each row: source, translation, label, source_lang, origin.

    python -m alexandria_models.evaluator.prepare_data --output_dir data/evaluator
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

import numpy as np

from ..common.download import download
from ..common.io import get_logger, write_json, write_jsonl
from ..common.sources import DEFAULT_PARALLEL, good_pair, iter_parallel
from ..common.text import normalize
from .perturb import perturb

log = get_logger("evaluator.prepare_data")

WMT23 = "https://raw.githubusercontent.com/WMT-QE-Task/wmt-qe-2023-data/main"
WMT23_TRAIN = f"{WMT23}/task_1/en-hi/train.enhi.df.short.tsv"
WMT23_DEV = f"{WMT23}/task_1/en-hi/dev.enhi.df.short.tsv"
WMT23_TEST_SRC = f"{WMT23}/test_data_2023/task1_sentence_level/en-hi/test.enhi.final.src"
WMT23_TEST_MT = f"{WMT23}/test_data_2023/task1_sentence_level/en-hi/test.enhi.final.mt"
WMT23_TEST_GOLD = f"{WMT23}/gold_labels/hallucinations_gold_T1s_header.tsv"


def _read_tsv(path: Path) -> list[dict]:
    # QUOTE_NONE: the WMT files contain unbalanced double quotes that the default
    # dialect would silently merge into neighbouring fields.
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def load_wmt23(cache_dir: str | None) -> dict[str, list[dict]]:
    """Return {'train','dev','test'} lists of {src_en, mt_hi, label in 0..1}."""
    out: dict[str, list[dict]] = {}
    z_pairs = []
    for split, url in (("train", WMT23_TRAIN), ("dev", WMT23_DEV)):
        rows = []
        for r in _read_tsv(download(url, cache_dir, f"wmt23_enhi_{split}.tsv")):
            try:
                mean, z = float(r["mean"]), float(r["z_mean"])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append({"src_en": r["original"], "mt_hi": r["translation"], "label": min(1.0, max(0.0, mean / 100))})
            if split == "train":
                z_pairs.append((z, mean))
        out[split] = rows
    # About a quarter of dev pairs also occur in train; drop them so validation is honest.
    train_keys = {(r["src_en"], r["mt_hi"]) for r in out["train"]}
    n_dev = len(out["dev"])
    out["dev"] = [r for r in out["dev"] if (r["src_en"], r["mt_hi"]) not in train_keys]
    log.info("WMT23 dev: dropped %d pairs also in train", n_dev - len(out["dev"]))
    # Test gold is released as z-scores only; map z -> raw DA with a linear fit on train.
    zs, raws = np.array([p[0] for p in z_pairs]), np.array([p[1] for p in z_pairs])
    slope, intercept = np.polyfit(zs, raws, 1) if len(zs) > 1 else (15.0, 80.0)
    src = download(WMT23_TEST_SRC, cache_dir, "wmt23_enhi_test.src").read_text(encoding="utf-8").splitlines()
    mt = download(WMT23_TEST_MT, cache_dir, "wmt23_enhi_test.mt").read_text(encoding="utf-8").splitlines()
    gold = [r for r in _read_tsv(download(WMT23_TEST_GOLD, cache_dir, "wmt23_gold_t1.tsv")) if r.get("lp") == "en-hi"]
    gold.sort(key=lambda r: int(r["sid"]))
    test = []
    for r in gold:
        sid = int(r["sid"])
        if sid >= min(len(src), len(mt)):
            continue
        if r["score"] == "hallucination":  # organisers' injected hallucinations: wrong translations
            label = 0.0
        else:
            label = float(np.clip((intercept + slope * float(r["score"])) / 100, 0, 1))
        if (src[sid], mt[sid]) not in train_keys:
            test.append({"src_en": src[sid], "mt_hi": mt[sid], "label": label})
    out["test"] = test
    log.info("WMT23 en-hi: %s", {k: len(v) for k, v in out.items()})
    return out


def human_rows(rows: list[dict], origin: str, both_directions: bool) -> list[dict]:
    out = []
    for r in rows:
        en, hi = normalize(r["src_en"]), normalize(r["mt_hi"])
        out.append({"source": en, "translation": hi, "label": r["label"], "source_lang": "en", "origin": origin})
        if both_directions:
            out.append({"source": hi, "translation": en, "label": r["label"], "source_lang": "hi", "origin": origin + "_swapped"})
    return out


def synthetic_rows(pairs: list[tuple[str, str]], rng: random.Random, negatives_per_pair: int = 2) -> list[dict]:
    """For each (hi, en) pair and each direction: the correct answer, perturbed answers and a mismatch."""
    rows: list[dict] = []
    by_len: dict[int, list[int]] = {}
    for i, (hi, _) in enumerate(pairs):
        by_len.setdefault(min(len(hi.split()), 30), []).append(i)

    def mismatch(i: int) -> int:
        """Another pair of similar length (harder than a random one)."""
        n = min(len(pairs[i][0].split()), 30)
        for d in (0, 1, -1, 2, -2, 3, -3):
            pool = by_len.get(n + d, [])
            if len(pool) > 1:
                j = rng.choice(pool)
                if j != i:
                    return j
        return rng.randrange(len(pairs))

    for i, (hi, en) in enumerate(pairs):
        for src_lang, src, tgt, tgt_lang in (("hi", hi, en, "en"), ("en", en, hi, "hi")):
            rows.append({"source": src, "translation": tgt, "label": 1.0, "source_lang": src_lang, "origin": "synthetic_correct"})
            for _ in range(negatives_per_pair):
                p = perturb(tgt, tgt_lang, rng)
                if p:
                    rows.append({"source": src, "translation": p[0], "label": p[1], "source_lang": src_lang,
                                 "origin": f"synthetic_{p[2]}"})
            if rng.random() < 0.5:
                j = mismatch(i)
                if pairs[j][0] != hi:
                    other = pairs[j][1] if tgt_lang == "en" else pairs[j][0]
                    rows.append({"source": src, "translation": other, "label": 0.02, "source_lang": src_lang,
                                 "origin": "synthetic_mismatch"})
            # Answer in the wrong language (copying the prompt) is not a translation.
            if rng.random() < 0.05:
                rows.append({"source": src, "translation": src, "label": 0.0, "source_lang": src_lang,
                             "origin": "synthetic_copy"})
    return rows


def prepare(args: argparse.Namespace) -> Path:
    rng = random.Random(args.seed)
    out = Path(args.output_dir)
    train: list[dict] = []
    val_human: list[dict] = []
    test_human: list[dict] = []
    info: dict = {}

    if not args.no_wmt:
        try:
            wmt = load_wmt23(args.cache_dir)
            for _ in range(args.human_repeat):
                train += human_rows(wmt["train"], "wmt23", both_directions=True)
            val_human = human_rows(wmt["dev"], "wmt23", both_directions=True)
            test_human = human_rows(wmt["test"], "wmt23_test", both_directions=False)
            info["wmt23"] = {k: len(v) for k, v in wmt.items()}
        except Exception as e:  # noqa: BLE001
            log.warning("could not load WMT23 QE data (%s: %s); training on synthetic data only", type(e).__name__, e)

    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    sources = [s for s in args.parallel_sources.split(",") if s]
    for _, hi, en in iter_parallel(sources, args.max_pairs_per_source, args.hf_token):
        if hi not in seen and good_pair(hi, en):
            seen.add(hi)
            pairs.append((hi, en))
    rng.shuffle(pairs)
    pairs = pairs[: args.synthetic_pairs]
    n_val = max(1, len(pairs) // 20) if len(pairs) >= 20 else 0
    synth_val = synthetic_rows(pairs[:n_val], rng)
    synth_train = synthetic_rows(pairs[n_val:], rng)
    train += synth_train
    rng.shuffle(train)
    info.update(
        synthetic_pairs=len(pairs),
        train=len(train),
        train_origins=dict(Counter(r["origin"].split("_")[0] for r in train)),
        val_human=len(val_human),
        val_synthetic=len(synth_val),
        test_human=len(test_human),
    )
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val_synthetic.jsonl", synth_val)
    if val_human:
        write_jsonl(out / "val_human.jsonl", val_human)
    if test_human:
        write_jsonl(out / "test_human.jsonl", test_human)
    write_json(out / "data_info.json", info)
    log.info("evaluator data: %s", info)
    return out


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--parallel_sources", default=",".join(DEFAULT_PARALLEL))
    p.add_argument("--max_pairs_per_source", type=int, default=20_000)
    p.add_argument("--synthetic_pairs", type=int, default=12_000, help="parallel pairs turned into synthetic examples")
    p.add_argument("--human_repeat", type=int, default=2, help="upsample the human-scored WMT rows")
    p.add_argument("--no_wmt", action="store_true", help="skip downloading WMT23 (synthetic data only)")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--hf_token", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> None:
    prepare(parse_args(argv))


if __name__ == "__main__":
    main()
