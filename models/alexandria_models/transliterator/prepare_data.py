"""Build word pairs for the transliterator (CPU only).

Sources:
  * AI4Bharat Aksharantar, Hindi (~1.3M romanized/Devanagari word pairs; ``hin.zip`` on the
    Hugging Face Hub, JSON lines with keys "english word" / "native word" / "score").
  * Google Dakshina, Hindi lexicons (~44K pairs, several romanizations per word, written
    by native speakers) and romanized Wikipedia sentences for a sentence-level dev set.
    The 2 GB tarball is streamed and reading stops right after the Hindi files.
  * Any local TSV with ``roman<TAB>native`` lines (``--extra``).

Words that occur in any dev/test set are removed from training (Aksharantar's training
split itself contains Dakshina words), so reported accuracy is honest.

Output (``--output_dir``): train.tsv, val.tsv, test.tsv (roman<TAB>native),
sentences_dev.tsv (roman sentence<TAB>native sentence), data_info.json.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from ..common.io import get_logger, write_json
from ..common.text import normalize

log = get_logger("transliterator.prepare_data")
ROMAN_RE = re.compile(r"^[a-z]+$")
NATIVE_RE = re.compile(r"^[ऀ-ॣॱ-ॿ]+$")
DAKSHINA_URL = "https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar"
DAKSHINA_HI = "dakshina_dataset_v1.0/hi/"


def clean_pair(roman: str, native: str, max_len: int = 30) -> tuple[str, str] | None:
    roman = roman.strip().lower()
    native = normalize(native.strip())
    if not ROMAN_RE.match(roman) or not NATIVE_RE.match(native):
        return None
    if len(roman) > max_len or len(native) > max_len:
        return None
    return roman, native


def load_aksharantar(min_score: float | None, token: str | None) -> dict[str, list[tuple[str, str]]]:
    from huggingface_hub import hf_hub_download

    zpath = hf_hub_download(repo_id="ai4bharat/Aksharantar", filename="hin.zip", repo_type="dataset", token=token)
    out: dict[str, list[tuple[str, str]]] = {}
    with zipfile.ZipFile(zpath) as zf:
        members = {m.rsplit("/", 1)[-1]: m for m in zf.namelist()}
        for split in ("train", "valid", "test"):
            name = members.get(f"hin_{split}.json")
            if not name:
                continue
            rows = []
            with zf.open(name) as fh:
                for line in io.TextIOWrapper(fh, encoding="utf-8"):
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    score = r.get("score")
                    # Mined pairs carry a model log-prob; low scores are likely wrong pairs.
                    if split == "train" and min_score is not None and score is not None and score < min_score:
                        continue
                    p = clean_pair(r.get("english word", ""), r.get("native word", ""))
                    if p:
                        rows.append(p)
            out[split] = rows
    log.info("Aksharantar: %s", {k: len(v) for k, v in out.items()})
    return out


def load_dakshina(cache_dir: Path) -> dict[str, list]:
    """Stream the Dakshina tarball, keep only Hindi lexicons + romanized sentences."""
    target = cache_dir / "dakshina_hi"
    wanted = {
        "lexicons/hi.translit.sampled.train.tsv", "lexicons/hi.translit.sampled.dev.tsv",
        "lexicons/hi.translit.sampled.test.tsv", "romanized/hi.romanized.rejoined.dev.roman.txt",
        "romanized/hi.romanized.rejoined.dev.native.txt",
    }
    if not all((target / Path(w).name).exists() for w in wanted):
        target.mkdir(parents=True, exist_ok=True)
        log.info("streaming Dakshina (stops after the Hindi files, ~600 MB)")
        req = urllib.request.Request(DAKSHINA_URL, headers={"User-Agent": "alexandria-models"})
        seen_hi = False
        with urllib.request.urlopen(req, timeout=300) as resp, tarfile.open(fileobj=resp, mode="r|") as tf:
            for m in tf:
                if m.name.startswith(DAKSHINA_HI):
                    seen_hi = True
                    rel = m.name[len(DAKSHINA_HI):]
                    if m.isfile() and rel in wanted:
                        (target / Path(rel).name).write_bytes(tf.extractfile(m).read())
                elif seen_hi:
                    break

    def lex(split: str) -> list[tuple[str, str]]:
        rows = []
        with (target / f"hi.translit.sampled.{split}.tsv").open(encoding="utf-8", newline="") as f:
            for parts in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                if len(parts) >= 2:
                    p = clean_pair(parts[1], parts[0])  # file order: native, roman, count
                    if p:
                        rows.append(p)
        return rows

    roman_lines = (target / "hi.romanized.rejoined.dev.roman.txt").read_text(encoding="utf-8").splitlines()
    native_lines = (target / "hi.romanized.rejoined.dev.native.txt").read_text(encoding="utf-8").splitlines()
    out = {
        "train": lex("train"),
        "dev": lex("dev"),
        "test": lex("test"),
        "sentences_dev": [(r, normalize(n)) for r, n in zip(roman_lines, native_lines)],
    }
    log.info("Dakshina: %s", {k: len(v) for k, v in out.items()})
    return out


def read_local(path: str) -> list[tuple[str, str]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                p = clean_pair(parts[0], parts[1])
                if p:
                    rows.append(p)
    return rows


def write_pairs(path: Path, pairs) -> None:
    with path.open("w", encoding="utf-8") as f:
        for a, b in pairs:
            f.write(f"{a}\t{b}\n")


def prepare(args: argparse.Namespace) -> Path:
    rng = random.Random(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir) if args.cache_dir else out / "_downloads"

    train: list[tuple[str, str]] = []
    val: list[tuple[str, str]] = []
    test: list[tuple[str, str]] = []
    sentences_dev: list[tuple[str, str]] = []
    info: dict = {}
    if not args.no_aksharantar:
        try:
            ak = load_aksharantar(args.min_score, args.hf_token)
            train += ak.get("train", [])
            val += ak.get("valid", [])
            test += ak.get("test", [])
            info["aksharantar"] = {k: len(v) for k, v in ak.items()}
        except Exception as e:  # noqa: BLE001
            log.warning("Aksharantar unavailable (%s: %s)", type(e).__name__, e)
    if not args.no_dakshina:
        try:
            dk = load_dakshina(cache)
            train += dk["train"]
            val += dk["dev"]
            test += dk["test"]
            sentences_dev = dk["sentences_dev"]
            info["dakshina"] = {k: len(v) for k, v in dk.items()}
        except Exception as e:  # noqa: BLE001
            log.warning("Dakshina unavailable (%s: %s)", type(e).__name__, e)
    for path in args.extra:
        rows = read_local(path)
        rng.shuffle(rows)
        n_hold = len(rows) // 10 if len(rows) >= 20 else 0
        val += rows[:n_hold]
        train += rows[n_hold:]
        info[f"local:{path}"] = len(rows)

    if not train:
        raise SystemExit("no training pairs found; check network access or pass --extra pairs.tsv")
    if not val:  # e.g. only local data: hold out by native word
        natives = sorted({n for _, n in train})
        rng.shuffle(natives)
        held = set(natives[: max(1, len(natives) // 20)])
        val = [p for p in train if p[1] in held]
        train = [p for p in train if p[1] not in held]

    held_out = {n for _, n in val} | {n for _, n in test}
    before = len(train)
    train = sorted(set(p for p in train if p[1] not in held_out))
    rng.shuffle(train)
    info.update(train=len(train), removed_for_leakage=before - len(train), val=len(set(val)), test=len(set(test)))
    write_pairs(out / "train.tsv", train)
    write_pairs(out / "val.tsv", sorted(set(val)))
    write_pairs(out / "test.tsv", sorted(set(test)))
    if sentences_dev:
        write_pairs(out / "sentences_dev.tsv", sentences_dev)
    write_json(out / "data_info.json", info)
    log.info("transliteration data: %s", info)
    return out


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--extra", nargs="*", default=[], help="local roman<TAB>native files")
    p.add_argument("--min_score", type=float, default=None,
                   help="drop mined Aksharantar pairs whose model score is below this log-prob")
    p.add_argument("--no_aksharantar", action="store_true")
    p.add_argument("--no_dakshina", action="store_true")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--hf_token", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> None:
    prepare(parse_args(argv))


if __name__ == "__main__":
    main()
