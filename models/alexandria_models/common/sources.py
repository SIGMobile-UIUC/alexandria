"""Registry of the corpora used for training, with streaming loaders.

Monolingual Hindi corpora feed word frequencies (difficulty scorer, transliteration
context). English-Hindi parallel corpora feed the generator and the evaluator.
Every source is optional: if one fails to load (network, gating, renamed on the Hub),
it is logged and skipped so a training run never dies on a single corpus.

Besides the named sources, any local file can be used as a source by passing its path:
  *.txt                       one Hindi sentence or paragraph per line (monolingual)
  *.tsv / *.csv               parallel; header must contain columns named 'hi' and 'en'
  *.jsonl                     parallel ({"hi": ..., "en": ...}) or monolingual ({"text": ...})
"""

from __future__ import annotations

import csv
import io
import itertools
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from .download import download
from .io import get_logger
from .text import clean_sentence_ok, normalize, split_sentences

log = get_logger("sources")


# --------------------------------------------------------------------------- helpers
def _hf_stream(path: str, name: str | None = None, split: str = "train", data_dir: str | None = None,
               data_files=None, token: str | None = None):
    from datasets import load_dataset

    kwargs = {"split": split, "streaming": True, "token": token}
    if name:
        kwargs["name"] = name
    if data_dir:
        kwargs["data_dir"] = data_dir
    if data_files:
        kwargs["data_files"] = data_files
    return load_dataset(path, **kwargs)


def _get(row: dict, dotted: str):
    cur = row
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


# --------------------------------------------------------------------------- monolingual
@dataclass(frozen=True)
class MonoSource:
    name: str
    description: str
    license: str
    load: Callable[[str | None], Iterator[str]]  # token -> iterator of raw documents


def _hf_text(path: str, name: str | None = None, split: str = "train", data_dir: str | None = None,
             field_name: str = "text"):
    def load(token: str | None) -> Iterator[str]:
        for row in _hf_stream(path, name, split, data_dir, token=token):
            text = row.get(field_name)
            if text:
                yield text

    return load


MONO_SOURCES: dict[str, MonoSource] = {
    "sangraha_verified": MonoSource(
        "sangraha_verified", "AI4Bharat Sangraha, human-verified Hindi web/PDF/transcript text", "CC-BY-4.0",
        _hf_text("ai4bharat/sangraha", data_dir="verified/hin"),
    ),
    "sangraha_synthetic": MonoSource(
        "sangraha_synthetic", "AI4Bharat Sangraha, English Wikipedia translated to Hindi", "CC-BY-4.0",
        _hf_text("ai4bharat/sangraha", data_dir="synthetic/hin_Deva"),
    ),
    "fineweb2": MonoSource(
        "fineweb2", "HuggingFaceFW FineWeb-2 Hindi (Devanagari) web text", "ODC-By-1.0",
        _hf_text("HuggingFaceFW/fineweb-2", name="hin_Deva"),
    ),
    "indiccorp_v2": MonoSource(
        "indiccorp_v2", "AI4Bharat IndicCorp v2 Hindi news/web sentences", "CC0",
        _hf_text("ai4bharat/IndicCorpV2", name="indiccorp_v2", split="hin_Deva"),
    ),
    "wikipedia": MonoSource(
        "wikipedia", "Hindi Wikipedia (Nov 2023 dump)", "CC-BY-SA-4.0",
        _hf_text("wikimedia/wikipedia", name="20231101.hi"),
    ),
}
DEFAULT_MONO = ["sangraha_verified", "fineweb2", "indiccorp_v2", "wikipedia"]


def _local_mono(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    if row.get("text") or row.get("hi"):
                        yield row.get("text") or row.get("hi")
        else:
            for line in f:
                if line.strip():
                    yield line


def iter_hindi_sentences(
    sources: list[str], max_sentences_per_source: int, token: str | None = None, max_words: int = 40
) -> Iterator[tuple[str, str]]:
    """Yield (source_name, clean Hindi sentence) from each source, up to the cap each."""
    for name in sources:
        if name in MONO_SOURCES:
            docs = MONO_SOURCES[name].load(token)
        elif Path(name).exists():
            docs = _local_mono(Path(name))
        else:
            log.warning("unknown monolingual source %r (not a registry name or a file); skipping", name)
            continue
        n = 0
        try:
            for doc in docs:
                for para in doc.split("\n"):
                    for sent in split_sentences(normalize(para)):
                        if clean_sentence_ok(sent, lang="hi", max_words=max_words):
                            yield name, sent
                            n += 1
                            if n >= max_sentences_per_source:
                                break
                    if n >= max_sentences_per_source:
                        break
                if n >= max_sentences_per_source:
                    break
        except Exception as e:  # noqa: BLE001 - one bad corpus must not stop the run
            log.warning("source %s failed after %d sentences: %s: %s", name, n, type(e).__name__, e)
        log.info("source %s: %d sentences", name, n)


# --------------------------------------------------------------------------- parallel
@dataclass(frozen=True)
class ParallelSource:
    name: str
    description: str
    license: str
    load: Callable[[str | None], Iterator[tuple[str, str]]]  # token -> (hi, en) pairs
    weight: float = 1.0  # sampling preference: clean conversational data > noisy mined data
    gated: bool = False  # needs a Hugging Face token + accepting the dataset's terms
    tags: tuple[str, ...] = field(default_factory=tuple)
    cap: int | None = None  # per-source maximum, on top of the run-wide cap
    skip: int = 0  # leading rows to skip (e.g. a block of software strings at the start)


def _hf_pairs(path: str, hi: str, en: str, name: str | None = None, split: str = "train",
              data_dir: str | None = None, data_files=None):
    def load(token: str | None) -> Iterator[tuple[str, str]]:
        for row in _hf_stream(path, name, split, data_dir, data_files, token=token):
            h, e = _get(row, hi), _get(row, en)
            if h and e:
                yield h, e

    return load


def _anki_hin_eng(token: str | None) -> Iterator[tuple[str, str]]:
    """ManyThings.org Anki export of Tatoeba; hin.txt lines are 'English<TAB>Hindi<TAB>attribution'."""
    zpath = download("https://www.manythings.org/anki/hin-eng.zip")
    with zipfile.ZipFile(zpath) as z, z.open("hin.txt") as f:
        for line in io.TextIOWrapper(f, encoding="utf-8"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                yield parts[1], parts[0]


def _tatoeba_exports(token: str | None) -> Iterator[tuple[str, str]]:
    """Tatoeba weekly exports: Hindi sentences joined to English through hin-eng links."""
    import bz2

    base = "https://downloads.tatoeba.org/exports/per_language"

    def rows(name: str):
        with bz2.open(download(f"{base}/{name}"), "rt", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    yield parts

    hin = {r[0]: r[2] for r in rows("hin/hin_sentences.tsv.bz2") if len(r) >= 3}
    links: dict[str, str] = {}
    for h_id, e_id, *_ in rows("hin/hin-eng_links.tsv.bz2"):
        if h_id in hin and h_id not in links:  # one English translation per Hindi sentence
            links[h_id] = e_id
    wanted = set(links.values())
    eng = {r[0]: r[2] for r in rows("eng/eng_sentences.tsv.bz2") if len(r) >= 3 and r[0] in wanted}
    for h_id, e_id in links.items():
        if e_id in eng:
            yield hin[h_id], eng[e_id]


def _bpcc(subdir: str):
    """AI4Bharat BPCC subset, read as a TSV file (gated: accept the terms on the Hub first)."""

    def load(token: str | None) -> Iterator[tuple[str, str]]:
        from datasets import load_dataset

        ds = load_dataset(
            "csv",
            data_files={"train": f"hf://datasets/ai4bharat/BPCC/{subdir}/hin_Deva.tsv"},
            delimiter="\t",
            quoting=3,  # csv.QUOTE_NONE: stray quotes would otherwise merge rows
            split="train",
            streaming=True,
            token=token,
        )
        for row in ds:
            if row.get("src_lang") == "eng_Latn" and row.get("tgt_lang") == "hin_Deva" and row.get("src") and row.get("tgt"):
                yield row["tgt"], row["src"]

    return load


def _in22(path: str):
    def load(token: str | None) -> Iterator[tuple[str, str]]:
        for row in _hf_stream(path, "default", "test", token=token):
            if row.get("hin_Deva") and row.get("eng_Latn"):
                yield row["hin_Deva"], row["eng_Latn"]

    return load


PARALLEL_SOURCES: dict[str, ParallelSource] = {
    "tatoeba": ParallelSource(
        "tatoeba", "Tatoeba Hindi-English pairs (weekly exports): short everyday sentences",
        "CC-BY-2.0-FR", _tatoeba_exports, weight=3.0, tags=("conversational",),
    ),
    "tatoeba_anki": ParallelSource(
        "tatoeba_anki", "Tatoeba pairs via the ManyThings Anki export (fallback if the exports are unreachable)",
        "CC-BY-2.0-FR", _anki_hin_eng, weight=3.0, tags=("conversational",),
    ),
    "bpcc_daily": ParallelSource(
        "bpcc_daily", "BPCC-H-Daily: human-translated everyday sentences", "CC-BY-4.0",
        _bpcc("daily"), weight=2.5, gated=True, tags=("conversational",),
    ),
    "bpcc_massive": ParallelSource(
        "bpcc_massive", "BPCC MASSIVE: short virtual-assistant requests and questions", "CC0",
        _bpcc("massive"), weight=1.5, gated=True, tags=("conversational",), cap=20_000,
    ),
    "bpcc_wiki": ParallelSource(
        "bpcc_wiki", "BPCC-H-Wiki: human-translated Wikipedia sentences", "CC-BY-4.0",
        _bpcc("wiki"), weight=1.5, gated=True,
    ),
    "opus100": ParallelSource(
        "opus100", "OPUS-100 en-hi (subtitles, software strings, religious text; noisy)", "varies (OPUS)",
        _hf_pairs("Helsinki-NLP/opus-100", hi="translation.hi", en="translation.en", name="en-hi"),
        weight=0.6, cap=100_000,
    ),
    "iitb": ParallelSource(
        "iitb", "IIT Bombay English-Hindi corpus (government, news, Wikipedia)", "CC-BY-NC-4.0",
        _hf_pairs("cfilt/iitb-english-hindi", hi="translation.hi", en="translation.en"), weight=1.0,
        skip=250_000,  # the split opens with ~240k GNOME/KDE UI strings
    ),
    "samanantar": ParallelSource(
        "samanantar", "AI4Bharat Samanantar en-hi (large, mined; noisier)", "CC-BY-NC-4.0",
        _hf_pairs("ai4bharat/samanantar", hi="tgt", en="src", name="hi"), weight=0.5, gated=True,
    ),
    # Benchmarks: keep them out of training and use them for evaluation.
    "in22_conv": ParallelSource(
        "in22_conv", "IN22-Conv: 1503 conversational sentences (evaluation set)", "CC-BY-4.0",
        _in22("ai4bharat/IN22-Conv"), gated=True, tags=("eval",),
    ),
    "in22_gen": ParallelSource(
        "in22_gen", "IN22-Gen: 1024 general-domain sentences (evaluation set)", "CC-BY-4.0",
        _in22("ai4bharat/IN22-Gen"), gated=True, tags=("eval",),
    ),
}
DEFAULT_PARALLEL = ["tatoeba", "tatoeba_anki", "bpcc_daily", "bpcc_massive", "bpcc_wiki", "opus100", "iitb", "samanantar"]


def _local_parallel(path: Path) -> Iterator[tuple[str, str]]:
    with path.open(encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    if row.get("hi") and row.get("en"):
                        yield row["hi"], row["en"]
        else:
            reader = csv.DictReader(f, delimiter="\t" if path.suffix == ".tsv" else ",")
            for row in reader:
                if row.get("hi") and row.get("en"):
                    yield row["hi"], row["en"]


def iter_parallel(
    sources: list[str], max_pairs_per_source: int, token: str | None = None
) -> Iterator[tuple[str, str, str]]:
    """Yield (source_name, hindi, english), normalized but not yet quality-filtered."""
    for name in sources:
        cap = max_pairs_per_source
        if name in PARALLEL_SOURCES:
            src = PARALLEL_SOURCES[name]
            pairs = itertools.islice(src.load(token), src.skip, None)
            cap = min(cap, src.cap or cap)
        elif Path(name).exists():
            pairs = _local_parallel(Path(name))
        else:
            log.warning("unknown parallel source %r (not a registry name or a file); skipping", name)
            continue
        n = 0
        try:
            for hi, en in pairs:
                yield name, normalize(hi), normalize(en)
                n += 1
                if n >= cap:
                    break
        except Exception as e:  # noqa: BLE001
            log.warning("source %s failed after %d pairs: %s: %s", name, n, type(e).__name__, e)
        log.info("source %s: %d pairs", name, n)


def source_weight(name: str) -> float:
    src = PARALLEL_SOURCES.get(name)
    return src.weight if src else 2.0  # user-provided local files are presumably curated


def good_pair(hi: str, en: str, max_words: int = 40) -> bool:
    """Both sides clean and of plausible relative length."""
    if not (clean_sentence_ok(hi, lang="hi", max_words=max_words) and clean_sentence_ok(en, lang="en", max_words=max_words)):
        return False
    nh, ne = len(hi.split()), len(en.split())
    return 0.4 <= ne / max(1, nh) <= 2.5 or (nh <= 3 and ne <= 4)
