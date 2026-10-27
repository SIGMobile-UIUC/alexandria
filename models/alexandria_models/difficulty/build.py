"""Build the difficulty scorer from corpora (step 1 of the pipeline; CPU only).

1. Stream Hindi sentences from monolingual corpora and count word (and word-pair)
   frequencies. Word rarity drives the lexical part of the difficulty score; the
   word-pair counts let the transliterator pick between readings using context.
2. Mine extra verb stems from the counts (better grammar detection).
3. Fit the level thresholds on a sample of the Hindi side of the parallel corpora,
   which are the sentences the generator will learn from, taking an equal share from
   each source so one big corpus cannot dominate.

Version 2 (the defaults) uses 50 absolute levels and also counts words in spoken-
style text (subtitles, Tatoeba, everyday conversation), see ``scorer.py``. Counting the
big web corpora takes most of the time; ``--reuse_from`` takes those counts from an
earlier build instead.

Output (``--output_dir``): scorer.json, word_freq.tsv.gz, spoken_freq.tsv.gz,
bigram_freq.tsv.gz, verb_stems.txt, build_info.json.

    python -m alexandria_models.difficulty.build --output_dir artifacts_v2/difficulty \
        --reuse_from artifacts/difficulty
"""

from __future__ import annotations

import argparse
import gzip
import random
import shutil
from collections import Counter
from pathlib import Path

from ..common.io import get_logger, write_json
from ..common.quality import dedupe_key, is_learner_sentence
from ..common.sources import (
    DEFAULT_MONO,
    DEFAULT_SPOKEN,
    LEARNER_PARALLEL,
    MONO_SOURCES,
    iter_hindi_sentences,
    iter_parallel,
)
from ..common.text import clean_sentence_ok, hindi_words
from .features import derive_verb_stems
from .scorer import FREQ_FILE, STEMS_FILE, DifficultyConfig, DifficultyScorer, FrequencyTable

log = get_logger("difficulty.build")
BIGRAM_FILE = "bigram_freq.tsv.gz"


def _prune(counter: Counter, keep: int) -> None:
    """Keep only the ``keep`` most frequent entries (bounds memory on Colab)."""
    if len(counter) > keep:
        for k, _ in counter.most_common()[keep:]:
            del counter[k]


def count_corpus(sentences, max_bigrams: int = 1_000_000) -> tuple[Counter, Counter, dict[str, int]]:
    words: Counter = Counter()
    bigrams: Counter = Counter()
    per_source: Counter = Counter()
    for i, (src, sent) in enumerate(sentences):
        toks = hindi_words(sent)
        words.update(toks)
        bigrams.update(zip(toks, toks[1:]))
        per_source[src] += 1
        if max_bigrams and len(bigrams) > 3 * max_bigrams:
            _prune(bigrams, max_bigrams)
        if (i + 1) % 500_000 == 0:
            log.info("%d sentences counted (%d word types)", i + 1, len(words))
    if max_bigrams:
        _prune(bigrams, max_bigrams)
    return words, bigrams, dict(per_source)


def save_bigrams(path: Path, bigrams: Counter, min_count: int = 2) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for (a, b), c in bigrams.most_common():
            if c < min_count:
                break
            f.write(f"{a}\t{b}\t{c}\n")


def count_spoken(sources: list[str], max_per_source: int, token: str | None) -> Counter:
    """Word counts from spoken-style text: monolingual sources and/or the Hindi side of
    conversational parallel corpora."""
    words: Counter = Counter()
    mono = [s for s in sources if s in MONO_SOURCES or (Path(s).exists() and not s.endswith(".tsv"))]
    par = [s for s in sources if s not in mono]
    for _, sent in iter_hindi_sentences(mono, max_per_source, token):
        words.update(hindi_words(sent))
    for _, hi, _ in iter_parallel(par, max_per_source, token):
        words.update(hindi_words(hi))
    return words


def reference_sample(sources: list[str], per_source: int, token: str | None, seed: int,
                     learner_only: bool = False) -> list[str]:
    """Hindi sentences for fitting levels: equal-size reservoir sample per parallel source."""
    rng = random.Random(seed)
    pools: dict[str, list[str]] = {}
    seen_counts: Counter = Counter()
    seen: set[str] = set()
    for src, hi, en in iter_parallel(sources, max_pairs_per_source=per_source * 5, token=token):
        if not clean_sentence_ok(hi, lang="hi"):
            continue
        if learner_only and not is_learner_sentence(hi, en):
            continue
        key = dedupe_key(hi)
        if key in seen:
            continue
        seen.add(key)
        seen_counts[src] += 1
        pool = pools.setdefault(src, [])
        if len(pool) < per_source:
            pool.append(hi)
        else:
            j = rng.randrange(seen_counts[src])
            if j < per_source:
                pool[j] = hi
    for src, pool in pools.items():
        log.info("level reference: %d sentences from %s", len(pool), src)
    return [s for pool in pools.values() for s in pool]


def wordfreq_fallback() -> Counter:
    """Hindi frequencies from the `wordfreq` package, if installed (no download needed)."""
    try:
        from wordfreq import get_frequency_dict
    except ImportError:
        return Counter()
    freqs = get_frequency_dict("hi", wordlist="small")
    return Counter({w: max(1, int(f * 1e9)) for w, f in freqs.items()})


def build(args: argparse.Namespace) -> DifficultyScorer:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mono = [s for s in args.mono_sources.split(",") if s]
    parallel = [s for s in args.parallel_sources.split(",") if s]

    if args.reuse_from:
        prev = Path(args.reuse_from)
        log.info("reusing web word/bigram counts and verb stems from %s", prev)
        freq = FrequencyTable.load(prev / FREQ_FILE)
        stems = frozenset((prev / STEMS_FILE).read_text(encoding="utf-8").split()) if (prev / STEMS_FILE).exists() else frozenset()
        if (prev / BIGRAM_FILE).exists() and prev.resolve() != out.resolve():
            shutil.copy(prev / BIGRAM_FILE, out / BIGRAM_FILE)
        bigrams, per_source = Counter(), {"reused_from": str(prev)}
    else:
        words, bigrams, per_source = count_corpus(
            iter_hindi_sentences(mono, args.max_sentences_per_source, args.hf_token), args.max_bigrams
        )
        if sum(words.values()) < 10_000:
            log.warning("only %d words counted from corpora; adding wordfreq's Hindi list as a fallback", sum(words.values()))
            words.update(wordfreq_fallback())
        freq = FrequencyTable.from_counter(words)
        stems = derive_verb_stems(freq.counts, min_infinitive_count=args.min_stem_count)
    log.info("%d word types, %d mined verb stems", len(freq), len(stems))

    spoken_sources = [s for s in args.spoken_sources.split(",") if s]
    spoken = FrequencyTable.from_counter(count_spoken(spoken_sources, args.max_spoken_per_source, args.hf_token))
    log.info("spoken frequencies: %d word types (%d tokens)", len(spoken), sum(spoken.counts.values()))
    if spoken.counts and sum(spoken.counts.values()) < 50_000:
        log.warning("spoken corpus is small (%d tokens); its ranks will be noisy", sum(spoken.counts.values()))

    cfg = DifficultyConfig(num_levels=args.num_levels, rank_cap=args.rank_cap, scheme=args.scheme,
                           spoken_weight=args.spoken_weight)
    scorer = DifficultyScorer(freq, cfg, verb_stems=stems, spoken=spoken)
    ref = reference_sample(parallel, args.reference_per_source, args.hf_token, args.seed,
                           learner_only=args.scheme == "absolute")
    if args.scheme == "quantile" and len(ref) < args.num_levels * 50:
        log.warning("only %d parallel sentences for fitting; topping up with monolingual sentences", len(ref))
        extra = [s for _, s in iter_hindi_sentences(mono, args.reference_per_source, args.hf_token)]
        ref += extra
    summaries = scorer.fit_levels(ref)
    for s in summaries:
        log.info("level %2d: n=%d mean_words=%.1f e.g. %s", s.level, s.count, s.mean_words, (s.examples or [""])[0])

    scorer.save(out)
    if not args.reuse_from:
        save_bigrams(out / BIGRAM_FILE, bigrams)
    write_json(
        out / "build_info.json",
        {
            "mono_sources": per_source,
            "parallel_reference_sentences": len(ref),
            "word_types": len(freq),
            "spoken_word_types": len(spoken),
            "scheme": args.scheme,
            "level_counts": [s.count for s in summaries],
            "bigram_types": len(bigrams),
            "mined_verb_stems": len(stems),
            "num_levels": args.num_levels,
        },
    )
    log.info("difficulty scorer saved to %s", out)
    return scorer


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mono_sources", default=",".join(DEFAULT_MONO),
                   help="comma-separated registry names and/or local files")
    p.add_argument("--parallel_sources", default=",".join(LEARNER_PARALLEL),
                   help="sentences used to summarize (and, for --scheme quantile, fit) the levels")
    p.add_argument("--spoken_sources", default=",".join(DEFAULT_SPOKEN),
                   help="spoken-style text for word ranks; empty string to use web counts only")
    p.add_argument("--max_spoken_per_source", type=int, default=2_000_000)
    p.add_argument("--spoken_weight", type=float, default=0.6)
    p.add_argument("--scheme", choices=["absolute", "quantile"], default="absolute")
    p.add_argument("--reuse_from", default=None, help="earlier difficulty dir whose web counts to reuse")
    p.add_argument("--max_sentences_per_source", type=int, default=1_000_000)
    p.add_argument("--reference_per_source", type=int, default=20_000)
    p.add_argument("--num_levels", type=int, default=50)
    p.add_argument("--rank_cap", type=int, default=30_000)
    p.add_argument("--max_bigrams", type=int, default=1_000_000)
    p.add_argument("--min_stem_count", type=int, default=20)
    p.add_argument("--hf_token", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> None:
    build(parse_args(argv))


if __name__ == "__main__":
    main()
