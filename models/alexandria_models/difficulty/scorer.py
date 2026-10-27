"""Continuous difficulty score for Hindi sentences and its mapping to discrete levels.

score = w_grammar * grammar + w_lexical * lexical + w_length * length + w_orthography * orthography

  grammar      hardest construction present (its teaching step / max step), plus a little
               for how many different constructions the sentence combines
  lexical      how rare the words are in the corpus: log-rank rarity, averaging the mean
               word with the two rarest words (one rare word is enough to make a sentence hard)
  length       number of words, saturating at ``length_cap``
  orthography  conjunct consonants per word (स्वास्थ्य, प्रतिनिधित्व are hard to read)

Every component is in [0, 1]. Two ways to turn this into levels (``config.scheme``):

  "quantile"  (v1) equal-frequency bins of the score over a reference sample. Level 1 is
              then just "the easiest 5% of the corpus", which for news-heavy corpora is
              still too hard for a beginner.
  "absolute"  (v2, default) each sentence gets the level at which a learner could read
              it: the highest of a vocabulary level (how far down a spoken-Hindi frequency
              list its rarest words are), a grammar level (when its hardest construction
              is taught) and a length level, plus a little for the other two. Level 1 is
              two or three of the ~200 most common words with है, e.g. यह घर है।
              The anchors below are on a 50-level scale and stretch to ``num_levels``.

Word rarity can blend web frequencies with spoken frequencies (subtitles, Tatoeba,
everyday conversation), so सरकार ("government") counts as rarer than पानी ("water")
even though news uses it constantly.
"""

from __future__ import annotations

import bisect
import gzip
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from ..common.io import read_json, write_json
from . import features as F

SCORER_FILE = "scorer.json"
FREQ_FILE = "word_freq.tsv.gz"
SPOKEN_FREQ_FILE = "spoken_freq.tsv.gz"
STEMS_FILE = "verb_stems.txt"

# Level (on a 50-level scale) at which each construction is introduced.
GRAMMAR_LEVELS_50 = {
    "copula_present": 1, "possessive_pronoun": 2, "question_word": 3, "negation": 4, "postposition": 5,
    "copula_past": 8, "imperative": 9, "habitual_present": 10, "progressive": 12, "future": 14,
    "dative_subject": 16, "simple_past": 18, "perfective_ne": 20, "habitual_past": 22, "ability_sakna": 23,
    "obligation_chahiye": 24, "compulsion": 26, "conjunctive_participle": 27, "compound_verb": 28,
    "complement_ki": 30, "conditional": 31, "completive_chukna": 32, "vaala": 33, "relative_correlative": 35,
    "subjunctive": 36, "participle_hua": 38, "ability_paana": 39, "passive": 40, "causative": 42,
    "counterfactual": 44, "urdu_connective": 46, "formal_register": 48,
}
# (word rank, level): the ~200 most common words are level 1, rank 1000 is about level 16.
VOCAB_ANCHORS_50 = [(200, 1), (500, 8), (1000, 16), (2000, 24), (4000, 32), (8000, 40), (16000, 48), (25000, 50)]
# (number of words, level)
LENGTH_ANCHORS_50 = [(3, 1), (4, 3), (5, 5), (6, 7), (8, 11), (10, 15), (12, 19), (15, 25), (20, 35), (25, 45), (30, 50)]


def _interp_log(x: float, anchors: list[tuple[float, float]]) -> float:
    """Piecewise-linear in log(x) through the anchors; flat beyond both ends."""
    if x <= anchors[0][0]:
        return anchors[0][1]
    for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
    return anchors[-1][1]


def _interp(x: float, anchors: list[tuple[float, float]]) -> float:
    if x <= anchors[0][0]:
        return anchors[0][1]
    for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return anchors[-1][1]


@dataclass
class DifficultyConfig:
    num_levels: int = 20
    w_grammar: float = 0.35
    w_lexical: float = 0.35
    w_length: float = 0.20
    w_orthography: float = 0.10
    rank_cap: int = 30000  # words rarer than this rank all count as maximally rare
    length_cap: int = 25  # sentences with this many words get length = 1.0
    conjuncts_per_word_cap: float = 0.5
    scheme: str = "quantile"  # "absolute" for v2 (see module docstring)
    spoken_weight: float = 0.6  # share of spoken (vs web) frequency in a word's effective rank
    secondary_weight: float = 0.15  # how much the two non-dominant dimensions add (absolute)


# Inflectional endings stripped to group word forms into families (पीता/पीना/पीती -> पी).
_INFLECTIONS = sorted(
    ("ियों", "ाओं", "ुओं", "ओं", "ों", "ियाँ", "ियां", "ाएँ", "ाएं", "एँ", "एं", "ें", "तीं", "ता", "ती", "ते",
     "ना", "नी", "ने", "कर", "ा", "ी", "े", "ीं"),
    key=len, reverse=True,
)


def lemma_key(word: str) -> str:
    """Crude stem: strip the longest inflectional ending, keeping at least two characters."""
    for suf in _INFLECTIONS:
        if word.endswith(suf) and len(word) - len(suf) >= 2:
            return word[: -len(suf)]
    return word


class FrequencyTable:
    """Word counts with 1-based frequency ranks (rank 1 = most frequent)."""

    def __init__(self, counts: Mapping[str, int] | None = None):
        self.counts: dict[str, int] = dict(counts or {})
        self._ranks: dict[str, int] | None = None
        self._family_ranks: dict[str, int] | None = None

    def __len__(self) -> int:
        return len(self.counts)

    @property
    def ranks(self) -> dict[str, int]:
        if self._ranks is None:
            ordered = sorted(self.counts.items(), key=lambda kv: (-kv[1], kv[0]))
            self._ranks = {w: i + 1 for i, (w, _) in enumerate(ordered)}
        return self._ranks

    def rank(self, word: str) -> int | None:
        return self.ranks.get(word)

    def family_rank(self, word: str) -> int | None:
        """Rank of the word's family (all forms sharing its stem), or of the word itself if
        that is better: an inflected form of a common verb is not a rare word."""
        if self._family_ranks is None:
            fam: Counter = Counter()
            for w, c in self.counts.items():
                fam[lemma_key(w)] += c
            ordered = sorted(fam.items(), key=lambda kv: (-kv[1], kv[0]))
            self._family_ranks = {k: i + 1 for i, (k, _) in enumerate(ordered)}
        own = self.ranks.get(word)
        fr = self._family_ranks.get(lemma_key(word))
        return min(r for r in (own, fr) if r is not None) if (own or fr) else None

    def most_common(self, n: int) -> list[tuple[str, int]]:
        return sorted(self.counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]

    def save(self, path: str | Path, max_words: int = 500_000) -> None:
        """Save the ``max_words`` most frequent words. Ranks of saved words are unchanged
        on reload; dropped words rank beyond ``max_words`` either way, which is past any
        sensible ``rank_cap``, so scores do not change."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            for w, c in self.most_common(max_words):
                f.write(f"{w}\t{c}\n")

    @classmethod
    def load(cls, path: str | Path) -> "FrequencyTable":
        counts: dict[str, int] = {}
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                w, _, c = line.rstrip("\n").partition("\t")
                if w and c:
                    counts[w] = int(c)
        return cls(counts)

    @classmethod
    def from_counter(cls, counter: Counter) -> "FrequencyTable":
        return cls(dict(counter))


@dataclass
class LevelSummary:
    level: int
    count: int = 0
    mean_score: float = 0.0
    mean_words: float = 0.0
    top_constructions: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)


class DifficultyScorer:
    def __init__(
        self,
        freq: FrequencyTable | None = None,
        config: DifficultyConfig | None = None,
        thresholds: list[float] | None = None,
        verb_stems: frozenset[str] | None = None,
        level_summaries: list[dict] | None = None,
        spoken: FrequencyTable | None = None,
    ):
        self.freq = freq or FrequencyTable()
        self.spoken = spoken or FrequencyTable()
        self.config = config or DifficultyConfig()
        self.verb_stems = F.BUILTIN_VERB_STEMS | (verb_stems or frozenset())
        n = self.config.num_levels
        # Until fit_levels() runs, use evenly spaced thresholds over the typical score range.
        self.thresholds = thresholds or [0.05 + 0.65 * i / n for i in range(1, n)]
        if self.config.scheme == "quantile" and len(self.thresholds) != n - 1:
            raise ValueError(f"expected {n - 1} thresholds for {n} levels, got {len(self.thresholds)}")
        self.level_summaries = level_summaries or []

    # ------------------------------------------------------------------ scoring
    def effective_rank(self, word: str) -> float:
        """Frequency rank, blending spoken and web ranks geometrically when both exist."""
        cap = self.config.rank_cap
        absolute = self.config.scheme == "absolute"
        rank = (lambda t, w: t.family_rank(w)) if absolute else (lambda t, w: t.rank(w))
        web = rank(self.freq, word) or cap if self.freq.counts else None
        if not self.spoken.counts:
            return float(web or cap)
        # Words absent from the (smaller) spoken corpus rank just past its vocabulary.
        spoken = rank(self.spoken, word) or max(len(self.spoken) * 2, cap)
        if web is None:
            return float(spoken)
        a = self.config.spoken_weight
        return math.exp(a * math.log(spoken) + (1 - a) * math.log(web))

    def word_rarity(self, word: str) -> float:
        if not self.freq.counts and not self.spoken.counts:
            return 0.5
        rank = min(self.effective_rank(word), self.config.rank_cap)
        return min(1.0, math.log(rank + 1) / math.log(self.config.rank_cap + 1))

    def _scale(self, level50: float) -> float:
        return 1 + (level50 - 1) * (self.config.num_levels - 1) / 49

    def level_dimensions(self, feats: F.SentenceFeatures) -> dict[str, float]:
        """Absolute scheme: the level each dimension alone would put the sentence at (1..num_levels)."""
        ranks = sorted((self.effective_rank(t) for t in feats.tokens), reverse=True)
        word_levels = [_interp_log(r, VOCAB_ANCHORS_50) for r in ranks] or [1.0]
        vocab = word_levels[0] if len(word_levels) == 1 else 0.7 * word_levels[0] + 0.3 * word_levels[1]
        grammar = max((GRAMMAR_LEVELS_50.get(c, 1) for c in feats.constructions), default=1)
        length = _interp(max(1, feats.n_tokens), LENGTH_ANCHORS_50)
        ortho = min(3.0, 4.0 * feats.n_conjuncts / max(1, feats.n_tokens))
        return {k: self._scale(v) for k, v in
                {"vocabulary": vocab, "grammar": grammar, "length": length}.items()} | {"orthography": ortho}

    def _absolute_level(self, dims: Mapping[str, float]) -> float:
        main = sorted((dims["vocabulary"], dims["grammar"], dims["length"]), reverse=True)
        lvl = main[0] + self.config.secondary_weight * ((main[1] - 1) + (main[2] - 1)) + dims["orthography"]
        return min(float(self.config.num_levels), max(1.0, lvl))

    def components(self, sentence: str) -> tuple[dict[str, float], F.SentenceFeatures]:
        cfg = self.config
        feats = F.extract(sentence, self.verb_stems)
        n = max(1, feats.n_tokens)
        rarities = sorted((self.word_rarity(t) for t in feats.tokens), reverse=True) or [0.0]
        lexical = 0.5 * (sum(rarities) / len(rarities)) + 0.5 * (sum(rarities[:2]) / len(rarities[:2]))
        grammar = 0.8 * feats.max_step / F.MAX_STEP + 0.2 * min(1.0, len(feats.constructions) / 8)
        length = min(1.0, (n - 1) / (cfg.length_cap - 1))
        orthography = min(1.0, feats.n_conjuncts / n / cfg.conjuncts_per_word_cap)
        return (
            {"grammar": grammar, "lexical": lexical, "length": length, "orthography": orthography},
            feats,
        )

    def _combine(self, comps: Mapping[str, float]) -> float:
        cfg = self.config
        total = cfg.w_grammar + cfg.w_lexical + cfg.w_length + cfg.w_orthography
        return (
            cfg.w_grammar * comps["grammar"]
            + cfg.w_lexical * comps["lexical"]
            + cfg.w_length * comps["length"]
            + cfg.w_orthography * comps["orthography"]
        ) / total

    def score(self, sentence: str) -> float:
        comps, _ = self.components(sentence)
        return self._combine(comps)

    def level_of_score(self, score: float) -> int:
        return 1 + bisect.bisect_right(self.thresholds, score)

    def level(self, sentence: str) -> int:
        return self.analyze(sentence)["level"]

    def analyze(self, sentence: str) -> dict:
        """Everything the app may want to show: level, score, components, grammar, rare words."""
        comps, feats = self.components(sentence)
        if self.config.scheme == "absolute":
            dims = self.level_dimensions(feats)
            raw = self._absolute_level(dims)
            level = int(round(raw))
            score = raw / self.config.num_levels
            comps = {**comps, **{f"level_{k}": v for k, v in dims.items()}}
            rare = sorted({t for t in feats.tokens if self.effective_rank(t) > 2000}, key=lambda t: -self.effective_rank(t))
        else:
            score = self._combine(comps)
            level = self.level_of_score(score)
            rare = sorted(
                {t for t in feats.tokens if self.word_rarity(t) >= 0.75},
                key=lambda t: -self.word_rarity(t),
            )
        return {
            "level": level,
            "num_levels": self.config.num_levels,
            "score": round(score, 4),
            "components": {k: round(v, 4) for k, v in comps.items()},
            "constructions": feats.constructions,
            "rare_words": rare,
            "n_words": feats.n_tokens,
        }

    # ------------------------------------------------------------------ fitting
    def fit_levels(self, sentences: Iterable[str], examples_per_level: int = 5, seed: int = 0) -> list[LevelSummary]:
        """Quantile scheme: set thresholds to equal-frequency quantiles of the scores of
        ``sentences``. Both schemes: record per-level counts and examples."""
        if self.config.scheme == "absolute":
            return self._summarize(sentences, examples_per_level, seed)
        scored = []
        for s in sentences:
            comps, feats = self.components(s)
            if feats.n_tokens:
                scored.append((self._combine(comps), s, feats))
        if len(scored) < self.config.num_levels * 2:
            raise ValueError(f"need at least {self.config.num_levels * 2} sentences to fit levels, got {len(scored)}")
        scored.sort(key=lambda x: x[0])
        n, L = len(scored), self.config.num_levels
        self.thresholds = [scored[min(n - 1, (n * k) // L)][0] for k in range(1, L)]
        # Ties can make thresholds equal; nudge them so every level stays reachable.
        for i in range(1, len(self.thresholds)):
            if self.thresholds[i] <= self.thresholds[i - 1]:
                self.thresholds[i] = self.thresholds[i - 1] + 1e-6

        summaries = [LevelSummary(level=i + 1) for i in range(L)]
        cons: list[Counter] = [Counter() for _ in range(L)]
        for score, s, feats in scored:
            summ = summaries[self.level_of_score(score) - 1]
            summ.count += 1
            summ.mean_score += score
            summ.mean_words += feats.n_tokens
            cons[summ.level - 1].update(feats.constructions)
            if len(summ.examples) < examples_per_level:
                summ.examples.append(s)
        for summ, c in zip(summaries, cons):
            if summ.count:
                summ.mean_score /= summ.count
                summ.mean_words /= summ.count
            summ.top_constructions = [name for name, _ in c.most_common(5)]
        self.level_summaries = [asdict(s) for s in summaries]
        return summaries

    def _summarize(self, sentences: Iterable[str], examples_per_level: int, seed: int) -> list[LevelSummary]:
        import random

        rng = random.Random(seed)
        L = self.config.num_levels
        summaries = [LevelSummary(level=i + 1) for i in range(L)]
        cons: list[Counter] = [Counter() for _ in range(L)]
        for s in dict.fromkeys(sentences):  # dedupe, keep order
            a = self.analyze(s)
            if not a["n_words"]:
                continue
            summ = summaries[a["level"] - 1]
            summ.count += 1
            summ.mean_score += a["score"]
            summ.mean_words += a["n_words"]
            cons[summ.level - 1].update(a["constructions"])
            # reservoir sample of examples, so they are not all from the first source
            if len(summ.examples) < examples_per_level:
                summ.examples.append(s)
            elif (j := rng.randrange(summ.count)) < examples_per_level:
                summ.examples[j] = s
        for summ, c in zip(summaries, cons):
            if summ.count:
                summ.mean_score /= summ.count
                summ.mean_words /= summ.count
            summ.top_constructions = [name for name, _ in c.most_common(5)]
        self.level_summaries = [asdict(s) for s in summaries]
        return summaries

    # ------------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        write_json(
            d / SCORER_FILE,
            {
                "config": asdict(self.config),
                "thresholds": self.thresholds,
                "level_summaries": self.level_summaries,
                "constructions": [
                    {"name": c.name, "step": c.step, "description": c.description} for c in F.CONSTRUCTIONS
                ],
            },
        )
        if self.freq.counts:
            self.freq.save(d / FREQ_FILE)
        if self.spoken.counts:
            self.spoken.save(d / SPOKEN_FREQ_FILE)
        extra = sorted(self.verb_stems - F.BUILTIN_VERB_STEMS)
        (d / STEMS_FILE).write_text("\n".join(extra) + ("\n" if extra else ""), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "DifficultyScorer":
        d = Path(directory)
        meta = read_json(d / SCORER_FILE)
        freq = FrequencyTable.load(d / FREQ_FILE) if (d / FREQ_FILE).exists() else None
        spoken = FrequencyTable.load(d / SPOKEN_FREQ_FILE) if (d / SPOKEN_FREQ_FILE).exists() else None
        stems = None
        if (d / STEMS_FILE).exists():
            stems = frozenset(w for w in (d / STEMS_FILE).read_text(encoding="utf-8").split() if w)
        return cls(
            freq=freq,
            config=DifficultyConfig(**meta["config"]),
            thresholds=meta["thresholds"],
            verb_stems=stems,
            level_summaries=meta.get("level_summaries"),
            spoken=spoken,
        )
