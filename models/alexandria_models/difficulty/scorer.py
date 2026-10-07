"""Continuous difficulty score for Hindi sentences and its mapping to discrete levels.

score = w_grammar * grammar + w_lexical * lexical + w_length * length + w_orthography * orthography

  grammar      hardest construction present (its teaching step / max step), plus a little
               for how many different constructions the sentence combines
  lexical      how rare the words are in the corpus: log-rank rarity, averaging the mean
               word with the two rarest words (one rare word is enough to make a sentence hard)
  length       number of words, saturating at ``length_cap``
  orthography  conjunct consonants per word (स्वास्थ्य, प्रतिनिधित्व are hard to read)

Every component is in [0, 1]. Levels are equal-frequency bins of the score over a
reference sample (``fit_levels``), so with 20 levels each level holds ~5% of the
reference sentences and the generator gets balanced training data per level.
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
STEMS_FILE = "verb_stems.txt"


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


class FrequencyTable:
    """Word counts with 1-based frequency ranks (rank 1 = most frequent)."""

    def __init__(self, counts: Mapping[str, int] | None = None):
        self.counts: dict[str, int] = dict(counts or {})
        self._ranks: dict[str, int] | None = None

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
    ):
        self.freq = freq or FrequencyTable()
        self.config = config or DifficultyConfig()
        self.verb_stems = F.BUILTIN_VERB_STEMS | (verb_stems or frozenset())
        n = self.config.num_levels
        # Until fit_levels() runs, use evenly spaced thresholds over the typical score range.
        self.thresholds = thresholds or [0.05 + 0.65 * i / n for i in range(1, n)]
        if len(self.thresholds) != n - 1:
            raise ValueError(f"expected {n - 1} thresholds for {n} levels, got {len(self.thresholds)}")
        self.level_summaries = level_summaries or []

    # ------------------------------------------------------------------ scoring
    def word_rarity(self, word: str) -> float:
        if not self.freq.counts:
            return 0.5
        rank = self.freq.rank(word) or self.config.rank_cap
        return min(1.0, math.log(rank + 1) / math.log(self.config.rank_cap + 1))

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
        return self.level_of_score(self.score(sentence))

    def analyze(self, sentence: str) -> dict:
        """Everything the app may want to show: level, score, components, grammar, rare words."""
        comps, feats = self.components(sentence)
        score = self._combine(comps)
        rare = sorted(
            {t for t in feats.tokens if self.word_rarity(t) >= 0.75},
            key=lambda t: -self.word_rarity(t),
        )
        return {
            "level": self.level_of_score(score),
            "num_levels": self.config.num_levels,
            "score": round(score, 4),
            "components": {k: round(v, 4) for k, v in comps.items()},
            "constructions": feats.constructions,
            "rare_words": rare,
            "n_words": feats.n_tokens,
        }

    # ------------------------------------------------------------------ fitting
    def fit_levels(self, sentences: Iterable[str], examples_per_level: int = 5) -> list[LevelSummary]:
        """Set level thresholds to equal-frequency quantiles of the scores of ``sentences``."""
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
        extra = sorted(self.verb_stems - F.BUILTIN_VERB_STEMS)
        (d / STEMS_FILE).write_text("\n".join(extra) + ("\n" if extra else ""), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "DifficultyScorer":
        d = Path(directory)
        meta = read_json(d / SCORER_FILE)
        freq = FrequencyTable.load(d / FREQ_FILE) if (d / FREQ_FILE).exists() else None
        stems = None
        if (d / STEMS_FILE).exists():
            stems = frozenset(w for w in (d / STEMS_FILE).read_text(encoding="utf-8").split() if w)
        return cls(
            freq=freq,
            config=DifficultyConfig(**meta["config"]),
            thresholds=meta["thresholds"],
            verb_stems=stems,
            level_summaries=meta.get("level_summaries"),
        )
