"""Sentence-level romanized Hindi -> Devanagari using the word model plus corpus context.

Each Latin word gets its top-k spellings from the character model; a word-bigram
language model built from the Hindi corpus (``difficulty/word_freq.tsv.gz`` and
``bigram_freq.tsv.gz``) then picks the most plausible combination. This is what
resolves ambiguous romanizations: "main" -> मैं (I) or में (in), "ki" -> की or कि,
"to" -> तो. Devanagari words, numbers and punctuation pass through unchanged.

    tr = Transliterator.load("artifacts/transliterator", difficulty_dir="artifacts/difficulty")
    tr.transliterate("main ghar ja raha hoon", k=3)
    # ['मैं घर जा रहा हूँ', 'में घर जा रहा हूँ', 'मैं घर जा रहा हूं']
"""

from __future__ import annotations

import gzip
import math
import re
import threading
from functools import lru_cache
from pathlib import Path

from ..common.io import resolve_device
from ..common.text import normalize
from .model import PAD, load_model

_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*|[ऀ-ॣॱ-ॿ]+|[0-9०-९]+|\S")
_LATIN_RE = re.compile(r"^[A-Za-z]+(?:'[A-Za-z]+)*$")
_NO_SPACE_BEFORE = set(".,!?;:)।॥")


class WordLM:
    """Stupid-backoff word bigram model over Devanagari words."""

    def __init__(self, unigrams: dict[str, int], bigrams: dict[tuple[str, str], int] | None = None):
        self.uni = unigrams
        self.bi = bigrams or {}
        self.total = max(1, sum(unigrams.values()))
        self.oov = math.log(0.1 / self.total)

    @classmethod
    def load(cls, difficulty_dir: str | Path) -> "WordLM | None":
        d = Path(difficulty_dir)
        uni_path, bi_path = d / "word_freq.tsv.gz", d / "bigram_freq.tsv.gz"
        if not uni_path.exists():
            return None
        uni: dict[str, int] = {}
        with gzip.open(uni_path, "rt", encoding="utf-8") as f:
            for line in f:
                w, _, c = line.rstrip("\n").partition("\t")
                if w and c:
                    uni[w] = int(c)
        bi: dict[tuple[str, str], int] = {}
        if bi_path.exists():
            with gzip.open(bi_path, "rt", encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) == 3:
                        bi[(parts[0], parts[1])] = int(parts[2])
        return cls(uni, bi)

    def logp(self, word: str, prev: str | None) -> float:
        if prev is not None:
            c = self.bi.get((prev, word))
            if c and self.uni.get(prev):
                return math.log(c / self.uni[prev])
        c = self.uni.get(word)
        return math.log(0.4) + (math.log(c / self.total) if c else self.oov)


class Transliterator:
    def __init__(self, model, src_vocab, tgt_vocab, device: str = "cpu", lm: WordLM | None = None,
                 lm_weight: float = 0.3, word_beam: int = 5, sentence_beam: int = 8):
        self.model = model
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.device = device
        self.lm = lm
        self.lm_weight = lm_weight
        self.word_beam = word_beam
        self.sentence_beam = sentence_beam
        self._lock = threading.Lock()
        self._cached_words = lru_cache(maxsize=50_000)(self._words_uncached)

    @classmethod
    def load(cls, transliterator_dir: str | Path, difficulty_dir: str | Path | None = None,
             device: str = "auto", **kwargs) -> "Transliterator":
        device = resolve_device(device)
        model, src_vocab, tgt_vocab = load_model(transliterator_dir, device)
        lm = WordLM.load(difficulty_dir) if difficulty_dir else None
        return cls(model, src_vocab, tgt_vocab, device, lm, **kwargs)

    # ---------------------------------------------------------------- words
    def _words_uncached(self, word: str) -> tuple[tuple[str, float], ...]:
        return tuple(self.transliterate_words([word])[0])

    def transliterate_words(self, words: list[str], k: int | None = None) -> list[list[tuple[str, float]]]:
        """Top-k Devanagari spellings with log-probabilities for each romanized word."""
        import torch

        k = k or self.word_beam
        max_len = self.model.cfg.max_len
        src = [self.src_vocab.encode(w.lower(), max_len) for w in words]
        S = max(map(len, src))
        src_t = torch.tensor([s + [PAD] * (S - len(s)) for s in src], device=self.device)
        with self._lock:
            hyps = self.model.beam_search(src_t, beam_size=k, max_len=max_len)
        out = []
        for row in hyps:
            seen: dict[str, float] = {}
            for ids, score in row:
                text = normalize(self.tgt_vocab.decode(ids))
                if text and text not in seen:
                    seen[text] = score
            out.append(sorted(seen.items(), key=lambda kv: -kv[1]))
        return out

    # ---------------------------------------------------------------- sentences
    def transliterate(self, text: str, k: int = 3) -> list[str]:
        """Return up to ``k`` Devanagari readings of ``text``, best first."""
        tokens = _TOKEN_RE.findall(text)
        if not tokens:
            return []
        latin = [t for t in dict.fromkeys(t.lower() for t in tokens if _LATIN_RE.match(t))]
        cands = {w: list(self._cached_words(w)) for w in latin}

        # Beam over the sentence: (score, tokens, last Devanagari word)
        beams: list[tuple[float, list[str], str | None]] = [(0.0, [], None)]
        for tok in tokens:
            if _LATIN_RE.match(tok):
                options = cands.get(tok.lower()) or [(tok, -10.0)]
            else:
                options = [("।" if tok == "." else tok, 0.0)]
            new = []
            for score, toks, prev in beams:
                for cand, lp in options:
                    s = score + lp
                    is_word = bool(re.match(r"[ऀ-ॿ]", cand))
                    if self.lm is not None and is_word:
                        s += self.lm_weight * self.lm.logp(cand, prev)
                    new.append((s, toks + [cand], cand if is_word else prev))
            new.sort(key=lambda b: -b[0])
            beams = new[: self.sentence_beam]

        results = []
        for _, toks, _ in beams:
            s = ""
            for t in toks:
                s += t if (not s or t in _NO_SPACE_BEFORE) else " " + t
            s = normalize(s)
            if s not in results:
                results.append(s)
            if len(results) >= k:
                break
        return results
