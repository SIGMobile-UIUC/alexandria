"""Synthetic wrong (and still-right) translations used to train the evaluator.

Human-scored data (WMT QE) covers machine translations; learners make different
mistakes. Each function takes a correct translation and returns a corrupted one plus
the score it should get (0 = unrelated, 1 = perfect), or None if it does not apply.
Scores are deliberately graded: a gender-agreement slip should cost far less than a
flipped negation or a missing half of the sentence.
"""

from __future__ import annotations

import random
import re
from typing import Callable

from ..common.text import normalize

Perturbation = Callable[[str, random.Random], "tuple[str, float] | None"]


def _words(s: str) -> list[str]:
    return s.split()


def _join(ws: list[str]) -> str:
    return " ".join(ws)


def _swap_pairs(text: str, pairs: dict[str, str], rng: random.Random, whole_word: bool = True) -> str | None:
    """Replace one random occurrence of any key in ``pairs`` by its value."""
    ws = _words(text)
    idx = [i for i, w in enumerate(ws) if re.sub(r"[।?!,.]+$", "", w) in pairs]
    if not idx:
        return None
    i = rng.choice(idx)
    core = re.sub(r"[।?!,.]+$", "", ws[i])
    ws[i] = pairs[core] + ws[i][len(core):]
    return _join(ws)


def _bidirectional(pairs: list[tuple[str, str]]) -> dict[str, str]:
    d: dict[str, str] = {}
    for a, b in pairs:
        d[normalize(a)] = normalize(b)
        d[normalize(b)] = normalize(a)
    return d


# --------------------------------------------------------------------------- Hindi
HI_GENDER = _bidirectional(
    [("रहा", "रही"), ("गया", "गई"), ("था", "थी"), ("अच्छा", "अच्छी"), ("बड़ा", "बड़ी"), ("छोटा", "छोटी"),
     ("मेरा", "मेरी"), ("तुम्हारा", "तुम्हारी"), ("उसका", "उसकी"), ("हमारा", "हमारी"), ("आपका", "आपकी"),
     ("का", "की"), ("करता", "करती"), ("जाता", "जाती"), ("आया", "आई"), ("सकता", "सकती"), ("चाहता", "चाहती"), ("नया", "नई"), ("पीता", "पीती"), ("खाता", "खाती")]
)
HI_NUMBER = _bidirectional([("है", "हैं"), ("था", "थे"), ("रहा", "रहे"), ("गया", "गए"), ("करता", "करते"), ("लड़का", "लड़के")])
HI_TENSE = {normalize(a): normalize(b) for a, b in [
    ("है", "था"), ("हैं", "थे"), ("था", "है"), ("थे", "हैं"), ("थी", "है"), ("जाएगा", "गया"), ("गया", "जाएगा"),
    ("आएगा", "आया"), ("आया", "आएगा"), ("करेगा", "किया"), ("जाएँगे", "गए"), ("गए", "जाएँगे")]}
HI_POSTPOSITION = {normalize(a): normalize(b) for a, b in [
    ("में", "पर"), ("पर", "में"), ("से", "को"), ("को", "से"), ("का", "से"), ("के", "से"), ("तक", "से"), ("ने", "को")]}
HI_PRONOUN = _bidirectional([("मैं", "तुम"), ("वह", "हम"), ("मुझे", "उसे"), ("हमने", "उन्होंने"), ("मैंने", "उसने"), ("यह", "वह")])
HI_NEGATIONS = {normalize(w) for w in ("नहीं", "न", "मत")}
# Learner spelling slips that do not change meaning: vowel length, nasalisation, nukta.
HI_SPELLING = [
    ("ी", "ि"), ("ि", "ी"), ("ू", "ु"), ("ु", "ू"), ("ँ", "ं"), ("ं", "ँ"), ("़", ""), ("ै", "े"), ("ो", "ौ"),
]

# --------------------------------------------------------------------------- English
EN_PRONOUN = _bidirectional([("I", "you"), ("he", "she"), ("He", "She"), ("we", "they"), ("We", "They"), ("my", "your"),
                             ("his", "her"), ("him", "her"), ("our", "their"), ("me", "him")])
EN_TENSE = _bidirectional([("is", "was"), ("are", "were"), ("am", "was"), ("will", "did"), ("has", "had"), ("go", "went"),
                           ("eat", "ate"), ("come", "came"), ("see", "saw"), ("do", "did")])
EN_PREPOSITION = _bidirectional([("in", "on"), ("to", "from"), ("at", "in"), ("with", "without"), ("before", "after")])
EN_ANTONYM = _bidirectional([("big", "small"), ("hot", "cold"), ("good", "bad"), ("old", "new"), ("early", "late"),
                             ("happy", "sad"), ("open", "close"), ("always", "never"), ("tomorrow", "yesterday"),
                             ("today", "tomorrow"), ("many", "few"), ("love", "hate"), ("like", "dislike")])
EN_NEGATION_RE = re.compile(r"\b(not|n't|never|no)\b", re.I)
EN_AUX_RE = re.compile(r"\b(is|are|am|was|were|will|can|should|do|does|did|has|have|had)\b")


def _drop_words(text: str, rng: random.Random) -> tuple[str, float] | None:
    ws = _words(text)
    if len(ws) < 4:
        return None
    k = max(1, round(len(ws) * rng.uniform(0.15, 0.5)))
    keep = sorted(rng.sample(range(len(ws)), len(ws) - k))
    return _join([ws[i] for i in keep]), max(0.1, 1.0 - 1.3 * k / len(ws))


def _truncate(text: str, rng: random.Random) -> tuple[str, float] | None:
    ws = _words(text)
    if len(ws) < 5:
        return None
    cut = rng.randint(max(2, len(ws) // 3), len(ws) - 2)
    return _join(ws[:cut]), 0.15 + 0.5 * cut / len(ws)


def _shuffle_local(text: str, rng: random.Random) -> tuple[str, float] | None:
    ws = _words(text)
    if len(ws) < 4:
        return None
    i = rng.randrange(len(ws) - 1)
    ws[i], ws[i + 1] = ws[i + 1], ws[i]
    return _join(ws), 0.8


def _shuffle_all(text: str, rng: random.Random) -> tuple[str, float] | None:
    ws = _words(text)
    if len(ws) < 5:
        return None
    rng.shuffle(ws)
    return _join(ws), 0.35


def _dup_word(text: str, rng: random.Random) -> tuple[str, float] | None:
    ws = _words(text)
    if len(ws) < 3:
        return None
    i = rng.randrange(len(ws))
    ws.insert(i, ws[i])
    return _join(ws), 0.85


def _pairs(table: dict[str, str], score: float) -> Perturbation:
    def f(text: str, rng: random.Random):
        out = _swap_pairs(text, table, rng)
        return (out, score) if out and out != text else None

    return f


def _hi_negation(text: str, rng: random.Random) -> tuple[str, float] | None:
    ws = _words(text)
    neg = [i for i, w in enumerate(ws) if w in HI_NEGATIONS]
    if neg:
        del ws[rng.choice(neg)]
        return _join(ws), 0.15
    # Insert नहीं before the final verb group (the last 1-2 words before a copula/end).
    if len(ws) < 2:
        return None
    pos = len(ws) - 2 if len(ws) >= 3 else len(ws) - 1
    ws.insert(pos, "नहीं")
    return _join(ws), 0.15


def _hi_spelling(text: str, rng: random.Random) -> tuple[str, float] | None:
    a, b = rng.choice(HI_SPELLING)
    idx = [m.start() for m in re.finditer(re.escape(a), text)]
    if not idx:
        return None
    i = rng.choice(idx)
    return text[:i] + b + text[i + len(a):], 0.9


def _en_negation(text: str, rng: random.Random) -> tuple[str, float] | None:
    if EN_NEGATION_RE.search(text):
        out = EN_NEGATION_RE.sub("", text, count=1)
        return re.sub(r"\s+", " ", out).strip(), 0.15
    m = EN_AUX_RE.search(text)
    if not m:
        return None
    return text[: m.end()] + " not" + text[m.end():], 0.15


def _en_case(text: str, rng: random.Random) -> tuple[str, float] | None:
    """Lowercase / no punctuation: still a correct answer."""
    out = re.sub(r"[.!?]+$", "", text).lower()
    return (out, 1.0) if out != text else None


HINDI_PERTURBATIONS: list[tuple[str, Perturbation]] = [
    ("drop_words", _drop_words),
    ("truncate", _truncate),
    ("swap_adjacent", _shuffle_local),
    ("scramble", _shuffle_all),
    ("duplicate_word", _dup_word),
    ("gender_agreement", _pairs(HI_GENDER, 0.75)),
    ("number_agreement", _pairs(HI_NUMBER, 0.75)),
    ("tense", _pairs(HI_TENSE, 0.45)),
    ("postposition", _pairs(HI_POSTPOSITION, 0.6)),
    ("pronoun", _pairs(HI_PRONOUN, 0.3)),
    ("negation", _hi_negation),
    ("spelling", _hi_spelling),
]
ENGLISH_PERTURBATIONS: list[tuple[str, Perturbation]] = [
    ("drop_words", _drop_words),
    ("truncate", _truncate),
    ("swap_adjacent", _shuffle_local),
    ("scramble", _shuffle_all),
    ("duplicate_word", _dup_word),
    ("pronoun", _pairs(EN_PRONOUN, 0.3)),
    ("tense", _pairs(EN_TENSE, 0.5)),
    ("preposition", _pairs(EN_PREPOSITION, 0.6)),
    ("antonym", _pairs(EN_ANTONYM, 0.15)),
    ("negation", _en_negation),
    ("casing", _en_case),
]


def perturb(text: str, lang: str, rng: random.Random, tries: int = 6) -> tuple[str, float, str] | None:
    """Apply one random applicable perturbation; returns (text, score, name) or None."""
    table = HINDI_PERTURBATIONS if lang == "hi" else ENGLISH_PERTURBATIONS
    for _ in range(tries):
        name, fn = rng.choice(table)
        out = fn(text, rng)
        if out and out[0].strip() and out[0] != text:
            return out[0], out[1], name
    return None
