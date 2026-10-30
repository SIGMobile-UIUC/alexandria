"""Is this sentence pair something to show a language learner?

The corpora are mostly news, Wikipedia, software strings, religious texts and voice-
assistant commands. ``learner_issues`` rejects what a teacher would never hand out:
fragments and headlines (no verb, no final punctuation), numbers, Latin letters inside
Hindi, bracketed asides, names-heavy news sentences and scripture. It is a cheap rule
filter; the naturalness model (``naturalness/``) then ranks what is left.

    learner_issues("मैं दूध पीता हूँ।", "I drink milk.")          -> []
    learner_issues("खाद्यान्न उत्पादन में विकास", "Growth in ...") -> ['hi_no_final_punct', 'no_predicate', ...]
"""

from __future__ import annotations

import re

from ..difficulty import features as F
from .text import hindi_words, normalize, script_counts, tokenize

# Scripture and archaic religious register (OPUS / Tanzil Quran translations etc.).
RELIGIOUS = F._set(
    "अल्लाह फ़िरऔन फिरऔन पैग़म्बर पैगम्बर इबलीस रसूल क़ुरआन कुरआन सजदा फ़रिश्ते फरिश्ते फ़रिश्तों "
    "फरिश्तों ईमान काफ़िर काफिर निस्संदेह यहोवा प्रभु परमेश्वर"
)
HI_FINAL = ("।", "?", "!", ".")
EN_FINAL = (".", "?", "!")
SUBJUNCTIVE_SUFFIXES = ("ें", "ूँ", "ूं", "एँ", "एं", "ए")
_BAD_CHARS = re.compile(r"[()\[\]{}<>/\\|_=*#@%&;:\"“”]|\.\.|--|^[-–—]")
_AUX_STEMS = frozenset({"पड़", "चुक", "पा", "लग"})  # पड़ेगा, चुका, पाया, लगा
_DIGITS = re.compile(r"[0-9०-९]")
_LATIN = re.compile(r"[A-Za-z]")


def has_predicate(tokens: list[str], stems: frozenset[str] = F.BUILTIN_VERB_STEMS) -> bool:
    """A finite verb or copula: what separates a sentence from a headline or a fragment."""
    ctx = F._Ctx(tokens, stems | _AUX_STEMS)
    if tokens and tokens[-1] in ("की", "कीं"):  # sentence-final की is करना, not "of": मदद की
        return True
    for t in tokens:
        if t in F.COPULA or t in F.PROGRESSIVE_AUX or t in F.SAKNA_FORMS or t in F.IRREGULAR_FUTURE \
                or t in F.PADNA_FORMS or t in F.CHUKNA_FORMS or t in F.PAANA_FORMS:
            return True
        if t in F.IRREGULAR_POLITE_IMPERATIVE or t in ("चाहिए", "चाहिये"):
            return True
        if ctx.habitual(t) or ctx.perfective(t):
            return True
        if t not in F.NOT_FUTURE and ctx.has_suffix_on_stem(t, F.FUTURE_SUFFIXES):
            return True
        if t not in F.NOT_POLITE_IMPERATIVE and ctx.has_suffix_on_stem(
            t, F.POLITE_IMPERATIVE_SUFFIXES + F.FAMILIAR_IMPERATIVE_SUFFIXES + SUBJUNCTIVE_SUFFIXES
        ):
            return True
    return False


def tidy_hindi(hi: str) -> str:
    """Normalize and use a danda instead of a final full stop."""
    hi = normalize(hi)
    return hi[:-1].rstrip() + "।" if hi.endswith(".") and not hi.endswith("..") else hi


def learner_issues(
    hi: str,
    en: str | None = None,
    *,
    min_words: int = 2,
    max_words: int = 20,
    stems: frozenset[str] = F.BUILTIN_VERB_STEMS,
) -> list[str]:
    """Reasons ``hi`` (with its English ``en``) is unsuitable for learners; [] if fine."""
    issues: list[str] = []
    hi = normalize(hi)
    toks = hindi_words(hi)
    if not min_words <= len(toks) <= max_words:
        issues.append("length")
    if len(tokenize(hi)) != len(toks) or _LATIN.search(hi):
        issues.append("hi_not_pure_devanagari")
    if _DIGITS.search(hi) or (en and _DIGITS.search(en)):
        issues.append("digits")
    if _BAD_CHARS.search(hi) or (en and _BAD_CHARS.search(en)):
        issues.append("brackets_or_symbols")
    if not hi.endswith(HI_FINAL):
        issues.append("hi_no_final_punct")
    if toks and not has_predicate(toks, stems):
        issues.append("no_predicate")
    if any(t in RELIGIOUS for t in toks):
        issues.append("religious")
    if en is not None:
        en = normalize(en)
        words = re.findall(r"[A-Za-z']+", en)
        dev, _ = script_counts(en)
        if dev:
            issues.append("en_has_devanagari")
        if not en.endswith(EN_FINAL):
            issues.append("en_no_final_punct")
        if words and not words[0][0].isupper():
            issues.append("en_lowercase_start")
        caps = [w for w in words[1:] if w[0].isupper() and w != "I"]
        if len(words) >= 4 and len(caps) / (len(words) - 1) > 0.5:
            issues.append("en_title_case")  # a headline
        elif len(caps) > 2:
            issues.append("en_many_names")  # news about named people and places
    return issues


def is_learner_sentence(hi: str, en: str | None = None, **kw) -> bool:
    return not learner_issues(hi, en, **kw)


def dedupe_key(text: str) -> str:
    """Spelling-insensitive key so near-identical copies (punctuation, nukta, nasal) collapse."""
    t = normalize(text).replace("़", "").replace("ँ", "ं")
    return " ".join(hindi_words(t)) or re.sub(r"\W+", " ", t.lower()).strip()
