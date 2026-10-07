"""Text helpers shared by every model: normalization, script detection, tokenization.

All Hindi text is normalized to Unicode NFC before it is counted, scored or fed to
a model. NFC decomposes the precomposed nukta letters (U+0958-U+095F, e.g. क़)
into base letter + nukta, so the same word always has a single spelling.
"""

from __future__ import annotations

import re
import unicodedata

VIRAMA = "्"
NUKTA = "़"
DANDA = "।"
DOUBLE_DANDA = "॥"
ZWJ, ZWNJ = "‍", "‌"

# Devanagari letters and combining marks, excluding the dandas and Devanagari digits.
_DEV_WORD = r"[ऀ-ॣॱ-ॿ]+"
_LATIN_WORD = r"[A-Za-z]+(?:['’][A-Za-z]+)*"
_NUMBER = r"[0-9०-९]+(?:[.,][0-9०-९]+)*"
WORD_RE = re.compile(f"{_DEV_WORD}|{_LATIN_WORD}|{_NUMBER}")
DEV_WORD_RE = re.compile(_DEV_WORD)

_DEV_CHAR = re.compile(r"[ऀ-ॿ]")
_LATIN_CHAR = re.compile(r"[A-Za-z]")
_DEV_CONSONANT = re.compile(r"[क-हक़-य़ॸ-ॿ]")
_DEV_INDEPENDENT_VOWEL = re.compile(r"[ऄ-औॠॡॲ-ॷ]")

_SENT_SPLIT_RE = re.compile(r"(?<=[।॥?!.])\s+")
_SPACE_RE = re.compile(r"\s+")
_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", " ": " "})


def normalize(text: str) -> str:
    """NFC-normalize, unify quotes/dandas and collapse whitespace."""
    text = unicodedata.normalize("NFC", text).translate(_QUOTES)
    text = text.replace(ZWJ, "").replace(ZWNJ, "")
    # A pipe right after Devanagari text is almost always a danda typed on a Latin keyboard.
    text = re.sub(r"(?<=[ऀ-ॣ])\s*[|]\s*", DANDA + " ", text)
    text = re.sub(r"(?<=[ऀ-ॣ])\s*।\s*", DANDA + " ", text)
    return _SPACE_RE.sub(" ", text).strip()


def script_counts(text: str) -> tuple[int, int]:
    """Return (devanagari_chars, latin_chars)."""
    return len(_DEV_CHAR.findall(text)), len(_LATIN_CHAR.findall(text))


def devanagari_ratio(text: str) -> float:
    """Fraction of script letters that are Devanagari (0.0 when there are none)."""
    dev, lat = script_counts(text)
    return dev / (dev + lat) if dev + lat else 0.0


def is_devanagari(text: str, threshold: float = 0.6) -> bool:
    return devanagari_ratio(text) >= threshold


def is_latin(text: str, threshold: float = 0.6) -> bool:
    dev, lat = script_counts(text)
    return (lat / (dev + lat) if dev + lat else 0.0) >= threshold


def tokenize(text: str) -> list[str]:
    """Split into word tokens (Devanagari words, Latin words, numbers); punctuation dropped."""
    return WORD_RE.findall(text)


def hindi_words(text: str) -> list[str]:
    """Only the Devanagari word tokens of ``text``."""
    return DEV_WORD_RE.findall(text)


def split_sentences(text: str) -> list[str]:
    """Split running text into sentences on danda, double danda, ?, ! and full stop."""
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]


def conjunct_count(word: str) -> int:
    """Number of conjunct consonants (virama followed by a consonant), e.g. 2 in 'स्वास्थ्य'."""
    return sum(
        1
        for i, ch in enumerate(word[:-1])
        if ch == VIRAMA and _DEV_CONSONANT.match(word[i + 1])
    )


def akshara_count(word: str) -> int:
    """Approximate orthographic syllables: consonant clusters and independent vowels."""
    count = 0
    prev = ""
    for ch in word:
        if _DEV_INDEPENDENT_VOWEL.match(ch):
            count += 1
        elif _DEV_CONSONANT.match(ch) and prev != VIRAMA:
            count += 1
        if ch != NUKTA:
            prev = ch
    return count


def clean_sentence_ok(
    text: str,
    *,
    lang: str,
    min_words: int = 1,
    max_words: int = 40,
    min_script_ratio: float = 0.85,
) -> bool:
    """Cheap quality filter for one sentence in ``lang`` ('hi' or 'en')."""
    words = tokenize(text)
    if not (min_words <= len(words) <= max_words):
        return False
    if re.search(r"https?://|www\.|@\w|#\w|[<>{}\[\]_=\\]", text):
        return False
    dev, lat = script_counts(text)
    if lang == "hi":
        return dev > 0 and dev / (dev + lat) >= min_script_ratio
    if lang == "en":
        return lat > 0 and dev == 0
    raise ValueError(f"unknown lang {lang!r}")
