"""Informal romanization variants, so the model handles how people actually type Hindi.

Dataset romanizations are fairly careful ("paanee", "kitaab"), but learners type
"pani", "kitab", "bohot", "shukriya"/"sukriya", "zindagi"/"jindagi". During training
some source words are replaced by a random variant produced by these rewrite rules.
"""

from __future__ import annotations

import random
import re

# (pattern, replacement) rewrites that map a spelling onto a common informal one.
_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(p), r)
    for p, r in [
        (r"aa", "a"),
        (r"ee", "i"),
        (r"ii", "i"),
        (r"oo", "u"),
        (r"uu", "u"),
        (r"w", "v"),
        (r"v", "w"),
        (r"z", "j"),
        (r"ph", "f"),
        (r"f", "ph"),
        (r"sh", "s"),
        (r"ksh", "x"),
        (r"q", "k"),
        (r"ai$", "e"),
        (r"ain$", "en"),
        (r"ein$", "en"),
        (r"aa$", "a"),
        (r"an$", "n"),
        (r"(?<=[^aeiou])a(?=[^aeiou]$)", ""),
        (r"([bcdfghjklmnpqrstvwxyz])\1", r"\1"),
        (r"th", "t"),
        (r"dh", "d"),
        (r"bh", "b"),
        (r"chh", "ch"),
        (r"ch", "c"),
        (r"n$", ""),
        (r"y$", "i"),
        (r"i$", "ee"),
        (r"u$", "oo"),
        (r"o$", "oh"),
    ]
]


def romanization_variants(word: str, max_variants: int = 3, rng: random.Random | None = None) -> list[str]:
    """Return up to ``max_variants`` distinct informal spellings of ``word`` (excluding itself)."""
    rng = rng or random
    seen = {word}
    out: list[str] = []
    rules = _RULES[:]
    rng.shuffle(rules)
    for pat, rep in rules:
        if len(out) >= max_variants:
            break
        if not pat.search(word):
            continue
        variant = pat.sub(rep, word, count=1)
        # Occasionally stack a second rewrite to mimic heavier abbreviation.
        if rng.random() < 0.3:
            pat2, rep2 = rng.choice(_RULES)
            variant = pat2.sub(rep2, variant, count=1)
        if variant and len(variant) >= max(1, len(word) // 2) and variant not in seen:
            seen.add(variant)
            out.append(variant)
    return out
