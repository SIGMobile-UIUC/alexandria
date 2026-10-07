"""Tokenizer loading that cannot silently destroy Hindi.

In transformers 5.x a BERT tokenizer rebuilds its normalizer from ``do_lower_case``,
which defaults to True; lowercasing also strips accents, and for Devanagari that deletes
vowel signs, virama and nukta ("हिन्दी किताबें" -> "हिनदी किताब"). MuRIL and IndicBERT
are cased models, so we always pass ``do_lower_case=False`` and verify a round trip.
"""

from __future__ import annotations

from ..common.text import normalize

_PROBE = "हिन्दी किताबें पढ़ूँगा"


def load_encoder_tokenizer(name_or_path):
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(name_or_path, do_lower_case=False, strip_accents=False)
    except TypeError:
        tok = AutoTokenizer.from_pretrained(name_or_path)
    check_devanagari_roundtrip(tok)
    return tok


def check_devanagari_roundtrip(tok) -> None:
    ids = tok(_PROBE, add_special_tokens=False)["input_ids"]
    decoded = normalize(tok.decode(ids, skip_special_tokens=True)).replace(" ", "")
    unk = getattr(tok, "unk_token_id", None)
    if decoded != normalize(_PROBE).replace(" ", "") and (unk is None or unk not in ids):
        raise ValueError(
            f"tokenizer {getattr(tok, 'name_or_path', '?')} mangles Devanagari "
            f"({_PROBE!r} -> {decoded!r}); it is probably lowercasing/stripping accents"
        )
