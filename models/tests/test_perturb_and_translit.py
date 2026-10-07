import random

from alexandria_models.evaluator.perturb import ENGLISH_PERTURBATIONS, HINDI_PERTURBATIONS, perturb
from alexandria_models.generator.prompts import build_completion, build_prompt, parse_completion
from alexandria_models.transliterator.augment import romanization_variants
from alexandria_models.transliterator.prepare_data import clean_pair


def test_perturbations_change_text_and_have_valid_scores(pairs):
    rng = random.Random(0)
    seen = set()
    for hi, en in pairs:
        for text, lang in ((hi, "hi"), (en, "en")):
            out = perturb(text, lang, rng)
            if out is None:
                continue
            new, score, name = out
            assert new != text and 0.0 <= score <= 1.0
            seen.add((lang, name))
    # Most perturbation types should fire somewhere on the fixture sentences.
    assert len({n for lang, n in seen if lang == "hi"}) >= len(HINDI_PERTURBATIONS) - 3
    assert len({n for lang, n in seen if lang == "en"}) >= len(ENGLISH_PERTURBATIONS) - 3


def test_negation_is_penalised_heavily():
    from alexandria_models.evaluator.perturb import _en_negation, _hi_negation

    rng = random.Random(0)
    assert _hi_negation("वह कल नहीं आएगा।", rng)[1] < 0.3
    text, score = _en_negation("He is coming tomorrow.", rng)
    assert "not" in text and score < 0.3


def test_prompt_roundtrip():
    prompt = build_prompt(7, 20, word="पानी", grammar="future")
    assert prompt.startswith("Level: 07/20") and prompt.endswith("Hindi:")
    assert parse_completion(build_completion("मैं पानी लाऊँगा।", "I will bring water.")) == (
        "मैं पानी लाऊँगा।", "I will bring water.",
    )
    assert parse_completion(" just garbage") is None


def test_romanization_variants():
    rng = random.Random(1)
    variants = romanization_variants("paani", max_variants=3, rng=rng)
    assert variants and "paani" not in variants
    assert all(v.isalpha() for v in variants)


def test_clean_pair():
    assert clean_pair("Paani", "पानी") == ("paani", "पानी")
    assert clean_pair("pa1ni", "पानी") is None
    assert clean_pair("paani", "pani") is None


def test_english_contractions_are_unnegated():
    from alexandria_models.evaluator.perturb import _en_negation

    rng = random.Random(0)
    assert _en_negation("I can't swim.", rng)[0] == "I can swim."
    assert _en_negation("He doesn't eat meat.", rng)[0] == "He does eat meat."
    assert _en_negation("Won't you come?", rng)[0] == "Will you come?"
