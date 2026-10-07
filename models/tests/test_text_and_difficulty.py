from collections import Counter

import pytest

from alexandria_models.common.text import (
    akshara_count,
    conjunct_count,
    devanagari_ratio,
    hindi_words,
    normalize,
    split_sentences,
    tokenize,
)
from alexandria_models.difficulty.features import derive_verb_stems, extract
from alexandria_models.difficulty.scorer import DifficultyConfig, DifficultyScorer, FrequencyTable


def test_normalize_unifies_nukta_and_danda():
    assert normalize("क़िताब") == normalize("क़िताब")
    assert normalize("मैं घर जा रहा हूँ |") == "मैं घर जा रहा हूँ।"
    assert normalize("  a   b ") == "a b"


def test_tokenize_and_scripts():
    assert hindi_words("मैं घर जा रहा हूँ।") == ["मैं", "घर", "जा", "रहा", "हूँ"]
    assert tokenize("I have 2 dogs, मेरे पास") == ["I", "have", "2", "dogs", "मेरे", "पास"]
    assert devanagari_ratio("मैं office जा रहा") > 0.5
    assert split_sentences("यह घर है। वह कौन है? ठीक है!") == ["यह घर है।", "वह कौन है?", "ठीक है!"]


def test_orthography_counts():
    assert conjunct_count("स्वास्थ्य") == 3
    assert conjunct_count("घर") == 0
    assert akshara_count("किताब") == 3
    assert akshara_count("प्रतिनिधित्व") == 5


@pytest.mark.parametrize(
    "sentence, construction",
    [
        ("मैं हर दिन दूध पीता हूँ।", "habitual_present"),
        ("मैं अभी खाना खा रहा हूँ।", "progressive"),
        ("हम कल दिल्ली जाएँगे।", "future"),
        ("मैंने एक नई किताब खरीदी।", "perfective_ne"),
        ("क्या आप मेरी मदद कर सकते हैं?", "ability_sakna"),
        ("तुम्हें डॉक्टर के पास जाना चाहिए।", "obligation_chahiye"),
        ("खाना खाकर हम बाज़ार गए।", "conjunctive_participle"),
        ("उसने अपना काम पूरा कर लिया है।", "compound_verb"),
        ("जो लड़का वहाँ खड़ा है, वह मेरा भाई है।", "relative_correlative"),
        ("यह पत्र मेरे दादाजी द्वारा लिखा गया था।", "passive"),
        ("मैंने उससे पत्र लिखवाया।", "causative"),
        ("काश मैं तुम्हारे साथ आ पाता।", "counterfactual"),
    ],
)
def test_constructions_detected(sentence, construction):
    assert construction in extract(sentence).constructions


@pytest.mark.parametrize(
    "sentence, wrong",
    [
        ("मेरे पास एक कुत्ता है।", "habitual_present"),  # कुत्ता is a noun, not a verb in -ता
        ("उसका जन्मदिन अगले हफ़्ते है।", "habitual_present"),
        ("रोगी की जान बच गई।", "future"),  # रोगी is not a future form
        ("तुम्हें जाना चाहिए।", "imperative"),  # चाहिए is not a polite imperative
    ],
)
def test_no_false_positives(sentence, wrong):
    assert wrong not in extract(sentence).constructions


def test_derive_verb_stems_rejects_nouns():
    counts = {"करना": 100, "करता": 50, "सपना": 100, "अपना": 500, "खाना": 80, "खाकर": 10}
    stems = derive_verb_stems(counts, min_infinitive_count=20, min_participle_count=5)
    assert "कर" in stems and "खा" in stems
    assert "सप" not in stems and "अप" not in stems


def _scorer(pairs, num_levels=20):
    counts = Counter(w for h, _ in pairs for w in hindi_words(h))
    s = DifficultyScorer(FrequencyTable.from_counter(counts), DifficultyConfig(num_levels=num_levels, rank_cap=300))
    s.fit_levels([h for h, _ in pairs])
    return s


def test_scorer_orders_easy_before_hard(pairs):
    s = _scorer(pairs)
    easy = ["यह मेरा घर है।", "मैं ठीक हूँ।", "आप कैसे हैं?"]
    hard = ["यदि समय पर उपचार किया जाता तो रोगी की जान बच सकती थी।", "यह पत्र मेरे दादाजी द्वारा लिखा गया था।"]
    assert max(s.score(x) for x in easy) < min(s.score(x) for x in hard)
    assert all(s.level(x) <= 5 for x in easy)
    assert all(s.level(x) >= 15 for x in hard)


def test_levels_cover_range_and_roundtrip(tmp_path, pairs):
    s = _scorer(pairs)
    levels = {s.level(h) for h, _ in pairs}
    assert min(levels) == 1 and max(levels) == 20
    assert len(s.level_summaries) == 20
    s.save(tmp_path)
    s2 = DifficultyScorer.load(tmp_path)
    for h, _ in pairs:
        assert s2.level(h) == s.level(h)
    a = s2.analyze("मैंने उससे पत्र लिखवाया।")
    assert set(a) >= {"level", "score", "components", "constructions", "rare_words"}
