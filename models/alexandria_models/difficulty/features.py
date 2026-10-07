"""Surface features of a Hindi sentence used to grade its difficulty.

Grammar is detected with token patterns on NFC-normalized Devanagari, ordered roughly
the way Hindi courses introduce it (copula -> postpositions -> present/past/future ->
ने-perfective -> modals -> compound verbs -> relative clauses -> passive -> causatives
-> counterfactuals -> formal Sanskritized register). Each construction carries a
``step`` (its place in that order); the hardest step present is one of the strongest
difficulty signals.

Most verb constructions are recognised as ``verb stem + suffix`` where the stem must be
a known verb stem. The built-in list covers common verbs, and ``derive_verb_stems``
adds stems mined from a corpus frequency table (a stem ``X`` is a verb when both the
infinitive ``Xना`` and a participle ``Xता``/``Xते``/``Xकर`` are frequent). Checking the
stem is what keeps nouns such as कुत्ता, सफलता, रोगी or उपयोगी from being read as verb
forms. The patterns trade some recall for precision: a construction that is missed only
makes a sentence look slightly easier, while a false positive can push an easy sentence
many levels up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

from ..common.text import akshara_count, conjunct_count, hindi_words, normalize

Tokens = Sequence[str]


def _set(words: str) -> frozenset[str]:
    return frozenset(normalize(w) for w in words.split())


BUILTIN_VERB_STEMS = _set(
    "कर खा पी सो जा आ दे ले रह हो कह बैठ उठ सुन देख बोल पढ़ लिख रख बना बन समझ मार गिर भाग "
    "चल रो हँस हंस सीख भूल पहुँच पहुंच निकल बच बचा बदल जल टूट तोड़ फेंक खो मिल छोड़ बता सुना "
    "लौट लौटा थक डर रुक बिक फट मान जान सोच पूछ भेज खोल काट धो बेच ख़रीद खरीद सह दौड़ खेल गा "
    "नाच तैर उड़ चढ़ उतर पकड़ छू घूम बढ़ बढ़ा घट घटा लग मना चाह सक पा दिखा सिखा पढ़ा बुला "
    "बिठा सुला जगा जाग नहा पहन ढूँढ ढूंढ खोज जीत हार लड़ चुन गिन माँग मांग बाँट बांट संभाल "
    "सँभाल भर डाल फैल बीत गुज़र गुजर सुधर सुधार पहचान सजा बिता ला पिला खिला चला झुक मुड़ हिल "
    "सूख भीग पक उबल जोड़ ठहर फिसल चिल्ला मुस्कुरा मुस्करा घबरा पछता दबा उठा गिरा कमा बजा "
    "सुलझा उलझ बिगड़ बिगाड़ समझा मिला दिला करा करवा लिखवा बनवा कटवा धुलवा सिलवा पिट पीट "
    "भूलना चुक बैठा लेट जीत छिप छिपा बदला बरस बह बहा"
)

COPULA_PRESENT = _set("है हैं हूँ हूं हो")
COPULA_PAST = _set("था थी थे थीं")
COPULA = COPULA_PRESENT | COPULA_PAST
POSTPOSITIONS = _set("में पर से को का की के तक लिए साथ बाद पहले पास")
POSSESSIVE_PRONOUNS = _set(
    "मेरा मेरी मेरे तेरा तेरी तेरे उसका उसकी उसके इसका इसकी इसके हमारा हमारी हमारे "
    "तुम्हारा तुम्हारी तुम्हारे आपका आपकी आपके उनका उनकी उनके इनका इनकी इनके अपना अपनी अपने"
)
QUESTION_WORDS = _set(
    "क्या कौन कहाँ कहां कब क्यों कैसे कैसा कैसी कितना कितनी कितने किसका किसकी किसके किसे किसको"
)
NEGATION = _set("नहीं नही न मत")
ERGATIVE_PRONOUNS = _set("मैंने तूने उसने इसने हमने तुमने आपने उन्होंने इन्होंने किसने जिसने जिन्होंने किन्होंने")
DATIVE_PRONOUNS = _set("मुझे मुझको तुझे उसे इसे हमें तुम्हें उन्हें इन्हें आपको किसे जिसे जिन्हें")
PROGRESSIVE_AUX = _set("रहा रही रहे रहीं")
VECTOR_VERBS = _set(
    "लिया ली लिए लिये लेना लेगा लेगी लेंगे लो लीजिए लूँगा लूंगा लेता लेती लेते "
    "दिया दी दिए दिये देना देगा देगी देंगे दो दीजिए दूँगा दूंगा देता देती देते "
    "गया गई गए गयी गये जाएगा जाएगी जाएँगे जाएंगे जाता जाती जाते जाना "
    "डाला डाली डाले डालो डालना बैठा बैठी बैठे पड़ा पड़ी पड़े उठा उठी उठे"
)
IRREGULAR_PERFECTIVE = _set(
    "गया गई गए गयी गये गईं हुआ हुई हुए हुईं किया किए किये कीं दिया दी दिए दिये दीं "
    "लिया ली लिए लिये लीं पिया पिए पिये"
)
IRREGULAR_FUTURE = _set("होगा होगी होंगे होंगी होगे होऊँगा देगा देगी देंगे दूँगा दूंगा लेगा लेगी लेंगे लूँगा लूंगा")
IRREGULAR_POLITE_IMPERATIVE = _set("कीजिए कीजिये दीजिए दीजिये लीजिए लीजिये पीजिए पीजिये")
NOT_POLITE_IMPERATIVE = _set("चाहिए चाहिये लिए लिये किए किये दिए दिये पिए पिये")
NOT_FUTURE = _set("आगे जागा जागी")
CONDITIONAL = _set("अगर यदि")
SUBJUNCTIVE_MARKERS = _set("ताकि शायद चाहे कदाचित् कदाचित")
RELATIVE = _set(
    "जो जिस जिसे जिसको जिसका जिसकी जिसके जिसने जिन जिन्हें जिनका जिनकी जिनके "
    "जिन्होंने जहाँ जहां जिधर जितना जितनी जितने जब जैसे जैसा जैसी"
)
CORRELATIVE = _set("वह वो वे उस उसे उन वहाँ वहां तब तो उतना उतनी उतने उधर वैसे वैसा वैसी")
PARTICIPLE_HUA = _set("हुआ हुई हुए हुईं")
VAALA = _set("वाला वाली वाले")
PASSIVE_AUX = _set("जाता जाती जाते जाएगा जाएगी जाएँगे जाएंगे गया गई गए गयी गये जाए जाएँ जाएं जाना")
NOT_PASSIVE_PARTICIPLE = _set("चला भागा")
CAUSATIVE_EXCLUDE = _set("दवाई हवाई कार्रवाई कार्यवाही गवाही रवाना")
FORMAL_CONNECTIVES = _set(
    "तथा एवं अथवा किंतु किन्तु परंतु परन्तु अतः अतएव यद्यपि तथापि हेतु द्वारा अंतर्गत अन्तर्गत "
    "फलस्वरूप उपरांत उपरान्त पश्चात् पश्चात समक्ष संबंधी सम्बन्धी"
)
URDU_CONNECTIVES = _set("हालाँकि हालांकि बल्कि मगर बावजूद चूँकि चूंकि वरना")

HABITUAL_SUFFIXES = ("तीं", "ता", "ती", "ते")
PERFECTIVE_SUFFIXES = ("यीं", "या", "यी", "ये", "ईं", "ई", "ए", "ीं", "ा", "ी", "े")
FUTURE_SUFFIXES = (
    "ूँगा", "ूंगा", "ूँगी", "ूंगी", "ेंगे", "ेंगी", "ेगा", "ेगी", "ोगे", "ोगी",
    "ऊँगा", "ऊंगा", "ऊँगी", "ऊंगी", "एँगे", "एंगे", "एँगी", "एंगी", "एगा", "एगी", "ओगे", "ओगी",
    "येगा", "येगी", "येंगे", "येंगी",
)
INFINITIVE_SUFFIXES = ("ना", "नी", "ने")
FAMILIAR_IMPERATIVE_SUFFIXES = ("ओ", "ो")
POLITE_IMPERATIVE_SUFFIXES = ("िए", "िये", "इए", "इये")
CONJ_PARTICIPLE_SUFFIXES = ("करके", "कर")
CAUSATIVE_SUFFIXES = tuple(
    "वा" + s for s in ("या", "यी", "ये", "ई", "ए", "ना", "नी", "ने", "ता", "ती", "ते", "ऊँगा", "एगा", "ओ", "इए")
)
SAKNA_FORMS = frozenset(
    "सक" + s for s in ("ता", "ती", "ते", "तीं", "ा", "ी", "े", "ीं", "ूँ", "ूं", "ें", "ेंगे", "ेगा", "ेगी",
                      "ूँगा", "ूंगा", "ूँगी", "ूंगी", "ोगे", "ोगी", "ना")
)
CHUKNA_FORMS = frozenset("चुक" + s for s in ("ा", "ी", "े", "ीं", "ता", "ती", "ते", "ेगा", "ेगी", "ेंगे", "ूँगा", "ूंगा"))
PAANA_FORMS = frozenset("पा" + s for s in ("ता", "ती", "ते", "या", "ई", "ए", "एगा", "एगी", "एँगे", "एंगे", "ऊँगा", "ऊंगा"))
PADNA_FORMS = frozenset("पड़" + s for s in ("ता", "ती", "ते", "ा", "ी", "े", "ेगा", "ेगी", "ेंगे"))


def derive_verb_stems(
    counts: Mapping[str, int], min_infinitive_count: int = 20, min_participle_count: int = 5
) -> frozenset[str]:
    """Mine verb stems from a word-frequency table.

    A stem ``X`` is accepted when ``Xना`` is frequent and at least one of ``Xता``,
    ``Xते``, ``Xती`` or ``Xकर`` also occurs. Nouns that happen to end in -ना
    (सपना, अपना, पुराना) fail the second test because they have no participles.
    """
    stems = set()
    for word, c in counts.items():
        if c < min_infinitive_count or not word.endswith("ना") or len(word) < 3:
            continue
        stem = word[:-2]
        if any(counts.get(stem + s, 0) >= min_participle_count for s in ("ता", "ते", "ती", "कर")):
            stems.add(stem)
    return frozenset(stems)


@dataclass(frozen=True)
class Construction:
    name: str
    step: int  # position in the teaching order; higher = learned later
    description: str
    detect: Callable[["_Ctx"], bool]


class _Ctx:
    """Tokens plus memoized morphological lookups for one sentence."""

    def __init__(self, tokens: Tokens, stems: frozenset[str]):
        self.toks = list(tokens)
        self.stems = stems
        self.pairs = list(zip(self.toks, self.toks[1:]))

    def has_suffix_on_stem(self, tok: str, suffixes: Iterable[str]) -> bool:
        for s in suffixes:
            if tok.endswith(s) and len(tok) > len(s) and tok[: -len(s)] in self.stems:
                return True
        return False

    def any_tok(self, vocab: frozenset[str]) -> bool:
        return any(t in vocab for t in self.toks)

    def habitual(self, tok: str) -> bool:
        return self.has_suffix_on_stem(tok, HABITUAL_SUFFIXES)

    def perfective(self, tok: str) -> bool:
        if tok in IRREGULAR_PERFECTIVE:
            return True
        if tok in PROGRESSIVE_AUX or tok.startswith("जा"):
            return False
        return self.has_suffix_on_stem(tok, PERFECTIVE_SUFFIXES)

    def infinitive(self, tok: str) -> bool:
        return self.has_suffix_on_stem(tok, INFINITIVE_SUFFIXES)


def _vocab(words: frozenset[str]) -> Callable[[_Ctx], bool]:
    return lambda c: c.any_tok(words)


def _imperative(c: _Ctx) -> bool:
    for t in c.toks:
        if t in IRREGULAR_POLITE_IMPERATIVE:
            return True
        if t in NOT_POLITE_IMPERATIVE:
            continue
        if c.has_suffix_on_stem(t, POLITE_IMPERATIVE_SUFFIXES + FAMILIAR_IMPERATIVE_SUFFIXES):
            return True
    return False


def _habitual_present(c: _Ctx) -> bool:
    return any(c.habitual(a) and b in COPULA_PRESENT and a not in SAKNA_FORMS for a, b in c.pairs)


def _habitual_past(c: _Ctx) -> bool:
    return any(c.habitual(a) and b in COPULA_PAST and a not in SAKNA_FORMS for a, b in c.pairs)


def _progressive(c: _Ctx) -> bool:
    return any(b in PROGRESSIVE_AUX and (a in c.stems) for a, b in c.pairs)


def _future(c: _Ctx) -> bool:
    return any(
        t not in NOT_FUTURE and (t in IRREGULAR_FUTURE or c.has_suffix_on_stem(t, FUTURE_SUFFIXES))
        for t in c.toks
    )


def _simple_past(c: _Ctx) -> bool:
    # A perfective verb form that is not just the vector of a compound verb.
    return any(c.perfective(t) for t in c.toks)


def _ergative(c: _Ctx) -> bool:
    return any(t in ERGATIVE_PRONOUNS or t == "ने" for t in c.toks)


def _dative_subject(c: _Ctx) -> bool:
    return c.any_tok(DATIVE_PRONOUNS)


def _obligation(c: _Ctx) -> bool:
    return "चाहिए" in c.toks or "चाहिये" in c.toks


def _compulsion(c: _Ctx) -> bool:
    return any(c.infinitive(a) and (b in PADNA_FORMS or b in _set("होगा होगी होंगे")) for a, b in c.pairs) or any(
        a in DATIVE_PRONOUNS and c.infinitive(b) and nxt in COPULA
        for a, b, nxt in zip(c.toks, c.toks[1:], c.toks[2:])
    ) or any(c.infinitive(a) and b in COPULA_PRESENT for a, b in c.pairs if a.endswith("ना"))


def _conj_participle(c: _Ctx) -> bool:
    if any(a in c.stems and b in _set("कर करके") for a, b in c.pairs):
        return True
    return any(c.has_suffix_on_stem(t, CONJ_PARTICIPLE_SUFFIXES) for t in c.toks)


def _compound_verb(c: _Ctx) -> bool:
    return any(a in c.stems and b in VECTOR_VERBS for a, b in c.pairs)


def _sakna(c: _Ctx) -> bool:
    return c.any_tok(SAKNA_FORMS)


def _chukna(c: _Ctx) -> bool:
    return c.any_tok(CHUKNA_FORMS)


def _paana(c: _Ctx) -> bool:
    return any(a in c.stems and b in PAANA_FORMS for a, b in c.pairs)


def _relative(c: _Ctx) -> bool:
    for i, t in enumerate(c.toks):
        if t in RELATIVE:
            return any(u in CORRELATIVE for u in c.toks[i + 1 :])
    return False


def _subjunctive(c: _Ctx) -> bool:
    return c.any_tok(SUBJUNCTIVE_MARKERS)


def _participle_hua(c: _Ctx) -> bool:
    return any((c.habitual(a) or c.perfective(a)) and b in PARTICIPLE_HUA for a, b in c.pairs) or any(
        c.habitual(a) and b == "ही" for a, b in c.pairs
    )


def _vaala(c: _Ctx) -> bool:
    return any(c.infinitive(a) and b in VAALA for a, b in c.pairs)


def _passive(c: _Ctx) -> bool:
    if "द्वारा" in c.toks:
        return True
    return any(
        b in PASSIVE_AUX and a not in NOT_PASSIVE_PARTICIPLE and a not in c.stems and c.perfective(a)
        for a, b in c.pairs
    )


def _causative(c: _Ctx) -> bool:
    for t in c.toks:
        if t in CAUSATIVE_EXCLUDE:
            continue
        for s in CAUSATIVE_SUFFIXES:
            if t.endswith(s) and len(t) > len(s) and akshara_count(t[: -len(s)]) >= 1:
                stem = t[: -len(s)]
                if stem in c.stems or stem + "वा" in c.stems:
                    return True
    return False


def _counterfactual(c: _Ctx) -> bool:
    if "काश" in c.toks:
        return True
    if not c.any_tok(CONDITIONAL) or "तो" not in c.toks:
        return False
    # "अगर तुम आते तो..." / "यदि ... किया जाता तो ...": an -ता/-ती/-ते participle
    # used as a finite verb, i.e. not followed by a copula.
    nxt = c.toks[1:] + [""]
    return any(
        c.habitual(a) and b not in COPULA and b not in PARTICIPLE_HUA and a not in SAKNA_FORMS
        for a, b in zip(c.toks, nxt)
    )


CONSTRUCTIONS: tuple[Construction, ...] = (
    Construction("copula_present", 1, "है/हैं/हूँ: 'X is Y'", _vocab(COPULA_PRESENT)),
    Construction("possessive_pronoun", 2, "मेरा/तुम्हारा/उसका", _vocab(POSSESSIVE_PRONOUNS)),
    Construction("question_word", 3, "क्या/कौन/कहाँ/कब/क्यों/कैसे", _vocab(QUESTION_WORDS)),
    Construction("negation", 4, "नहीं/न/मत", _vocab(NEGATION)),
    Construction("postposition", 5, "में/पर/से/को/का/की/के/तक", _vocab(POSTPOSITIONS)),
    Construction("copula_past", 6, "था/थी/थे", _vocab(COPULA_PAST)),
    Construction("imperative", 7, "करो / बैठिए / कीजिए", _imperative),
    Construction("habitual_present", 8, "-ता/-ती/-ते + है: पीता हूँ", _habitual_present),
    Construction("progressive", 9, "रहा/रही/रहे: खा रहा हूँ", _progressive),
    Construction("future", 10, "-गा/-गी/-गे: जाएँगे", _future),
    Construction("dative_subject", 11, "मुझे/तुम्हें + पसंद/लगना/चाहिए", _dative_subject),
    Construction("simple_past", 12, "perfective: गया, पहुँचे, हुई", _simple_past),
    Construction("perfective_ne", 13, "ने-construction: मैंने खरीदी", _ergative),
    Construction("habitual_past", 14, "-ता था: रहता था", _habitual_past),
    Construction("ability_sakna", 15, "सकना: कर सकते हैं", _sakna),
    Construction("obligation_chahiye", 16, "चाहिए: जाना चाहिए", _obligation),
    Construction("compulsion", 17, "-ना है / -ना पड़ेगा", _compulsion),
    Construction("conjunctive_participle", 18, "-कर/-के: खाकर", _conj_participle),
    Construction("compound_verb", 19, "कर लिया / सो गया / दे दिया", _compound_verb),
    Construction("complement_ki", 20, "कि-clause: कहा कि", _vocab(_set("कि"))),
    Construction("conditional", 21, "अगर/यदि ... तो", _vocab(CONDITIONAL)),
    Construction("completive_chukna", 22, "चुकना: छूट चुकी थी", _chukna),
    Construction("vaala", 23, "-ने वाला: करने वाले लोग", _vaala),
    Construction("relative_correlative", 24, "जो ... वह / जब ... तब", _relative),
    Construction("subjunctive", 25, "ताकि/शायद/चाहे + subjunctive", _subjunctive),
    Construction("participle_hua", 26, "-ते हुए / -ते ही / बैठा हुआ", _participle_hua),
    Construction("ability_paana", 27, "पाना: आ पाता", _paana),
    Construction("passive", 28, "लिखा गया / किया जाता / द्वारा", _passive),
    Construction("causative", 29, "-वा- causatives: लिखवाया", _causative),
    Construction("counterfactual", 30, "काश / अगर ... आते तो", _counterfactual),
    Construction("urdu_connective", 31, "हालाँकि/बल्कि/बावजूद", _vocab(URDU_CONNECTIVES)),
    Construction("formal_register", 32, "तथा/एवं/अतः/द्वारा/हेतु", _vocab(FORMAL_CONNECTIVES)),
)
MAX_STEP = max(c.step for c in CONSTRUCTIONS)
CONSTRUCTION_NAMES = tuple(c.name for c in CONSTRUCTIONS)
_STEP = {c.name: c.step for c in CONSTRUCTIONS}


@dataclass
class SentenceFeatures:
    tokens: list[str]
    n_tokens: int
    n_aksharas: int
    n_conjuncts: int
    constructions: list[str] = field(default_factory=list)
    max_step: int = 0

    def as_dict(self) -> dict:
        return {
            "n_tokens": self.n_tokens,
            "n_aksharas": self.n_aksharas,
            "n_conjuncts": self.n_conjuncts,
            "constructions": self.constructions,
            "max_step": self.max_step,
        }


def detect_constructions(tokens: Tokens, verb_stems: frozenset[str] = BUILTIN_VERB_STEMS) -> list[str]:
    ctx = _Ctx(tokens, verb_stems)
    return [c.name for c in CONSTRUCTIONS if c.detect(ctx)]


def extract(sentence: str, verb_stems: frozenset[str] = BUILTIN_VERB_STEMS) -> SentenceFeatures:
    tokens = hindi_words(normalize(sentence))
    found = detect_constructions(tokens, verb_stems)
    return SentenceFeatures(
        tokens=tokens,
        n_tokens=len(tokens),
        n_aksharas=sum(akshara_count(t) for t in tokens),
        n_conjuncts=sum(conjunct_count(t) for t in tokens),
        constructions=found,
        max_step=max((_STEP[n] for n in found), default=0),
    )
