# Alexandria models

Training and inference code for the Hindi learning app's models. Everything trains on a
free Google Colab **T4** GPU and is saved to plain directories that the FastAPI backend
loads with one call.

| Part | What it does | Model |
|---|---|---|
| **Difficulty scorer** | Grades any Hindi sentence into **20 levels** and names the grammar it uses | Rules + word frequencies from Hindi corpora (no GPU) |
| **Sentence generator** | Produces a Hindi sentence **and its English translation** at a requested level, optionally practising a given word or grammar construction | LoRA fine-tune of a free open-weight LLM (default Llama-3.2-3B) on English-Hindi parallel data |
| **Translation evaluator** | Scores a learner's translation from 0 to 1, in both directions (Hindi→English and English→Hindi) | MuRIL cross-encoder trained on human quality scores + synthetic learner mistakes |
| **Transliterator** | Turns romanized Hindi ("main ghar ja raha hoon") into Devanagari (मैं घर जा रहा हूँ) so it can be scored | Small character-level Transformer trained from scratch, plus corpus word statistics for context |

## How the pieces fit together

```
Hindi corpora ──► difficulty scorer (20 levels, grammar tags, word frequencies)
                         │
parallel en-hi ──────────┼──► generator data (pairs balanced across levels) ──► generator (LoRA)
corpora                  │                                   └─► sentence bank (real sentences per level)
                         │
WMT23 human scores ──────┴──► evaluator data (+ synthetic learner mistakes) ──► evaluator
Aksharantar + Dakshina ──────► transliterator
```

At serving time:

```python
from alexandria_models.inference import AlexandriaModels

models = AlexandriaModels.load("artifacts")                 # once, at startup

models.sentences(level=5, n=3)                               # [{"hindi", "english", "level", ...}]
models.sentences(level=12, n=3, word="पानी")                 # practise a word
models.sentences(level=14, n=3, grammar="perfective_ne")     # practise a construction
models.evaluate("मैं घर जा रहा हूँ।", "I am going home")      # {"score": 0.96, ...}
models.evaluate("I am going home", "main ghar ja raha hoon") # romanized Hindi answer works
models.difficulty("यह पत्र मेरे दादाजी द्वारा लिखा गया था।")   # {"level": 19, "constructions": [...]}
models.transliterate("kya haal hai", k=3)
```

Or mount the ready-made FastAPI router in your backend:

```python
from fastapi import FastAPI
from alexandria_models.api import lifespan_for, router

app = FastAPI(lifespan=lifespan_for("artifacts"))
app.include_router(router, prefix="/ml")
# GET /ml/health, GET /ml/levels, POST /ml/sentences, /ml/evaluate, /ml/difficulty, /ml/transliterate
```

## Training on Colab

Open [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) in Colab
(Runtime → Change runtime type → **T4 GPU**) and run the cells top to bottom. Everything
is written to Google Drive, and every training step can resume after a disconnect
(`--resume`). The commands it runs, in order:

```bash
pip install -e "models[train,serve]"
A=/content/drive/MyDrive/alexandria/artifacts   D=/content/drive/MyDrive/alexandria/data

# 1. difficulty scorer: word frequencies from Hindi corpora + level thresholds (CPU, ~30-60 min)
python -m alexandria_models.difficulty.build --output_dir $A/difficulty

# 2. generator: level the parallel data, then LoRA fine-tune (GPU, a few hours for the 3B model)
python -m alexandria_models.generator.prepare_data --difficulty_dir $A/difficulty --output_dir $D/generator
python -m alexandria_models.generator.train --data_dir $D/generator --output_dir $A/generator

# 3. evaluator (GPU, ~30 min)
python -m alexandria_models.evaluator.prepare_data --output_dir $D/evaluator
python -m alexandria_models.evaluator.train --data_dir $D/evaluator --output_dir $A/evaluator

# 4. transliterator (GPU, ~30 min)
python -m alexandria_models.transliterator.prepare_data --output_dir $D/transliterator
python -m alexandria_models.transliterator.train --data_dir $D/transliterator --output_dir $A/transliterator

# 5. (recommended) pre-generate sentences so a CPU server never has to run the LLM
python -m alexandria_models.generator.generate --generator_dir $A/generator \
    --difficulty_dir $A/difficulty --evaluator_dir $A/evaluator \
    --per_level 200 --out $A/generator/generated_bank.jsonl
```

Times are rough estimates for a free T4; run a short `--max_steps 50` trial first (the
notebook does) to check memory and loss before a long run. Every script has `--help`.

### Hugging Face access (one-time)

Some models and datasets are free but **gated**: you click "Agree" on their Hugging Face
page with your account, then give Colab a token (Colab sidebar → 🔑 Secrets → add
`HF_TOKEN` from <https://huggingface.co/settings/tokens>).

| Needs "Agree" | Used for | If you skip it |
|---|---|---|
| [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) | default generator base model | use `--preset qwen3-4b` (no sign-up) |
| [ai4bharat/BPCC](https://huggingface.co/datasets/ai4bharat/BPCC) | everyday human-translated sentences | skipped automatically (less conversational data) |
| [ai4bharat/samanantar](https://huggingface.co/datasets/ai4bharat/samanantar) | extra long/formal pairs for high levels | skipped automatically |

Every corpus is optional: one that fails to load is logged and skipped, so a run never
dies on a single source. Check the logs for `source ... failed` lines.

## Choosing the generator's base model

All of these are free, open-weight models. Pass one with `--preset`, or any Hugging Face
causal LM with `--base_model`.

| Preset | Model | License | Gated | Hindi tokens/word | Notes |
|---|---|---|---|---|---|
| `llama3.2-3b` (default) | meta-llama/Llama-3.2-3B | Llama 3.2 Community | yes | 2.8 | Hindi officially supported; stable in fp16 on a T4 |
| `llama3.2-1b` | meta-llama/Llama-3.2-1B | Llama 3.2 Community | yes | 2.8 | ~3x faster; good first run |
| `qwen3-4b` | Qwen/Qwen3-4B-Base | Apache-2.0 | no | 4.7 | best no-sign-up option |
| `qwen3-1.7b` | Qwen/Qwen3-1.7B-Base | Apache-2.0 | no | 4.7 | smaller, weaker Hindi |
| `pragna-1b` | soketlabs/pragna-1b | Apache-2.0 | no | 1.6 | Hindi-focused 1.25B model; weaker overall |
| `qwen3.5-2b` | Qwen/Qwen3.5-2B-Base | Apache-2.0 | no | 3.3 | experimental: slow on a T4 (no fast kernels for its linear attention) |

"Tokens per word" is how many pieces the model's tokenizer cuts an average Hindi word into
(from public FLORES/FLEURS comparisons); fewer means faster training and usually better
Hindi. Not included on purpose: **Gemma 3/4** (their activations overflow float16 on a T4
unless you use Unsloth's patches) and **Sarvam-1** (best Hindi tokenizer, but a
non-commercial license).

If training logs a NaN loss, the run stops with a message; re-run with `--fp32` or switch preset.

## Difficulty levels

`difficulty/features.py` detects 32 grammar constructions, ordered roughly the way Hindi
courses teach them: copula है → possessives → questions → negation → postpositions →
past था → imperatives → habitual present → progressive → future → मुझे-constructions →
simple past → ने-perfective → … → relative clauses (जो…वह) → subjunctive → participles
(-ते हुए) → passive → causatives (लिखवाया) → counterfactuals (काश…) → formal register
(तथा, एवं, द्वारा). Most verb forms are matched as *known verb stem + suffix*; verb stems
are mined from the corpus, which stops nouns like कुत्ता or सफलता being read as verbs.

A sentence's score combines:

* **grammar**: the hardest construction it contains, and how many it combines
* **vocabulary**: how rare its words are in the corpus (the two rarest words count extra)
* **length**: number of words
* **script**: conjunct consonants per word (स्वास्थ्य, प्रतिनिधित्व)

The weights are in `DifficultyConfig`. Levels are equal-sized slices of the score
distribution over the training sentences, so each of the 20 levels gets the same amount of
training data. `GET /levels` (or `scorer.level_summaries`) shows example sentences and the
typical grammar for every level; this is the place to start when building the skill tree.
To change the number of levels, pass `--num_levels` to `difficulty.build` and re-run the
generator steps. **The scoring is a heuristic you should tune**: read the per-level examples
it logs and adjust the weights or the construction order.

## How the evaluator handles Hinglish

* The direction is detected from the script of the sentence shown to the learner.
* If the learner should answer in Hindi but typed Latin letters, the transliterator
  produces the top 3 Devanagari readings. The evaluator scores each one and keeps the best,
  so a transliteration slip does not mark a correct answer wrong. The response shows how
  the answer was read (`interpreted_as`), which is useful feedback in the app.
* English words inside a Hinglish answer get transliterated phonetically (office → ऑफिस);
  MuRIL was pretrained on such text and handles it reasonably.
* Training data includes learner-style slips with graded scores. Spelling slips (ि/ी, ं/ँ)
  score ~0.9. Gender or number agreement errors score ~0.75, a wrong postposition ~0.6,
  wrong tense ~0.45, missing half the sentence ~0.4, a wrong pronoun ~0.3, and flipped
  negation ~0.15. Unrelated answers score ~0.

## Serving on a CPU-only server

* The evaluator (~240M parameters), transliterator (~7M) and difficulty scorer run fine
  on a CPU.
* The generator does not. A 3B model in float32 needs ~12 GB RAM and takes seconds per
  sentence. Pre-generate sentences on Colab (step 5) and load without it:
  `AlexandriaModels.load("artifacts", components=("difficulty", "evaluator", "transliterator"))`.
  `sentences()` then serves from `sentence_bank.jsonl` (real corpus sentences) and
  `generated_bank.jsonl`.
* Endpoints are synchronous functions, so FastAPI runs them in its thread pool; each
  model guards itself with a lock.

## Output layout

```
artifacts/
  difficulty/      scorer.json, word_freq.tsv.gz, bigram_freq.tsv.gz, verb_stems.txt
  generator/       merged/ (full fp16 model), adapter/ (LoRA only), generator_config.json,
                   sentence_bank.jsonl, generated_bank.jsonl
  evaluator/       model/ (transformers model + tokenizer), evaluator_config.json
  transliterator/  model.pt, config.json, vocab.json
```

## Data sources

| Source | Kind | License | Used for |
|---|---|---|---|
| AI4Bharat Sangraha (verified Hindi) | monolingual | CC-BY-4.0 | word frequencies |
| FineWeb-2 (hin_Deva) | monolingual | ODC-By | word frequencies |
| AI4Bharat IndicCorp v2 | monolingual | CC0 | word frequencies |
| Hindi Wikipedia | monolingual | CC-BY-SA | word frequencies |
| Tatoeba (exports / ManyThings) | parallel | CC-BY 2.0 FR | short everyday sentences |
| AI4Bharat BPCC (daily, massive, wiki) | parallel | CC-BY-4.0 / CC0 | everyday + general sentences (gated) |
| OPUS-100 en-hi | parallel | mixed | extra short sentences (noisy, capped) |
| IIT Bombay en-hi | parallel | CC-BY-NC-4.0 | longer, formal sentences |
| AI4Bharat Samanantar | parallel | CC-BY-NC-4.0 | longer sentences (gated) |
| WMT23 QE en-hi | human quality scores | see repo | evaluator |
| AI4Bharat Aksharantar | transliteration pairs | see card | transliterator |
| Google Dakshina | transliteration pairs | CC-BY-SA-4.0 | transliterator |

Several sources are **non-commercial** (IITB, Samanantar). That is fine for a club project,
but drop them (`--sources` / `--parallel_sources`) if the app is ever used commercially. Any
local file can also be used as a source: a `.txt` with one Hindi sentence per line, or a
`.tsv`/`.jsonl` with `hi` and `en` columns.

## Honest limitations

* **Generated sentences can be wrong.** The generator's output is filtered: it must parse,
  look like clean Hindi/English, measure within 2 levels of the request, and (with the
  evaluator loaded) have a matching translation. That does not guarantee perfect Hindi.
  Real corpus sentences from the bank are the safe default (`mode="bank"`); review
  generated ones before relying on them.
* **The evaluator's human data is machine-translation quality scores** (WMT23, English→Hindi
  only). Learner mistakes and the Hindi→English direction are covered by synthetic data, so
  treat scores as a guide. Thresholds like "≥0.8 = correct" should be calibrated on real
  learner answers.
* **Transliteration of informal romanization is hard.** The best published systems get
  ~60% of Hindi words exactly right on Dakshina. That is why the evaluator tries several
  readings. Expect this model to be somewhat below that.

## Tests

```bash
cd models
pip install -e ".[train,serve,dev]"
pytest -q            # ~30 tests; includes a CPU end-to-end run with tiny random models
```

The end-to-end test trains tiny randomly initialised models on the fixtures in
`tests/fixtures/` and checks that every script runs and every artifact loads and answers
through `AlexandriaModels` and the FastAPI router. It checks plumbing, not model quality.
