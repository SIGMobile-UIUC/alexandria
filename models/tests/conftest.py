"""Shared fixtures: tiny randomly-initialised models built locally (no downloads), so the
whole pipeline can be smoke-tested on a CPU in a couple of minutes."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def fixture_pairs() -> list[tuple[str, str]]:
    with (FIXTURES / "parallel.tsv").open(encoding="utf-8") as f:
        return [(r["hi"], r["en"]) for r in csv.DictReader(f, delimiter="\t")]


@pytest.fixture(scope="session")
def pairs() -> list[tuple[str, str]]:
    return fixture_pairs()


@pytest.fixture(scope="session")
def hindi_corpus_file(tmp_path_factory, pairs) -> Path:
    path = tmp_path_factory.mktemp("corpus") / "hindi.txt"
    # Repeat so word counts are non-trivial.
    path.write_text("\n".join(h for h, _ in pairs * 5) + "\n", encoding="utf-8")
    return path


def _train_bpe_tokenizer(texts: list[str], out_dir: Path, vocab_size: int = 600):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(texts, trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, bos_token="<s>", eos_token="</s>", pad_token="<pad>", unk_token="<unk>"
    )
    fast.save_pretrained(out_dir)
    return fast


@pytest.fixture(scope="session")
def tiny_causal_lm(tmp_path_factory, pairs) -> Path:
    """A 2-layer Llama with a BPE tokenizer trained on the fixture sentences."""
    from transformers import LlamaConfig, LlamaForCausalLM

    from alexandria_models.generator.prompts import build_completion, build_prompt

    out = tmp_path_factory.mktemp("tiny_lm")
    texts = [build_prompt(5, 20, "पानी", "future") + build_completion(h, e) for h, e in pairs]
    tok = _train_bpe_tokenizer(texts * 3, out)
    cfg = LlamaConfig(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=512,
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
    )
    LlamaForCausalLM(cfg).save_pretrained(out)
    return out


@pytest.fixture(scope="session")
def tiny_encoder(tmp_path_factory, pairs) -> Path:
    """A 2-layer BERT with a WordPiece vocabulary covering the fixture sentences."""
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    out = tmp_path_factory.mktemp("tiny_encoder")
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    trainer = trainers.WordPieceTrainer(vocab_size=800, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    tok.train_from_iterator([t for p in pairs for t in p] * 3, trainer)
    vocab = sorted(tok.get_vocab().items(), key=lambda kv: kv[1])
    (out / "vocab.txt").write_text("\n".join(w for w, _ in vocab) + "\n", encoding="utf-8")
    fast = BertTokenizerFast(vocab_file=str(out / "vocab.txt"), do_lower_case=False, strip_accents=False)
    fast.save_pretrained(out)
    cfg = BertConfig(vocab_size=len(fast), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=256, num_labels=1)
    BertForSequenceClassification(cfg).save_pretrained(out)
    return out
