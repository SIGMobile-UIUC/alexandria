"""End-to-end smoke test of every training script and the serving layer, on CPU, with
tiny randomly initialised models and the fixture corpora (no network needed).

It checks that the pipeline runs and the saved artifacts load and answer; it says
nothing about model quality (the models are random)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import FIXTURES

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory, hindi_corpus_file, tiny_causal_lm, tiny_encoder) -> Path:
    from alexandria_models.difficulty import build as difficulty_build
    from alexandria_models.evaluator import prepare_data as eval_data, train as eval_train
    from alexandria_models.generator import prepare_data as gen_data, train as gen_train
    from alexandria_models.transliterator import prepare_data as tr_data, train as tr_train

    root = tmp_path_factory.mktemp("run")
    art, data = root / "artifacts", root / "data"
    par = str(FIXTURES / "parallel.tsv")

    difficulty_build.main([
        "--output_dir", str(art / "difficulty"), "--mono_sources", str(hindi_corpus_file),
        "--parallel_sources", par, "--reference_per_source", "100", "--rank_cap", "300",
        "--min_stem_count", "2",
    ])
    gen_data.main([
        "--difficulty_dir", str(art / "difficulty"), "--output_dir", str(data / "generator"),
        "--sources", par, "--per_level", "10",
    ])
    gen_train.main([
        "--data_dir", str(data / "generator"), "--output_dir", str(art / "generator"),
        "--base_model", str(tiny_causal_lm), "--max_steps", "3", "--batch_size", "4", "--grad_accum", "1",
        "--eval_steps", "2", "--logging_steps", "1",
    ])
    eval_data.main([
        "--output_dir", str(data / "evaluator"), "--no_wmt", "--parallel_sources", par, "--synthetic_pairs", "60",
    ])
    eval_train.main([
        "--data_dir", str(data / "evaluator"), "--output_dir", str(art / "evaluator"),
        "--base_model", str(tiny_encoder), "--max_steps", "3", "--batch_size", "8", "--eval_steps", "2",
    ])
    tr_data.main([
        "--output_dir", str(data / "transliterator"), "--no_aksharantar", "--no_dakshina",
        "--extra", str(FIXTURES / "translit_words.tsv"),
    ])
    tr_train.main([
        "--data_dir", str(data / "transliterator"), "--output_dir", str(art / "transliterator"),
        "--epochs", "2", "--batch_size", "16", "--d_model", "32", "--nhead", "2", "--layers", "1",
        "--log_every", "1",
    ])
    return art


def test_artifacts_written(artifacts):
    assert (artifacts / "difficulty" / "scorer.json").exists()
    assert (artifacts / "difficulty" / "word_freq.tsv.gz").exists()
    assert (artifacts / "difficulty" / "bigram_freq.tsv.gz").exists()
    cfg = json.loads((artifacts / "generator" / "generator_config.json").read_text())
    assert cfg["merged"] and cfg["num_levels"] == 20
    assert (artifacts / "generator" / "merged" / "config.json").exists()
    assert (artifacts / "generator" / "adapter" / "adapter_config.json").exists()
    assert (artifacts / "generator" / "sentence_bank.jsonl").exists()
    assert (artifacts / "evaluator" / "model" / "config.json").exists()
    assert (artifacts / "transliterator" / "model.pt").exists()


def test_inference(artifacts):
    from alexandria_models.inference import AlexandriaModels

    m = AlexandriaModels.load(artifacts, device="cpu")
    status = m.status()
    assert all(status[k] for k in ("difficulty", "generator", "evaluator", "transliterator"))
    assert status["num_levels"] == 20 and status["sentence_bank"] > 0

    level = next(iter(m.bank.by_level))
    got = m.sentences(level=level, n=2, mode="bank")
    assert got and all(s["hindi"] and s["english"] and s["level"] == level for s in got)
    # Random weights rarely produce a parseable sentence; this only checks it runs.
    assert isinstance(m.sentences(level=5, n=1, mode="generate"), list)

    r = m.evaluate("मैं घर जा रहा हूँ।", "I am going home")
    assert 0.0 <= r["score"] <= 1.0 and r["source_lang"] == "hi"
    r = m.evaluate("I am going home.", "main ghar ja raha hoon")
    assert r["romanized"] and r["source_lang"] == "en" and r["readings"]
    assert isinstance(m.transliterate("main ghar ja raha hoon", k=2), list)
    d = m.difficulty("यह पत्र मेरे दादाजी द्वारा लिखा गया था।")
    assert 1 <= d["level"] <= 20 and "passive" in d["constructions"]


def test_api(artifacts):
    from fastapi.testclient import TestClient

    from alexandria_models.api import create_app

    with TestClient(create_app(str(artifacts), device="cpu")) as client:
        assert client.get("/health").json()["evaluator"] is True
        levels = client.get("/levels").json()
        assert levels["num_levels"] == 20 and levels["constructions"]
        level = levels["levels"][0]["level"]
        r = client.post("/sentences", json={"level": level, "n": 1, "mode": "bank"})
        assert r.status_code == 200
        assert client.post("/sentences", json={"level": 99}).status_code == 422
        r = client.post("/evaluate", json={"source": "I am fine.", "answer": "main theek hoon"})
        assert r.status_code == 200 and 0 <= r.json()["score"] <= 1
        assert client.post("/difficulty", json={"sentence": "मैं ठीक हूँ।"}).json()["level"] >= 1
        assert "readings" in client.post("/transliterate", json={"text": "kya"}).json()
