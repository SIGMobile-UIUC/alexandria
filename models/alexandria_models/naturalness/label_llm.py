"""Rate sentences 1-5 for learner suitability with a large open LLM (GPU).

No text is generated: the model reads the prompt once and we read its probabilities for
the answers "1".."5". The label is the expected rating mapped to 0..1, which is smoother
than the single most likely digit and costs one forward pass per sentence.

The default judge, Qwen3-8B (Apache-2.0, no access form), fits an L4 or A100 in bf16;
on a T4 add ``--load_in_4bit``. Any chat model with good Hindi works (``--model``).
Results are appended as they are produced, so a disconnect loses at most one batch:
re-run the same command to continue.

    python -m alexandria_models.naturalness.label_llm \\
        --input data_v2/naturalness/to_judge.jsonl --output data_v2/naturalness/judged.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..common.io import get_logger, read_jsonl

log = get_logger("naturalness.label_llm")
DEFAULT_JUDGE = "Qwen/Qwen3-8B"

INSTRUCTIONS = """You are an experienced Hindi teacher choosing example sentences for an app that teaches Hindi to English speakers.
Rate the Hindi sentence from 1 to 5.
5 = a complete, natural sentence a native speaker would really say or write in everyday life; correct and useful to learn
4 = natural and correct, but a little formal, specific or less useful
3 = understandable but awkward, bookish, news-style, or needs context to make sense
2 = unnatural: sounds translated, a headline or title, a fragment, a command for a device, or religious/archaic language
1 = ungrammatical, broken, not really Hindi, offensive, or nonsense"""


def build_prompt(tokenizer, hi: str, en: str | None) -> str:
    body = f"{INSTRUCTIONS}\n\nHindi: {hi}\n"
    if en:
        body += f"English meaning: {en}\n"
    body += "\nAnswer with a single digit from 1 to 5."
    if getattr(tokenizer, "chat_template", None):
        msgs = [{"role": "user", "content": body}]
        try:  # Qwen3: answer directly, without a thinking block
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return body + "\nRating:"


def digit_token_ids(tokenizer) -> list[list[int]]:
    """For each of 1..5, the token ids that start that answer ("3" and " 3")."""
    out = []
    for d in "12345":
        ids = set()
        for variant in (d, " " + d):
            enc = tokenizer.encode(variant, add_special_tokens=False)
            if enc:
                ids.add(enc[-1])
        out.append(sorted(ids))
    return out


def label(args: argparse.Namespace) -> Path:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.output)
    done = {r["id"] for r in read_jsonl(out)} if out.exists() else set()
    rows = [r for r in read_jsonl(args.input) if r["id"] not in done]
    if args.limit:
        rows = rows[: max(0, args.limit - len(done))]
    log.info("%d already judged, %d to go", len(done), len(rows))
    if not rows:
        return out

    cuda = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(args.model, token=args.hf_token)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kwargs: dict = {"token": args.hf_token}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        )
        kwargs["device_map"] = {"": 0}
    else:
        bf16 = cuda and torch.cuda.is_bf16_supported()
        kwargs["dtype"] = torch.bfloat16 if bf16 else (torch.float16 if cuda else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if not args.load_in_4bit:
        model.to("cuda" if cuda else "cpu")
    model.eval()
    device = next(model.parameters()).device
    digits = digit_token_ids(tok)
    scale = torch.arange(1, 6, dtype=torch.float32)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i : i + args.batch_size]
            prompts = [build_prompt(tok, r["hi"], r.get("en") if args.with_english else None) for r in batch]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            with torch.inference_mode():
                # logits_to_keep=1: only the last position (full logits would be seq x 150k vocab)
                logits = model(**enc, logits_to_keep=1).logits[:, -1, :].float()
            logp = torch.log_softmax(logits, dim=-1).cpu()
            per_digit = torch.stack([torch.logsumexp(logp[:, ids], dim=-1) for ids in digits], dim=-1)
            probs = torch.softmax(per_digit, dim=-1)  # renormalize over the five answers
            expected = (probs * scale).sum(-1)
            for r, e, p in zip(batch, expected.tolist(), probs.tolist()):
                f.write(json.dumps({**r, "rating": round(e, 3), "label": round((e - 1) / 4, 4),
                                    "probs": [round(x, 3) for x in p]}, ensure_ascii=False) + "\n")
            f.flush()
            if (i // args.batch_size) % 20 == 0:
                log.info("judged %d / %d", i + len(batch), len(rows))
    return out


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="to_judge.jsonl from naturalness.prepare_data")
    p.add_argument("--output", required=True)
    p.add_argument("--model", default=DEFAULT_JUDGE)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--limit", type=int, default=None, help="judge at most this many in total")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--with_english", action=argparse.BooleanOptionalAction, default=True,
                   help="show the English meaning to the judge (helps spot mistranslations)")
    p.add_argument("--hf_token", default=None)
    return p.parse_args(argv)


def main(argv=None) -> None:
    label(parse_args(argv))


if __name__ == "__main__":
    main()
