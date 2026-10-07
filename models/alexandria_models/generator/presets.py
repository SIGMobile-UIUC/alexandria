"""Named base models for the generator. All are free, open-weight checkpoints.

``tokens_per_word`` is how many tokens the tokenizer spends on an average Hindi word
(lower = faster training, more text per sequence, usually better Hindi). Numbers come
from public tokenizer comparisons on FLORES/FLEURS Hindi; see the README.

Batch sizes are per-device defaults that fit a 16 GB T4 with 4-bit QLoRA. Models with a
very large vocabulary need smaller batches, because the training loss keeps a float32
copy of the full logits (tokens x vocabulary).

Not included on purpose: Gemma 3/4 (activations overflow float16 on a T4 without
Unsloth's patches) and Sarvam-1 (excellent Hindi tokenizer, non-commercial license).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    model_id: str
    license: str
    gated: bool
    tokens_per_word: float
    batch_size: int
    grad_accum: int
    max_seq_len: int
    notes: str


PRESETS: dict[str, Preset] = {
    "llama3.2-3b": Preset(
        "meta-llama/Llama-3.2-3B", "Llama 3.2 Community License", True, 2.8, 16, 2, 192,
        "Default. Hindi is officially supported; standard architecture, stable in fp16 on a T4. "
        "Gated: accept the license on its Hugging Face page and set HF_TOKEN.",
    ),
    "llama3.2-1b": Preset(
        "meta-llama/Llama-3.2-1B", "Llama 3.2 Community License", True, 2.8, 16, 2, 192,
        "Same family, ~3x faster; good for a quick first run. Gated.",
    ),
    "qwen3-4b": Preset(
        "Qwen/Qwen3-4B-Base", "Apache-2.0", False, 4.7, 8, 4, 256,
        "Best option that needs no sign-up. Hindi tokenization is inefficient (longer sequences).",
    ),
    "qwen3-1.7b": Preset(
        "Qwen/Qwen3-1.7B-Base", "Apache-2.0", False, 4.7, 8, 4, 256,
        "Smaller ungated option; faster, weaker Hindi.",
    ),
    "pragna-1b": Preset(
        "soketlabs/pragna-1b", "Apache-2.0", False, 1.6, 16, 2, 192,
        "Hindi/English-focused 1.25B Llama-style model with a very efficient Hindi tokenizer; "
        "weaker general ability than the others.",
    ),
    "qwen3.5-2b": Preset(
        "Qwen/Qwen3.5-2B-Base", "Apache-2.0", False, 3.3, 8, 4, 192,
        "Experimental: newer model (201 languages, better Hindi tokenizer than Qwen3) but its "
        "linear-attention layers fall back to slow PyTorch code on a T4.",
    ),
}
DEFAULT_PRESET = "llama3.2-3b"


def resolve_base_model(preset: str | None, base_model: str | None) -> str:
    if base_model:
        return base_model
    return PRESETS[preset or DEFAULT_PRESET].model_id
