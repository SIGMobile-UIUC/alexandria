"""Plain-text prompt format shared by generator training and inference.

A training example looks like::

    Level: 07/20
    Word: पानी
    Grammar: future
    Hindi: मैं कल पानी लाऊँगा।
    English: I will bring water tomorrow.<eos>

``Word`` and ``Grammar`` are optional controls (present in a random subset of training
examples) so the app can ask for a sentence that practises a given word or grammar
construction. Everything up to and including ``Hindi:`` is the prompt; the model is
trained only on the completion. Plain text is used instead of new special tokens so no
embedding rows have to be added or trained, which LoRA handles poorly.
"""

from __future__ import annotations

import re

PROMPT_FORMAT_VERSION = 1
_COMPLETION_RE = re.compile(r"^\s*(?P<hi>[^\n]+?)\s*\nEnglish:\s*(?P<en>[^\n]+?)\s*(?:\n|$)")


def build_prompt(level: int, num_levels: int, word: str | None = None, grammar: str | None = None) -> str:
    width = len(str(num_levels))
    lines = [f"Level: {level:0{width}d}/{num_levels}"]
    if word:
        lines.append(f"Word: {word}")
    if grammar:
        lines.append(f"Grammar: {grammar}")
    lines.append("Hindi:")
    return "\n".join(lines)


def build_completion(hindi: str, english: str) -> str:
    return f" {hindi}\nEnglish: {english}"


def parse_completion(text: str) -> tuple[str, str] | None:
    """Extract (hindi, english) from generated text; None if the format is broken."""
    m = _COMPLETION_RE.match(text)
    if not m:
        return None
    hi, en = m.group("hi").strip(), m.group("en").strip()
    return (hi, en) if hi and en else None


def bos_prefix(tokenizer) -> list[int]:
    """[bos_token_id] if this tokenizer normally starts sequences with BOS, else [].

    Probes real tokenization instead of trusting ``add_bos_token``: Llama-3 tokenizers
    add BOS through the tokenizer.json post-processor and report add_bos_token=False."""
    bos = tokenizer.bos_token_id
    if bos is None:
        return []
    ids = tokenizer("a")["input_ids"]
    return [bos] if ids and ids[0] == bos else []
