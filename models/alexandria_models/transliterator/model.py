"""Character-level Transformer that transliterates one romanized word into Devanagari.

Small (~7M parameters) so it trains in well under an hour on a T4 and runs fast on a
CPU server. ``beam_search`` returns the top-k spellings with log-probabilities so the
sentence-level decoder (``transliterate.py``) can pick between them using context.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<s>", "</s>", "<unk>"]
CONFIG_FILE = "config.json"
WEIGHTS_FILE = "model.pt"
VOCAB_FILE = "vocab.json"


class CharVocab:
    def __init__(self, chars: list[str]):
        self.itos = SPECIALS + [c for c in chars if c not in SPECIALS]
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, words, min_count: int = 1) -> "CharVocab":
        counts: dict[str, int] = {}
        for w in words:
            for ch in w:
                counts[ch] = counts.get(ch, 0) + 1
        return cls(sorted(c for c, n in counts.items() if n >= min_count))

    def encode(self, word: str, max_len: int, add_bos: bool = False) -> list[int]:
        ids = [self.stoi.get(ch, UNK) for ch in word][: max_len - 1 - int(add_bos)]
        return ([BOS] if add_bos else []) + ids + [EOS]

    def decode(self, ids) -> str:
        out = []
        for i in ids:
            i = int(i)
            if i == EOS:
                break
            if i in (PAD, BOS):
                continue
            out.append(self.itos[i] if i < len(self.itos) else "")
        return "".join(out)


@dataclass
class TranslitConfig:
    src_vocab_size: int = 64
    tgt_vocab_size: int = 128
    d_model: int = 256
    nhead: int = 4
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_len: int = 32


class TransliterationModel(nn.Module):
    def __init__(self, cfg: TranslitConfig):
        super().__init__()
        self.cfg = cfg
        self.src_emb = nn.Embedding(cfg.src_vocab_size, cfg.d_model, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(cfg.tgt_vocab_size, cfg.d_model, padding_idx=PAD)
        self.pos_emb = nn.Embedding(cfg.max_len + 2, cfg.d_model)
        with warnings.catch_warnings():  # nested-tensor fast path is unused with norm_first
            warnings.simplefilter("ignore", UserWarning)
            self.transformer = nn.Transformer(
                d_model=cfg.d_model,
                nhead=cfg.nhead,
                num_encoder_layers=cfg.num_encoder_layers,
                num_decoder_layers=cfg.num_decoder_layers,
                dim_feedforward=cfg.dim_feedforward,
                dropout=cfg.dropout,
                batch_first=True,
                norm_first=True,
            )
        self.out = nn.Linear(cfg.d_model, cfg.tgt_vocab_size)
        self.scale = math.sqrt(cfg.d_model)

    def _embed(self, emb: nn.Embedding, ids: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
        return emb(ids) * self.scale + self.pos_emb(pos)

    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        src_pad = src.eq(PAD)
        memory = self.transformer.encoder(self._embed(self.src_emb, src), src_key_padding_mask=src_pad)
        return memory, src_pad

    def decode(self, tgt_in: torch.Tensor, memory: torch.Tensor, src_pad: torch.Tensor) -> torch.Tensor:
        n = tgt_in.size(1)
        causal = torch.triu(torch.ones(n, n, dtype=torch.bool, device=tgt_in.device), diagonal=1)
        h = self.transformer.decoder(
            self._embed(self.tgt_emb, tgt_in),
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_in.eq(PAD),
            memory_key_padding_mask=src_pad,
        )
        return self.out(h)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        memory, src_pad = self.encode(src)
        return self.decode(tgt_in, memory, src_pad)

    @torch.no_grad()
    def beam_search(self, src: torch.Tensor, beam_size: int = 5, max_len: int | None = None):
        """Return, for each source row, a list of (token_ids, logprob) sorted best first."""
        max_len = max_len or self.cfg.max_len
        B, device = src.size(0), src.device
        memory, src_pad = self.encode(src)
        K = beam_size
        memory = memory.repeat_interleave(K, dim=0)
        src_pad = src_pad.repeat_interleave(K, dim=0)
        seqs = torch.full((B * K, 1), BOS, dtype=torch.long, device=device)
        scores = torch.full((B, K), float("-inf"), device=device)
        scores[:, 0] = 0.0  # start with one live beam per row
        scores = scores.view(-1)
        finished = torch.zeros(B * K, dtype=torch.bool, device=device)

        for _ in range(max_len):
            logits = self.decode(seqs, memory, src_pad)[:, -1, :].float()
            logp = torch.log_softmax(logits, dim=-1)
            # A finished beam can only extend with PAD at no cost.
            logp[finished] = float("-inf")
            logp[finished, PAD] = 0.0
            V = logp.size(-1)
            cand = (scores.unsqueeze(1) + logp).view(B, K * V)
            top_scores, top_idx = cand.topk(K, dim=-1)
            beam_idx = (top_idx // V) + torch.arange(B, device=device).unsqueeze(1) * K
            tok = (top_idx % V).view(-1, 1)
            seqs = torch.cat([seqs[beam_idx.view(-1)], tok], dim=1)
            finished = finished[beam_idx.view(-1)] | tok.view(-1).eq(EOS) | tok.view(-1).eq(PAD)
            scores = top_scores.view(-1)
            if bool(finished.all()):
                break

        out = []
        seqs = seqs.view(B, K, -1)
        scores = scores.view(B, K)
        for b in range(B):
            hyps = []
            for k in range(K):
                ids = seqs[b, k, 1:].tolist()
                end = next((i for i, t in enumerate(ids) if t in (EOS, PAD)), None)
                length = (end if end is not None else len(ids)) + 1
                # Light length normalisation so long words are not unfairly penalised;
                # a hypothesis that never emitted EOS is a cut-off word, so rank it last.
                score = float(scores[b, k]) / (length ** 0.6) - (1e4 if end is None else 0.0)
                hyps.append((ids, score))
            hyps.sort(key=lambda h: -h[1])
            out.append(hyps)
        return out


def save_model(directory: str | Path, model: TransliterationModel, src_vocab: CharVocab, tgt_vocab: CharVocab, extra: dict | None = None) -> None:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), d / WEIGHTS_FILE)
    (d / CONFIG_FILE).write_text(json.dumps({**asdict(model.cfg), **(extra or {})}, indent=2), encoding="utf-8")
    (d / VOCAB_FILE).write_text(
        json.dumps({"src": src_vocab.itos, "tgt": tgt_vocab.itos}, ensure_ascii=False), encoding="utf-8"
    )


def load_model(directory: str | Path, device: str = "cpu") -> tuple[TransliterationModel, CharVocab, CharVocab]:
    d = Path(directory)
    raw = json.loads((d / CONFIG_FILE).read_text(encoding="utf-8"))
    cfg = TranslitConfig(**{k: v for k, v in raw.items() if k in TranslitConfig.__dataclass_fields__})
    vocab = json.loads((d / VOCAB_FILE).read_text(encoding="utf-8"))
    src_vocab, tgt_vocab = CharVocab(vocab["src"][len(SPECIALS):]), CharVocab(vocab["tgt"][len(SPECIALS):])
    model = TransliterationModel(cfg)
    model.load_state_dict(torch.load(d / WEIGHTS_FILE, map_location=device, weights_only=True))
    return model.to(device).eval(), src_vocab, tgt_vocab
