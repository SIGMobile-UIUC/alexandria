"""Train the romanized-Hindi -> Devanagari word transliterator (plain PyTorch).

Reads ``train.tsv`` / ``val.tsv`` (``roman<TAB>native`` per line) produced by
``transliterator.prepare_data``. About 20-40 minutes on a T4 for ~1M word pairs.
During training a share of source words is replaced by an informal spelling variant
(``augment.py``) so the model copes with "pani"/"paani", "bohot"/"bahut" and so on.

Output: ``model.pt`` + ``config.json`` + ``vocab.json`` in ``--output_dir``.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

from ..common.io import get_logger, set_seed
from .augment import romanization_variants
from .model import PAD, CharVocab, TranslitConfig, TransliterationModel, save_model

log = get_logger("transliterator.train")


def read_pairs(path: str | Path, max_len: int) -> list[tuple[str, str]]:
    pairs = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            roman, native = parts[0].strip().lower(), parts[1].strip()
            if roman and native and len(roman) < max_len - 1 and len(native) < max_len - 1:
                pairs.append((roman, native))
    return pairs


def batches(pairs, src_vocab, tgt_vocab, batch_size, max_len, rng, augment_prob=0.0, shuffle=True):
    """Length-bucketed batches of (src, tgt_in, tgt_out) tensors."""
    import torch

    idx = list(range(len(pairs)))
    if shuffle:
        rng.shuffle(idx)
    # Sort within large chunks so batches have similar lengths (less padding).
    chunk = batch_size * 50
    for c in range(0, len(idx), chunk):
        part = sorted(idx[c : c + chunk], key=lambda i: len(pairs[i][0]))
        groups = [part[i : i + batch_size] for i in range(0, len(part), batch_size)]
        if shuffle:
            rng.shuffle(groups)
        for g in groups:
            srcs, tgts = [], []
            for i in g:
                roman, native = pairs[i]
                if augment_prob and rng.random() < augment_prob:
                    variants = romanization_variants(roman, max_variants=1, rng=rng)
                    if variants:
                        roman = variants[0]
                srcs.append(src_vocab.encode(roman, max_len))
                tgts.append(tgt_vocab.encode(native, max_len + 1, add_bos=True))
            S = max(map(len, srcs))
            T = max(map(len, tgts))
            src = torch.tensor([s + [PAD] * (S - len(s)) for s in srcs])
            tgt = torch.tensor([t + [PAD] * (T - len(t)) for t in tgts])
            yield src, tgt[:, :-1], tgt[:, 1:]


def word_accuracy(model, pairs, src_vocab, tgt_vocab, device, max_len, limit=5000, beam=1) -> float:
    """Top-1 accuracy per romanized input, counted correct if the prediction is any of
    that input's gold spellings (one romanization can map to several Hindi words)."""
    import torch

    gold: dict[str, set[str]] = {}
    for roman, native in pairs:
        gold.setdefault(roman, set()).add(native)
    romans = list(gold)[:limit]
    model.eval()
    correct = 0
    with torch.inference_mode():
        for i in range(0, len(romans), 256):
            chunk = romans[i : i + 256]
            src = [src_vocab.encode(r, max_len) for r in chunk]
            S = max(map(len, src))
            src_t = torch.tensor([s + [PAD] * (S - len(s)) for s in src], device=device)
            hyps = model.beam_search(src_t, beam_size=beam, max_len=max_len)
            for roman, h in zip(chunk, hyps):
                correct += tgt_vocab.decode(h[0][0]) in gold[roman]
    model.train()
    return correct / max(1, len(romans))


def train(args: argparse.Namespace) -> Path:
    import torch

    set_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir, out = Path(args.data_dir), Path(args.output_dir)
    train_pairs = read_pairs(data_dir / "train.tsv", args.max_len)
    val_pairs = read_pairs(data_dir / "val.tsv", args.max_len)
    if args.max_train_pairs:
        rng.shuffle(train_pairs)
        train_pairs = train_pairs[: args.max_train_pairs]
    rng.shuffle(val_pairs)
    log.info("train pairs=%d val pairs=%d device=%s", len(train_pairs), len(val_pairs), device)

    # Source vocabulary covers a-z plus anything seen; augmentation only uses a-z.
    src_vocab = CharVocab.build([r for r, _ in train_pairs] + ["abcdefghijklmnopqrstuvwxyz'"])
    tgt_vocab = CharVocab.build(n for _, n in train_pairs)
    cfg = TranslitConfig(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.layers,
        num_decoder_layers=args.layers,
        dim_feedforward=args.d_model * 4,
        dropout=args.dropout,
        max_len=args.max_len,
    )
    model = TransliterationModel(cfg).to(device)
    log.info("parameters: %.1fM", sum(p.numel() for p in model.parameters()) / 1e6)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train_pairs) / args.batch_size)
    total = steps_per_epoch * args.epochs
    warmup = min(args.warmup_steps, max(1, total // 10))

    def lr_at(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    use_amp = device == "cuda" and not args.fp32
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)

    best, step = -1.0, 0
    for epoch in range(args.epochs):
        t0, running = time.time(), 0.0
        for src, tgt_in, tgt_out in batches(
            train_pairs, src_vocab, tgt_vocab, args.batch_size, args.max_len, rng, args.augment_prob
        ):
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(src, tgt_in)
                loss = loss_fn(logits.float().reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            running += loss.item()
            if step % args.log_every == 0:
                log.info("epoch %d step %d loss %.4f lr %.2e", epoch + 1, step, running / args.log_every, sched.get_last_lr()[0])
                running = 0.0
        acc = word_accuracy(model, val_pairs, src_vocab, tgt_vocab, device, args.max_len)
        log.info("epoch %d done in %.0fs: val top-1 word accuracy %.4f", epoch + 1, time.time() - t0, acc)
        if acc > best:
            best = acc
            save_model(out, model, src_vocab, tgt_vocab, extra={"val_top1_accuracy": acc, "epoch": epoch + 1})
            log.info("saved best model to %s", out)
    return out


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_len", type=int, default=32)
    p.add_argument("--augment_prob", type=float, default=0.25)
    p.add_argument("--max_train_pairs", type=int, default=None)
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
