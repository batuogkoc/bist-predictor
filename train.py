"""Minimal, extensible PyTorch training script.

Extensibility hooks:
- add `--model` choices to expand architectures
- integrate `wandb.init()` and config logging where noted
- replace argparse with Hydra in future
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from parquet_torch_dataset import build_dataset_from_parquet
from models import SimpleMLP, SimpleRNN, SimpleTransformer
from engine import train_epoch, eval_epoch


def get_model(name: str, seq_len: int, **kwargs):
    name = name.lower()
    if name == "mlp":
        return SimpleMLP(seq_len=seq_len, **kwargs)
    if name == "rnn":
        return SimpleRNN(seq_len=seq_len, **kwargs)
    if name == "transformer":
        return SimpleTransformer(seq_len=seq_len, **kwargs)
    raise ValueError(f"Unknown model: {name}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", default="data/bist/raw/prices_raw.parquet")
    p.add_argument("--seq-len", type=int, default=50)
    p.add_argument("--model", default="mlp", choices=["mlp", "rnn", "transformer"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--fraction", type=float, default=1.0,
                   help="Fraction of the dataset to keep for overfit/test purposes")
    p.add_argument("--ticker-index", type=int, help="select single ticker by index")
    p.add_argument("--ticker", help="select single ticker by name")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading dataset from {args.parquet} (seq_len={args.seq_len})")
    selection = args.ticker if args.ticker is not None else args.ticker_index
    dataset = build_dataset_from_parquet(args.parquet, seq_len=args.seq_len, ticker_selection=selection)

    n = len(dataset)
    if args.fraction <= 0.0 or args.fraction > 1.0:
        raise ValueError("--fraction must be in the interval (0.0, 1.0]")
    if args.fraction < 1.0:
        keep = max(1, int(n * args.fraction))
        subset_idx = torch.randperm(n)[:keep].tolist()
        dataset = torch.utils.data.Subset(dataset, subset_idx)
        n = len(dataset)

    print(f"Dataset size: {n} samples")
    model = get_model(args.model, seq_len=args.seq_len)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model}, device={device}, params={num_params:,}")

    # simple train/val split
    split = int(n * 0.9)
    indices = list(range(n))
    train_idx, val_idx = indices[:split], indices[split:]

    train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_idx))
    val_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=torch.utils.data.SubsetRandomSampler(val_idx))

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, device)
        t1 = time.time()
        print(
            f"Epoch {epoch}/{args.epochs} - "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} - "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({t1-t0:.1f}s)"
        )

    # save a small checkpoint
    ckpt = {
        "model_state": model.state_dict(),
        "args": vars(args),
    }
    out = Path("checkpoints")
    out.mkdir(exist_ok=True)
    torch.save(ckpt, out / "last_checkpoint.pt")
    print("Saved checkpoint to checkpoints/last_checkpoint.pt")


if __name__ == "__main__":
    main()
