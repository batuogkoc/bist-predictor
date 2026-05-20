"""Toy MNIST training script that reuses the same training engine and model code."""
from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from engine import train_epoch, eval_epoch
from models import SimpleMLP, SimpleRNN, SimpleTransformer


def get_model(name: str, seq_len: int, input_size: int = 1, **kwargs):
    name = name.lower()
    if name == "mlp":
        return SimpleMLP(seq_len=seq_len, **kwargs)
    if name == "rnn":
        return SimpleRNN(seq_len=seq_len, input_size=input_size, **kwargs)
    if name == "transformer":
        return SimpleTransformer(seq_len=seq_len, input_size=input_size, **kwargs)
    raise ValueError(f"Unknown model: {name}")


class MNISTWrapper(Dataset):
    def __init__(self, dataset: Dataset, mode: str):
        self.dataset = dataset
        self.mode = mode

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if self.mode == "mlp":
            x = x.view(-1)
        else:
            x = x.squeeze(0)
        return x, y


def parse_args():
    parser = argparse.ArgumentParser(description="Train toy MNIST models with the current engine and model code.")
    parser.add_argument("--model", default="mlp", choices=["mlp", "rnn", "transformer"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--download", action="store_true", help="Download MNIST if needed")
    parser.add_argument("--train-size", type=int, default=10000, help="Number of MNIST training samples to use")
    return parser.parse_args()


def build_datasets(train_size: int, download: bool):
    transform = transforms.ToTensor()
    full_train = datasets.MNIST(root="data/mnist", train=True, transform=transform, download=download)
    test_set = datasets.MNIST(root="data/mnist", train=False, transform=transform, download=download)

    train_size = min(train_size, len(full_train))
    val_size = min(2000, len(full_train) - train_size)
    train_subset, val_subset, _ = random_split(full_train, [train_size, val_size, len(full_train) - train_size - val_size])

    return train_subset, val_subset, test_set


def main():
    args = parse_args()
    device = torch.device(args.device)

    if args.model == "mlp":
        seq_len = 28 * 28
        input_size = 1
    else:
        seq_len = 28
        input_size = 28

    train_dataset, val_dataset, test_dataset = build_datasets(args.train_size, args.download)
    train_dataset = MNISTWrapper(train_dataset, args.model)
    val_dataset = MNISTWrapper(val_dataset, args.model)
    test_dataset = MNISTWrapper(test_dataset, args.model)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = get_model(args.model, seq_len=seq_len, input_size=input_size, num_classes=10)
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model}, params={num_params:,}, device={device}")
    print(f"Train size: {len(train_loader.dataset)}, val size: {len(val_loader.dataset)}, test size: {len(test_loader.dataset)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc, _ = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, device)
        t1 = time.time()
        print(
            f"Epoch {epoch}/{args.epochs} - "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} - "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({t1-t0:.1f}s)"
        )

    test_loss, test_acc = eval_epoch(model, test_loader, device)
    print(f"Test loss={test_loss:.4f} test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
