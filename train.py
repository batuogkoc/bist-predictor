"""Minimal, extensible PyTorch training script."""
from __future__ import annotations

from pathlib import Path
import time

import hydra
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

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


def build_mnist_dataset(model_name: str, download: bool):
    from torchvision import datasets, transforms
    raw = datasets.MNIST(root="data/mnist", train=True,
                         transform=transforms.ToTensor(), download=download)
    if model_name == "mlp":
        seq_len = 28 * 28
        dataset_model_kwargs = {"num_classes": 10}
    else:
        seq_len = 28
        dataset_model_kwargs = {"num_classes": 10, "input_size": 28}
    return MNISTWrapper(raw, model_name), seq_len, dataset_model_kwargs


def plot_history(history: dict, epochs: int, out: Path):
    epoch_nums = list(range(1, epochs + 1))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epoch_nums, history["train_loss"], label="train")
    axes[0].plot(epoch_nums, history["val_loss"], label="val")
    axes[0].set_title("Loss per epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epoch_nums, history["train_acc"], label="train")
    axes[1].plot(epoch_nums, history["val_acc"], label="val")
    axes[1].set_title("Accuracy per epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    all_batch_losses = []
    epoch_boundaries = [0]
    for bl in history["batch_losses"]:
        all_batch_losses.extend(bl)
        epoch_boundaries.append(len(all_batch_losses))

    axes[2].plot(all_batch_losses, linewidth=0.8)
    for i, b in enumerate(epoch_boundaries[1:-1], start=1):
        axes[2].axvline(x=b, color="gray", linestyle="--", linewidth=0.5, alpha=0.6,
                        label="epoch boundary" if i == 1 else None)
    axes[2].set_title("Batch loss (within epoch)")
    axes[2].set_xlabel("Batch (global)")
    axes[2].set_ylabel("Loss")
    axes[2].legend()

    plt.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=150)
    print(f"Saved training curves to {out / 'training_curves.png'}")
    plt.show()


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if cfg.device == "auto" else torch.device(cfg.device)

    # Separate model architecture params (from YAML) from the model name
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_name = model_cfg.pop("name")

    if cfg.dataset == "mnist":
        print(f"Loading MNIST dataset (model={model_name})")
        dataset, seq_len, dataset_model_kwargs = build_mnist_dataset(model_name, cfg.download)
    else:
        print(f"Loading dataset from {cfg.parquet} (seq_len={cfg.seq_len})")
        selection = cfg.ticker if cfg.ticker is not None else cfg.ticker_index
        dataset = build_dataset_from_parquet(
            cfg.parquet, seq_len=cfg.seq_len,
            ticker_selection=selection,
            normalize=cfg.normalize,
        )
        seq_len = cfg.seq_len
        dataset_model_kwargs = {}

    n = len(dataset)
    if cfg.fraction <= 0.0 or cfg.fraction > 1.0:
        raise ValueError("fraction must be in the interval (0.0, 1.0]")
    if cfg.fraction < 1.0:
        keep = max(1, int(n * cfg.fraction))
        dataset = torch.utils.data.Subset(dataset, list(range(keep)))
        n = len(dataset)
        print(f"Using deterministic subset: first {keep} samples ({cfg.fraction:.2%} of dataset)")
    else:
        print("Using full dataset")

    print(f"Dataset size: {n} samples")
    # model_cfg has arch hyperparams; dataset_model_kwargs has num_classes / input_size
    model = get_model(model_name, seq_len=seq_len, **model_cfg, **dataset_model_kwargs)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_name}, device={device}, params={num_params:,}")
    print(OmegaConf.to_yaml(cfg.model))

    indices = list(range(n))
    split = int(n * 0.9)
    train_idx, val_idx = indices[:split], indices[split:]
    train_loader = DataLoader(dataset, batch_size=cfg.batch_size,
                              sampler=torch.utils.data.SubsetRandomSampler(train_idx))
    val_loader = DataLoader(dataset, batch_size=cfg.batch_size,
                            sampler=torch.utils.data.SubsetRandomSampler(val_idx))

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "batch_losses": []}

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss, train_acc, batch_losses = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, device)
        t1 = time.time()
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["batch_losses"].append(batch_losses)
        print(
            f"Epoch {epoch}/{cfg.epochs} - "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} - "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({t1-t0:.1f}s)"
        )

    ckpt = {
        "model_state": model.state_dict(),
        "cfg": OmegaConf.to_container(cfg, resolve=True),
    }
    out = Path("checkpoints")
    out.mkdir(exist_ok=True)
    torch.save(ckpt, out / "last_checkpoint.pt")
    print("Saved checkpoint to checkpoints/last_checkpoint.pt")

    plot_history(history, cfg.epochs, out)


if __name__ == "__main__":
    main()
