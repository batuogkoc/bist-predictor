"""Training and evaluation engine utilities."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def train_epoch(model, dataloader: DataLoader, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    batch_losses = []
    for batch in dataloader:
        if len(batch) == 2:
            x, y = batch
        else:
            x, y, _ = batch
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y.to(torch.long))
        loss.backward()
        optimizer.step()

        batch_losses.append(loss.item())
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return total_loss / max(total, 1), correct / max(total, 1), batch_losses


def eval_epoch(model, dataloader: DataLoader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 2:
                x, y = batch
            else:
                x, y, _ = batch
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)
