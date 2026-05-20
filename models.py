"""Basic model architectures: simple MLP and simple RNN (GRU) for sequence classification.

Design goals:
- Small and dependency-free
- Easy to extend with new architectures (Transformer, deeper MLP, etc.)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self, seq_len: int, hidden_size: int = 256, num_layers: int = 5, dropout: float = 0.1, num_classes: int = 2):
        super().__init__()
        layers = []
        input_size = seq_len
        for i in range(num_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_size = hidden_size
        layers.append(nn.Linear(input_size, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len) -- flatten for MLP
        if x.dim() == 3 and x.size(-1) == 1:
            x = x.squeeze(-1)
        return self.net(x)


class SimpleRNN(nn.Module):
    def __init__(
        self,
        seq_len: int,
        input_size: int = 1,
        rnn_hidden: int = 128,
        num_layers: int = 3,
        bidirectional: bool = False,
        dropout: float = 0.0,
        num_classes: int = 2,
    ):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=rnn_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        factor = 2 if bidirectional else 1
        self.fc = nn.Linear(rnn_hidden * factor, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect x shape: (batch, seq_len, input_size) or (batch, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.fc(last)


# Placeholder for a transformer-like small model
class SimpleTransformer(nn.Module):
    def __init__(
        self,
        seq_len: int,
        input_size: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        num_classes: int = 2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size) or (batch, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x = self.input_proj(x)
        x = x.permute(1, 0, 2)
        out = self.encoder(x)
        out = out[-1]
        return self.classifier(out)
