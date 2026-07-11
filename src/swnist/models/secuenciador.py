"""Secuenciador: red recurrente (retroalimentada) que reconoce la entrada por partes.

Entradas por paso: posición (x, y) normalizada + salida (features) del dimensionador
sobre la ventana en esa posición. Mantiene un estado interno h que se retroalimenta,
y emite logits del dígito reconocido en cada paso.
"""

import torch
import torch.nn as nn


class ElmanCell(nn.Module):
    """Celda recurrente simple: h' = tanh(W_x x + W_h h + b)."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.lin = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.lin(torch.cat([x, h], dim=-1)))


class Secuenciador(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 10,
        cell: str = "gru",
    ):
        super().__init__()
        self.config = {
            "feature_dim": feature_dim,
            "hidden_dim": hidden_dim,
            "num_classes": num_classes,
            "cell": cell,
        }
        input_dim = feature_dim + 2  # features + (x, y)
        if cell == "gru":
            self.cell = nn.GRUCell(input_dim, hidden_dim)
        elif cell == "elman":
            self.cell = ElmanCell(input_dim, hidden_dim)
        else:
            raise ValueError(f"Celda desconocida: {cell!r} (usa 'gru' o 'elman')")
        self.hidden_dim = hidden_dim
        self.head = nn.Linear(hidden_dim, num_classes)

    @classmethod
    def from_config(cls, config: dict) -> "Secuenciador":
        return cls(**config)

    def init_state(self, batch_size: int, device=None) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def step(
        self, feature: torch.Tensor, pos: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Un paso: feature (B, D), pos (B, 2), h (B, H) -> (logits (B, C), h' (B, H))."""
        h = self.cell(torch.cat([feature, pos], dim=-1), h)
        return self.head(h), h

    def forward(
        self,
        features_seq: torch.Tensor,
        positions_seq: torch.Tensor,
        h0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """features_seq (B, T, D), positions_seq (B, T, 2) -> (logits (B, T, C), h_T)."""
        B, T, _ = features_seq.shape
        h = h0 if h0 is not None else self.init_state(B, features_seq.device)
        logits = []
        for t in range(T):
            logit_t, h = self.step(features_seq[:, t], positions_seq[:, t], h)
            logits.append(logit_t)
        return torch.stack(logits, dim=1), h
