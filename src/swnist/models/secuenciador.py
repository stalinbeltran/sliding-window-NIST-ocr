"""Secuenciador: red recurrente (retroalimentada) que reconoce la entrada por partes.

Entradas por paso: el **desplazamiento relativo** (dx, dy) respecto al paso anterior
(no la posición absoluta: ver `relative_positions`) + la salida (features) del
dimensionador sobre la ventana en esa posición. Mantiene un estado interno h que se
retroalimenta, y emite logits del dígito reconocido en cada paso.

El dataset sigue entregando las posiciones ABSOLUTAS normalizadas (T, 2): el paso a
relativas lo hace el modelo, de modo que el recorrido (trace, visualizador, datasets
custom) se sigue describiendo en coordenadas de la imagen.
"""

import torch
import torch.nn as nn

# Codificación de la posición que consume la red. Es parte de la config del modelo
# (viaja en el checkpoint) porque cambia el SIGNIFICADO de la entrada: un
# secuenciador entrenado con posición absoluta tiene los mismos shapes pero no es
# reutilizable aquí (ver from_config).
POSITION_ENCODING = "delta"
POSITION_DIM = 2  # (dx, dy)


def relative_positions(positions_seq: torch.Tensor) -> torch.Tensor:
    """Posiciones absolutas (B, T, 2) -> desplazamiento por paso (B, T, 2).

    El paso 0 no tiene anterior: su desplazamiento es (0, 0). Consecuencia buscada:
    la entrada posicional es invariante a trasladar el recorrido completo (el mismo
    trazo dibujado más a la derecha produce la misma secuencia de deltas).
    """
    deltas = torch.zeros_like(positions_seq)
    deltas[:, 1:] = positions_seq[:, 1:] - positions_seq[:, :-1]
    return deltas


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
        position_encoding: str = POSITION_ENCODING,
    ):
        super().__init__()
        if position_encoding != POSITION_ENCODING:
            raise ValueError(
                f"position_encoding={position_encoding!r} no soportado: el "
                f"secuenciador recibe el desplazamiento relativo (dx, dy) respecto "
                f"al paso anterior ({POSITION_ENCODING!r}), no la posición absoluta.")
        self.config = {
            "feature_dim": feature_dim,
            "hidden_dim": hidden_dim,
            "num_classes": num_classes,
            "cell": cell,
            "position_encoding": position_encoding,
        }
        input_dim = feature_dim + POSITION_DIM  # features + (dx, dy)
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
        if "position_encoding" not in config:
            raise ValueError(
                "Este secuenciador se entrenó con la posición ABSOLUTA (x, y) como "
                "entrada, que ya no se usa: ahora recibe el desplazamiento relativo "
                "(dx, dy) respecto al paso anterior. Sus pesos tienen el shape "
                "correcto pero la entrada significa otra cosa, así que no se pueden "
                "reutilizar. Entrena un secuenciador nuevo (el dimensionador sí se "
                "reutiliza tal cual).")
        return cls(**config)

    def init_state(self, batch_size: int, device=None) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def step(
        self, feature: torch.Tensor, delta: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Un paso: feature (B, D), delta (B, 2) = (dx, dy) respecto al paso anterior,
        h (B, H) -> (logits (B, C), h' (B, H))."""
        h = self.cell(torch.cat([feature, delta], dim=-1), h)
        return self.head(h), h

    def forward(
        self,
        features_seq: torch.Tensor,
        positions_seq: torch.Tensor,
        h0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """features_seq (B, T, D), positions_seq (B, T, 2) ABSOLUTAS (el modelo las
        convierte a desplazamientos) -> (logits (B, T, C), h_T)."""
        B, T, _ = features_seq.shape
        deltas = relative_positions(positions_seq)
        h = h0 if h0 is not None else self.init_state(B, features_seq.device)
        logits = []
        for t in range(T):
            logit_t, h = self.step(features_seq[:, t], deltas[:, t], h)
            logits.append(logit_t)
        return torch.stack(logits, dim=1), h
