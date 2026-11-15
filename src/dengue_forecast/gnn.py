"""Modelo espaço-temporal em grafos (GNN) para previsão quantílica de dengue.

Esta camada implementa a preparação de tensores espaço-temporais e a arquitetura
de rede neural em grafos (ST-GNN) descrita no plano da Fase 4 (docs/fase4-plano.md).
"""

from typing import Any

import numpy as np
from scipy import sparse

from dengue_forecast.baselines import HORIZONS
from dengue_forecast.data import PanelData
from dengue_forecast.metrics import QUANTILES


def normalized_adjacency(adjacency: sparse.csr_matrix) -> sparse.csr_matrix:
    """Calcula a matriz de adjacência normalizada simétrica com auto-laços.

    A_hat = D_tilde^(-1/2) * (A + I) * D_tilde^(-1/2)
    """
    n = adjacency.shape[0]
    # Adiciona auto-laços (identidade)
    eye = sparse.eye(n, format="csr", dtype=np.float32)
    a_tilde = adjacency.astype(np.float32) + eye

    # Calcula graus
    degrees = np.asarray(a_tilde.sum(axis=1)).ravel()
    inv_sqrt_deg = np.zeros_like(degrees, dtype=np.float32)
    mask = degrees > 0
    inv_sqrt_deg[mask] = 1.0 / np.sqrt(degrees[mask])

    d_mat = sparse.diags(inv_sqrt_deg, format="csr", dtype=np.float32)
    return d_mat @ a_tilde @ d_mat


def pinball_loss_np(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantiles: tuple[float, ...] = QUANTILES,
) -> float:
    """Cálculo exato da perda multi-quantílica Pinball (Check function) em NumPy.

    y_true: [..., 1] ou [...]
    y_pred: [..., len(quantiles)]
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    if y_true.ndim == y_pred.ndim - 1:
        y_true = y_true[..., None]

    errors = y_true - y_pred
    losses = []
    for i, q in enumerate(quantiles):
        err = errors[..., i]
        loss_q = np.maximum(q * err, (q - 1.0) * err)
        losses.append(np.mean(loss_q))
    return float(np.mean(losses))


def prepare_spatiotemporal_window(
    data: PanelData,
    origin_col: int,
    lookback: int = 12,
    horizons: tuple[int, ...] = HORIZONS,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Extrai o tensor espaço-temporal de entrada e o vetor de alvos para uma semana de origem.

    Entrada X: [N, lookback, F] onde F canais são:
      0: log1p(casos)
      1: incidência por 100k
      2: temperatura média normalizada (ou zero se ausente)
      3: umidade média normalizada (ou zero se ausente)
    Alvo Y: [N, len(horizons)] com variação log1p(casos[t+h]) - log1p(casos[t]), ou None.
    """
    n_geo = data.cases.shape[0]
    n_weeks = data.n_weeks

    if origin_col < 0 or origin_col >= n_weeks:
        raise ValueError(f"origin_col {origin_col} fora dos limites [0, {n_weeks})")

    start_col = origin_col - lookback + 1
    # Cria canais
    channels = []

    # Canal 0: log1p(casos)
    log_cases = np.log1p(data.cases)
    # Canal 1: incidência
    incidence = np.log1p(data.cases / data.population * 1e5)
    channels.append(log_cases)
    channels.append(incidence)

    # Clima (se disponível em extras)
    if "temp_mean" in data.extras:
        temp = data.extras["temp_mean"]
        channels.append(np.nan_to_num(temp, nan=22.0) / 40.0)
    else:
        channels.append(np.zeros_like(log_cases))

    if "humid_mean" in data.extras:
        humid = data.extras["humid_mean"]
        channels.append(np.nan_to_num(humid, nan=70.0) / 100.0)
    else:
        channels.append(np.zeros_like(log_cases))

    stacked = np.stack(channels, axis=-1)  # [N, n_weeks, F]
    f_dim = stacked.shape[-1]

    # Fatiamento temporal causal
    x_tensor = np.zeros((n_geo, lookback, f_dim), dtype=np.float32)
    if start_col < 0:
        # Preenchimento com zero para semanas anteriores ao início da série
        valid_start = 0
        pad_len = -start_col
        x_tensor[:, pad_len:, :] = stacked[:, valid_start : origin_col + 1, :]
    else:
        x_tensor = stacked[:, start_col : origin_col + 1, :].astype(np.float32)

    # Alvos multi-horizonte (se disponíveis no painel)
    max_h = max(horizons)
    if origin_col + max_h < n_weeks:
        base_log = log_cases[:, origin_col]
        y_targets = np.empty((n_geo, len(horizons)), dtype=np.float32)
        for j, h in enumerate(horizons):
            y_targets[:, j] = log_cases[:, origin_col + h] - base_log
    else:
        y_targets = None

    return x_tensor, y_targets


# -------------------------------------------------------------------------
# Implementação PyTorch (importada sob demanda se PyTorch estiver disponível)
# -------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True

    class SpatioTemporalGNN(nn.Module):
        """Rede Neural em Grafos Espaço-Temporal para regressão quantílica multi-horizonte."""

        def __init__(
            self,
            in_channels: int = 4,
            hidden_dim: int = 64,
            num_horizons: int = len(HORIZONS),
            num_quantiles: int = len(QUANTILES),
        ):
            super().__init__()
            self.temporal_fc = nn.Linear(in_channels, hidden_dim)
            self.spatial_weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
            nn.init.xavier_uniform_(self.spatial_weight)

            self.gru = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                batch_first=True,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_horizons * num_quantiles),
            )
            self.num_horizons = num_horizons
            self.num_quantiles = num_quantiles

        def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
            """Passada direta: x [N, T, F], a_hat [N, N].

            Retorna [N, num_horizons, num_quantiles].
            """
            n, t, _ = x.shape
            # 1. Projeção temporal inicial
            h = F.relu(self.temporal_fc(x))  # [N, T, hidden_dim]

            # 2. Convolução espacial ao longo do grafo A_hat
            # h_spatial = A_hat @ H @ W
            h_trans = h.transpose(0, 1)  # [T, N, hidden_dim]
            h_conv = torch.matmul(a_hat, h_trans)  # [T, N, hidden_dim]
            h_spatial = torch.matmul(h_conv, self.spatial_weight).transpose(0, 1)

            # 3. Recorrência GRU
            gru_out, _ = self.gru(h_spatial)  # [N, T, hidden_dim]
            last_step = gru_out[:, -1, :]  # [N, hidden_dim]

            # 4. Decodificação quantílica
            out = self.head(last_step)  # [N, H * Q]
            out = out.view(n, self.num_horizons, self.num_quantiles)
            # Ordena quantis para garantir monotonicidade estrita
            out_sorted, _ = torch.sort(out, dim=-1)
            return out_sorted

    class PinballLossTorch(nn.Module):
        """Função de perda Pinball multi-horizonte multi-quantil em PyTorch."""

        def __init__(self, quantiles: tuple[float, ...] = QUANTILES):
            super().__init__()
            self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32))

        def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
            """y_pred: [N, H, Q], y_true: [N, H]."""
            errors = y_true.unsqueeze(-1) - y_pred
            losses = torch.max(self.quantiles * errors, (self.quantiles - 1.0) * errors)
            return torch.mean(losses)

except ImportError:
    TORCH_AVAILABLE = False
    SpatioTemporalGNN = Any  # type: ignore
    PinballLossTorch = Any  # type: ignore
