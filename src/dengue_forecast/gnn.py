"""Modelo espaço-temporal em grafos (GNN) para previsão quantílica de dengue.

Esta camada implementa a preparação de tensores espaço-temporais e a arquitetura
de rede neural em grafos (ST-GNN) descrita no plano da Fase 4 (docs/fase4-plano.md).
"""

from typing import Any

import numpy as np
from scipy import sparse

from dengue_forecast.baselines import HORIZONS
from dengue_forecast.data import PanelData
from dengue_forecast.features import GROUPS, LOOKBACK, FeatureBuilder
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


def node_features(
    builder: FeatureBuilder,
    cols,
    horizons,
) -> tuple[np.ndarray, list[str]]:
    """Matriz de features de nós com forma [len(cols), n_geo, F] float32.

    Colunas não sazonais de um único horizonte seguidas pelas colunas sazonais
    de cada horizonte em ordem (com sufixo _h{h}). A coluna categórica 'uf'
    é removida pois é tratada via embedding.
    """
    cols = np.asarray(cols, dtype=np.int64)
    horizons = tuple(horizons)
    n_cols = len(cols)
    n_geo = builder.data.cases.shape[0]

    season_set = set(builder.season_names)
    non_season_indices = [
        i for i, name in enumerate(builder.names) if name not in season_set and name != "uf"
    ]
    non_season_names = [name for name in builder.names if name not in season_set and name != "uf"]
    season_indices = [i for i, name in enumerate(builder.names) if name in season_set]
    season_names = [name for name in builder.names if name in season_set]

    if n_cols == 0:
        names = list(non_season_names)
        for h in horizons:
            names.extend([f"{name}_h{h}" for name in season_names])
        return np.empty((0, n_geo, len(names)), dtype=np.float32), names

    ref_h = horizons[0] if len(horizons) > 0 else 1
    m_first = builder.matrix(cols, ref_h).reshape(n_cols, n_geo, -1)

    blocks = []
    names = list(non_season_names)
    if non_season_indices:
        blocks.append(m_first[:, :, non_season_indices])

    for h in horizons:
        if season_indices:
            if h == ref_h:
                m_h = m_first
            else:
                m_h = builder.matrix(cols, h).reshape(n_cols, n_geo, -1)
            blocks.append(m_h[:, :, season_indices])
            names.extend([f"{name}_h{h}" for name in season_names])

    if blocks:
        x = np.concatenate(blocks, axis=-1).astype(np.float32)
    else:
        x = np.empty((n_cols, n_geo, 0), dtype=np.float32)

    return x, names


def node_targets(builder: FeatureBuilder, cols, horizons) -> np.ndarray:
    """Alvos de nós com forma [len(cols), n_geo, H] float32.

    Contém NaN onde cols + h >= n_weeks ou onde o alvo original é NaN.
    Nunca chama builder.target fora dos limites do painel.
    """
    cols = np.asarray(cols, dtype=np.int64)
    horizons = tuple(horizons)
    n_cols = len(cols)
    n_geo = builder.data.cases.shape[0]
    n_weeks = builder.data.n_weeks

    out = np.full((n_cols, n_geo, len(horizons)), np.nan, dtype=np.float32)
    for j, h in enumerate(horizons):
        valid_mask = cols + h < n_weeks
        valid_indices = np.flatnonzero(valid_mask)
        if len(valid_indices) > 0:
            valid_cols = cols[valid_indices]
            t = builder.target(valid_cols, h).reshape(len(valid_indices), n_geo)
            out[valid_indices, :, j] = t.astype(np.float32)

    return out


# -------------------------------------------------------------------------
# Implementação PyTorch (importada sob demanda se PyTorch estiver disponível)
# -------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True

    class STGNN(nn.Module):
        """Rede Neural em Grafos Espaço-Temporal para regressão quantílica."""

        def __init__(
            self,
            in_dim: int,
            n_uf: int,
            hidden: int = 128,
            layers: int = 2,
            n_horizons: int = len(HORIZONS),
            n_quantiles: int = len(QUANTILES),
            dropout: float = 0.1,
            use_graph: bool = True,
        ):
            super().__init__()
            self.in_dim = in_dim
            self.n_uf = n_uf
            self.hidden = hidden
            self.layers = layers
            self.n_horizons = n_horizons
            self.n_quantiles = n_quantiles
            self.use_graph = use_graph

            emb_dim = min(16, hidden)
            self.emb_uf = nn.Embedding(n_uf, emb_dim)
            self.in_proj = nn.Linear(in_dim + emb_dim, hidden)

            self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
            self.convs = nn.ModuleList([nn.Linear(2 * hidden, hidden) for _ in range(layers)])
            self.dropout = nn.Dropout(dropout)

            self.head = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_horizons * n_quantiles),
            )

        def forward(
            self,
            x: torch.Tensor,
            uf: torch.Tensor,
            a_hat: torch.Tensor | None = None,
        ) -> torch.Tensor:
            b_size, n_nodes, _ = x.shape

            emb = self.emb_uf(uf)
            emb_expanded = emb.unsqueeze(0).expand(b_size, -1, -1)

            h = F.relu(self.in_proj(torch.cat([x, emb_expanded], dim=-1)))

            for norm, conv in zip(self.norms, self.convs, strict=True):
                h_norm = norm(h)
                if self.use_graph and a_hat is not None:
                    h_perm = h_norm.permute(1, 0, 2).reshape(n_nodes, b_size * self.hidden)
                    m_2d = torch.sparse.mm(a_hat, h_perm)
                    m = m_2d.view(n_nodes, b_size, self.hidden).permute(1, 0, 2)
                else:
                    m = torch.zeros_like(h_norm)
                h = h + self.dropout(F.relu(conv(torch.cat([h_norm, m], dim=-1))))

            out = self.head(h)
            out = out.view(b_size, n_nodes, self.n_horizons, self.n_quantiles)
            out_sorted, _ = torch.sort(out, dim=-1)
            return out_sorted

except ImportError:
    TORCH_AVAILABLE = False
    STGNN = Any  # type: ignore


class STGNNQuantile:
    """Modelo global quantílico baseado em ST-GNN com a mesma interface de LGBMQuantile."""

    def __init__(
        self,
        adjacency: sparse.csr_matrix | None = None,
        groups: tuple[str, ...] = GROUPS,
        hidden: int = 128,
        layers: int = 2,
        dropout: float = 0.1,
        epochs: int = 30,
        batch_origins: int = 8,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        min_origin: int = 104,
        horizons=HORIZONS,
        use_graph: bool = True,
        seed: int = 0,
        device: str | None = None,
        name: str = "gnn",
        log: Any = None,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("STGNNQuantile exige PyTorch: uv sync --group gnn")
        self.adjacency = adjacency
        self.groups = tuple(groups)
        self.hidden = hidden
        self.layers = layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_origins = batch_origins
        self.lr = lr
        self.weight_decay = weight_decay
        self.min_origin = min_origin
        self.horizons = tuple(horizons)
        self.use_graph = use_graph
        self.seed = seed
        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu") if device is None else str(device)
        )
        self.name = name
        self.log = log

        self.model: STGNN | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.feature_names: list[str] = []
        self.uf_values: np.ndarray | None = None
        self.history: list[float] = []
        self.a_hat: Any = None

    def _graph(self, n_geo: int):
        """Adjacência normalizada como tensor esparso no dispositivo do modelo."""
        adjacency = self.adjacency
        if adjacency is None:
            adjacency = sparse.csr_matrix((n_geo, n_geo), dtype=np.float32)
        coo = normalized_adjacency(adjacency).tocoo()
        indices = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
        values = torch.tensor(coo.data, dtype=torch.float32)
        return torch.sparse_coo_tensor(indices, values, coo.shape, device=self.device).coalesce()

    def fit(self, data: PanelData) -> None:
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        builder = FeatureBuilder(data, self.adjacency, self.groups)
        n_weeks = data.n_weeks
        start_col = min(self.min_origin, max(n_weeks - 2, 0))
        end_col = n_weeks - 1
        cols = np.arange(start_col, end_col)

        if len(cols) == 0:
            return

        chunks = np.array_split(cols, max(len(cols) // 40, 1))
        xs = []
        fnames: list[str] = []
        for chunk in chunks:
            xc, fnames = node_features(builder, chunk, self.horizons)
            xs.append(xc)
        x = np.concatenate(xs, axis=0)
        self.feature_names = fnames

        mean = np.nanmean(x, axis=(0, 1))
        std = np.nanstd(x, axis=(0, 1))
        std = np.where((std < 1e-6) | np.isnan(std), 1.0, std)
        mean = np.where(np.isnan(mean), 0.0, mean)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

        x = (x - self.mean) / self.std
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        y = node_targets(builder, cols, self.horizons)

        uf_raw = builder.data.uf
        self.uf_values, uf_indices = np.unique(uf_raw, return_inverse=True)
        n_uf = len(self.uf_values)
        uf_tensor = torch.tensor(uf_indices, dtype=torch.long, device=self.device)

        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)

        self.a_hat = self._graph(len(builder.data.geocodes))

        self.model = STGNN(
            in_dim=x.shape[-1],
            n_uf=n_uf,
            hidden=self.hidden,
            layers=self.layers,
            n_horizons=len(self.horizons),
            n_quantiles=len(QUANTILES),
            dropout=self.dropout,
            use_graph=self.use_graph,
        ).to(self.device)

        n_origins = len(cols)
        batch_size = self.batch_origins
        n_batches = (n_origins + batch_size - 1) // batch_size
        total_steps = max(1, self.epochs * n_batches)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)

        quantiles = torch.tensor(QUANTILES, dtype=torch.float32, device=self.device)

        self.history = []
        self.model.train()

        for epoch in range(self.epochs):
            perm = torch.randperm(n_origins, generator=generator)
            epoch_losses = []
            for start in range(0, n_origins, batch_size):
                batch_idx = perm[start : start + batch_size].to(self.device)
                bx = x_tensor[batch_idx]
                by = y_tensor[batch_idx]

                optimizer.zero_grad()
                pred = self.model(bx, uf_tensor, self.a_hat)

                mask = torch.isfinite(by)
                if mask.any():
                    diff = by.unsqueeze(-1) - pred
                    loss_all = torch.max(quantiles * diff, (quantiles - 1.0) * diff)
                    mask_q = mask.unsqueeze(-1).expand_as(loss_all)
                    loss = loss_all[mask_q].mean()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    epoch_losses.append(loss.item())
                else:
                    scheduler.step()

            epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            self.history.append(epoch_loss)
            if self.log is not None:
                self.log(f"epoca {epoch} loss {epoch_loss:.4f}")

        self.model.eval()

    def predict(self, data: PanelData, horizons) -> np.ndarray:
        if self.model is None or self.mean is None or self.std is None or self.uf_values is None:
            raise RuntimeError("chame fit() antes de predict()")

        recent = data.tail(LOOKBACK)
        builder = FeatureBuilder(recent, self.adjacency, self.groups)

        origin_col = recent.n_weeks - 1
        x_np, _ = node_features(builder, [origin_col], self.horizons)

        x_norm = (x_np - self.mean) / self.std
        x_norm = np.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        uf_raw = builder.data.uf
        uf_indices = np.searchsorted(self.uf_values, uf_raw)
        uf_indices = np.clip(uf_indices, 0, len(self.uf_values) - 1)
        uf_tensor = torch.tensor(uf_indices, dtype=torch.long, device=self.device)

        x_tensor = torch.tensor(x_norm, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            pred = self.model(x_tensor, uf_tensor, self.a_hat)

        delta_all = pred.squeeze(0).cpu().numpy()

        h_indices = [self.horizons.index(h) for h in horizons]
        delta = delta_all[:, h_indices, :]

        base = np.log1p(recent.cases[:, -1])
        out = np.clip(np.expm1(base[:, None, None] + delta), 0.0, None)
        return out.astype(np.float64)
