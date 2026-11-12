"""Features causais para os modelos globais.

Toda feature da coluna `t` usa apenas dados com semana ≤ t. As operações são defasagens e
janelas móveis para trás, então a matriz da origem `t` é a mesma quer venha do painel completo,
quer do painel truncado em `t` — o teste de causalidade verifica exatamente isso.
"""

import numpy as np
from scipy import sparse

from dengue_forecast import epiweek
from dengue_forecast.data import PanelData

GROUPS = ("own", "season", "static", "neighbors", "climate")
CLIMATE = ("temp_mean", "temp_min", "humid_mean")
SEASON_LAG = 52
CLIM_YEARS = 5
LOOKBACK = SEASON_LAG * CLIM_YEARS + 16  # semanas de história que as features consultam
CATEGORICAL = ("uf",)


def _lag(m: np.ndarray, k: int) -> np.ndarray:
    """Valor de `k` semanas atrás (NaN onde não há história)."""
    if k == 0:
        return m
    out = np.full_like(m, np.nan, dtype=float)
    if k < m.shape[1]:
        out[:, k:] = m[:, :-k]
    return out


def _roll_mean(m: np.ndarray, window: int) -> np.ndarray:
    """Média móvel para trás (inclui a semana corrente), ignorando NaN."""
    valid = ~np.isnan(m)
    total = np.cumsum(np.where(valid, m, 0.0), axis=1)
    count = np.cumsum(valid, axis=1).astype(float)
    total[:, window:] -= total[:, :-window].copy()
    count[:, window:] -= count[:, :-window].copy()
    return np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)


class FeatureBuilder:
    """Pré-calcula as séries derivadas de um painel e monta a matriz de features por origem."""

    def __init__(
        self,
        data: PanelData,
        adjacency: sparse.csr_matrix | None = None,
        groups: tuple[str, ...] = GROUPS,
    ):
        unknown = set(groups) - set(GROUPS)
        if unknown:
            raise ValueError(f"grupos de features desconhecidos: {sorted(unknown)}")
        self.data = data
        self.groups = tuple(g for g in GROUPS if g in groups)
        self.log = np.log1p(data.cases)
        self.base: dict[str, np.ndarray] = {}
        self.static: dict[str, np.ndarray] = {}
        incidence = np.log1p(data.cases / data.population * 1e5)

        if "own" in self.groups:
            for k in range(9):
                self.base[f"l{k}"] = _lag(self.log, k)
            for w in (4, 8, 12, 52):
                self.base[f"roll{w}"] = _roll_mean(self.log, w)
            self.base["d1"] = self.log - _lag(self.log, 1)
            self.base["d4"] = self.log - _lag(self.log, 4)
            self.base["inc"] = incidence
            self.base["zero12"] = _roll_mean((data.cases == 0).astype(float), 12)

        if "static" in self.groups:
            self.base["logpop"] = np.log(data.population)
            self.static["band"] = data.band.astype(float)
            self.static["uf"] = data.uf.astype(float)

        if "neighbors" in self.groups:
            if adjacency is None:
                raise ValueError("o grupo 'neighbors' exige a matriz de adjacência")
            isolated = np.asarray(adjacency.sum(axis=1)).ravel() == 0
            nb = np.asarray(adjacency @ incidence)
            nb[isolated] = np.nan
            self.base["nb0"] = nb
            self.base["nb1"] = _lag(nb, 1)
            self.base["nb_roll4"] = _roll_mean(nb, 4)
            self.base["nb_rel"] = nb - incidence
            ufs, inverse = np.unique(data.uf, return_inverse=True)
            onehot = sparse.csr_matrix(
                (np.ones(len(inverse)), (inverse, np.arange(len(inverse)))),
                shape=(len(ufs), len(inverse)),
            )
            uf_inc = np.log1p((onehot @ data.cases) / (onehot @ data.population) * 1e5)[inverse]
            self.base["uf0"] = uf_inc
            self.base["uf_d4"] = uf_inc - _lag(uf_inc, 4)
            self.base["uf_rel"] = uf_inc - incidence

        if "climate" in self.groups:
            for name in CLIMATE:
                smooth = _roll_mean(data.extras[name], 4)
                for k in (0, 4, 8):
                    self.base[f"{name}_r4_l{k}"] = _lag(smooth, k)

        self.season_names = (
            ["tw_sin", "tw_cos", "seas_ref", "seas_growth", "clim", "crosses_year"]
            if "season" in self.groups
            else []
        )
        self.names = [*self.base, *self.static, *self.season_names]

    def _season(self, cols: np.ndarray, horizon: int) -> list[np.ndarray]:
        """Features que dependem do alvo t+h; só consultam colunas ≤ t."""
        log, n_weeks = self.log, self.data.n_weeks
        target_week = epiweek.add(self.data.epiweeks[cols], horizon) % 100
        angle = 2 * np.pi * target_week / 52.0
        n_geo = log.shape[0]

        def at(idx: np.ndarray) -> np.ndarray:  # log[:, idx] com NaN fora da história
            ok = (idx >= 0) & (idx < n_weeks)
            out = np.full((n_geo, len(idx)), np.nan)
            out[:, ok] = log[:, idx[ok]]
            return out

        current = log[:, cols]
        ref = at(cols + horizon - SEASON_LAG)
        years = np.stack([at(cols + horizon - SEASON_LAG * k) for k in range(1, CLIM_YEARS + 1)])
        seen = (~np.isnan(years)).sum(axis=0)
        clim = np.divide(
            np.nansum(years, axis=0), seen, out=np.full(current.shape, np.nan), where=seen > 0
        )
        origin_week = self.data.epiweeks[cols]
        crosses = origin_week // 100 != epiweek.add(origin_week, horizon) // 100
        ones = np.ones((n_geo, 1))
        return [
            ones * np.sin(angle)[None, :],
            ones * np.cos(angle)[None, :],
            ref - current,
            ref - at(cols - SEASON_LAG),
            clim - current,
            ones * crosses[None, :].astype(float),
        ]

    def matrix(self, cols, horizon: int) -> np.ndarray:
        """Features das origens `cols` para o horizonte `horizon`.

        Devolve [len(cols) * n_geo, n_features] em float32; a linha `i * n_geo + g` é a origem
        `cols[i]` do município `g`. Os nomes das colunas estão em `self.names`.
        """
        cols = np.asarray(cols, dtype=np.int64)
        n_geo = self.log.shape[0]
        blocks = [m[:, cols] for m in self.base.values()]
        blocks += [np.repeat(v[:, None], len(cols), axis=1) for v in self.static.values()]
        if self.season_names:
            blocks += self._season(cols, horizon)
        out = np.empty((len(cols) * n_geo, len(blocks)), dtype=np.float32)
        for j, block in enumerate(blocks):
            out[:, j] = block.T.reshape(-1)
        return out

    def target(self, cols, horizon: int) -> np.ndarray:
        """Alvo de treino: variação em escala log entre a origem e t+h (mesma ordem de `matrix`)."""
        cols = np.asarray(cols, dtype=np.int64)
        return (self.log[:, cols + horizon] - self.log[:, cols]).T.reshape(-1)
