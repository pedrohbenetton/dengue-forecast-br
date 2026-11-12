"""Baselines probabilísticos. Todo modelo do projeto segue a mesma interface:

    fit(data)                  -> ajusta com o painel truncado no fim do treino
    predict(data, horizons)    -> [n_municípios, n_horizontes, n_quantis], com o painel
                                  truncado na semana de origem (a última coluna é a origem)
"""

import numpy as np

from dengue_forecast.data import PanelData
from dengue_forecast.metrics import QUANTILES
from dengue_forecast.panel import POP_LABELS

HORIZONS = (1, 2, 4, 8)
SEASON_LAG = 52


def _residual_quantiles(errors: np.ndarray, band: np.ndarray) -> np.ndarray:
    """Quantis dos erros (escala log) por faixa de população -> [n_faixas, n_quantis]."""
    out = np.zeros((len(POP_LABELS), len(QUANTILES)))
    for code in range(len(POP_LABELS)):
        rows = errors[band == code]
        if rows.size:
            out[code] = np.quantile(rows, QUANTILES)
    return out


def _apply(point: np.ndarray, resid: np.ndarray) -> np.ndarray:
    """Previsão pontual + quantis de erro em escala log1p, de volta à escala de casos."""
    return np.clip(np.expm1(np.log1p(point)[:, None] + resid), 0.0, None)


class Naive:
    """Persistência: a previsão pontual para t+h é o valor observado em t.

    A incerteza vem dos erros históricos de persistência em escala log, por faixa de população
    e horizonte, medidos apenas no treino.
    """

    name = "naive"

    def __init__(self, horizons=HORIZONS):
        self.horizons = tuple(horizons)
        self.resid: dict[int, np.ndarray] | None = None

    def fit(self, data: PanelData) -> None:
        log = np.log1p(data.cases)
        self.resid = {
            h: _residual_quantiles(log[:, h:] - log[:, :-h], data.band) for h in self.horizons
        }

    def predict(self, data: PanelData, horizons) -> np.ndarray:
        if self.resid is None:
            raise RuntimeError("chame fit() antes de predict()")
        last = data.cases[:, -1]
        return np.stack([_apply(last, self.resid[h][data.band]) for h in horizons], axis=1)


class SeasonalNaive:
    """Sazonal: a previsão pontual para a semana s é o valor de s - 52 semanas.

    Usa defasagem fixa de 52 semanas (em anos de 53 semanas o alinhamento desliza uma semana).
    """

    name = "seasonal"

    def __init__(self):
        self.resid: np.ndarray | None = None

    def fit(self, data: PanelData) -> None:
        log = np.log1p(data.cases)
        self.resid = _residual_quantiles(log[:, SEASON_LAG:] - log[:, :-SEASON_LAG], data.band)

    def predict(self, data: PanelData, horizons) -> np.ndarray:
        if self.resid is None:
            raise RuntimeError("chame fit() antes de predict()")
        origin = data.n_weeks - 1
        resid = self.resid[data.band]
        out = []
        for h in horizons:
            ref = origin + h - SEASON_LAG
            point = data.cases[:, ref] if 0 <= ref <= origin else data.cases[:, -1]
            out.append(_apply(point, resid))
        return np.stack(out, axis=1)


class Climatology:
    """Histórico por semana: quantis empíricos da mesma SE (±1) nos anos anteriores do município.

    Não tem parâmetros ajustados; usa tudo o que foi observado até a origem.
    """

    name = "climatology"

    def __init__(self, window: int = 1):
        self.window = window

    def fit(self, data: PanelData) -> None:
        return None

    def predict(self, data: PanelData, horizons) -> np.ndarray:
        origin = data.n_weeks - 1
        out = []
        for h in horizons:
            target = origin + h
            idx = [
                target - SEASON_LAG * k + d
                for k in range(1, target // SEASON_LAG + 2)
                for d in range(-self.window, self.window + 1)
                if 0 <= target - SEASON_LAG * k + d <= origin
            ]
            if idx:
                out.append(np.quantile(data.cases[:, idx], QUANTILES, axis=1).T)
            else:
                out.append(np.zeros((len(data.geocodes), len(QUANTILES))))
        return np.stack(out, axis=1)


LEVEL_EDGES = np.array([1, 3, 6, 11, 26, 51, 101, 501])  # 0 | 1-2 | 3-5 | ... | 101-500 | 501+


def level_bin(cases: np.ndarray) -> np.ndarray:
    """Faixa do nível recente de casos (0 = zero casos, 8 = mais de 500)."""
    return np.digitize(cases, LEVEL_EDGES)


class NaiveLevel:
    """Persistência com incerteza condicionada ao nível: os erros históricos em escala log são
    agrupados pelo número de casos na origem (e pelo horizonte), não pelo porte do município.

    Um município grande em semana de zero casos e outro em pleno surto têm erros de persistência
    muito diferentes; esta versão captura isso.
    """

    name = "naive_level"

    def __init__(self, horizons=HORIZONS):
        self.horizons = tuple(horizons)
        self.resid: dict[int, np.ndarray] | None = None

    def fit(self, data: PanelData) -> None:
        log = np.log1p(data.cases)
        n_bins = len(LEVEL_EDGES) + 1
        self.resid = {}
        for h in self.horizons:
            errors = log[:, h:] - log[:, :-h]
            bins = level_bin(data.cases[:, :-h])
            table = np.zeros((n_bins, len(QUANTILES)))
            for b in range(n_bins):
                values = errors[bins == b]
                if values.size:
                    table[b] = np.quantile(values, QUANTILES)
            self.resid[h] = table

    def predict(self, data: PanelData, horizons) -> np.ndarray:
        if self.resid is None:
            raise RuntimeError("chame fit() antes de predict()")
        last = data.cases[:, -1]
        bins = level_bin(last)
        return np.stack([_apply(last, self.resid[h][bins]) for h in horizons], axis=1)


ALL = {m.name: m for m in (Naive, NaiveLevel, SeasonalNaive, Climatology)}
