"""LightGBM global quantílico: um modelo por (horizonte, quantil), treinado em todos os municípios.

O alvo é a variação em escala log entre a origem e t+h, não o nível de casos. Árvores não
extrapolam além dos valores vistos no treino; prever a variação permite acompanhar um surto
maior que qualquer um do passado (caso de 2023-24).
"""

import lightgbm as lgb
import numpy as np
from scipy import sparse

from dengue_forecast.baselines import HORIZONS
from dengue_forecast.data import PanelData
from dengue_forecast.features import CATEGORICAL, GROUPS, LOOKBACK, FeatureBuilder
from dengue_forecast.metrics import QUANTILES

DEFAULT_PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 255,
    "verbose": -1,
}


class LGBMQuantile:
    def __init__(
        self,
        adjacency: sparse.csr_matrix | None = None,
        groups: tuple[str, ...] = GROUPS,
        params: dict | None = None,
        n_estimators: int = 300,
        max_rows: int = 1_200_000,
        min_origin: int = 104,
        horizons=HORIZONS,
        seed: int = 0,
        name: str = "lgbm",
    ):
        self.adjacency = adjacency
        self.groups = tuple(groups)
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.n_estimators = n_estimators
        self.max_rows = max_rows
        self.min_origin = min_origin
        self.horizons = tuple(horizons)
        self.seed = seed
        self.name = name
        self.models: dict[tuple[int, float], lgb.Booster] = {}
        self.feature_names: list[str] = []

    def training_set(self, builder: FeatureBuilder, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        """Features e alvo de todas as origens cujo alvo já foi observado, subamostrados
        uniformemente até `max_rows` linhas."""
        n_weeks = builder.data.n_weeks
        cols = np.arange(min(self.min_origin, max(n_weeks - horizon - 1, 0)), n_weeks - horizon)
        n_geo = len(builder.data.geocodes)
        keep = min(1.0, self.max_rows / (len(cols) * n_geo))
        rng = np.random.default_rng([self.seed, horizon])
        xs, ys = [], []
        for chunk in np.array_split(cols, max(len(cols) // 40, 1)):
            x, y = builder.matrix(chunk, horizon), builder.target(chunk, horizon)
            if keep < 1.0:
                mask = rng.random(len(y)) < keep
                x, y = x[mask], y[mask]
            xs.append(x)
            ys.append(y)
        return np.concatenate(xs), np.concatenate(ys)

    def fit(self, data: PanelData) -> None:
        builder = FeatureBuilder(data, self.adjacency, self.groups)
        self.feature_names = builder.names
        categorical = [n for n in CATEGORICAL if n in builder.names]
        self.models = {}
        for h in self.horizons:
            x, y = self.training_set(builder, h)
            dataset = lgb.Dataset(
                x, label=y, feature_name=builder.names, categorical_feature=categorical,
                free_raw_data=False,
            )  # fmt: skip
            for q in QUANTILES:
                params = {**self.params, "objective": "quantile", "alpha": float(q)}
                params["seed"] = self.seed
                self.models[(h, float(q))] = lgb.train(params, dataset, self.n_estimators)

    def predict(self, data: PanelData, horizons) -> np.ndarray:
        if not self.models:
            raise RuntimeError("chame fit() antes de predict()")
        recent = data.tail(LOOKBACK)
        builder = FeatureBuilder(recent, self.adjacency, self.groups)
        base = np.log1p(recent.cases[:, -1])
        out = []
        for h in horizons:
            x = builder.matrix([recent.n_weeks - 1], h)
            delta = np.stack([self.models[(h, float(q))].predict(x) for q in QUANTILES], axis=1)
            delta.sort(axis=1)  # impede cruzamento de quantis
            out.append(np.clip(np.expm1(base[:, None] + delta), 0.0, None))
        return np.stack(out, axis=1)

    def importance(self) -> dict[str, float]:
        """Ganho total de cada feature, somado sobre horizontes e quantis, normalizado."""
        total = np.zeros(len(self.feature_names))
        for booster in self.models.values():
            total += booster.feature_importance(importance_type="gain")
        share = total / total.sum() if total.sum() > 0 else total
        return dict(sorted(zip(self.feature_names, share, strict=True), key=lambda kv: -kv[1]))
