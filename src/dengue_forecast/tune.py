"""Escolha dos hiperparâmetros do LightGBM na temporada de validação (2021-22).

Roda uma grade pequena uma única vez; a configuração escolhida fica congelada em
`lgbm.DEFAULT_PARAMS` para as quatro temporadas de teste, que não participam da escolha.

Uso:
    uv run python -m dengue_forecast.tune
"""

import argparse
from pathlib import Path

import pandas as pd

from dengue_forecast import backtest, geo
from dengue_forecast.lgbm import LGBMQuantile

VALIDATION_SEASON = 2022
HORIZONS = (1, 8)  # os extremos bastam para ordenar as configurações
CONFIGS: dict[str, dict] = {
    "folhas63_300": {},
    "folhas255_300": {"params": {"num_leaves": 255, "min_data_in_leaf": 500}},
    "folhas31_600": {"params": {"num_leaves": 31}, "n_estimators": 600},
}


def evaluate(
    data, adjacency, configs=CONFIGS, season=VALIDATION_SEASON, horizons=HORIZONS, log=print
):
    """WIS (bruto e em log) e cobertura de cada configuração na temporada de validação."""
    rows = []
    for name, kwargs in configs.items():
        model = LGBMQuantile(adjacency, horizons=horizons, name=name, **kwargs)
        scored = backtest.score(backtest.run_season(model, data, season, horizons))
        summary = scored.groupby("horizon")[["wis", "wis_log", "cov50", "cov90"]].mean()
        log(f"{name}\n{summary.round(4).to_string()}")
        rows.append(summary.assign(config=name).reset_index())
    return pd.concat(rows, ignore_index=True)[
        ["config", "horizon", "wis", "wis_log", "cov50", "cov90"]
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--panel", default="data/processed/panel.parquet")
    parser.add_argument("--mesh", default="data/raw/ibge/malha_municipios.topo.json")
    parser.add_argument("--out", default="reports/fase2/tuning.csv")
    args = parser.parse_args(argv)
    data = backtest.load_data(args.panel)
    adjacency = geo.adjacency_matrix(data.geocodes, geo.neighbors(geo.fetch_mesh(args.mesh)))
    table = evaluate(data, adjacency)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False, float_format="%.4f")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
