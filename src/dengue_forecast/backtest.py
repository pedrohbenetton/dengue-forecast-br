"""Harness de avaliação: backtest por origem móvel, pontuação e agregação.

Uso:
    uv run python -m dengue_forecast.backtest            # todos os baselines, 4 temporadas
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dengue_forecast import baselines, metrics
from dengue_forecast.baselines import HORIZONS
from dengue_forecast.data import PanelData
from dengue_forecast.metrics import OUTBREAK_INCIDENCE, QUANTILES
from dengue_forecast.panel import POP_LABELS

TEST_SEASONS = (2023, 2024, 2025, 2026)  # 2022-23, 2023-24, 2024-25, 2025-26
QCOLS = [f"q{q:g}" for q in QUANTILES]
INTERVALS = (0.5, 0.8, 0.9, 0.95)
REFERENCE = "naive"


def run_season(model, data: PanelData, season: int, horizons=HORIZONS) -> pd.DataFrame:
    """Previsões de `model` para todas as semanas-alvo da temporada, em todos os horizontes.

    O modelo é ajustado com o painel truncado na véspera da temporada (janela expansiva) e,
    para cada origem t, recebe o painel truncado em t. Uma linha por (município, origem,
    horizonte), com a verdade `y` (casos consolidados no alvo) ao lado dos quantis.
    """
    targets = data.season_positions(season)
    if len(targets) == 0:
        raise ValueError(f"temporada {season} fora do painel")
    first, last = int(targets[0]), int(targets[-1])
    model.fit(data.until(first - 1))

    n_geo = len(data.geocodes)
    parts: dict[str, list] = {k: [] for k in ("origin", "target", "horizon", "y", "pop", "q")}
    for origin in range(max(first - max(horizons), 0), last):
        valid = [h for h in horizons if first <= origin + h <= last]
        if not valid:
            continue
        q = model.predict(data.until(origin), valid)
        for j, h in enumerate(valid):
            parts["origin"].append(np.full(n_geo, data.epiweeks[origin]))
            parts["target"].append(np.full(n_geo, data.epiweeks[origin + h]))
            parts["horizon"].append(np.full(n_geo, h, dtype=np.int8))
            parts["y"].append(data.cases[:, origin + h])
            parts["pop"].append(data.population[:, origin])
            parts["q"].append(q[:, j, :])

    n_blocks = len(parts["origin"])
    out = pd.DataFrame(
        {
            "geocode": np.tile(data.geocodes, n_blocks),
            "season": np.int16(season),
            "origin": np.concatenate(parts["origin"]),
            "target": np.concatenate(parts["target"]),
            "horizon": np.concatenate(parts["horizon"]),
            "band": pd.Categorical.from_codes(
                np.tile(data.band, n_blocks), categories=POP_LABELS, ordered=True
            ),
            "region": pd.Categorical(np.tile(data.region, n_blocks)),
            "population": np.concatenate(parts["pop"]),
            "y": np.concatenate(parts["y"]),
        }
    )
    out["crosses_year"] = out["origin"] // 100 != out["target"] // 100
    out[QCOLS] = np.concatenate(parts["q"])
    return out


def score(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta as métricas por linha: WIS, WIS em escala log, erro absoluto, cobertura e alerta.

    O alerta é "incidência no alvo > 300/100 mil", com a população conhecida na origem.
    """
    out = forecasts.copy()
    y = out["y"].to_numpy(dtype=float)
    q = out[QCOLS].to_numpy(dtype=float)
    out["wis"] = metrics.wis(y, q)
    out["wis_log"] = metrics.wis(np.log1p(y), np.log1p(q))
    out["ae"] = np.abs(y - q[:, len(QUANTILES) // 2])
    for nominal in INTERVALS:
        out[f"cov{nominal * 100:g}"] = metrics.covered(y, q, nominal)
    threshold = OUTBREAK_INCIDENCE * out["population"].to_numpy(dtype=float) / 1e5
    out["p_outbreak"] = metrics.exceedance_probability(q, threshold)
    out["outbreak"] = y > threshold
    return out


def summarize(scored: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Médias por (modelo, *by): WIS, WIS log, MAE, coberturas e número de previsões."""
    grouped = scored.groupby(["model", *by], observed=True)
    out = grouped.agg(
        wis=("wis", "mean"),
        wis_log=("wis_log", "mean"),
        mae=("ae", "mean"),
        **{f"cov{n * 100:g}": (f"cov{n * 100:g}", "mean") for n in INTERVALS},
        n=("wis", "size"),
    )
    return out


def relative_wis(summary: pd.DataFrame, reference: str = REFERENCE, col: str = "wis") -> pd.Series:
    """WIS de cada modelo dividido pelo do modelo de referência no mesmo grupo (< 1 é melhor)."""
    values = summary[col]
    ref = values.xs(reference, level="model")
    if not isinstance(ref, pd.Series):  # sem estratos: um único número
        return values / ref
    by = [name for name in values.index.names if name != "model"]
    aligned = values.reset_index().merge(ref.rename("ref").reset_index(), on=by)
    aligned = aligned.set_index(values.index.names).reindex(values.index)
    return (aligned[col] / aligned["ref"]).rename(f"rel_{col}")


def pr_auc(scored: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """PR-AUC do alerta de surto por (modelo, *by), com a prevalência ao lado."""
    rows = []
    for key, group in scored.groupby(["model", *by], observed=True):
        labels = group["outbreak"].to_numpy()
        rows.append(
            (*key, metrics.average_precision(labels, group["p_outbreak"].to_numpy()), labels.mean())
        )
    return pd.DataFrame(rows, columns=["model", *by, "pr_auc", "prevalence"]).set_index(
        ["model", *by]
    )


# ------------------------------------------------------------------------------------ CLI

_KEEP = [
    "model", "season", "horizon", "band", "region", "crosses_year", "wis", "wis_log", "ae",
    *[f"cov{n * 100:g}" for n in INTERVALS], "p_outbreak", "outbreak",
]  # fmt: skip


def run_all(
    data: PanelData,
    models: list,
    seasons=TEST_SEASONS,
    forecast_dir: Path | None = None,
    log=print,
    horizons=HORIZONS,
) -> pd.DataFrame:
    """Roda todos os modelos em todas as temporadas; devolve as métricas por linha (compactas)."""
    scored_parts = []
    for model in models:
        for season in seasons:
            forecasts = run_season(model, data, season, horizons)
            if forecast_dir is not None:
                forecast_dir.mkdir(parents=True, exist_ok=True)
                compact = forecasts.astype({c: "float32" for c in [*QCOLS, "y", "population"]})
                compact.to_parquet(forecast_dir / f"{model.name}_{season}.parquet", index=False)
            scored = score(forecasts).assign(model=model.name)[_KEEP]
            scored = scored.astype({c: "float32" for c in ("wis", "wis_log", "ae", "p_outbreak")})
            scored_parts.append(scored)
            log(f"{model.name} {season}: WIS médio {scored['wis'].mean():.2f}, n={len(scored)}")
    out = pd.concat(scored_parts, ignore_index=True)
    out["model"] = pd.Categorical(out["model"], categories=[m.name for m in models])
    return out


def load_scored(forecast_dir: Path | str, models: list[str]) -> pd.DataFrame:
    """Junta as métricas por linha já calculadas de cada modelo (scored/{modelo}.parquet)."""
    parts = [pd.read_parquet(Path(forecast_dir) / "scored" / f"{name}.parquet") for name in models]
    out = pd.concat(parts, ignore_index=True)
    out["model"] = pd.Categorical(out["model"].astype(str), categories=models)
    return out


def write_tables(
    scored: pd.DataFrame, out_dir: Path, reference: str = REFERENCE
) -> dict[str, pd.DataFrame]:
    """Tabelas da avaliação (CSV): por horizonte, faixa, região, temporada e virada do ano."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {}
    for name, by in {
        "por_horizonte": ["horizon"],
        "por_faixa": ["band", "horizon"],
        "por_regiao": ["region", "horizon"],
        "por_temporada": ["season", "horizon"],
        "virada_do_ano": ["crosses_year", "horizon"],
    }.items():
        table = summarize(scored, by)
        table["rel_wis"] = relative_wis(table, reference)
        table["rel_wis_log"] = relative_wis(table, reference, col="wis_log")
        tables[name] = table
    tables["alerta"] = pr_auc(scored, ["horizon"])
    tables["alerta_por_faixa"] = pr_auc(scored, ["band", "horizon"])
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", float_format="%.4f")
    return tables


def load_data(panel_path: Path | str) -> PanelData:
    """Painel em forma de matriz, com as séries de clima usadas pelas features."""
    from dengue_forecast.features import CLIMATE

    columns = ["geocode", "epiweek", "cases", "population", "pop_band", "region", *CLIMATE]
    return PanelData.from_panel(pd.read_parquet(panel_path, columns=columns), extras=CLIMATE)


def make_model(name: str, data: PanelData, horizons=HORIZONS, mesh_path: Path | str | None = None):
    """Instancia um modelo pelo nome: um baseline ou uma variante do LightGBM."""
    if name in baselines.ALL:
        return baselines.ALL[name]()
    from dengue_forecast import geo
    from dengue_forecast.lgbm import VARIANTS, LGBMQuantile

    if name not in VARIANTS:
        raise ValueError(f"modelo desconhecido: {name}")
    adjacency = geo.adjacency_matrix(data.geocodes, geo.neighbors(geo.fetch_mesh(mesh_path)))
    return LGBMQuantile(adjacency, groups=VARIANTS[name], horizons=horizons, name=name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--panel", default="data/processed/panel.parquet")
    parser.add_argument("--mesh", default="data/raw/ibge/malha_municipios.topo.json")
    parser.add_argument("--models", default=",".join(baselines.ALL))
    parser.add_argument("--seasons", default=",".join(map(str, TEST_SEASONS)))
    parser.add_argument("--horizons", default=",".join(map(str, HORIZONS)))
    parser.add_argument("--forecast-dir", default="data/forecasts")
    parser.add_argument("--out-dir", default=None, help="se dado, grava as tabelas CSV aqui")
    parser.add_argument("--reference", default=REFERENCE)
    parser.add_argument("--force", action="store_true", help="refaz modelos já pontuados")
    args = parser.parse_args(argv)

    names = args.models.split(",")
    seasons = tuple(int(s) for s in args.seasons.split(","))
    horizons = tuple(int(h) for h in args.horizons.split(","))
    forecast_dir = Path(args.forecast_dir)
    (forecast_dir / "scored").mkdir(parents=True, exist_ok=True)
    data = None
    for name in names:
        path = forecast_dir / "scored" / f"{name}.parquet"
        if path.exists() and not args.force:
            print(f"{name}: já pontuado, pulando (use --force para refazer)")
            continue
        data = data if data is not None else load_data(args.panel)
        model = make_model(name, data, horizons, args.mesh)
        scored = run_all(data, [model], seasons, forecast_dir, horizons=horizons)
        scored.to_parquet(path, index=False)
        if hasattr(model, "importance"):
            importance = pd.Series(model.importance(), name="gain_share")
            importance.to_csv(forecast_dir / f"importance_{name}.csv", index_label="feature")

    if args.out_dir:
        tables = write_tables(load_scored(forecast_dir, names), Path(args.out_dir), args.reference)
        pd.set_option("display.width", 200)
        print(tables["por_horizonte"].round(3).to_string())
        print(tables["alerta"].round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
