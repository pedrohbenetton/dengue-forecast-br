"""Painel consolidado (município × semana epidemiológica) a partir dos parquets brutos.

Uso:
    uv run python -m dengue_forecast.panel
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dengue_forecast import ibge

SEASON_START_WEEK = 41  # temporada Y = SE 41 de Y-1 até SE 40 de Y
POP_BINS = [0, 5_000, 20_000, 100_000, 500_000, np.inf]
POP_LABELS = ["<5 mil", "5–20 mil", "20–100 mil", "100–500 mil", ">500 mil"]
_DROP = ["record_id", "locality_id", "municipality_name"]
_LEADING = [
    "geocode", "epiweek", "week_start", "year", "week", "season",
    "name", "uf", "region", "pop_band",
]  # fmt: skip


def season_of(epiweek: pd.Series) -> pd.Series:
    """Temporada epidemiológica (ano em que termina) de cada semana YYYYWW."""
    year, week = epiweek // 100, epiweek % 100
    return year + (week >= SEASON_START_WEEK).astype("int64")


def pop_band(population: pd.Series) -> pd.Series:
    """Faixa de população (categórica ordenada), intervalos fechados à esquerda."""
    return pd.cut(population.astype("float64"), bins=POP_BINS, labels=POP_LABELS, right=False)


def build_panel(raw_dir: Path | str, municipalities: pd.DataFrame) -> pd.DataFrame:
    """Concatena `{raw_dir}/*.parquet`, junta UF/região do IBGE e deriva colunas de calendário.

    `pop_band` usa a população mais recente de cada município (constante no tempo), para que
    um município não troque de estrato no meio do backtest.
    """
    files = sorted(Path(raw_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"nenhum parquet em {raw_dir}")
    frames = [f for f in (pd.read_parquet(p) for p in files) if len(f)]
    df = pd.concat(frames, ignore_index=True).drop(columns=_DROP)

    unknown = sorted(set(df["geocode"]) - set(municipalities["geocode"]))
    if unknown:
        raise ValueError(f"geocodes fora da lista do IBGE: {unknown[:5]}")

    df = df.sort_values(["geocode", "epiweek"], kind="stable").reset_index(drop=True)
    df["year"] = df["epiweek"] // 100
    df["week"] = df["epiweek"] % 100
    df["season"] = season_of(df["epiweek"])

    latest_pop = df.groupby("geocode")["population"].last()
    info = municipalities.set_index("geocode")[["name", "uf", "region"]]
    info["pop_band"] = pop_band(latest_pop).reindex(info.index)
    df = df.join(info, on="geocode")
    df["name"] = df["name"].astype("string")
    for col in ["uf", "region"]:
        df[col] = df[col].astype("category")

    return df[_LEADING + [c for c in df.columns if c not in _LEADING]]


def validate_panel(df: pd.DataFrame, municipalities: pd.DataFrame) -> dict:
    """Checagens de integridade do painel; devolve um relatório serializável."""
    all_weeks = df["epiweek"].nunique()
    weeks_per_city = df.groupby("geocode")["epiweek"].nunique()
    gaps = (all_weeks - weeks_per_city)[lambda s: s > 0]
    without_data = sorted(set(municipalities["geocode"]) - set(df["geocode"]))
    return {
        "n_rows": int(len(df)),
        "n_municipalities_expected": int(len(municipalities)),
        "n_municipalities_with_data": int(df["geocode"].nunique()),
        "municipalities_without_data": [int(g) for g in without_data],
        "first_epiweek": int(df["epiweek"].min()),
        "last_epiweek": int(df["epiweek"].max()),
        "n_epiweeks": int(all_weeks),
        "municipalities_with_gaps": {int(g): int(n) for g, n in gaps.items()},
        "duplicate_keys": int(df.duplicated(["geocode", "epiweek"]).sum()),
        "negative_cases": int((df["cases"] < 0).sum()),
        "null_cases": int(df["cases"].isna().sum()),
        "null_population": int(df["population"].isna().sum()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args(argv)

    raw = Path(args.raw_dir)
    municipalities = ibge.fetch_municipalities(cache_path=raw / "ibge" / "municipios.json")
    df = build_panel(raw / "infodengue", municipalities)
    report = validate_panel(df, municipalities)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "panel.parquet", index=False)
    (out / "panel_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {k: v for k, v in report.items() if not isinstance(v, list | dict)}
    summary["n_without_data"] = len(report["municipalities_without_data"])
    summary["n_with_gaps"] = len(report["municipalities_with_gaps"])
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
