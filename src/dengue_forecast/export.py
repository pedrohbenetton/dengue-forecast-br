"""Exportação dos dados reais para o site estático (GitHub Pages).

Tudo o que o site mostra sai daqui, a partir do painel e das previsões do backtest:

- `municipios.topo.json`: malha IBGE simplificada (propriedade `codarea` = geocode).
- `municipalities.json`: nome, UF, região e população, na ordem das colunas dos binários.
- `map/meta.json` + `map/*.bin`: camadas do mapa, uma matriz uint8 [semana, município] por
  camada. Incidência por 100 mil codificada como `round(log1p(inc) * INC_SCALE)`, probabilidade
  como `round(p * 254)`; 255 marca ausência.
- `fan/{geocode}.json`: série observada e quantis previstos por semana-alvo, por horizonte.
- `benchmarks.json`: tabela de métricas lida dos CSVs dos relatórios.

Uso:
    uv run python -m dengue_forecast.export
"""

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from dengue_forecast import backtest
from dengue_forecast.baselines import HORIZONS
from dengue_forecast.data import PanelData
from dengue_forecast.metrics import OUTBREAK_INCIDENCE, exceedance_probability

INC_SCALE = 32.0
MISSING = 255
FAN_QUANTILES = ("q0.05", "q0.25", "q0.5", "q0.75", "q0.95")
FAN_HORIZONS = HORIZONS
SERVED = "lgbm_cqr_level"

# Ordem e rótulos da tabela de benchmarks do site.
LABELS = {
    "naive": "Ingênuo (última semana)",
    "naive_level": "Ingênuo por nível (baseline)",
    "seasonal": "Sazonal (mesma SE do ano anterior)",
    "climatology": "Média histórica por SE",
    "lgbm": "LightGBM global quantílico",
    "lgbm_cqr": "LightGBM + CQR por horizonte",
    "lgbm_cqr_level": "LightGBM + CQR por nível (servido)",
    "lgbm_aci": "LightGBM + ACI por horizonte",
    "lgbm_aci_level": "LightGBM + ACI por nível",
    "lgbm_immunity": "LightGBM + dívida imunológica",
    "gnn": "Rede em grafo (GNN)",
    "gnn_nograph": "Mesma rede, sem grafo",
}


def encode_incidence(inc) -> np.ndarray:
    """Incidência por 100 mil → uint8 em escala log1p; NaN → 255."""
    inc = np.asarray(inc, dtype=float)
    code = np.round(np.log1p(np.clip(np.nan_to_num(inc, nan=0.0), 0.0, None)) * INC_SCALE)
    code = np.clip(code, 0, MISSING - 1).astype(np.uint8)
    code[np.isnan(inc)] = MISSING
    return code


def encode_probability(p) -> np.ndarray:
    """Probabilidade em [0, 1] → uint8 (0..254); NaN → 255."""
    p = np.asarray(p, dtype=float)
    code = np.clip(np.round(np.nan_to_num(p, nan=0.0) * 254), 0, 254).astype(np.uint8)
    code[np.isnan(p)] = MISSING
    return code


def _grid(forecasts: pd.DataFrame, data: PanelData, weeks: np.ndarray, values) -> np.ndarray:
    """Espalha uma coluna das previsões numa matriz [semana, município] (NaN onde falta)."""
    out = np.full((len(weeks), len(data.geocodes)), np.nan)
    row = np.searchsorted(weeks, forecasts["target"].to_numpy())
    order = np.argsort(data.geocodes)
    col = order[np.searchsorted(data.geocodes, forecasts["geocode"].to_numpy(), sorter=order)]
    out[row, col] = np.asarray(values, dtype=float)
    return out


def _observed(data: PanelData, weeks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Casos e população nas semanas pedidas, como matrizes [semana, município]."""
    cols = np.searchsorted(data.epiweeks, weeks)
    if not (data.epiweeks[cols] == weeks).all():
        raise ValueError("semanas-alvo fora do painel")
    return data.cases[:, cols].T, data.population[:, cols].T


def export_map(
    forecasts: pd.DataFrame,
    data: PanelData,
    out_dir: Path | str,
    horizons=HORIZONS,
) -> dict:
    """Camadas do mapa: incidência observada e, por horizonte, mediana e P(surto)."""
    out = Path(out_dir) / "map"
    out.mkdir(parents=True, exist_ok=True)
    weeks = np.sort(forecasts["target"].unique())
    cases, pop = _observed(data, weeks)
    encode_incidence(cases / pop * 1e5).tofile(out / "obs.bin")
    for h in horizons:
        sub = forecasts[forecasts["horizon"] == h]
        q = sub[backtest.QCOLS].to_numpy(dtype=float)
        sub_pop = sub["population"].to_numpy(dtype=float)
        p = exceedance_probability(q, OUTBREAK_INCIDENCE * sub_pop / 1e5)
        q50 = _grid(sub, data, weeks, sub["q0.5"].to_numpy() / sub_pop * 1e5)
        encode_incidence(q50).tofile(out / f"q50_h{h}.bin")
        encode_probability(_grid(sub, data, weeks, p)).tofile(out / f"p_h{h}.bin")
    meta = {
        "weeks": [int(w) for w in weeks],
        "geocodes": [int(g) for g in data.geocodes],
        "horizons": [int(h) for h in horizons],
        "inc_scale": INC_SCALE,
        "latest_week": int(weeks[-1]),
    }
    (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def _clean(values: np.ndarray, digits: int = 1) -> list:
    return [None if math.isnan(v) else round(float(v), digits) for v in values]


def export_fancharts(
    forecasts: pd.DataFrame,
    data: PanelData,
    out_dir: Path | str,
    horizons=FAN_HORIZONS,
) -> list[Path]:
    """Um JSON por município: casos observados e quantis previstos por semana-alvo."""
    out = Path(out_dir) / "fan"
    out.mkdir(parents=True, exist_ok=True)
    weeks = np.sort(forecasts["target"].unique())
    cases, _ = _observed(data, weeks)
    grids = {}
    for h in horizons:
        sub = forecasts[forecasts["horizon"] == h]
        grids[h] = {q: _grid(sub, data, weeks, sub[q].to_numpy()) for q in FAN_QUANTILES}
    week_list = [int(w) for w in weeks]
    paths = []
    for g, geocode in enumerate(data.geocodes):
        record: dict = {"weeks": week_list, "obs": _clean(cases[:, g])}
        for h in horizons:
            record[f"h{h}"] = {q: _clean(grids[h][q][:, g]) for q in FAN_QUANTILES}
        path = out / f"{int(geocode)}.json"
        path.write_text(json.dumps(record, separators=(",", ":")), encoding="utf-8")
        paths.append(path)
    return paths


def _read_all(paths) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in map(Path, paths) if p.exists()]
    if not frames:
        return pd.DataFrame(columns=["model", "horizon"])
    # o primeiro arquivo que traz um (modelo, horizonte) vence
    return pd.concat(frames, ignore_index=True).drop_duplicates(["model", "horizon"])


def _num(value) -> float | None:
    return None if value is None or pd.isna(value) else round(float(value), 4)


def export_benchmarks(
    wis_tables,
    alert_tables,
    out_file: Path | str,
    reference: str = "naive_level",
) -> dict:
    """Tabela de benchmarks a partir dos CSVs dos relatórios (nenhum número digitado à mão)."""
    wis = _read_all(wis_tables)
    alert = _read_all(alert_tables)
    ref = wis[(wis["model"] == reference) & (wis["horizon"] == 8)]["wis"]
    ref8 = float(ref.iloc[0]) if len(ref) else None
    rows = []
    for model, label in LABELS.items():
        sub = wis[wis["model"] == model].set_index("horizon")
        if sub.empty:
            continue
        by_h = {str(h): _num(sub["wis"].get(h)) for h in HORIZONS}
        cov = sub["cov90"].dropna() if "cov90" in sub else pd.Series(dtype=float)
        pr = alert[(alert["model"] == model) & (alert["horizon"] == 8)]["pr_auc"]
        rows.append(
            {
                "model": model,
                "label": label,
                "wis": by_h,
                "rel_wis_8": _num(by_h["8"] / ref8) if by_h["8"] and ref8 else None,
                "cov90": _num(cov.mean()) if len(cov) else None,
                "pr_auc_8": _num(pr.iloc[0]) if len(pr) else None,
            }
        )
    result = {"rows": rows, "reference": reference, "source": "backtest 2022-23 a 2025-26"}
    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


def export_municipalities(data: PanelData, names: pd.DataFrame, out_file: Path | str) -> list:
    """Índice de municípios na ordem do painel (a mesma das colunas dos binários)."""
    lookup = names.set_index("geocode")
    records = []
    for g, geocode in enumerate(data.geocodes):
        known = int(geocode) in lookup.index
        records.append(
            {
                "geocode": int(geocode),
                "name": str(lookup.at[int(geocode), "name"]) if known else str(int(geocode)),
                "uf": str(lookup.at[int(geocode), "uf"]) if known else "",
                "region": str(data.region[g]),
                "population": int(data.population[g, -1]),
            }
        )
    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return records


def load_forecasts(forecast_dir: Path | str, model: str = SERVED, seasons=None) -> pd.DataFrame:
    """Previsões de todas as temporadas de teste de um modelo, concatenadas."""
    seasons = seasons or backtest.TEST_SEASONS
    cols = ["geocode", "origin", "target", "horizon", "population", *backtest.QCOLS]
    frames = [
        pd.read_parquet(Path(forecast_dir) / f"{model}_{s}.parquet", columns=cols) for s in seasons
    ]
    return pd.concat(frames, ignore_index=True)


def export_site(
    data: PanelData,
    forecasts: pd.DataFrame,
    names: pd.DataFrame,
    mesh_path: Path | str,
    out_dir: Path | str,
    reports_dir: Path | str = "reports",
) -> dict:
    """Gera todos os arquivos que o site consome."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(mesh_path, out / "municipios.topo.json")
    export_municipalities(data, names, out / "municipalities.json")
    meta = export_map(forecasts, data, out)
    export_fancharts(forecasts, data, out)
    reports = Path(reports_dir)
    export_benchmarks(
        [reports / f"{d}/{f}" for d, f in [
            ("fase3", "por_horizonte.csv"),
            ("fase2", "por_horizonte.csv"),
            ("fase4", "por_horizonte.csv"),
            ("fase6", "ablacao.csv"),
        ]],
        [reports / f"{d}/alerta.csv" for d in ("fase3", "fase2", "fase4", "fase6")],
        out / "benchmarks.json",
    )  # fmt: skip
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--panel", default="data/processed/panel.parquet")
    parser.add_argument("--forecast-dir", default="data/forecasts")
    parser.add_argument("--model", default=SERVED)
    parser.add_argument("--names", default="data/raw/ibge/municipios.json")
    parser.add_argument("--mesh", default="data/raw/ibge/malha_municipios.topo.json")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--out-dir", default="web/public/data")
    args = parser.parse_args(argv)

    from dengue_forecast.ibge import parse_municipalities

    data = backtest.load_data(args.panel)
    names = parse_municipalities(Path(args.names).read_text(encoding="utf-8"))
    forecasts = load_forecasts(args.forecast_dir, args.model)
    meta = export_site(data, forecasts, names, args.mesh, args.out_dir, args.reports_dir)
    print(f"site: {len(meta['weeks'])} semanas x {len(meta['geocodes'])} municípios")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
