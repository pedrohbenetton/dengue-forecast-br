"""Exportação dos dados reais para o site estático (GitHub Pages).

Tudo o que o site mostra sai daqui, a partir do painel e das previsões do backtest:

- `municipios.topo.json`: malha IBGE simplificada (propriedade `codarea` = geocode).
- `municipalities.json`: nome, UF, região e população, na ordem das colunas dos binários.
- `map/meta.json` + `map/*.bin`: camadas do mapa, uma matriz uint8 [semana, município] por
  camada. Incidência por 100 mil codificada como `round(log1p(inc) * INC_SCALE)`, probabilidade
  como `round(p * 254)` e resíduo assinado em escala log como `round(127 + r * RES_SCALE)`; 255
  marca ausência.
- `fan/{geocode}.json`: série observada e quantis previstos por semana-alvo, por horizonte.
- `benchmarks.json`: tabela de métricas lida dos CSVs dos relatórios.
- `diagnostics.json`: estratificações, skill por UF, calibração, PIT, curvas PR e análise de erro.

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

from dengue_forecast import backtest, metrics
from dengue_forecast.baselines import HORIZONS
from dengue_forecast.data import PanelData
from dengue_forecast.metrics import OUTBREAK_INCIDENCE, exceedance_probability

INC_SCALE = 32.0
RES_SCALE = 32.0
RES_ZERO = 127
MISSING = 255
FAN_QUANTILES = ("q0.05", "q0.25", "q0.5", "q0.75", "q0.95")
FAN_HORIZONS = HORIZONS
SERVED = "lgbm_cqr_level"
REGION_ORDER = ("Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul")

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


def encode_residual(obs_inc, pred_inc) -> np.ndarray:
    """Resíduo assinado log1p(obs_inc) - log1p(pred_inc) → uint8 centrado em 127; NaN → 255.

    Valores > 127 indicam subestimação pelo modelo (surto maior que a mediana prevista);
    valores < 127 indicam superestimação.
    """
    obs = np.asarray(obs_inc, dtype=float)
    pred = np.asarray(pred_inc, dtype=float)
    miss = np.isnan(obs) | np.isnan(pred)
    obs_log = np.log1p(np.clip(np.nan_to_num(obs, nan=0.0), 0.0, None))
    pred_log = np.log1p(np.clip(np.nan_to_num(pred, nan=0.0), 0.0, None))
    diff = obs_log - pred_log
    code = np.clip(np.round(RES_ZERO + diff * RES_SCALE), 0, MISSING - 1).astype(np.uint8)
    code[miss] = MISSING
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
    """Camadas do mapa: incidência observada e, por horizonte, mediana, P(surto) e resíduo."""
    out = Path(out_dir) / "map"
    out.mkdir(parents=True, exist_ok=True)
    weeks = np.sort(forecasts["target"].unique())
    cases, pop = _observed(data, weeks)
    obs_inc = cases / pop * 1e5
    encode_incidence(obs_inc).tofile(out / "obs.bin")
    for h in horizons:
        sub = forecasts[forecasts["horizon"] == h]
        q = sub[backtest.QCOLS].to_numpy(dtype=float)
        sub_pop = sub["population"].to_numpy(dtype=float)
        p = exceedance_probability(q, OUTBREAK_INCIDENCE * sub_pop / 1e5)
        q50 = _grid(sub, data, weeks, sub["q0.5"].to_numpy() / sub_pop * 1e5)
        encode_incidence(q50).tofile(out / f"q50_h{h}.bin")
        encode_probability(_grid(sub, data, weeks, p)).tofile(out / f"p_h{h}.bin")
        encode_residual(obs_inc, q50).tofile(out / f"err_h{h}.bin")
    meta = {
        "weeks": [int(w) for w in weeks],
        "geocodes": [int(g) for g in data.geocodes],
        "horizons": [int(h) for h in horizons],
        "inc_scale": INC_SCALE,
        "res_scale": RES_SCALE,
        "res_zero": RES_ZERO,
        "latest_week": int(weeks[-1]),
    }
    (out / "meta.json").write_text(json.dumps(meta, allow_nan=False), encoding="utf-8")
    return meta


def _clean(values: np.ndarray, digits: int = 1) -> list:
    return [None if not math.isfinite(float(v)) else round(float(v), digits) for v in values]


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
        path.write_text(
            json.dumps(record, separators=(",", ":"), allow_nan=False), encoding="utf-8"
        )
        paths.append(path)
    return paths


def _read_all(paths, keys: list[str] | None = None) -> pd.DataFrame:
    keys = keys or ["model", "horizon"]
    frames = [pd.read_csv(p) for p in map(Path, paths) if p.exists()]
    if not frames:
        return pd.DataFrame(columns=keys)
    # o primeiro arquivo que traz uma chave vence
    return pd.concat(frames, ignore_index=True).drop_duplicates(keys)


def _num(value, digits: int = 4) -> float | None:
    if value is None or pd.isna(value):
        return None
    fv = float(value)
    return None if not math.isfinite(fv) else round(fv, digits)


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
    out.write_text(
        json.dumps(result, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8"
    )
    return result


def _stratified_records(
    reports: Path, filename: str, group_col: str, reference: str = "naive_level"
) -> list[dict]:
    """Lê tabelas estratificadas dos relatórios recalibrando rel_wis contra `reference`."""
    dirs = ("fase3", "fase2", "fase4", "fase6")
    df = _read_all([reports / d / filename for d in dirs], keys=["model", group_col, "horizon"])
    if df.empty or group_col not in df.columns:
        return []
    ref_df = df[df["model"] == reference].set_index([group_col, "horizon"])
    ref_wis = ref_df["wis"] if "wis" in ref_df else pd.Series(dtype=float)
    ref_log = ref_df["wis_log"] if "wis_log" in ref_df else pd.Series(dtype=float)

    records = []
    for model, label in LABELS.items():
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        for row in sub.itertuples(index=False):
            g_val = getattr(row, group_col)
            h_val = int(row.horizon)
            rw = ref_wis.get((g_val, h_val))
            rl = ref_log.get((g_val, h_val))
            wis_v = float(row.wis)
            log_v = float(row.wis_log) if hasattr(row, "wis_log") else None
            rel_log_val = _num(log_v / rl) if log_v is not None and rl and rl > 0 else None
            records.append(
                {
                    "model": model,
                    "label": label,
                    "stratum": str(g_val),
                    "horizon": h_val,
                    "wis": _num(wis_v),
                    "wis_log": _num(log_v),
                    "rel_wis": _num(wis_v / rw) if rw and rw > 0 else None,
                    "rel_wis_log": rel_log_val,
                    "mae": _num(getattr(row, "mae", None)),
                    "cov90": _num(getattr(row, "cov90", None)),
                }
            )
    return records


def _uf_wis_sum(forecast_dir: Path, model: str, seasons) -> pd.DataFrame | None:
    cols = ["geocode", "horizon", "y", *backtest.QCOLS]
    parts = []
    for s in seasons:
        p = forecast_dir / f"{model}_{s}.parquet"
        if not p.exists():
            return None
        f = pd.read_parquet(p, columns=cols)
        w = metrics.wis(f["y"].to_numpy(dtype=float), f[backtest.QCOLS].to_numpy(dtype=float))
        uf_code = f["geocode"].to_numpy(dtype=np.int64) // 100_000
        tmp = pd.DataFrame({"uf_code": uf_code, "horizon": f["horizon"].to_numpy(), "wis": w})
        parts.append(tmp.groupby(["uf_code", "horizon"])["wis"].agg(["sum", "count"]))
    total = pd.concat(parts).groupby(level=[0, 1]).sum()
    return (total["sum"] / total["count"]).unstack("horizon")


def _build_uf_skill(
    names: pd.DataFrame,
    data: PanelData,
    forecast_dir: Path | None,
    seasons,
    reference: str = "naive_level",
) -> dict:
    if forecast_dir is None:
        return {"models": [], "ufs": []}
    ref_mean = _uf_wis_sum(forecast_dir, reference, seasons)
    if ref_mean is None:
        return {"models": [], "ufs": []}

    candidate_models = ("lgbm_cqr_level", "lgbm", "lgbm_immunity", "gnn_nograph", "gnn")
    skill_by_model: dict[str, pd.DataFrame] = {}
    for m in candidate_models:
        m_mean = _uf_wis_sum(forecast_dir, m, seasons)
        if m_mean is not None:
            skill_by_model[m] = m_mean / ref_mean

    if not skill_by_model:
        return {"models": [], "ufs": []}

    geo_df = pd.DataFrame({"geocode": data.geocodes, "region": data.region})
    muni = names[["geocode", "uf"]].merge(geo_df, on="geocode", how="inner")
    muni["uf_code"] = muni["geocode"] // 100_000
    uf_meta = muni.drop_duplicates("uf_code").copy()
    reg_rank = {r: i for i, r in enumerate(REGION_ORDER)}
    uf_meta["order"] = uf_meta["region"].map(lambda r: reg_rank.get(str(r), 99))
    uf_meta = uf_meta.sort_values(["order", "uf"])

    ufs = []
    for row in uf_meta.itertuples(index=False):
        code = int(row.uf_code)
        if code not in ref_mean.index:
            continue
        skills = {}
        for m, tbl in skill_by_model.items():
            if code in tbl.index:
                skills[m] = {str(h): _num(tbl.at[code, h], 3) for h in tbl.columns}
        ufs.append(
            {"uf_code": code, "uf": str(row.uf), "region": str(row.region), "skills": skills}
        )

    models_info = [{"model": m, "label": LABELS.get(m, m)} for m in skill_by_model]
    return {"models": models_info, "ufs": ufs}


def _build_calibration(reports: Path, forecast_dir: Path | None) -> dict:
    dirs = ("fase3", "fase2", "fase4", "fase6")
    wis = _read_all([reports / d / "por_horizonte.csv" for d in dirs])
    nominals = [0.5, 0.8, 0.9, 0.95]
    cov_cols = ["cov50", "cov80", "cov90", "cov95"]

    curves = []
    if not wis.empty and all(c in wis.columns for c in cov_cols):
        for model in ("naive", "naive_level", "lgbm", "lgbm_cqr_level", "lgbm_aci", "gnn_nograph"):
            sub = wis[wis["model"] == model].sort_values("horizon")
            if sub.empty:
                continue
            by_h = {
                str(int(r.horizon)): [_num(getattr(r, c), 4) for c in cov_cols]
                for r in sub.itertuples(index=False)
            }
            curves.append(
                {
                    "model": model,
                    "label": LABELS.get(model, model),
                    "overall": [_num(sub[c].mean(), 4) for c in cov_cols],
                    "by_horizon": by_h,
                }
            )

    weekly_2024: dict = {"origins": [], "series": []}
    if forecast_dir is not None:
        origins_ref: list[int] | None = None
        for model in ("naive_level", "lgbm", "lgbm_cqr_level", "lgbm_aci", "gnn_nograph"):
            p = forecast_dir / f"{model}_2024.parquet"
            if not p.exists():
                continue
            df = pd.read_parquet(p, columns=["origin", "horizon", "y", "q0.05", "q0.95"])
            cov = (df["q0.05"] <= df["y"]) & (df["y"] <= df["q0.95"])
            df = df.assign(cov90=cov)
            w_all = df.groupby("origin")["cov90"].mean()
            w_h8 = df[df["horizon"] == 8].groupby("origin")["cov90"].mean()
            if origins_ref is None:
                origins_ref = [int(o) for o in sorted(w_all.index)]
            weekly_2024["series"].append(
                {
                    "model": model,
                    "label": LABELS.get(model, model),
                    "all": [_num(w_all.get(o), 4) for o in origins_ref],
                    "h8": [_num(w_h8.get(o), 4) for o in origins_ref],
                }
            )
        if origins_ref is not None:
            weekly_2024["origins"] = origins_ref

    return {"nominals": nominals, "curves": curves, "weekly_2024": weekly_2024}


def _build_pit_and_pr(forecast_dir: Path | None, seasons) -> tuple[dict, dict]:
    pit_out: dict = {"bins": 10, "models": []}
    pr_out: dict = {"recall": [round(float(x), 3) for x in np.linspace(0.0, 1.0, 51)], "models": []}
    if forecast_dir is None:
        return pit_out, pr_out

    pit_models = ("naive_level", "lgbm", "lgbm_cqr_level", "gnn_nograph")
    for model in pit_models:
        paths = [forecast_dir / f"{model}_{s}.parquet" for s in seasons]
        if not all(p.exists() for p in paths):
            continue
        frames = [
            pd.read_parquet(p, columns=["horizon", "y", *backtest.QCOLS]).assign(season=s)
            for p, s in zip(paths, seasons, strict=True)
        ]
        full = pd.concat(frames, ignore_index=True)
        y_all = full["y"].to_numpy(dtype=float)
        q_all = full[backtest.QCOLS].to_numpy(dtype=float)
        h_all = full["horizon"].to_numpy()
        s_all = full["season"].to_numpy()

        hists: dict[str, list[float]] = {}
        for h in (1, 4, 8):
            m = h_all == h
            if m.any():
                hists[f"h{h}"] = [
                    round(float(v), 4)
                    for v in metrics.pit_histogram(y_all[m], q_all[m], bins=10, seed=0)
                ]
        m_2024 = (h_all == 8) & (s_all == 2024)
        if m_2024.any():
            hists["h8_2024"] = [
                round(float(v), 4)
                for v in metrics.pit_histogram(y_all[m_2024], q_all[m_2024], bins=10, seed=0)
            ]
        m_pos = (h_all == 8) & (y_all >= 10.0)
        if m_pos.any():
            hists["h8_active"] = [
                round(float(v), 4)
                for v in metrics.pit_histogram(y_all[m_pos], q_all[m_pos], bins=10, seed=0)
            ]
        pit_out["models"].append({"model": model, "label": LABELS.get(model, model), **hists})

    pr_models = ("naive_level", "lgbm", "lgbm_cqr_level", "lgbm_immunity", "gnn_nograph")
    scored_dir = forecast_dir / "scored"
    prevalence_by_h: dict[str, float | None] = {}
    for model in pr_models:
        sp = scored_dir / f"{model}.parquet"
        if not sp.exists():
            continue
        sdf = pd.read_parquet(sp, columns=["horizon", "outbreak", "p_outbreak"])
        entry: dict = {"model": model, "label": LABELS.get(model, model), "horizons": {}}
        for h in HORIZONS:
            sub = sdf[sdf["horizon"] == h]
            if sub.empty:
                continue
            lbl = sub["outbreak"].to_numpy(dtype=bool)
            scr = sub["p_outbreak"].to_numpy(dtype=float)
            _, prec = metrics.pr_curve(lbl, scr, n_points=51)
            auc = metrics.average_precision(lbl, scr)
            prevalence_by_h[str(h)] = _num(lbl.mean(), 4)
            entry["horizons"][str(h)] = {
                "pr_auc": _num(auc, 4),
                "precision": [_num(v, 4) for v in prec],
            }
        pr_out["models"].append(entry)
    pr_out["prevalence"] = prevalence_by_h
    return pit_out, pr_out


def _build_error_analysis(forecasts: pd.DataFrame, data: PanelData) -> dict:
    """Decomposição quantitativa dos erros do modelo servido por regime epidemiológico e semana."""
    geo_order = np.argsort(data.geocodes)
    g_idx = geo_order[
        np.searchsorted(data.geocodes, forecasts["geocode"].to_numpy(), sorter=geo_order)
    ]
    o_col = np.searchsorted(data.epiweeks, forecasts["origin"].to_numpy())
    t_col = np.searchsorted(data.epiweeks, forecasts["target"].to_numpy())

    orig_cases = data.cases[g_idx, np.clip(o_col, 0, data.n_weeks - 1)]
    y = (
        forecasts["y"].to_numpy(dtype=float)
        if "y" in forecasts.columns
        else data.cases[g_idx, np.clip(t_col, 0, data.n_weeks - 1)]
    )
    q_cols = [c for c in backtest.QCOLS if c in forecasts.columns]
    q50 = forecasts["q0.5"].to_numpy(dtype=float)
    q05 = forecasts["q0.05"].to_numpy(dtype=float)
    q95 = forecasts["q0.95"].to_numpy(dtype=float)
    horizons = forecasts["horizon"].to_numpy()

    wis_arr = (
        metrics.wis(y, forecasts[q_cols].to_numpy(dtype=float))
        if len(q_cols) == len(backtest.QCOLS)
        else np.abs(y - q50)
    )
    wis_log_arr = (
        metrics.wis(np.log1p(y), np.log1p(forecasts[q_cols].to_numpy(dtype=float)))
        if len(q_cols) == len(backtest.QCOLS)
        else np.abs(np.log1p(y) - np.log1p(q50))
    )

    # Regimes mutuamente exclusivos baseados na dinâmica observada entre origem e alvo
    quiet = (orig_cases < 5.0) & (y < 5.0)
    onset = (~quiet) & (y >= np.maximum(10.0, 2.0 * np.maximum(orig_cases, 1.0)))
    decline = (~quiet) & (~onset) & (orig_cases >= 10.0) & (y <= 0.5 * orig_cases)
    active = (~quiet) & (~onset) & (~decline)

    regime_defs = [
        ("quiet", "Silêncio / Endemia baixa (<5 casos)", quiet),
        ("onset", "Início / Explosão de surto (≥2× origem)", onset),
        ("active", "Transmissão sustentada / Platô", active),
        ("decline", "Descida pós-pico (≤0,5× origem)", decline),
    ]

    regimes_by_h: dict[str, list[dict]] = {}
    for h in (1, 4, 8):
        hm = horizons == h
        if not hm.any():
            continue
        total_h = int(hm.sum())
        rows = []
        for key, label, rmask in regime_defs:
            m = hm & rmask
            cnt = int(m.sum())
            if cnt == 0:
                continue
            yh, q50h, q05h, q95h = y[m], q50[m], q05[m], q95[m]
            rows.append(
                {
                    "regime": key,
                    "label": label,
                    "n": cnt,
                    "share": _num(cnt / total_h, 4),
                    "wis": _num(wis_arr[m].mean(), 3),
                    "wis_log": _num(wis_log_arr[m].mean(), 4),
                    "mae": _num(np.abs(yh - q50h).mean(), 3),
                    "bias_cases": _num((yh - q50h).mean(), 3),
                    "cov90": _num(((q05h <= yh) & (yh <= q95h)).mean(), 4),
                    "under90": _num((yh > q95h).mean(), 4),
                    "over90": _num((yh < q05h).mean(), 4),
                }
            )
        regimes_by_h[str(h)] = rows

    # Semanas críticas para inspeção direta no mapa de resíduos (h=8)
    notable: list[dict] = []
    hm8 = horizons == 8
    if hm8.any():
        df8 = pd.DataFrame(
            {
                "target": forecasts["target"].to_numpy()[hm8],
                "y": y[hm8],
                "q50": q50[hm8],
                "under": (y[hm8] > q95[hm8]).astype(float),
            }
        )
        wk = df8.groupby("target").agg(
            obs_total=("y", "sum"),
            pred_total=("q50", "sum"),
            under_share=("under", "mean"),
        )
        candidates = [
            (202401, "Degrau da virada de ano e aceleração inicial de 2024"),
            (202408, "Subida explosiva de fevereiro/2024 (subestimação em h=8)"),
            (202412, "Pico histórico nacional (637 mil casos na semana)"),
            (202422, "Descida pós-pico de 2024 (superestimação residual)"),
            (202514, "Pico da temporada 2024-25 (pós-choque)"),
        ]
        for week_id, desc in candidates:
            if week_id in wk.index:
                r = wk.loc[week_id]
                notable.append(
                    {
                        "week": int(week_id),
                        "description": desc,
                        "obs_total": int(round(float(r["obs_total"]))),
                        "pred_total": int(round(float(r["pred_total"]))),
                        "under90_share": _num(r["under_share"], 4),
                    }
                )

    return {"regimes_by_horizon": regimes_by_h, "notable_weeks": notable}


def export_diagnostics(
    forecasts: pd.DataFrame,
    data: PanelData,
    names: pd.DataFrame,
    out_file: Path | str,
    reports_dir: Path | str = "reports",
    forecast_dir: Path | str | None = None,
    seasons=None,
    reference: str = "naive_level",
) -> dict:
    """Consolida diagnósticos avançados de avaliação e erro para o painel interativo do site."""
    reports = Path(reports_dir)
    fdir = Path(forecast_dir) if forecast_dir is not None else None
    seasons = seasons or backtest.TEST_SEASONS

    pit_data, pr_data = _build_pit_and_pr(fdir, seasons)
    payload = {
        "reference": reference,
        "stratified": {
            "by_band": _stratified_records(reports, "por_faixa.csv", "band", reference),
            "by_region": _stratified_records(reports, "por_regiao.csv", "region", reference),
            "by_season": _stratified_records(reports, "por_temporada.csv", "season", reference),
            "by_year_turn": _stratified_records(
                reports, "virada_do_ano.csv", "crosses_year", reference
            ),
        },
        "uf_skill": _build_uf_skill(names, data, fdir, seasons, reference),
        "calibration": _build_calibration(reports, fdir),
        "pit": pit_data,
        "pr_curves": pr_data,
        "error_analysis": _build_error_analysis(forecasts, data),
    }
    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    return payload


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
    out.write_text(json.dumps(records, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return records


def load_forecasts(forecast_dir: Path | str, model: str = SERVED, seasons=None) -> pd.DataFrame:
    """Previsões de todas as temporadas de teste de um modelo, concatenadas."""
    seasons = seasons or backtest.TEST_SEASONS
    cols = ["geocode", "origin", "target", "horizon", "population", "y", *backtest.QCOLS]
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
    forecast_dir: Path | str | None = None,
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
        [
            reports / f"{d}/{f}"
            for d, f in [
                ("fase3", "por_horizonte.csv"),
                ("fase2", "por_horizonte.csv"),
                ("fase4", "por_horizonte.csv"),
                ("fase6", "por_horizonte.csv"),
                ("fase6", "ablacao.csv"),
            ]
        ],
        [reports / f"{d}/alerta.csv" for d in ("fase3", "fase2", "fase4", "fase6")],
        out / "benchmarks.json",
    )
    export_diagnostics(
        forecasts,
        data,
        names,
        out / "diagnostics.json",
        reports_dir=reports,
        forecast_dir=forecast_dir,
    )
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
    meta = export_site(
        data,
        forecasts,
        names,
        args.mesh,
        args.out_dir,
        args.reports_dir,
        forecast_dir=args.forecast_dir,
    )
    print(f"site: {len(meta['weeks'])} semanas x {len(meta['geocodes'])} municípios")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
