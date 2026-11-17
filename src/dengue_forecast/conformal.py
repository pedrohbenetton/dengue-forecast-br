"""Calibração conformal (CQR e ACI) das previsões probabilísticas do LightGBM.

Uso: uv run python -m dengue_forecast.conformal
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dengue_forecast import backtest, baselines
from dengue_forecast.data import PanelData

INTERVALS: dict[float, tuple[int, int]] = {
    0.5: (3, 5),
    0.8: (2, 6),
    0.9: (1, 7),
    0.95: (0, 8),
}


def conformity_scores(df: pd.DataFrame) -> np.ndarray:
    """Escore CQR em escala log1p para cada intervalo central, [n, 4].

    s = max(log1p(lo) - log1p(y), log1p(y) - log1p(hi)). Negativo quando y está dentro do intervalo.
    """
    y = df["y"].to_numpy(dtype=float)
    log_y = np.log1p(y)
    q = df[backtest.QCOLS].to_numpy(dtype=float)
    scores = np.empty((len(df), len(INTERVALS)), dtype=float)
    for j, (lo_idx, hi_idx) in enumerate(INTERVALS.values()):
        log_lo = np.log1p(q[:, lo_idx])
        log_hi = np.log1p(q[:, hi_idx])
        scores[:, j] = np.maximum(log_lo - log_y, log_y - log_hi)
    return scores


def conformal_quantile(scores: np.ndarray, nominal: float) -> float:
    """Quantil com correção de amostra finita: nível k = ceil((n + 1) * nominal)."""
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0:
        return 0.0
    k = int(np.ceil((n + 1) * nominal))
    if k <= 0:
        return float(np.min(scores))
    if k > n:
        return float(np.max(scores))
    return float(np.partition(scores, k - 1)[k - 1])


def apply_margins(q: np.ndarray, margins: np.ndarray) -> np.ndarray:
    """Aplica margens em escala log1p aos quantis q [n, 9] preservando a mediana.

    margins [n, 4] ou [4] na ordem de INTERVALS. Retorna cópia ajustada, clipped >= 0,
    monótona não decrescente de dentro para fora, sem alterar a mediana (índice 4).
    """
    q_arr = np.asarray(q, dtype=float)
    out = q_arr.copy()
    m_arr = np.asarray(margins, dtype=float)
    if m_arr.ndim == 1:
        m_arr = np.broadcast_to(m_arr, (len(out), len(INTERVALS)))
    for j, (lo_idx, hi_idx) in enumerate(INTERVALS.values()):
        m = m_arr[:, j]
        out[:, lo_idx] = np.expm1(np.log1p(q_arr[:, lo_idx]) - m)
        out[:, hi_idx] = np.expm1(np.log1p(q_arr[:, hi_idx]) + m)

    out = np.clip(out, 0.0, None)
    for idx in (3, 2, 1, 0):
        out[:, idx] = np.minimum(out[:, idx], out[:, idx + 1])
    for idx in (5, 6, 7, 8):
        out[:, idx] = np.maximum(out[:, idx], out[:, idx - 1])
    return out


def attach_origin_cases(df: pd.DataFrame, data: PanelData) -> pd.DataFrame:
    """Anexa colunas origin_cases e level (faixa de casos) a partir da semana de origem."""
    geo_to_row = {int(g): i for i, g in enumerate(data.geocodes)}
    week_to_col = {int(w): j for j, w in enumerate(data.epiweeks)}

    geos = df["geocode"].to_numpy(dtype=np.int64)
    origins = df["origin"].to_numpy(dtype=np.int64)

    missing_geos = set(geos) - set(geo_to_row.keys())
    if missing_geos:
        first = next(iter(missing_geos))
        raise ValueError(f"geocode {first} não encontrado no painel")

    missing_origins = set(origins) - set(week_to_col.keys())
    if missing_origins:
        first = next(iter(missing_origins))
        raise ValueError(f"semana de origem {first} não encontrada no painel")

    row_idx = np.array([geo_to_row[g] for g in geos], dtype=np.intp)
    col_idx = np.array([week_to_col[o] for o in origins], dtype=np.intp)
    origin_cases = data.cases[row_idx, col_idx]

    out = df.copy()
    out["origin_cases"] = origin_cases
    out["level"] = baselines.level_bin(origin_cases)
    return out


class StaticCQR:
    """CQR estático por grupo: margens calculadas na calibração e aplicadas ao teste."""

    def __init__(self, by: tuple[str, ...] = ("horizon",), name: str = "lgbm_cqr"):
        self.by = tuple(by)
        self.name = name
        self.margins_: dict[tuple, np.ndarray] = {}
        self.horizon_margins_: dict[int, np.ndarray] = {}
        self.global_margin_: np.ndarray | None = None

    def fit(self, cal: pd.DataFrame) -> "StaticCQR":
        scores = conformity_scores(cal)
        nominals = list(INTERVALS.keys())

        # Margem global
        self.global_margin_ = np.array(
            [conformal_quantile(scores[:, j], nom) for j, nom in enumerate(nominals)],
            dtype=float,
        )

        # Margem por horizonte (fallback)
        self.horizon_margins_ = {}
        if "horizon" in cal.columns:
            for h, idxs in cal.groupby("horizon", observed=True).indices.items():
                self.horizon_margins_[int(h)] = np.array(
                    [conformal_quantile(scores[idxs, j], nom) for j, nom in enumerate(nominals)],
                    dtype=float,
                )

        # Margens por grupo de self.by
        self.margins_ = {}
        grouped = cal.groupby(list(self.by), observed=True)
        for key, idxs in grouped.indices.items():
            norm_key = key if isinstance(key, tuple) else (key,)
            self.margins_[norm_key] = np.array(
                [conformal_quantile(scores[idxs, j], nom) for j, nom in enumerate(nominals)],
                dtype=float,
            )
        return self

    def to_frame(self) -> pd.DataFrame:
        """Margens em tabela (uma linha por grupo), para guardar no repositório."""
        if self.global_margin_ is None:
            raise RuntimeError("chame fit() antes de to_frame()")
        cols = [f"m{int(nom * 100)}" for nom in INTERVALS]
        rows = [{"kind": "global", **dict(zip(cols, self.global_margin_, strict=True))}]
        for h, m in self.horizon_margins_.items():
            rows.append({"kind": "horizon", "horizon": h, **dict(zip(cols, m, strict=True))})
        for key, m in self.margins_.items():
            group = dict(zip(self.by, (int(k) for k in key), strict=True))
            rows.append({"kind": "group", **group, **dict(zip(cols, m, strict=True))})
        return pd.DataFrame(rows, columns=["kind", *self.by, *cols])

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, by: tuple[str, ...], name: str) -> "StaticCQR":
        cqr = cls(by=by, name=name)
        cols = [f"m{int(nom * 100)}" for nom in INTERVALS]
        for row in frame.itertuples(index=False):
            m = np.array([getattr(row, c) for c in cols], dtype=float)
            if row.kind == "global":
                cqr.global_margin_ = m
            elif row.kind == "horizon":
                cqr.horizon_margins_[int(row.horizon)] = m
            else:
                cqr.margins_[tuple(int(getattr(row, k)) for k in by)] = m
        if cqr.global_margin_ is None:
            raise ValueError("tabela de margens sem a linha global")
        return cqr

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.global_margin_ is None:
            raise RuntimeError("chame fit() antes de transform()")
        margins = np.empty((len(df), len(INTERVALS)), dtype=float)
        grouped = df.groupby(list(self.by), observed=True)
        for key, idxs in grouped.indices.items():
            norm_key = key if isinstance(key, tuple) else (key,)
            if norm_key in self.margins_:
                m = self.margins_[norm_key]
            elif "horizon" in df.columns:
                h = int(df["horizon"].iloc[idxs[0]])
                m = self.horizon_margins_.get(h, self.global_margin_)
            else:
                m = self.global_margin_
            margins[idxs] = m

        q = df[backtest.QCOLS].to_numpy(dtype=float)
        adjusted_q = apply_margins(q, margins)
        out = df.copy()
        out[backtest.QCOLS] = adjusted_q
        return out


class AdaptiveCQR:
    """Adaptive Conformal Inference (ACI, Gibbs & Candès 2021) aplicado por grupo."""

    def __init__(
        self, gamma: float = 0.05, by: tuple[str, ...] = ("horizon",), name: str = "lgbm_aci"
    ):
        self.gamma = gamma
        self.by = tuple(by)
        self.name = name

    def run(self, cal: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
        scores_cal = conformity_scores(cal)
        nominals = list(INTERVALS.keys())
        n_intervals = len(nominals)

        # Escores de calibração fixos e ordenados por grupo
        cal_group_scores: dict[tuple, list[np.ndarray]] = {}
        grouped_cal = cal.groupby(list(self.by), observed=True)
        for key, idxs in grouped_cal.indices.items():
            norm_key = key if isinstance(key, tuple) else (key,)
            cal_group_scores[norm_key] = [np.sort(scores_cal[idxs, j]) for j in range(n_intervals)]

        cal_horizon_scores: dict[int, list[np.ndarray]] = {}
        if "horizon" in cal.columns:
            for h, idxs in cal.groupby("horizon", observed=True).indices.items():
                cal_horizon_scores[int(h)] = [
                    np.sort(scores_cal[idxs, j]) for j in range(n_intervals)
                ]

        cal_global_scores = [np.sort(scores_cal[:, j]) for j in range(n_intervals)]

        def get_scores(norm_key: tuple, h: int, j: int) -> np.ndarray:
            if norm_key in cal_group_scores and len(cal_group_scores[norm_key][j]) > 0:
                return cal_group_scores[norm_key][j]
            if h in cal_horizon_scores and len(cal_horizon_scores[h][j]) > 0:
                return cal_horizon_scores[h][j]
            return cal_global_scores[j]

        # Estado alpha_eff por grupo
        alpha_eff: dict[tuple, np.ndarray] = {}
        test_grouped = test.groupby(list(self.by), observed=True)
        for key in test_grouped.indices:
            norm_key = key if isinstance(key, tuple) else (key,)
            alpha_eff[norm_key] = np.array([1.0 - nom for nom in nominals], dtype=float)

        n_test = len(test)
        adjusted_q = np.empty((n_test, len(backtest.QCOLS)), dtype=float)
        alpha_eff_90 = np.empty(n_test, dtype=float)

        test_y = test["y"].to_numpy(dtype=float)
        test_orig = test["origin"].to_numpy()
        test_target = test["target"].to_numpy()
        test_horizons = (
            test["horizon"].to_numpy(dtype=int)
            if "horizon" in test.columns
            else np.zeros(n_test, dtype=int)
        )

        by_cols = list(self.by)
        if len(by_cols) == 1:
            vals = test[by_cols[0]].to_numpy()
            row_keys = [(v,) for v in vals]
        else:
            row_keys = [tuple(row) for row in test[by_cols].itertuples(index=False, name=None)]

        target_to_rows: dict[int, list[int]] = {}
        origin_to_rows: dict[int, list[int]] = {}
        for i in range(n_test):
            tgt = int(test_target[i])
            orig = int(test_orig[i])
            target_to_rows.setdefault(tgt, []).append(i)
            origin_to_rows.setdefault(orig, []).append(i)

        unique_origins = np.sort(np.unique(test_orig))

        for t in unique_origins:
            t_int = int(t)

            # 1. Atualizar alpha_eff com alvos observados (target == t emitidos antes)
            if t_int in target_to_rows:
                observed_rows = [i for i in target_to_rows[t_int] if test_orig[i] < t_int]
                if observed_rows:
                    obs_by_group: dict[tuple, list[int]] = {}
                    for r in observed_rows:
                        obs_by_group.setdefault(row_keys[r], []).append(r)

                    for g_key, g_rows in obs_by_group.items():
                        if g_key not in alpha_eff:
                            continue
                        g_y = test_y[g_rows]
                        g_adj = adjusted_q[g_rows]
                        for j, (lo_idx, hi_idx) in enumerate(INTERVALS.values()):
                            outside = (g_y < g_adj[:, lo_idx]) | (g_y > g_adj[:, hi_idx])
                            err = float(np.mean(outside))
                            alpha_nom = 1.0 - nominals[j]
                            alpha_eff[g_key][j] += self.gamma * (alpha_nom - err)

            # 2. Emitir ajuste para origem t
            orig_rows = origin_to_rows[t_int]
            emit_by_group: dict[tuple, list[int]] = {}
            for r in orig_rows:
                emit_by_group.setdefault(row_keys[r], []).append(r)

            for g_key, g_rows in emit_by_group.items():
                h_val = test_horizons[g_rows[0]]
                m_vec = np.empty(n_intervals, dtype=float)
                for j in range(n_intervals):
                    eff_nominal = float(np.clip(1.0 - alpha_eff[g_key][j], 0.0, 1.0))
                    sc = get_scores(g_key, h_val, j)
                    m_vec[j] = conformal_quantile(sc, eff_nominal)

                q_raw = test.iloc[g_rows][backtest.QCOLS].to_numpy(dtype=float)
                adj = apply_margins(q_raw, m_vec)
                adjusted_q[g_rows] = adj
                alpha_eff_90[g_rows] = alpha_eff[g_key][2]

        out = test.copy()
        out[backtest.QCOLS] = adjusted_q
        out["alpha_eff_90"] = alpha_eff_90
        return out


def run(
    forecast_dir: Path | str = "data/forecasts",
    val_dir: Path | str = "data/forecasts/val",
    panel_path: Path | str = "data/processed/panel.parquet",
    out_dir: Path | str = "reports/fase3",
    gamma: float = 0.05,
    reference: str = "naive_level",
    log=print,
) -> dict[str, pd.DataFrame]:
    """Aplica calibração conformal CQR e ACI ao LightGBM em todas as temporadas de teste."""
    forecast_dir = Path(forecast_dir)
    val_dir = Path(val_dir)
    out_dir = Path(out_dir)
    (forecast_dir / "scored").mkdir(parents=True, exist_ok=True)

    data = backtest.load_data(panel_path)

    test_dfs: dict[int, pd.DataFrame] = {}
    cal_dfs: dict[int, pd.DataFrame] = {}
    for season in backtest.TEST_SEASONS:
        test_p = forecast_dir / f"lgbm_{season}.parquet"
        if not test_p.exists():
            raise FileNotFoundError(
                f"previsões lgbm_{season}.parquet não encontradas em {forecast_dir}"
            )
        test_dfs[season] = attach_origin_cases(pd.read_parquet(test_p), data)

        if season == 2023:
            cal_p = val_dir / "lgbm_2022.parquet"
        else:
            cal_p = forecast_dir / f"lgbm_{season - 1}.parquet"
        if not cal_p.exists():
            raise FileNotFoundError(f"calibração {cal_p} não encontrada")
        cal_dfs[season] = attach_origin_cases(pd.read_parquet(cal_p), data)

    variants: dict[str, StaticCQR | AdaptiveCQR] = {
        "lgbm_cqr": StaticCQR(by=("horizon",), name="lgbm_cqr"),
        "lgbm_cqr_level": StaticCQR(by=("horizon", "level"), name="lgbm_cqr_level"),
        "lgbm_aci": AdaptiveCQR(gamma=gamma, by=("horizon",), name="lgbm_aci"),
        "lgbm_aci_level": AdaptiveCQR(gamma=gamma, by=("horizon", "level"), name="lgbm_aci_level"),
    }

    for name, method in variants.items():
        log(f"Ajustando {name}...")
        season_adjs: list[pd.DataFrame] = []
        for season in backtest.TEST_SEASONS:
            cal_df = cal_dfs[season]
            test_df = test_dfs[season]
            if isinstance(method, StaticCQR):
                method.fit(cal_df)
                adj = method.transform(test_df)
            else:
                adj = method.run(cal_df, test_df)
            season_adjs.append(adj)
            adj.to_parquet(forecast_dir / f"{name}_{season}.parquet", index=False)

        full_adj = pd.concat(season_adjs, ignore_index=True)
        scored = backtest.score(full_adj).assign(model=name)[backtest._KEEP]
        scored = scored.astype({c: "float32" for c in ("wis", "wis_log", "ae", "p_outbreak")})
        scored.to_parquet(forecast_dir / "scored" / f"{name}.parquet", index=False)

    all_models = [reference, "lgbm", *variants.keys()]
    scored_all = backtest.load_scored(forecast_dir, all_models)
    tables = backtest.write_tables(scored_all, out_dir, reference)
    pd.set_option("display.width", 200)
    log(tables["por_horizonte"].round(3).to_string())
    return tables


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--forecast-dir", default="data/forecasts")
    parser.add_argument("--val-dir", default="data/forecasts/val")
    parser.add_argument("--panel", default="data/processed/panel.parquet")
    parser.add_argument("--out-dir", default="reports/fase3")
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--reference", default="naive_level")
    args = parser.parse_args(argv)

    run(
        forecast_dir=args.forecast_dir,
        val_dir=args.val_dir,
        panel_path=args.panel,
        out_dir=args.out_dir,
        gamma=args.gamma,
        reference=args.reference,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
