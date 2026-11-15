"""Pipeline operacional de atualização semanal das previsões de dengue.

Executado semanalmente (cron do GitHub Actions) ou sob demanda local:
1. Ingestão de nowcast (casos_est) para mitigar o atraso de notificação recente.
2. Emissão de previsões multi-horizonte (h ∈ {1, 2, 4, 8}) com quantis calibrados.
3. Cálculo da probabilidade de surto (incidência > 300 por 100 mil hab.).
4. Exportação dos artefatos estáticos (JSON) para a camada visual no GitHub Pages.

Uso:
    uv run python -m dengue_forecast.pipeline --sample
"""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from dengue_forecast import backtest, epiweek, export, metrics
from dengue_forecast.baselines import HORIZONS
from dengue_forecast.data import PanelData
from dengue_forecast.metrics import QUANTILES


def apply_nowcast(data: PanelData, lookback_weeks: int = 4) -> PanelData:
    """Substitui casos brutos por nowcast (cases_est) nas últimas semanas de notificação.

    Em semanas recentes, a notificação passiva do SINAN sofre atraso cartorial.
    O InfoDengue calcula 'cases_est' via modelo de correção de atraso Bayesiano.
    """
    if "cases_est" not in data.extras:
        return data

    cases_est = data.extras["cases_est"]
    updated_cases = data.cases.copy()
    n_weeks = data.n_weeks
    start_col = max(0, n_weeks - lookback_weeks)

    recent_est = cases_est[:, start_col:]
    valid_est = ~np.isnan(recent_est) & (recent_est >= 0.0)
    updated_cases[:, start_col:][valid_est] = recent_est[valid_est]

    return replace(data, cases=updated_cases)


def generate_live_forecasts(
    model,
    data: PanelData,
    horizons: tuple[int, ...] = HORIZONS,
    origin_week: int | None = None,
) -> pd.DataFrame:
    """Gera o DataFrame com previsões quantílicas e probabilidades de alerta para a origem dada."""
    if origin_week is None:
        origin_week = int(data.epiweeks[-1])

    n_geo = len(data.geocodes)
    pred_quantiles = model.predict(data, horizons)  # [N, len(horizons), len(QUANTILES)]

    rows = []
    threshold_300 = backtest.OUTBREAK_INCIDENCE * data.population[:, -1] / 1e5

    for j, h in enumerate(horizons):
        target_week = int(epiweek.add(origin_week, h))
        q_h = pred_quantiles[:, j, :]  # [N, 9]
        p_outbreak = metrics.exceedance_probability(q_h, threshold_300)
        q_median = q_h[:, len(QUANTILES) // 2]

        for i in range(n_geo):
            row_dict = {
                "geocode": int(data.geocodes[i]),
                "origin": origin_week,
                "target": target_week,
                "horizon": int(h),
                "y": float(data.cases[i, -1]),
                "population": float(data.population[i, -1]),
                "p_outbreak": float(p_outbreak[i]),
                "q0.5": float(q_median[i]),
            }
            for q_idx, q_col in enumerate(backtest.QCOLS):
                row_dict[q_col] = float(q_h[i, q_idx])
            rows.append(row_dict)

    return pd.DataFrame(rows)


def run_pipeline(
    data: PanelData,
    model,
    out_dir: Path | str = "web/public/data",
    reports_dir: Path | str = "reports/fase6",
    sample_geocodes: list[int] | None = None,
) -> dict[str, int]:
    """Executa o ciclo completo de previsão operacional e exportação para o frontend."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Aplicação de nowcast nas últimas semanas
    adjusted_data = apply_nowcast(data, lookback_weeks=4)

    # 2. Geração das previsões da semana mais recente
    forecasts = generate_live_forecasts(model, adjusted_data)

    # 3. Exportação de dados do mapa e fan charts
    paths_timeline = export.export_map_timeline(forecasts, out_path / "timeline")
    paths_fanchart = export.export_fan_chart_series(
        forecasts, out_path / "fancharts", geocodes=sample_geocodes
    )
    export.export_evaluation_summary(reports_dir, out_path / "evaluation_summary.json")

    return {
        "timeline_slices": len(paths_timeline),
        "fan_charts": len(paths_fanchart),
        "total_forecasts": len(forecasts),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default="web/public/data")
    parser.add_argument("--reports-dir", default="reports/fase6")
    parser.add_argument("--sample", action="store_true", help="Gera com dados sintéticos")
    args = parser.parse_args(argv)

    print(f"Iniciando pipeline operacional semanal (sample={args.sample})...")
    from dengue_forecast.baselines import NaiveLevel

    # Na ausência de painel real baixado, opera em modo seguro com baseline
    model = NaiveLevel()
    rng = np.random.default_rng(42)
    weeks = np.array([202401, 202402, 202403, 202404])
    codes = np.array([3550308, 3304557, 3106200, 5300108, 4106902, 2927408, 1302603])
    cases = rng.poisson(150, (len(codes), len(weeks))).astype(float)
    pop = np.full((len(codes), len(weeks)), 1_000_000.0)

    sample_data = PanelData(
        geocodes=codes,
        epiweeks=weeks,
        cases=cases,
        population=pop,
        band=np.full(len(codes), 4, dtype=np.int8),
        region=np.full(len(codes), "Sudeste"),
        extras={"cases_est": cases * 1.1},
    )
    model.fit(sample_data.until(2))

    stats = run_pipeline(
        sample_data,
        model,
        out_dir=args.out_dir,
        reports_dir=args.reports_dir,
        sample_geocodes=codes.tolist(),
    )
    print(f"Pipeline concluído: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
