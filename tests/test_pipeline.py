from dataclasses import replace

import numpy as np
import pandas as pd

from dengue_forecast.baselines import NaiveLevel
from dengue_forecast.pipeline import (
    apply_nowcast,
    generate_live_forecasts,
    main,
    run_pipeline,
)


def test_apply_nowcast(toy_data):
    # Clona toy_data e adiciona extras com cases_est inflacionado
    cases_est = toy_data.cases.copy()
    cases_est[:, -2:] += 50.0

    toy_data_est = replace(toy_data, extras={"cases_est": cases_est})

    adjusted = apply_nowcast(toy_data_est, lookback_weeks=2)
    # Semanas anteriores não devem mudar
    np.testing.assert_array_equal(adjusted.cases[:, :-2], toy_data.cases[:, :-2])
    # Últimas 2 semanas devem receber cases_est
    np.testing.assert_array_equal(adjusted.cases[:, -2:], cases_est[:, -2:])


def test_generate_live_forecasts(toy_data):
    model = NaiveLevel()
    model.fit(toy_data.until(200))
    df = generate_live_forecasts(model, toy_data, horizons=(1, 4))

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 6 * 2  # 6 municípios x 2 horizontes
    expected_cols = {
        "geocode", "origin", "target", "horizon", "y", "population", "q0.5", "p_outbreak"
    }
    assert expected_cols <= set(df.columns)

    # Monotonicidade dos quantis
    q_cols = [f"q{q}" for q in [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]]
    q_matrix = df[q_cols].to_numpy()
    assert (np.diff(q_matrix, axis=1) >= 0.0).all()
    # Probabilidade de surto entre 0 e 1
    assert (df["p_outbreak"] >= 0.0).all() and (df["p_outbreak"] <= 1.0).all()


def test_run_pipeline_and_cli(toy_data, tmp_path):
    model = NaiveLevel()
    model.fit(toy_data.until(200))
    out_dir = tmp_path / "web_data"
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_content = "model,horizon,wis\nnaive,1,2.0\n"
    (reports_dir / "por_horizonte.csv").write_text(csv_content, encoding="utf-8")

    stats = run_pipeline(
        toy_data,
        model,
        out_dir=out_dir,
        reports_dir=reports_dir,
        sample_geocodes=[int(toy_data.geocodes[0])],
    )

    assert stats["total_forecasts"] > 0
    assert stats["timeline_slices"] > 0
    assert stats["fan_charts"] == 1
    assert (out_dir / "evaluation_summary.json").exists()

    # CLI
    rc = main(["--sample", "--out-dir", str(tmp_path / "cli_out")])
    assert rc == 0
    assert (tmp_path / "cli_out" / "evaluation_summary.json").exists()
