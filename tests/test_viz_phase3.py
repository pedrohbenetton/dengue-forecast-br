import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from dengue_forecast.panel import POP_LABELS
from dengue_forecast.viz import phase3


def _make_mock_scored(models: list[str]) -> pd.DataFrame:
    rows = []
    horizons = [1, 2, 4, 8]
    for m in models:
        for h in horizons:
            for s in [2023, 2024]:
                n = 10
                rows.append(
                    pd.DataFrame(
                        {
                            "model": m,
                            "season": s,
                            "horizon": h,
                            "band": pd.Categorical.from_codes(
                                np.zeros(n, dtype=int), categories=POP_LABELS, ordered=True
                            ),
                            "region": "Sudeste",
                            "crosses_year": False,
                            "wis": np.full(n, 2.0 if m == "naive_level" else 1.5),
                            "wis_log": np.full(n, 0.2),
                            "ae": np.full(n, 3.0),
                            "cov50": np.full(n, 0.50 if "cqr" in m else 0.65),
                            "cov80": np.full(n, 0.80 if "cqr" in m else 0.85),
                            "cov90": np.full(n, 0.90 if "cqr" in m or "aci" in m else 0.93),
                            "cov95": np.full(n, 0.95 if "cqr" in m else 0.96),
                            "p_outbreak": np.full(n, 0.1),
                            "outbreak": np.full(n, False),
                        }
                    )
                )
    df = pd.concat(rows, ignore_index=True)
    df["model"] = pd.Categorical(df["model"], categories=models)
    return df


def test_plot_calibration_comparison():
    models = ["naive_level", "lgbm", "lgbm_cqr", "lgbm_aci"]
    scored = _make_mock_scored(models)
    fig = phase3.plot_calibration_comparison(scored, models)
    assert isinstance(fig, Figure)


def test_plot_skill_comparison():
    models = ["naive_level", "lgbm", "lgbm_cqr", "lgbm_aci"]
    scored = _make_mock_scored(models)
    fig = phase3.plot_skill_comparison(scored, reference="naive_level")
    assert isinstance(fig, Figure)


def test_plot_weekly_coverage_2024():
    origins = [202341 + i for i in range(10)]
    forecasts = {
        "lgbm": pd.DataFrame({"origin": origins, "cov90": np.linspace(0.92, 0.75, 10)}),
        "lgbm_cqr": pd.DataFrame({"origin": origins, "cov90": np.linspace(0.90, 0.80, 10)}),
        "lgbm_aci": pd.DataFrame({"origin": origins, "cov90": np.linspace(0.90, 0.88, 10)}),
    }
    fig = phase3.plot_weekly_coverage_2024(forecasts)
    assert isinstance(fig, Figure)


def test_plot_alpha_trajectory_2024():
    origins = [202341 + i for i in range(10)]
    df = pd.DataFrame(
        {
            "origin": origins * 2,
            "horizon": [1] * 10 + [4] * 10,
            "alpha_eff_90": np.linspace(0.10, 0.04, 20),
        }
    )
    fig = phase3.plot_alpha_trajectory_2024(df)
    assert isinstance(fig, Figure)


def test_make_all_creates_files(tmp_path):
    models = ["naive_level", "lgbm", "lgbm_cqr", "lgbm_aci"]
    scored = _make_mock_scored(models)

    origins = [202341 + i for i in range(10)]
    forecasts_2024 = {
        "lgbm": pd.DataFrame({"origin": origins, "cov90": np.linspace(0.92, 0.75, 10)}),
        "lgbm_aci": pd.DataFrame(
            {
                "origin": origins,
                "horizon": 1,
                "cov90": np.linspace(0.90, 0.88, 10),
                "alpha_eff_90": np.linspace(0.10, 0.05, 10),
            }
        ),
    }

    paths = phase3.make_all(scored, tmp_path, forecasts_2024=forecasts_2024)
    file_names = {p.name for p in paths}
    assert "fase3_calibracao.png" in file_names
    assert "fase3_skill_wis.png" in file_names
    assert "fase3_cobertura_semanal_2024.png" in file_names
    assert "fase3_alpha_eff_aci.png" in file_names
    for p in paths:
        assert p.exists()
        assert p.stat().st_size > 0
