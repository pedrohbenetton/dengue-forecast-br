import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from dengue_forecast.panel import POP_LABELS
from dengue_forecast.viz import phase4


def _synthetic_scored() -> pd.DataFrame:
    rows = []
    models = {"naive_level": 1.0, "lgbm": 0.78, "gnn": 0.73, "gnn_nograph": 0.72}
    for model, factor in models.items():
        for season in (2023, 2024):
            for h in (1, 8):
                for band in POP_LABELS[:3]:
                    rows.append(
                        {
                            "model": model,
                            "season": season,
                            "horizon": h,
                            "band": band,
                            "region": "Sudeste",
                            "crosses_year": False,
                            "wis": (2.0 + h * 0.8) * factor,
                            "wis_log": 0.15 * factor,
                            "ae": 3.0 * factor,
                            "cov50": 0.60,
                            "cov80": 0.84,
                            "cov90": 0.92,
                            "cov95": 0.96,
                            "p_outbreak": 0.1,
                            "outbreak": False,
                        }
                    )
    df = pd.DataFrame(rows)
    df["band"] = pd.Categorical(df["band"], categories=POP_LABELS, ordered=True)
    df["model"] = pd.Categorical(df["model"], categories=list(models))
    return df


def test_phase4_plots_and_make_all(tmp_path):
    scored = _synthetic_scored()
    fig1 = phase4.plot_graph_vs_control(scored)
    assert isinstance(fig1, Figure)
    fig2 = phase4.plot_skill_by_band_phase4(scored)
    assert isinstance(fig2, Figure)

    paths = phase4.make_all(scored, out_dir=tmp_path)
    assert len(paths) == 2
    for p in paths:
        assert p.exists() and p.stat().st_size > 0
    assert np.isfinite(scored["wis"]).all()
