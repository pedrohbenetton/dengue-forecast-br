import matplotlib

matplotlib.use("Agg")

import pandas as pd
from matplotlib.figure import Figure

from dengue_forecast.viz import phase6


def _mock_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    ablacao = pd.DataFrame(
        [
            {"model": "lgbm_own", "horizon": 1, "wis": 1.87, "rel": 0.89},
            {"model": "lgbm_own", "horizon": 8, "wis": 7.31, "rel": 0.85},
            {"model": "lgbm_season", "horizon": 1, "wis": 1.79, "rel": 0.85},
            {"model": "lgbm_season", "horizon": 8, "wis": 6.26, "rel": 0.73},
            {"model": "lgbm", "horizon": 1, "wis": 1.80, "rel": 0.86},
            {"model": "lgbm", "horizon": 8, "wis": 6.43, "rel": 0.75},
            {"model": "lgbm_immunity", "horizon": 1, "wis": 1.79, "rel": 0.85},
            {"model": "lgbm_immunity", "horizon": 8, "wis": 6.40, "rel": 0.74},
        ]
    )
    por_temporada = pd.DataFrame(
        [
            {"model": "lgbm", "season": 2023, "horizon": 8, "rel_wis": 0.71},
            {"model": "lgbm_immunity", "season": 2023, "horizon": 8, "rel_wis": 0.71},
            {"model": "lgbm", "season": 2024, "horizon": 8, "rel_wis": 0.77},
            {"model": "lgbm_immunity", "season": 2024, "horizon": 8, "rel_wis": 0.76},
        ]
    )
    importance = pd.Series(
        {
            "roll4": 0.28,
            "roll8": 0.20,
            "l0": 0.14,
            "reemergence_boost": 0.05,
            "immune_debt": 0.04,
            "susceptible_gap": 0.03,
            "quiet_ratio": 0.02,
        }
    )
    return ablacao, por_temporada, importance


def test_plot_immunity_ablation():
    ablacao, _, _ = _mock_tables()
    fig = phase6.plot_immunity_ablation(ablacao)
    assert isinstance(fig, Figure)


def test_plot_season_shock_comparison():
    _, por_temporada, _ = _mock_tables()
    fig = phase6.plot_season_shock_comparison(por_temporada)
    assert isinstance(fig, Figure)


def test_plot_immunity_importance():
    _, _, importance = _mock_tables()
    fig = phase6.plot_immunity_importance(importance)
    assert isinstance(fig, Figure)


def test_make_all_and_main(tmp_path):
    ablacao, por_temporada, importance = _mock_tables()
    out_dir = tmp_path / "figures"
    paths = phase6.make_all(ablacao, por_temporada, importance, out_dir=out_dir)
    assert len(paths) == 3
    for p in paths:
        assert p.exists() and p.stat().st_size > 0

    f6 = tmp_path / "f6"
    f6.mkdir()
    ablacao.to_csv(f6 / "ablacao.csv", index=False)
    por_temporada.to_csv(f6 / "por_temporada.csv", index=False)
    imp_frame = pd.DataFrame(
        {"feature": importance.index, "group": "own", "importance": importance.values}
    )
    imp_frame.to_csv(f6 / "importancia.csv", index=False)
    rc = phase6.main(
        [
            "--forecast-dir",
            str(tmp_path / "no_forecasts"),
            "--fase6-dir",
            str(f6),
            "--out-dir",
            str(tmp_path / "cli_figures"),
        ]
    )
    assert rc == 0
    assert (tmp_path / "cli_figures" / "fase6_ablacao.png").exists()
