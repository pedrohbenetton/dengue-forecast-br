import matplotlib

matplotlib.use("Agg")

from matplotlib.figure import Figure

from dengue_forecast.viz import diagnostics


def _mock_diagnostics() -> dict:
    return {
        "pit": {
            "bins": 10,
            "models": [
                {
                    "model": "lgbm_cqr_level",
                    "label": "LightGBM + CQR por nível",
                    "h8": [0.95, 0.98, 1.02, 1.05, 1.04, 1.02, 0.99, 0.98, 0.99, 0.98],
                    "h8_2024": [0.70, 0.78, 0.85, 0.90, 0.94, 0.98, 1.05, 1.12, 1.25, 1.43],
                }
            ],
        },
        "pr_curves": {
            "recall": [0.0, 0.25, 0.5, 0.75, 1.0],
            "prevalence": {"8": 0.0326},
            "models": [
                {
                    "model": "naive_level",
                    "label": "Ingênuo por nível",
                    "horizons": {"8": {"pr_auc": 0.234, "precision": [0.8, 0.4, 0.2, 0.1, 0.03]}},
                },
                {
                    "model": "lgbm_cqr_level",
                    "label": "LightGBM + CQR por nível",
                    "horizons": {
                        "8": {"pr_auc": 0.440, "precision": [0.95, 0.7, 0.45, 0.22, 0.03]}
                    },
                },
            ],
        },
        "error_analysis": {
            "regimes_by_horizon": {
                "8": [
                    {
                        "regime": "quiet",
                        "label": "Silêncio",
                        "under90": 0.02,
                        "over90": 0.01,
                        "wis_log": 0.12,
                    },
                    {
                        "regime": "onset",
                        "label": "Explosão",
                        "under90": 0.28,
                        "over90": 0.005,
                        "wis_log": 0.62,
                    },
                ]
            }
        },
    }


def test_diagnostics_plots_and_make_all(tmp_path):
    payload = _mock_diagnostics()
    assert isinstance(diagnostics.plot_pit_and_pr(payload), Figure)
    assert isinstance(diagnostics.plot_error_regimes(payload), Figure)
    paths = diagnostics.make_all(payload, out_dir=tmp_path)
    assert len(paths) == 2
    for p in paths:
        assert p.exists() and p.stat().st_size > 0
