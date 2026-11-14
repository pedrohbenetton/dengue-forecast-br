import json

import pandas as pd

from dengue_forecast import backtest
from dengue_forecast.export import (
    export_evaluation_summary,
    export_fan_chart_series,
    export_map_timeline,
    export_municipalities_index,
    main,
)


def test_export_evaluation_summary(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()

    df_h = pd.DataFrame({"model": ["lgbm"], "horizon": [1], "wis": [1.8]})
    df_h.to_csv(reports_dir / "por_horizonte.csv", index=False)

    df_t = pd.DataFrame({"model": ["lgbm"], "season": [2024], "wis": [2.1]})
    df_t.to_csv(reports_dir / "por_temporada.csv", index=False)

    out_file = tmp_path / "summary.json"
    data_summary = export_evaluation_summary(reports_dir, out_file)

    assert out_file.exists()
    assert len(data_summary["por_horizonte"]) == 1
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert len(data["por_horizonte"]) == 1
    assert data["por_horizonte"][0]["wis"] == 1.8
    assert len(data["por_temporada"]) == 1
    assert data["por_temporada"][0]["season"] == 2024


def test_export_municipalities_index(tmp_path):
    df = pd.DataFrame(
        {
            "geocode": [3550308, 3304557],
            "name": ["São Paulo", "Rio de Janeiro"],
            "uf": ["SP", "RJ"],
            "region": ["Sudeste", "Sudeste"],
        }
    )
    out_file = tmp_path / "municipalities.json"
    res = export_municipalities_index(df, out_file)

    assert out_file.exists()
    assert len(res) == 2
    assert res[0]["name"] == "São Paulo"
    assert res[1]["uf"] == "RJ"


def test_export_map_timeline(tmp_path):
    rows = []
    for geo in [3550308, 3304557]:
        for h in [1, 2]:
            rows.append(
                {
                    "geocode": geo,
                    "origin": 202401,
                    "target": 202401 + h,
                    "horizon": h,
                    "y": 10.0,
                    **{q_col: 10.0 for q_col in backtest.QCOLS},
                    "q0.5": 12.0,
                    "p_outbreak": 0.05,
                }
            )
    df = pd.DataFrame(rows)
    out_dir = tmp_path / "timeline"
    paths = export_map_timeline(df, out_dir)

    assert len(paths) == 1
    assert paths[0].name == "week_202401.json"
    content = json.loads(paths[0].read_text(encoding="utf-8"))
    assert "3550308" in content
    assert "1" in content["3550308"]["forecasts"]
    assert content["3550308"]["forecasts"]["1"]["q50"] == 12.0


def test_export_fan_chart_series(tmp_path):
    rows = []
    geo = 3550308
    for orig in [202401, 202402]:
        for h in [1, 2]:
            rows.append(
                {
                    "geocode": geo,
                    "origin": orig,
                    "target": orig + h,
                    "horizon": h,
                    "y": 15.0,
                    **{q_col: 15.0 for q_col in backtest.QCOLS},
                }
            )
    df = pd.DataFrame(rows)
    out_dir = tmp_path / "fans"
    paths = export_fan_chart_series(df, out_dir, geocodes=[geo])

    assert len(paths) == 1
    assert paths[0].name == f"geo_{geo}.json"
    content = json.loads(paths[0].read_text(encoding="utf-8"))
    assert len(content) == 4
    assert content[0]["geocode"] if "geocode" in content[0] else content[0]["origin"] == 202401


def test_export_main_cli(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    df_h = pd.DataFrame({"model": ["lgbm"], "horizon": [1], "wis": [1.8]})
    df_h.to_csv(reports_dir / "por_horizonte.csv", index=False)

    out_dir = tmp_path / "public_data"
    code = main(["--reports-dir", str(reports_dir), "--out-dir", str(out_dir)])
    assert code == 0
    assert (out_dir / "evaluation_summary.json").exists()
