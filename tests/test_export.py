import json

import numpy as np
import pandas as pd
import pytest

from dengue_forecast import backtest, export
from dengue_forecast.baselines import NaiveLevel


@pytest.fixture
def forecasts(toy_data):
    """Previsões reais do harness (ingênuo por nível) para a temporada 2023 do painel sintético."""
    return backtest.run_season(NaiveLevel(), toy_data, 2023)


def test_encoding_roundtrip_and_missing():
    inc = np.array([0.0, 1.0, 37.5, 300.0, np.nan, -2.0, 1e9])
    codes = export.encode_incidence(inc)
    assert codes.dtype == np.uint8
    assert codes[4] == 255 and codes[5] == 0 and codes[6] == 254
    back = np.expm1(codes[:4] / export.INC_SCALE)
    np.testing.assert_allclose(back, inc[:4], rtol=0.04, atol=0.02)
    p = export.encode_probability(np.array([0.0, 0.5, 1.0, np.nan]))
    assert p.tolist() == [0, 127, 254, 255]


def test_export_map_layers(toy_data, forecasts, tmp_path):
    meta = export.export_map(forecasts, toy_data, tmp_path, horizons=(1, 8))
    weeks = sorted(forecasts["target"].unique())
    assert meta["weeks"] == [int(w) for w in weeks]
    assert meta["geocodes"] == [int(g) for g in toy_data.geocodes]
    assert meta["latest_week"] == weeks[-1]
    assert json.loads((tmp_path / "map" / "meta.json").read_text())["horizons"] == [1, 8]

    n_weeks, n_geo = len(weeks), len(toy_data.geocodes)
    obs = np.fromfile(tmp_path / "map" / "obs.bin", dtype=np.uint8).reshape(n_weeks, n_geo)
    col = int(np.flatnonzero(toy_data.epiweeks == weeks[10])[0])
    inc = toy_data.cases[:, col] / toy_data.population[:, col] * 1e5
    np.testing.assert_array_equal(obs[10], export.encode_incidence(inc))

    q50 = np.fromfile(tmp_path / "map" / "q50_h8.bin", dtype=np.uint8).reshape(n_weeks, n_geo)
    row = forecasts[(forecasts["horizon"] == 8) & (forecasts["target"] == weeks[10])]
    row = row.set_index("geocode").loc[toy_data.geocodes]
    expected = export.encode_incidence(row["q0.5"] / row["population"] * 1e5)
    np.testing.assert_array_equal(q50[10], expected)

    p = np.fromfile(tmp_path / "map" / "p_h1.bin", dtype=np.uint8)
    assert p.size == n_weeks * n_geo and (p <= 254).all()


def test_export_fancharts(toy_data, forecasts, tmp_path):
    paths = export.export_fancharts(forecasts, toy_data, tmp_path, horizons=(1, 8))
    assert len(paths) == len(toy_data.geocodes)
    geo = int(toy_data.geocodes[4])
    fan = json.loads((tmp_path / "fan" / f"{geo}.json").read_text())
    weeks = sorted(forecasts["target"].unique())
    assert fan["weeks"] == [int(w) for w in weeks]
    col = int(np.flatnonzero(toy_data.epiweeks == weeks[0])[0])
    assert fan["obs"][0] == pytest.approx(toy_data.cases[4, col])
    assert set(fan) == {"weeks", "obs", "h1", "h8"}
    assert set(fan["h8"]) == {"q0.05", "q0.25", "q0.5", "q0.75", "q0.95"}
    sel = forecasts[(forecasts["geocode"] == geo) & (forecasts["horizon"] == 8)]
    sel = sel.set_index("target")
    assert fan["h8"]["q0.5"][5] == pytest.approx(sel.loc[weeks[5], "q0.5"], abs=0.05)


def test_export_benchmarks(tmp_path):
    a = pd.DataFrame(
        {
            "model": ["naive_level"] * 2 + ["lgbm"] * 2,
            "horizon": [1, 8, 1, 8],
            "wis": [2.0, 8.0, 1.8, 6.0],
            "cov90": [0.93, 0.92, 0.94, 0.90],
        }
    )
    b = pd.DataFrame({"model": ["lgbm_immunity"], "horizon": [8], "wis": [5.9]})
    alert = pd.DataFrame({"model": ["lgbm", "lgbm"], "horizon": [1, 8], "pr_auc": [0.9, 0.43]})
    a.to_csv(tmp_path / "a.csv", index=False)
    b.to_csv(tmp_path / "b.csv", index=False)
    alert.to_csv(tmp_path / "alert.csv", index=False)
    out = export.export_benchmarks(
        [tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "missing.csv"],
        [tmp_path / "alert.csv"],
        tmp_path / "benchmarks.json",
    )
    rows = {r["model"]: r for r in out["rows"]}
    assert list(rows) == ["naive_level", "lgbm", "lgbm_immunity"]  # ordem de LABELS
    assert rows["lgbm"]["wis"] == {"1": 1.8, "2": None, "4": None, "8": 6.0}
    assert rows["lgbm"]["rel_wis_8"] == pytest.approx(0.75)
    assert rows["lgbm"]["cov90"] == pytest.approx(0.92)
    assert rows["lgbm"]["pr_auc_8"] == pytest.approx(0.43)
    assert rows["lgbm_immunity"]["cov90"] is None and rows["naive_level"]["pr_auc_8"] is None
    assert json.loads((tmp_path / "benchmarks.json").read_text())["reference"] == "naive_level"


def test_export_municipalities(toy_data, tmp_path):
    names = pd.DataFrame(
        {"geocode": toy_data.geocodes[:5], "name": [f"M{i}" for i in range(5)], "uf": "XX"}
    )
    records = export.export_municipalities(toy_data, names, tmp_path / "m.json")
    assert [r["geocode"] for r in records] == [int(g) for g in toy_data.geocodes]
    assert records[0]["name"] == "M0" and records[0]["population"] == 3000
    assert records[5]["name"] == str(toy_data.geocodes[5])  # sem nome: usa o geocode
    assert records[5]["region"] == "Sudeste"
