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

    err = np.fromfile(tmp_path / "map" / "err_h8.bin", dtype=np.uint8).reshape(n_weeks, n_geo)
    assert err.shape == (n_weeks, n_geo)
    # Resíduo zero dá exatamente RES_ZERO (127); subestimação > 127, superestimação < 127
    assert export.encode_residual([50.0, 200.0, 10.0, np.nan], [50.0, 20.0, 100.0, 5.0]).tolist()[
        0
    ] == export.RES_ZERO
    sample_res = export.encode_residual([200.0, 10.0, np.nan], [20.0, 100.0, 5.0])
    assert sample_res[0] > export.RES_ZERO
    assert sample_res[1] < export.RES_ZERO
    assert sample_res[2] == 255


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


def test_export_diagnostics(toy_data, forecasts, tmp_path):
    names = pd.DataFrame(
        {
            "geocode": toy_data.geocodes,
            "name": [f"M{i}" for i in range(6)],
            "uf": "SP",
            "region": toy_data.region,
        }
    )
    fdir = tmp_path / "forecasts"
    (fdir / "scored").mkdir(parents=True)
    forecasts.to_parquet(fdir / "naive_level_2023.parquet", index=False)
    forecasts.to_parquet(fdir / "lgbm_cqr_level_2023.parquet", index=False)
    backtest.score(forecasts).assign(model="lgbm_cqr_level").to_parquet(
        fdir / "scored" / "lgbm_cqr_level.parquet", index=False
    )

    reports = tmp_path / "reports"
    (reports / "fase3").mkdir(parents=True)
    pd.DataFrame(
        {
            "model": ["naive_level", "lgbm_cqr_level"],
            "band": ["<5 mil", "<5 mil"],
            "horizon": [8, 8],
            "wis": [1.2, 0.9],
            "wis_log": [0.2, 0.15],
            "mae": [1.5, 1.1],
            "cov90": [0.94, 0.92],
        }
    ).to_csv(reports / "fase3" / "por_faixa.csv", index=False)

    diag = export.export_diagnostics(
        forecasts,
        toy_data,
        names,
        tmp_path / "diagnostics.json",
        reports_dir=reports,
        forecast_dir=fdir,
        seasons=(2023,),
    )
    assert "stratified" in diag and len(diag["stratified"]["by_band"]) == 2
    by_model = {r["model"]: r for r in diag["stratified"]["by_band"]}
    assert by_model["lgbm_cqr_level"]["rel_wis"] == pytest.approx(0.75)
    assert len(diag["uf_skill"]["ufs"]) == 1
    assert len(diag["pit"]["models"]) == 2
    assert len(diag["pr_curves"]["models"]) == 1
    assert "8" in diag["error_analysis"]["regimes_by_horizon"]
    shares = [r["share"] for r in diag["error_analysis"]["regimes_by_horizon"]["8"]]
    assert sum(shares) == pytest.approx(1.0, abs=1e-3)


def test_clean_and_num_reject_non_finite_and_archive_is_strict_json():
    assert export._clean(np.array([1.25, np.nan, np.inf, -np.inf])) == [1.2, None, None, None]
    assert export._num(float("inf")) is None
    assert export._num(float("-inf")) is None
    assert export._num(float("nan")) is None

    import tarfile
    from pathlib import Path

    tar_path = Path("site/history.tar.gz")
    if not tar_path.exists():
        pytest.skip("site/history.tar.gz ausente")

    def reject_nan_const(c):
        raise AssertionError(f"JSON inválido com constante {c}")

    with tarfile.open(tar_path, "r:gz") as tf:
        members = {m.lstrip("./"): m for m in tf.getnames()}
        for required in (
            "municipalities.json",
            "benchmarks.json",
            "diagnostics.json",
            "live.json",
            "map/meta.json",
            "map/obs.bin",
            "map/err_h8.bin",
            "fan/3552205.json",
        ):
            assert required in members, f"{required} ausente em site/history.tar.gz"
            if required.endswith(".json"):
                raw = tf.extractfile(members[required]).read().decode("utf-8")
                json.loads(raw, parse_constant=reject_nan_const)

