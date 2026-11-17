import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from dengue_forecast import backtest, pipeline
from dengue_forecast.baselines import NaiveLevel
from dengue_forecast.conformal import StaticCQR, attach_origin_cases


def test_apply_nowcast_only_touches_recent_weeks(toy_data):
    cases_est = toy_data.cases.copy()
    cases_est[:, -2:] += 50.0
    cases_est[0, -1] = np.nan  # sem estimativa: mantém o bruto
    data = replace(toy_data, extras={**toy_data.extras, "cases_est": cases_est})
    adjusted = pipeline.apply_nowcast(data, weeks=2)
    np.testing.assert_array_equal(adjusted.cases[:, :-2], toy_data.cases[:, :-2])
    assert adjusted.cases[1, -1] == cases_est[1, -1]
    assert adjusted.cases[0, -1] == toy_data.cases[0, -1]
    assert pipeline.apply_nowcast(toy_data) is toy_data  # sem a coluna, nada muda


@pytest.fixture
def cqr(toy_data):
    cal = backtest.run_season(NaiveLevel(), toy_data, 2022)
    return StaticCQR(by=("horizon", "level"), name="cqr").fit(attach_origin_cases(cal, toy_data))


def test_margins_roundtrip(cqr, toy_data, tmp_path):
    frame = cqr.to_frame()
    frame.to_csv(tmp_path / "m.csv", index=False)
    back = StaticCQR.from_frame(pd.read_csv(tmp_path / "m.csv"), ("horizon", "level"), "cqr")
    test = attach_origin_cases(backtest.run_season(NaiveLevel(), toy_data, 2023), toy_data)
    pd.testing.assert_frame_equal(cqr.transform(test), back.transform(test))


def test_forecast_latest(toy_data, cqr):
    data = replace(toy_data, extras={**toy_data.extras, "cases_est": toy_data.cases * 2})
    out = pipeline.forecast_latest(NaiveLevel(), data, cqr, horizons=(1, 4))
    assert len(out) == 6 * 2
    assert (out["origin"] == data.epiweeks[-1]).all()
    assert set(out["horizon"]) == {1, 4}
    q = out[backtest.QCOLS].to_numpy()
    assert (np.diff(q, axis=1) >= 0).all() and (q >= 0).all()
    assert out["p_outbreak"].between(0, 1).all()
    # a previsão parte do nowcast (casos dobrados), não do bruto
    naive = NaiveLevel()
    naive.fit(data.until(data.n_weeks - 1 - pipeline.NOWCAST_WEEKS))
    raw = naive.predict(pipeline.apply_nowcast(data), (1,))[:, 0, 4]
    np.testing.assert_allclose(out[out["horizon"] == 1]["q0.5"].to_numpy(), raw)


def test_forecast_latest_does_not_train_on_incomplete_weeks(toy_data, cqr):
    seen = {}

    class Spy(NaiveLevel):
        def fit(self, data):
            seen["n_weeks"] = data.n_weeks
            super().fit(data)

    pipeline.forecast_latest(Spy(), toy_data, cqr, horizons=(1,))
    assert seen["n_weeks"] == toy_data.n_weeks - pipeline.NOWCAST_WEEKS


def test_export_live(toy_data, cqr, tmp_path):
    out = pipeline.forecast_latest(NaiveLevel(), toy_data, cqr, horizons=(1, 4))
    live = pipeline.export_live(out, toy_data, tmp_path / "live.json")
    stored = json.loads((tmp_path / "live.json").read_text())
    assert stored == live
    assert live["origin"] == int(toy_data.epiweeks[-1])
    assert live["geocodes"] == [int(g) for g in toy_data.geocodes]
    assert live["targets"] == {"1": 202401, "4": 202404}
    h4 = out[out["horizon"] == 4].set_index("geocode").loc[toy_data.geocodes]
    assert live["h4"]["q0.5"] == pytest.approx(h4["q0.5"].round(1).tolist())
    assert live["h4"]["p"] == pytest.approx(h4["p_outbreak"].round(3).tolist())
    assert len(live["obs"]) == 6 and len(live["obs"][0]) == pipeline.LIVE_HISTORY
