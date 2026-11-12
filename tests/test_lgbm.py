from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from dengue_forecast import backtest
from dengue_forecast.lgbm import LGBMQuantile
from dengue_forecast.metrics import QUANTILES

FAST = dict(n_estimators=15, params={"num_leaves": 7, "min_data_in_leaf": 5, "num_threads": 1})


def _model(adjacency, **kwargs):
    return LGBMQuantile(adjacency, **{**FAST, **kwargs})


def test_fit_predict_shape_and_validity(toy_data, toy_adjacency):
    model = _model(toy_adjacency)
    model.fit(toy_data.until(200))
    out = model.predict(toy_data.until(220), (1, 2, 4, 8))
    assert out.shape == (6, 4, len(QUANTILES))
    assert np.isfinite(out).all() and (out >= 0).all()
    assert (np.diff(out, axis=2) >= 0).all()
    assert len(model.models) == 4 * len(QUANTILES)


def test_predict_requires_fit(toy_data, toy_adjacency):
    with pytest.raises(RuntimeError):
        _model(toy_adjacency).predict(toy_data, (1,))


def test_training_set_only_uses_observed_targets(toy_data, toy_adjacency):
    from dengue_forecast.features import FeatureBuilder

    data = toy_data.until(200)
    model = _model(toy_adjacency, min_origin=104)
    x, y = model.training_set(FeatureBuilder(data, toy_adjacency), 8)
    n_origins = (201 - 8) - 104  # origens 104..192: o alvo t+8 cai dentro do painel truncado
    assert len(y) == n_origins * 6 == len(x)
    assert np.isfinite(y).all()


def test_subsampling_caps_rows_and_is_deterministic(toy_data, toy_adjacency):
    from dengue_forecast.features import FeatureBuilder

    builder = FeatureBuilder(toy_data.until(200), toy_adjacency)
    a = _model(toy_adjacency, max_rows=200).training_set(builder, 1)
    b = _model(toy_adjacency, max_rows=200).training_set(builder, 1)
    assert 120 < len(a[1]) < 280
    np.testing.assert_array_equal(a[0], b[0])


def test_deterministic_given_seed(toy_data, toy_adjacency):
    outs = []
    for _ in range(2):
        model = _model(toy_adjacency)
        model.fit(toy_data.until(200))
        outs.append(model.predict(toy_data.until(220), (4,)))
    np.testing.assert_array_equal(outs[0], outs[1])


def test_no_leakage_from_the_future(toy_data, toy_adjacency):
    cut_week = 202310
    cut = int(np.flatnonzero(toy_data.epiweeks == cut_week)[0])
    tampered = toy_data.cases.copy()
    tampered[:, cut + 1 :] = tampered[:, cut + 1 :] * 50 + 1000
    a = backtest.run_season(_model(toy_adjacency), toy_data, 2023)
    b = backtest.run_season(_model(toy_adjacency), replace(toy_data, cases=tampered), 2023)
    qcols = backtest.QCOLS
    before = a["origin"] <= cut_week
    pd.testing.assert_frame_equal(a.loc[before, qcols], b.loc[before, qcols])
    assert not a.loc[~before, qcols].equals(b.loc[~before, qcols])


def test_feature_groups_and_importance(toy_data, toy_adjacency):
    model = _model(None, groups=("own", "season"), name="lgbm_own")
    model.fit(toy_data.until(200))
    assert model.name == "lgbm_own"
    importance = model.importance()
    assert set(importance) == set(model.feature_names)
    assert "nb0" not in importance
    assert sum(importance.values()) == pytest.approx(1.0)
    assert list(importance.values()) == sorted(importance.values(), reverse=True)
