from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from dengue_forecast import backtest  # noqa: E402
from dengue_forecast.features import FeatureBuilder  # noqa: E402
from dengue_forecast.gnn import STGNNQuantile, node_features, node_targets  # noqa: E402
from dengue_forecast.metrics import QUANTILES  # noqa: E402

FAST = dict(device="cpu", hidden=16, layers=1, epochs=3, dropout=0.0)


def _model(adjacency, **kwargs):
    return STGNNQuantile(adjacency, **{**FAST, **kwargs})


def test_fit_predict_shape_and_validity(toy_data, toy_adjacency):
    model = _model(toy_adjacency)
    model.fit(toy_data.until(200))
    out = model.predict(toy_data.until(220), (1, 2, 4, 8))
    assert out.shape == (6, 4, len(QUANTILES))
    assert np.isfinite(out).all() and (out >= 0).all()
    assert (np.diff(out, axis=2) >= 0).all()
    assert len(model.history) == 3


def test_predict_subset_of_horizons_matches_full(toy_data, toy_adjacency):
    model = _model(toy_adjacency)
    model.fit(toy_data.until(200))
    full = model.predict(toy_data.until(220), (1, 2, 4, 8))
    part = model.predict(toy_data.until(220), (2, 8))
    np.testing.assert_allclose(part, full[:, [1, 3], :], rtol=1e-6)


def test_predict_requires_fit(toy_data, toy_adjacency):
    with pytest.raises(RuntimeError):
        _model(toy_adjacency).predict(toy_data, (1,))


def test_deterministic_given_seed(toy_data, toy_adjacency):
    outs = []
    for _ in range(2):
        model = _model(toy_adjacency)
        model.fit(toy_data.until(200))
        outs.append(model.predict(toy_data.until(220), (4,)))
    np.testing.assert_allclose(outs[0], outs[1], rtol=1e-5)


def test_scaler_is_frozen_after_fit(toy_data, toy_adjacency):
    model = _model(toy_adjacency)
    model.fit(toy_data.until(200))
    mean, std = model.mean.copy(), model.std.copy()
    model.predict(toy_data.until(240), (1,))
    np.testing.assert_array_equal(mean, model.mean)
    np.testing.assert_array_equal(std, model.std)


def test_no_leakage_from_the_future(toy_data, toy_adjacency):
    cut_week = 202310
    cut = int(np.flatnonzero(toy_data.epiweeks == cut_week)[0])
    tampered = toy_data.cases.copy()
    tampered[:, cut + 1 :] = tampered[:, cut + 1 :] * 50 + 1000
    a = backtest.run_season(_model(toy_adjacency), toy_data, 2023)
    b = backtest.run_season(_model(toy_adjacency), replace(toy_data, cases=tampered), 2023)
    qcols = backtest.QCOLS
    before = a["origin"] <= cut_week
    pd.testing.assert_frame_equal(
        a.loc[before, qcols], b.loc[before, qcols], check_exact=False, rtol=1e-4
    )
    assert not np.allclose(a.loc[~before, qcols], b.loc[~before, qcols], rtol=1e-3)


def test_node_features_layout(toy_data, toy_adjacency):
    builder = FeatureBuilder(toy_data, toy_adjacency)
    cols = np.array([150, 200])
    x, names = node_features(builder, cols, (1, 8))
    assert x.shape == (2, 6, len(names))
    assert "uf" not in names and "seas_ref_h1" in names and "seas_ref_h8" in names
    assert len(names) == len(builder.names) - 1 + len(builder.season_names)
    # origem cols[1], município 3: mesma célula que a matriz tabular
    flat = builder.matrix(cols, 8)[1 * 6 + 3]
    assert x[1, 3, names.index("l0")] == flat[builder.names.index("l0")]
    assert x[1, 3, names.index("seas_ref_h8")] == flat[builder.names.index("seas_ref")]


def test_node_targets_mask_unobserved(toy_data, toy_adjacency):
    data = toy_data.until(200)
    builder = FeatureBuilder(data, toy_adjacency)
    cols = np.array([150, 195, 199])
    y = node_targets(builder, cols, (1, 8))
    assert y.shape == (3, 6, 2)
    assert np.isfinite(y[0]).all()
    assert np.isfinite(y[1, :, 0]).all() and np.isnan(y[1, :, 1]).all()
    assert np.isnan(y[2, :, 1]).all() and np.isfinite(y[2, :, 0]).all()
    expected = np.log1p(data.cases[:, 158]) - np.log1p(data.cases[:, 150])
    np.testing.assert_allclose(y[0, :, 1], expected, rtol=1e-5)


def test_graph_switch_changes_forecast(toy_data, toy_adjacency):
    outs = []
    for use_graph in (True, False):
        model = _model(toy_adjacency, use_graph=use_graph)
        model.fit(toy_data.until(200))
        outs.append(model.predict(toy_data.until(220), (4,)))
    assert not np.allclose(outs[0], outs[1])
