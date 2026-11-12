import numpy as np
import pytest

from dengue_forecast.features import LOOKBACK, FeatureBuilder, _lag, _roll_mean


def test_lag_and_rolling_mean():
    m = np.array([[1.0, 2.0, np.nan, 4.0, 5.0]])
    assert np.isnan(_lag(m, 2)[0, :2]).all()
    assert _lag(m, 2)[0, 2:4].tolist() == [1.0, 2.0]
    assert np.isnan(_lag(m, 2)[0, 4])
    np.testing.assert_allclose(_roll_mean(m, 2), [[1.0, 1.5, 2.0, 4.0, 4.5]])
    np.testing.assert_allclose(_roll_mean(m, 10), [[1.0, 1.5, 1.5, 7 / 3, 3.0]])


def test_matrix_layout_and_names(toy_data, toy_adjacency):
    fb = FeatureBuilder(toy_data, toy_adjacency)
    cols = np.array([150, 200])
    x = fb.matrix(cols, 4)
    assert x.shape == (2 * 6, len(fb.names))
    assert x.dtype == np.float32
    assert len(set(fb.names)) == len(fb.names)
    names = fb.names
    # linha i * n_geo + g = origem cols[i], município g
    row = x[1 * 6 + 3]
    assert row[names.index("l0")] == pytest.approx(np.log1p(toy_data.cases[3, 200]))
    assert row[names.index("l3")] == pytest.approx(np.log1p(toy_data.cases[3, 197]))
    assert row[names.index("uf")] == 10
    assert row[names.index("band")] == 3
    expected = np.log1p(toy_data.cases[3, 200 + 4 - 52]) - np.log1p(toy_data.cases[3, 200])
    assert row[names.index("seas_ref")] == pytest.approx(expected, rel=1e-5)
    inc = np.log1p(toy_data.cases[:, 200] / toy_data.population[:, 200] * 1e5)
    assert row[names.index("nb0")] == pytest.approx((inc[2] + inc[4]) / 2, rel=1e-5)


def test_target(toy_data, toy_adjacency):
    fb = FeatureBuilder(toy_data, toy_adjacency)
    y = fb.target(np.array([150, 200]), 8)
    expected = np.log1p(toy_data.cases[3, 208]) - np.log1p(toy_data.cases[3, 200])
    assert y[6 + 3] == pytest.approx(expected)


@pytest.mark.parametrize("horizon", [1, 8])
def test_features_are_causal(toy_data, toy_adjacency, horizon):
    """A origem t tem as mesmas features no painel completo e no painel truncado em t."""
    t = 230
    full = FeatureBuilder(toy_data, toy_adjacency).matrix([t], horizon)
    truncated = toy_data.until(t)
    cut = FeatureBuilder(truncated, toy_adjacency).matrix([truncated.n_weeks - 1], horizon)
    np.testing.assert_array_equal(full, cut)


def test_features_from_tail_match_full_history():
    """Montar features só com as últimas LOOKBACK semanas dá o mesmo resultado."""
    from dengue_forecast import epiweek, geo
    from dengue_forecast.data import PanelData

    rng = np.random.default_rng(1)
    weeks = np.array(
        [y * 100 + w for y in range(2012, 2024) for w in range(1, epiweek.weeks_in_year(y) + 1)]
    )
    shape = (4, len(weeks))
    codes = np.arange(3_500_001, 3_500_005)
    long = PanelData(
        geocodes=codes,
        epiweeks=weeks,
        cases=rng.poisson(20, shape).astype(float),
        population=np.full(shape, 50_000.0),
        band=np.full(4, 2, dtype=np.int8),
        region=np.full(4, "Sudeste"),
        extras={k: rng.normal(20, 3, shape) for k in ("temp_mean", "temp_min", "humid_mean")},
    )
    ring = {int(c): [int(codes[i - 1])] for i, c in enumerate(codes)}
    adjacency = geo.adjacency_matrix(codes, ring)
    tail = long.tail(LOOKBACK)
    assert tail.n_weeks == LOOKBACK < long.n_weeks
    for horizon in (1, 8):
        a = FeatureBuilder(long, adjacency).matrix([long.n_weeks - 1], horizon)
        b = FeatureBuilder(tail, adjacency).matrix([tail.n_weeks - 1], horizon)
        np.testing.assert_allclose(a, b, rtol=1e-6, equal_nan=True)


def test_groups_select_features(toy_data, toy_adjacency):
    own = FeatureBuilder(toy_data, groups=("own",))
    assert "l0" in own.names and "nb0" not in own.names and "tw_sin" not in own.names
    with pytest.raises(ValueError, match="adjacência"):
        FeatureBuilder(toy_data, groups=("neighbors",))
    with pytest.raises(ValueError, match="desconhecidos"):
        FeatureBuilder(toy_data, groups=("magic",))
    everything = FeatureBuilder(toy_data, toy_adjacency)
    assert {"temp_mean_r4_l8", "uf_rel", "crosses_year", "logpop"} <= set(everything.names)
