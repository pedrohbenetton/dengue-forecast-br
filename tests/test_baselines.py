import numpy as np
import pytest

from dengue_forecast.baselines import Climatology, Naive, SeasonalNaive
from dengue_forecast.metrics import QUANTILES

HORIZONS = (1, 2, 4, 8)
MEDIAN = 4


@pytest.mark.parametrize("model_cls", [Naive, SeasonalNaive, Climatology])
def test_shape_monotone_nonnegative(toy_data, model_cls):
    model = model_cls()
    model.fit(toy_data.until(180))
    out = model.predict(toy_data.until(200), HORIZONS)
    assert out.shape == (6, 4, len(QUANTILES))
    assert (out >= 0).all()
    assert (np.diff(out, axis=2) >= -1e-9).all()  # quantis não se cruzam
    assert np.isfinite(out).all()


def test_naive_median_tracks_last_observation(toy_data):
    model = Naive()
    model.fit(toy_data.until(180))
    data = toy_data.until(200)
    out = model.predict(data, HORIZONS)
    big = out[5, 0, MEDIAN]  # município grande, h=1: mediana perto do último valor
    assert big == pytest.approx(data.cases[5, -1], rel=0.25)
    width = out[:, :, -1] - out[:, :, 0]
    assert (width[5, 3] > width[5, 0]).all()  # incerteza cresce com o horizonte


def test_naive_requires_fit(toy_data):
    with pytest.raises(RuntimeError):
        Naive().predict(toy_data, HORIZONS)


def test_seasonal_naive_uses_same_week_last_year(toy_data):
    model = SeasonalNaive()
    model.fit(toy_data.until(180))
    data = toy_data.until(200)
    out = model.predict(data, (4,))
    reference = data.cases[5, 200 + 4 - 52]
    assert out[5, 0, MEDIAN] == pytest.approx(reference, rel=0.25)


def test_climatology_quantiles_are_empirical(toy_data):
    data = toy_data.until(200)
    out = Climatology().predict(data, (2,))
    target = 200 + 2
    idx = [target - 52 * k + d for k in (1, 2, 3) for d in (-1, 0, 1) if target - 52 * k + d >= 0]
    expected = np.quantile(data.cases[3, idx], QUANTILES)
    assert out[3, 0] == pytest.approx(expected)


def test_climatology_without_history_predicts_zero(toy_data):
    out = Climatology().predict(toy_data.until(10), (1,))
    assert (out == 0).all()
