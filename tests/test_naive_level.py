from dataclasses import replace

import numpy as np
import pytest

from dengue_forecast.baselines import NaiveLevel, level_bin
from dengue_forecast.metrics import QUANTILES

HORIZONS = (1, 2, 4, 8)


def test_level_bins():
    cases = np.array([0, 1, 2, 3, 10, 11, 100, 101, 500, 501, 1e6])
    assert level_bin(cases).tolist() == [0, 1, 1, 2, 3, 4, 6, 7, 7, 8, 8]


def test_uncertainty_depends_on_recent_level(toy_data):
    model = NaiveLevel()
    model.fit(toy_data.until(180))
    data = toy_data.until(200)
    out = model.predict(data, HORIZONS)
    assert out.shape == (6, 4, len(QUANTILES))
    assert (np.diff(out, axis=2) >= -1e-9).all()

    # mesmo município, nível recente zerado -> outra tabela de erros
    low = data.cases.copy()
    low[5, -1] = 0.0
    out_low = model.predict(replace(data, cases=low), HORIZONS)
    width = (np.log1p(out[5, 0, -1]) - np.log1p(out[5, 0, 0]))
    width_low = (np.log1p(out_low[5, 0, -1]) - np.log1p(out_low[5, 0, 0]))
    assert width != pytest.approx(width_low)
    assert out_low[5, 0, 0] == 0.0  # partindo de zero, o quantil inferior fica em zero


def test_requires_fit(toy_data):
    with pytest.raises(RuntimeError):
        NaiveLevel().predict(toy_data, HORIZONS)
