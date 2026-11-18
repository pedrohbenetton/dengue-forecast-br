import numpy as np
import pytest

from dengue_forecast import metrics
from dengue_forecast.metrics import QUANTILES


def _interval_wis(y, q, levels):
    """Definição por intervalos (Bracher et al. 2021), escrita de forma independente."""
    k = len(levels) // 2
    total = 0.5 * abs(y - q[k])
    for i in range(k):
        alpha = 2 * levels[i]
        low, up = q[i], q[-1 - i]
        score = (up - low) + (2 / alpha) * max(low - y, 0) + (2 / alpha) * max(y - up, 0)
        total += (alpha / 2) * score
    return total / (k + 0.5)


def test_wis_hand_computed():
    # mediana 8, intervalo de 50% [5, 9], y = 10 -> (0.5*2 + 0.25*(4 + 4*1)) / 1.5 = 2
    out = metrics.wis(np.array([10.0]), np.array([[5.0, 8.0, 9.0]]), np.array([0.25, 0.5, 0.75]))
    assert out.tolist() == pytest.approx([2.0])


def test_wis_matches_interval_definition():
    rng = np.random.default_rng(0)
    q = np.sort(rng.gamma(2.0, 10.0, size=(200, len(QUANTILES))), axis=1)
    y = rng.gamma(2.0, 10.0, size=200)
    expected = [_interval_wis(yi, qi, QUANTILES) for yi, qi in zip(y, q, strict=True)]
    assert metrics.wis(y, q) == pytest.approx(expected)


def test_wis_properties():
    y = np.array([7.0])
    exact = np.full((1, len(QUANTILES)), 7.0)
    assert metrics.wis(y, exact) == pytest.approx([0.0])
    # previsão pontual (todos os quantis iguais) -> WIS = erro absoluto
    assert metrics.wis(y, exact + 3) == pytest.approx([3.0])
    # intervalo mais largo, igualmente centrado, é penalizado
    narrow = 7 + np.linspace(-1, 1, len(QUANTILES))[None, :]
    assert metrics.wis(y, narrow * 1.0) < metrics.wis(y, 7 + 5 * (narrow - 7))


def test_wis_rejects_asymmetric_levels():
    with pytest.raises(ValueError):
        metrics.wis(np.array([1.0]), np.array([[1.0, 2.0]]), np.array([0.1, 0.5]))


def test_coverage():
    q = np.tile(np.arange(9.0), (4, 1))  # q0.25 = 3, q0.75 = 5; q0.05 = 1, q0.95 = 7
    y = np.array([4.0, 3.0, 6.0, 8.0])
    assert metrics.covered(y, q, 0.5).tolist() == [True, True, False, False]
    assert metrics.covered(y, q, 0.9).tolist() == [True, True, True, False]
    with pytest.raises(ValueError):
        metrics.covered(y, q, 0.7)


def test_exceedance_probability():
    q = np.tile(np.arange(9.0), (4, 1))
    thr = np.array([4.0, 4.5, -1.0, 100.0])
    assert metrics.exceedance_probability(q, thr) == pytest.approx([0.5, 0.375, 0.975, 0.025])


def test_exceedance_probability_with_tied_quantiles():
    q = np.zeros((1, 9))
    assert metrics.exceedance_probability(q, np.array([0.0])) == pytest.approx([0.025])
    assert metrics.exceedance_probability(q, np.array([-0.5])) == pytest.approx([0.975])


def test_average_precision():
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    labels = np.array([1, 0, 1, 0], dtype=bool)
    assert metrics.average_precision(labels, scores) == pytest.approx(0.5 * 1 + 0.5 * 2 / 3)
    assert metrics.average_precision(labels, -scores) == pytest.approx(0.5 * 0.5 + 0.5 * 0.5)
    # empate total: a precisão é a prevalência
    assert metrics.average_precision(labels, np.ones(4)) == pytest.approx(0.5)
    assert np.isnan(metrics.average_precision(np.zeros(4, dtype=bool), scores))


def test_pit_uniform_for_calibrated_continuous_and_zero_inflated():
    from scipy import stats

    rng = np.random.default_rng(123)
    # 1. Contínua calibrada (Gamma)
    y_cont = rng.gamma(shape=3.0, scale=5.0, size=20_000)
    q_cont = np.tile(stats.gamma.ppf(QUANTILES, a=3.0, scale=5.0), (len(y_cont), 1))
    hist_cont = metrics.pit_histogram(y_cont, q_cont, bins=10, seed=7)
    assert hist_cont.mean() == pytest.approx(1.0)
    assert np.max(np.abs(hist_cont - 1.0)) < 0.08

    # 2. Degenerada toda em zero (municípios pequenos silenciosos) -> uniforme exata em [0, 1]
    y_zero = np.zeros(25_000)
    q_zero = np.zeros((25_000, len(QUANTILES)))
    hist_zero = metrics.pit_histogram(y_zero, q_zero, bins=10, seed=7)
    assert np.max(np.abs(hist_zero - 1.0)) < 0.06

    # 3. Invariância estrita a transformações monótonas crescentes (log1p)
    assert metrics.pit_values(np.log1p(y_cont), np.log1p(q_cont), seed=7) == pytest.approx(
        metrics.pit_values(y_cont, q_cont, seed=7)
    )


def test_pr_curve_properties():
    scores = np.array([0.95, 0.80, 0.60, 0.40, 0.10])
    labels = np.array([1, 1, 0, 1, 0], dtype=bool)
    rec, prec = metrics.pr_curve(labels, scores, n_points=11)
    assert len(rec) == 11 and len(prec) == 11
    assert rec[0] == 0.0 and rec[-1] == 1.0
    # Envelope é não crescente e termina acima ou igual à prevalência (3/5 = 0.6)
    assert (np.diff(prec) <= 1e-12).all()
    assert prec[0] == pytest.approx(1.0)
    assert prec[-1] >= 0.6 - 1e-12
    _, empty_prec = metrics.pr_curve(np.zeros(4, dtype=bool), np.ones(4), n_points=5)
    assert np.isnan(empty_prec).all()
