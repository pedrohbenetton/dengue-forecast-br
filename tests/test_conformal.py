import numpy as np
import pandas as pd
import pytest

from dengue_forecast import backtest, baselines
from dengue_forecast.conformal import (
    AdaptiveCQR,
    StaticCQR,
    apply_margins,
    attach_origin_cases,
    conformal_quantile,
    conformity_scores,
)
from dengue_forecast.panel import POP_LABELS

Z_NORM = np.array([-1.96, -1.645, -1.282, -0.674, 0.0, 0.674, 1.282, 1.645, 1.96])


def _make_toy_df(
    n: int = 10,
    y: np.ndarray | None = None,
    q: np.ndarray | None = None,
    m: np.ndarray | float | None = None,
    scale: float = 0.3,
    horizon: int = 1,
    origin: int = 202301,
    target: int = 202302,
    level: int = 0,
) -> pd.DataFrame:
    if y is not None:
        n = len(y)
        y_arr = np.asarray(y, dtype=float)
    else:
        y_arr = np.full(n, 10.0, dtype=float)

    if q is not None:
        q_arr = np.asarray(q, dtype=float)
    else:
        if m is not None:
            m_arr = np.broadcast_to(np.asarray(m, dtype=float), (n,))
        else:
            m_arr = np.full(n, 10.0, dtype=float)
        q_arr = np.clip(np.expm1(np.log1p(m_arr)[:, None] + Z_NORM[None, :] * scale), 0.0, None)

    df = pd.DataFrame(
        {
            "geocode": np.arange(1_000_001, 1_000_001 + n, dtype=np.int64),
            "season": np.full(n, 2023, dtype=np.int16),
            "origin": np.full(n, origin, dtype=np.int64),
            "target": np.full(n, target, dtype=np.int64),
            "horizon": np.full(n, horizon, dtype=np.int8),
            "band": pd.Categorical.from_codes(
                np.zeros(n, dtype=int), categories=POP_LABELS, ordered=True
            ),
            "region": pd.Categorical(np.full(n, "Sudeste")),
            "population": np.full(n, 100_000.0, dtype=float),
            "y": y_arr,
            "crosses_year": np.full(n, False, dtype=bool),
            "level": np.full(n, level, dtype=int),
            "origin_cases": np.full(n, float(level), dtype=float),
        }
    )
    for j, col in enumerate(backtest.QCOLS):
        df[col] = q_arr[:, j]
    return df


def test_conformity_scores_sign():
    # q com 9 quantis: [1, 2, 3, 4, 10, 15, 20, 25, 30]
    q_single = np.array([[1.0, 2.0, 3.0, 4.0, 10.0, 15.0, 20.0, 25.0, 30.0]])

    # 1. y dentro de todos os intervalos centrais (y = 10)
    df_inside = _make_toy_df(n=1, y=np.array([10.0]), q=q_single)
    s_inside = conformity_scores(df_inside)
    assert s_inside.shape == (1, 4)
    assert np.all(s_inside < 0), "Todos os escores devem ser negativos quando y está dentro"

    # 2. y abaixo de lo (y = 0.5 < lo=1.0 para o nível 0.95 com índices (0, 8))
    df_below = _make_toy_df(n=1, y=np.array([0.5]), q=q_single)
    s_below = conformity_scores(df_below)
    # Índice 3 em INTERVALS é o nominal 0.95 (lo=0 -> q=1.0, hi=8 -> q=30.0)
    assert s_below[0, 3] > 0
    expected_below = np.log1p(1.0) - np.log1p(0.5)
    assert np.isclose(s_below[0, 3], expected_below)

    # 3. y acima de hi (y = 50.0 > hi=30.0)
    df_above = _make_toy_df(n=1, y=np.array([50.0]), q=q_single)
    s_above = conformity_scores(df_above)
    assert s_above[0, 3] > 0
    expected_above = np.log1p(50.0) - np.log1p(30.0)
    assert np.isclose(s_above[0, 3], expected_above)


def test_conformal_quantile_finite_sample():
    # n=9, nominal 0.9 -> k = ceil(10 * 0.9) = 9 -> maior escore
    scores = np.arange(1.0, 10.0)
    assert conformal_quantile(scores, 0.9) == 9.0

    # n=0 -> 0.0
    assert conformal_quantile(np.array([]), 0.9) == 0.0

    # n=4, nominal 0.5 -> k = ceil(5 * 0.5) = 3 -> terceiro menor
    scores_4 = np.array([40.0, 10.0, 30.0, 20.0])
    assert conformal_quantile(scores_4, 0.5) == 30.0


def test_apply_margins_monotone_nonnegative_median_fixed():
    rng = np.random.default_rng(0)
    n = 100
    m = rng.uniform(5, 50, n)
    q = np.clip(np.expm1(np.log1p(m)[:, None] + Z_NORM[None, :] * 0.3), 0.0, None)
    q_orig = q.copy()

    # 1. Margens positivas (alargam)
    m_pos = np.full((n, 4), 0.5)
    out_pos = apply_margins(q, m_pos)
    assert np.all(out_pos >= 0)
    assert np.all(np.diff(out_pos, axis=1) >= 0)
    assert np.allclose(out_pos[:, 4], q[:, 4])
    np.testing.assert_array_equal(q, q_orig)

    # 2. Margens negativas grandes (encolhem além do razoável)
    m_neg = np.full((n, 4), -5.0)
    out_neg = apply_margins(q, m_neg)
    assert np.all(out_neg >= 0)
    assert np.all(np.diff(out_neg, axis=1) >= 0)
    assert np.allclose(out_neg[:, 4], q[:, 4])

    # 3. Margens mistas
    m_mix = rng.uniform(-1.0, 1.0, (n, 4))
    out_mix = apply_margins(q, m_mix)
    assert np.all(out_mix >= 0)
    assert np.all(np.diff(out_mix, axis=1) >= 0)
    assert np.allclose(out_mix[:, 4], q[:, 4])


def test_static_cqr_hits_nominal_on_exchangeable_data():
    rng = np.random.default_rng(42)
    n = 20_000
    y_cal = rng.lognormal(mean=2.5, sigma=0.8, size=n)
    y_test = rng.lognormal(mean=2.5, sigma=0.8, size=n)

    # Intervalos estreitos demais de propósito
    z_narrow = np.array([-0.3, -0.2, -0.15, -0.05, 0.0, 0.05, 0.15, 0.2, 0.3])
    m_cal = np.full(n, np.expm1(2.5))
    m_test = np.full(n, np.expm1(2.5))
    q_cal = np.expm1(np.log1p(m_cal)[:, None] + z_narrow[None, :] * 0.2)
    q_test = np.expm1(np.log1p(m_test)[:, None] + z_narrow[None, :] * 0.2)

    df_cal = _make_toy_df(y=y_cal, q=q_cal, horizon=1)
    df_test = _make_toy_df(y=y_test, q=q_test, horizon=1)

    cqr = StaticCQR(by=("horizon",)).fit(df_cal)
    adj_test = cqr.transform(df_test)

    cov90 = ((adj_test["q0.05"] <= adj_test["y"]) & (adj_test["y"] <= adj_test["q0.95"])).mean()
    cov50 = ((adj_test["q0.25"] <= adj_test["y"]) & (adj_test["y"] <= adj_test["q0.75"])).mean()

    assert 0.87 <= cov90 <= 0.93, f"cov90={cov90}"
    assert 0.46 <= cov50 <= 0.54, f"cov50={cov50}"


def test_static_cqr_shrinks_too_wide_intervals():
    rng = np.random.default_rng(123)
    n = 5_000
    y_cal = rng.lognormal(mean=2.0, sigma=0.5, size=n)
    y_test = rng.lognormal(mean=2.0, sigma=0.5, size=n)

    # Intervalos largos demais: cobertura inicial 100%
    z_wide = np.array([-5.0, -4.0, -3.0, -2.0, 0.0, 2.0, 3.0, 4.0, 5.0])
    m = np.full(n, np.expm1(2.0))
    q_cal = np.expm1(np.log1p(m)[:, None] + z_wide[None, :] * 2.0)
    q_test = np.expm1(np.log1p(m)[:, None] + z_wide[None, :] * 2.0)

    df_cal = _make_toy_df(y=y_cal, q=q_cal, horizon=1)
    df_test = _make_toy_df(y=y_test, q=q_test, horizon=1)

    cqr = StaticCQR(by=("horizon",)).fit(df_cal)
    # Margens devem ser negativas
    assert np.all(cqr.margins_[(1,)] < 0)

    adj_test = cqr.transform(df_test)
    cov90 = ((adj_test["q0.05"] <= adj_test["y"]) & (adj_test["y"] <= adj_test["q0.95"])).mean()
    assert 0.85 <= cov90 <= 0.95, f"cov90={cov90}"


def test_static_cqr_fallback_for_unseen_group():
    df_cal = _make_toy_df(n=50, horizon=1, level=0)
    cqr = StaticCQR(by=("horizon", "level")).fit(df_cal)

    df_test = _make_toy_df(n=20, horizon=1, level=5)  # level 5 ausente na calibração
    adj_test = cqr.transform(df_test)

    expected_cqr_h = StaticCQR(by=("horizon",)).fit(df_cal)
    expected_adj = expected_cqr_h.transform(df_test)
    np.testing.assert_allclose(
        adj_test[backtest.QCOLS].to_numpy(), expected_adj[backtest.QCOLS].to_numpy()
    )


def test_aci_no_leakage():
    rng = np.random.default_rng(42)
    n_cal = 200
    m_cal = np.full(n_cal, 10.0)
    y_cal = rng.lognormal(np.log(10), 0.35, size=n_cal)
    cal = _make_toy_df(n=n_cal, y=y_cal, m=m_cal, horizon=2)

    origins = [202300 + w for w in range(1, 21)]
    rows = []
    for orig in origins:
        y_orig = rng.lognormal(np.log(10), 0.35, size=10)
        m_orig = np.full(10, 10.0)
        rows.append(
            _make_toy_df(n=10, y=y_orig, m=m_orig, horizon=2, origin=orig, target=orig + 2)
        )
    test = pd.concat(rows, ignore_index=True)

    T = 202310
    aci = AdaptiveCQR(by=("horizon",))
    res1 = aci.run(cal, test)

    test2 = test.copy()
    mask = test2["target"] > T
    test2.loc[mask, "y"] *= 50.0

    res2 = aci.run(cal, test2)

    # Linhas com origin <= T devem ser idênticas
    mask_le = test["origin"] <= T
    np.testing.assert_allclose(
        res1.loc[mask_le, backtest.QCOLS].to_numpy(),
        res2.loc[mask_le, backtest.QCOLS].to_numpy(),
        err_msg="Vazamento detectado: previsões com origin <= T mudaram!",
    )

    # Linhas com origin > T devem mudar
    mask_gt = test["origin"] > T
    assert not np.allclose(
        res1.loc[mask_gt, backtest.QCOLS].to_numpy(),
        res2.loc[mask_gt, backtest.QCOLS].to_numpy(),
    ), "Previsões com origin > T deveriam ter mudado!"


def test_aci_adapts_to_regime_shift():
    rng = np.random.default_rng(99)
    n_cal = 2000
    m_cal = np.full(n_cal, 10.0)
    y_cal = rng.lognormal(np.log(10), 0.5, size=n_cal)
    cal = _make_toy_df(n=n_cal, y=y_cal, m=m_cal, horizon=1)

    # 40 semanas, horizonte 1: target = origin + 1
    # Origens 1..20: normal (y ≈ 10)
    # Origens 21..40: choque (y = 22, fora do intervalo não ajustado)
    origins = [202300 + w for w in range(1, 41)]
    test_rows = []
    for orig in origins:
        m_w = np.full(20, 10.0)
        if orig <= 202320:
            y_w = rng.lognormal(np.log(10), 0.5, size=20)
        else:
            y_w = np.full(20, 22.0)
        test_rows.append(_make_toy_df(n=20, y=y_w, m=m_w, horizon=1, origin=orig, target=orig + 1))
    test = pd.concat(test_rows, ignore_index=True)

    static_cqr = StaticCQR(by=("horizon",)).fit(cal)
    res_static = static_cqr.transform(test)

    aci = AdaptiveCQR(gamma=0.05, by=("horizon",))
    res_aci = aci.run(cal, test)

    second_half = test["origin"] > 202320
    cov_static_2nd = (
        (res_static.loc[second_half, "q0.05"] <= res_static.loc[second_half, "y"])
        & (res_static.loc[second_half, "y"] <= res_static.loc[second_half, "q0.95"])
    ).mean()

    cov_aci_2nd = (
        (res_aci.loc[second_half, "q0.05"] <= res_aci.loc[second_half, "y"])
        & (res_aci.loc[second_half, "y"] <= res_aci.loc[second_half, "q0.95"])
    ).mean()

    assert cov_aci_2nd > cov_static_2nd, f"ACI ({cov_aci_2nd}) deve ser > Static ({cov_static_2nd})"

    last_origin_mask = test["origin"] == origins[-1]
    last_alpha_eff_90 = res_aci.loc[last_origin_mask, "alpha_eff_90"].iloc[0]
    assert last_alpha_eff_90 < 0.1, f"alpha_eff_90={last_alpha_eff_90} deve ser < 0.1"


def test_attach_origin_cases(toy_data):
    df = pd.DataFrame(
        {
            "geocode": [toy_data.geocodes[0], toy_data.geocodes[1]],
            "origin": [toy_data.epiweeks[0], toy_data.epiweeks[5]],
            "target": [toy_data.epiweeks[1], toy_data.epiweeks[6]],
            "horizon": [1, 1],
        }
    )
    out = attach_origin_cases(df, toy_data)
    assert "origin_cases" in out.columns
    assert "level" in out.columns
    assert out.loc[0, "origin_cases"] == toy_data.cases[0, 0]
    assert out.loc[1, "origin_cases"] == toy_data.cases[1, 5]
    assert out.loc[0, "level"] == baselines.level_bin(toy_data.cases[0, 0])
    assert out.loc[1, "level"] == baselines.level_bin(toy_data.cases[1, 5])

    bad_df = pd.DataFrame(
        {
            "geocode": [9_999_999],
            "origin": [toy_data.epiweeks[0]],
        }
    )
    with pytest.raises(ValueError, match="geocode"):
        attach_origin_cases(bad_df, toy_data)
