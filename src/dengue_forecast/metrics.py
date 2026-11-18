"""Métricas de previsão probabilística por quantis (vetorizadas, uma linha por previsão)."""

import numpy as np

QUANTILES = np.array([0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975])
OUTBREAK_INCIDENCE = 300.0  # casos por 100 mil hab. na semana


def _check_levels(levels: np.ndarray) -> np.ndarray:
    levels = np.asarray(levels, dtype=float)
    symmetric = len(levels) % 2 == 1 and np.allclose(levels + levels[::-1], 1.0)
    if not symmetric or (np.diff(levels) <= 0).any():
        raise ValueError("níveis devem ser crescentes, simétricos e incluir a mediana")
    return levels


def wis(y: np.ndarray, q: np.ndarray, levels: np.ndarray = QUANTILES) -> np.ndarray:
    """Weighted Interval Score (Bracher et al., 2021). Menor é melhor; mesma unidade de `y`.

    Usa a identidade WIS = média, sobre os quantis, de 2 × pinball loss — equivalente à soma
    ponderada dos interval scores com pesos alpha/2 e peso 1/2 na mediana.
    """
    levels = _check_levels(levels)
    diff = np.asarray(y, dtype=float)[:, None] - np.asarray(q, dtype=float)
    pinball = diff * (levels[None, :] - (diff < 0))
    return 2.0 * pinball.mean(axis=1)


def covered(y: np.ndarray, q: np.ndarray, nominal: float, levels=QUANTILES) -> np.ndarray:
    """`y` caiu dentro do intervalo central de cobertura nominal `nominal`?"""
    levels = _check_levels(levels)
    low = np.flatnonzero(np.isclose(levels, (1 - nominal) / 2))
    if len(low) == 0:
        raise ValueError(f"não há quantis para o intervalo de {nominal:.0%}")
    i = int(low[0])
    y = np.asarray(y, dtype=float)
    return (q[:, i] <= y) & (y <= q[:, len(levels) - 1 - i])


def exceedance_probability(q: np.ndarray, threshold: np.ndarray, levels=QUANTILES) -> np.ndarray:
    """P(Y > threshold) pela CDF linear por partes que passa pelos quantis.

    Fora do intervalo coberto pelos quantis a probabilidade satura (aqui, em 0,975 e 0,025).
    """
    levels = _check_levels(levels)
    q = np.asarray(q, dtype=float)
    thr = np.asarray(threshold, dtype=float)
    n_below = (q <= thr[:, None]).sum(axis=1)  # quantos quantis estão em ou abaixo do limiar
    lo = np.clip(n_below - 1, 0, len(levels) - 2)
    rows = np.arange(len(q))
    q_lo, q_hi = q[rows, lo], q[rows, lo + 1]
    width = np.where(q_hi > q_lo, q_hi - q_lo, 1.0)
    frac = np.clip((thr - q_lo) / width, 0.0, 1.0)
    cdf = levels[lo] + (levels[lo + 1] - levels[lo]) * frac
    cdf = np.where(n_below == 0, levels[0], np.where(n_below == len(levels), levels[-1], cdf))
    return 1.0 - cdf


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Área sob a curva precisão-recall (average precision), com empates tratados em bloco."""
    labels = np.asarray(labels, dtype=bool)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    sorted_scores = np.asarray(scores, dtype=float)[order]
    tp = np.cumsum(labels[order])
    last_of_tie = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(sorted_scores) - 1]
    tp_at = tp[last_of_tie]
    precision = tp_at / (last_of_tie + 1)
    recall_gain = np.diff(np.r_[0, tp_at]) / n_pos
    return float((precision * recall_gain).sum())


def pit_values(
    y: np.ndarray, q: np.ndarray, levels=QUANTILES, seed: int = 0
) -> np.ndarray:
    """PIT aleatorizado em [0, 1] a partir de quantis preditivos (invariante a escala monótona).

    Quando `y` cai estritamente entre dois quantis `q_{k-1} < y < q_k`, sorteia uniformemente em
    `[levels[k-1], levels[k]]`, eliminando viés de corda linear em distribuições assimétricas. Em
    empates discretos (ex.: massa em zero), sorteia na célula delimitada pelos pontos médios entre
    níveis adjacentes (Czado et al., 2009). Sob calibração perfeita, o PIT é Uniforme(0, 1).
    """
    levels = _check_levels(levels)
    y_arr = np.asarray(y, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    k_len = len(levels)

    n_lt = (q_arr < y_arr[:, None]).sum(axis=1)
    n_le = (q_arr <= y_arr[:, None]).sum(axis=1)

    mids = np.empty(k_len + 1, dtype=float)
    mids[0] = 0.0
    mids[1:-1] = 0.5 * (levels[:-1] + levels[1:])
    mids[-1] = 1.0

    knots = np.r_[0.0, levels, 1.0]
    u_lo = knots[n_lt]
    u_hi = knots[n_lt + 1]

    tied = n_le > n_lt
    u_lo[tied] = mids[n_lt[tied]]
    u_hi[tied] = mids[n_le[tied]]

    rng = np.random.default_rng(seed)
    u = u_lo + (u_hi - u_lo) * rng.random(len(y_arr))
    return np.clip(u, 0.0, 1.0)


def pit_histogram(
    y: np.ndarray, q: np.ndarray, bins: int = 10, levels=QUANTILES, seed: int = 0
) -> np.ndarray:
    """Densidades normalizadas (média = 1 sob uniformidade) do histograma PIT em `bins` faixas."""
    u = pit_values(y, q, levels=levels, seed=seed)
    counts, _ = np.histogram(u, bins=np.linspace(0.0, 1.0, bins + 1))
    total = counts.sum()
    if total == 0:
        return np.zeros(bins, dtype=float)
    return counts.astype(float) / total * bins


def pr_curve(
    labels: np.ndarray, scores: np.ndarray, n_points: int = 101
) -> tuple[np.ndarray, np.ndarray]:
    """Curva precisão-recall reamostrada em grade uniforme de recall [0, 1] (empates em bloco)."""
    recall_grid = np.linspace(0.0, 1.0, n_points)
    labels_b = np.asarray(labels, dtype=bool)
    n_pos = int(labels_b.sum())
    if n_pos == 0 or len(labels_b) == 0:
        return recall_grid, np.full(n_points, np.nan)

    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    sorted_scores = np.asarray(scores, dtype=float)[order]
    tp = np.cumsum(labels_b[order])
    last_of_tie = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(sorted_scores) - 1]

    rec = np.r_[0.0, tp[last_of_tie] / n_pos]
    prec = np.r_[1.0, tp[last_of_tie] / (last_of_tie + 1)]
    # Envelope monótono decrescente padrão para interpolação em grade de recall
    prec_env = np.maximum.accumulate(prec[::-1])[::-1]
    idx = np.searchsorted(rec, recall_grid, side="left")
    idx = np.clip(idx, 0, len(prec_env) - 1)
    prec_out = prec_env[idx].copy()
    prec_out[0] = prec_env[1] if len(prec_env) > 1 else prec_env[0]
    return recall_grid, prec_out
