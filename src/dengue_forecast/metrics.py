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
