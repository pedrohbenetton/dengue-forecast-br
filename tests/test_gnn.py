import numpy as np
import pytest
from scipy import sparse

from dengue_forecast.gnn import (
    TORCH_AVAILABLE,
    normalized_adjacency,
    pinball_loss_np,
    prepare_spatiotemporal_window,
)


def test_normalized_adjacency(toy_adjacency):
    norm_adj = normalized_adjacency(toy_adjacency)
    assert norm_adj.shape == toy_adjacency.shape
    assert isinstance(norm_adj, sparse.csr_matrix)

    dense = norm_adj.toarray()
    # Matriz simétrica
    np.testing.assert_allclose(dense, dense.T, atol=1e-6)
    # Diagonal estritamente positiva por conta dos auto-laços
    assert (np.diag(dense) > 0.0).all()
    # Elementos normalizados entre 0 e 1
    assert (dense >= 0.0).all() and (dense <= 1.0).all()


def test_pinball_loss_np():
    y_true = np.array([[10.0], [20.0]])
    # Previsão exata -> perda deve ser zero
    y_pred_exact = np.repeat(y_true, 9, axis=1)
    assert pinball_loss_np(y_true, y_pred_exact) == pytest.approx(0.0)

    # Previsão enviesada para baixo (subestimação)
    y_pred_under = y_pred_exact - 5.0
    loss_under = pinball_loss_np(y_true, y_pred_under)
    assert loss_under > 0.0

    # Monotonicidade da perda com maior erro
    y_pred_worse = y_pred_exact - 10.0
    assert pinball_loss_np(y_true, y_pred_worse) > loss_under


def test_prepare_spatiotemporal_window(toy_data):
    # Semana intermediária com histórico suficiente e alvos no painel
    origin = 150
    x, y = prepare_spatiotemporal_window(toy_data, origin, lookback=12, horizons=(1, 2, 4, 8))
    n_geo = toy_data.cases.shape[0]

    assert x.shape == (n_geo, 12, 4)
    assert np.isfinite(x).all()
    assert y is not None
    assert y.shape == (n_geo, 4)
    assert np.isfinite(y).all()

    # Canal 0 no último passo do histórico deve ser log1p(casos[t])
    np.testing.assert_allclose(x[:, -1, 0], np.log1p(toy_data.cases[:, origin]), rtol=1e-5)


def test_prepare_spatiotemporal_window_out_of_bounds(toy_data):
    with pytest.raises(ValueError, match="fora dos limites"):
        prepare_spatiotemporal_window(toy_data, -1)

    # Última semana: histórico válido, mas sem alvos futuros observados
    last_col = toy_data.n_weeks - 1
    x, y = prepare_spatiotemporal_window(toy_data, last_col, lookback=8)
    assert x.shape == (toy_data.cases.shape[0], 8, 4)
    assert y is None


def test_torch_conditional_import():
    # Verifica que TORCH_AVAILABLE é booleano e que o módulo carrega sem quebrar
    assert isinstance(TORCH_AVAILABLE, bool)
