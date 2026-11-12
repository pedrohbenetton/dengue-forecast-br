import json

import numpy as np
import pytest

from dengue_forecast import geo

# Três municípios em fila: A | B | C. A e B dividem o arco 1; B e C dividem o arco 3.
TOPO = json.dumps(
    {
        "type": "Topology",
        "arcs": [[[0, 0], [1, 0]], [[1, 0], [0, 1]], [[1, 1], [1, 0]], [[2, 0], [0, 1]], [[3, 0]]],
        "objects": {
            "BR": {
                "type": "GeometryCollection",
                "geometries": [
                    {"type": "Polygon", "arcs": [[0, 1]], "properties": {"codarea": "1000001"}},
                    {
                        "type": "MultiPolygon",
                        "arcs": [[[-2, 2, 3]]],
                        "properties": {"codarea": "1000002"},
                    },
                    {"type": "Polygon", "arcs": [[-4, 4]], "properties": {"codarea": "1000003"}},
                ],
            }
        },
    }
)


def test_neighbors_from_shared_arcs():
    assert geo.neighbors(TOPO) == {
        1000001: [1000002],
        1000002: [1000001, 1000003],
        1000003: [1000002],
    }


def test_adjacency_matrix_is_row_normalized():
    codes = np.array([1000003, 1000001, 1000002, 1000009])  # ordem arbitrária + um desconhecido
    dense = geo.adjacency_matrix(codes, geo.neighbors(TOPO)).toarray()
    assert dense.shape == (4, 4)
    assert dense[2].tolist() == [0.5, 0.5, 0.0, 0.0]  # B: média de C e A
    assert dense[0].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert dense.sum(axis=1).tolist() == pytest.approx([1.0, 1.0, 1.0, 0.0])


def test_fetch_mesh_uses_cache(tmp_path):
    import httpx

    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, text=TOPO)

    cache = tmp_path / "malha.topo.json"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert geo.fetch_mesh(cache, client=client) == TOPO
        assert geo.fetch_mesh(cache, client=client) == TOPO
    assert len(calls) == 1
