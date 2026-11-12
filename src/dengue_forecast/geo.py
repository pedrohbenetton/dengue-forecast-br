"""Vizinhança entre municípios a partir da malha do IBGE em TopoJSON.

No TopoJSON as fronteiras são arcos compartilhados: dois municípios são vizinhos se, e somente
se, usam o mesmo arco. Isso dá a adjacência exata sem nenhuma operação geométrica.
"""

import json
from collections import defaultdict
from pathlib import Path

import httpx
import numpy as np
from scipy import sparse

from dengue_forecast._http import get_with_retry

MESH_URL = (
    "https://servicodados.ibge.gov.br/api/v3/malhas/paises/BR"
    "?formato=application/json&qualidade=minima&intrarregiao=municipio"
)


def fetch_mesh(cache_path: Path | str, *, client: httpx.Client | None = None) -> str:
    """Baixa a malha municipal (TopoJSON) do IBGE, ou lê do cache se já existir."""
    cache = Path(cache_path)
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    owns_client = client is None
    client = client or httpx.Client(timeout=300.0)
    try:
        text = get_with_retry(client, MESH_URL).text
    finally:
        if owns_client:
            client.close()
    neighbors(text)  # valida antes de gravar
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")
    return text


def _arc_ids(arcs) -> set[int]:
    """Índices de arco de uma geometria; o TopoJSON usa ~i (negativo) para o arco i invertido."""
    out: set[int] = set()
    stack = [arcs]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(item)
        else:
            out.add(item if item >= 0 else ~item)
    return out


def neighbors(topojson: str) -> dict[int, list[int]]:
    """Mapa geocode -> lista ordenada de geocodes vizinhos (que dividem ao menos um arco)."""
    topology = json.loads(topojson)
    users: dict[int, set[int]] = defaultdict(set)
    codes: set[int] = set()
    for collection in topology["objects"].values():
        for geometry in collection["geometries"]:
            code = int(geometry["properties"]["codarea"])
            codes.add(code)
            for arc in _arc_ids(geometry.get("arcs", [])):
                users[arc].add(code)
    out: dict[int, set[int]] = {code: set() for code in codes}
    for sharing in users.values():
        for code in sharing:
            out[code] |= sharing - {code}
    return {code: sorted(adjacent) for code, adjacent in sorted(out.items())}


def adjacency_matrix(geocodes: np.ndarray, adjacency: dict[int, list[int]]) -> sparse.csr_matrix:
    """Matriz [n, n] normalizada por linha: `A @ x` dá a média de `x` entre os vizinhos.

    Segue a ordem de `geocodes`; vizinhos fora da lista são ignorados e municípios sem vizinho
    conhecido ficam com a linha zerada.
    """
    position = {int(code): i for i, code in enumerate(geocodes)}
    rows, cols = [], []
    for code, i in position.items():
        for other in adjacency.get(code, []):
            if other in position:
                rows.append(i)
                cols.append(position[other])
    n = len(geocodes)
    matrix = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    degree = np.asarray(matrix.sum(axis=1)).ravel()
    scale = sparse.diags(np.divide(1.0, degree, out=np.zeros(n), where=degree > 0))
    return (scale @ matrix).tocsr()
