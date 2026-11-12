"""Lista de municípios brasileiros via API de localidades do IBGE."""

import json
from pathlib import Path

import httpx
import pandas as pd

from dengue_forecast._http import get_with_retry

MUNICIPALITIES_URL = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios"
COLUMNS = ["geocode", "name", "uf", "uf_name", "region"]


def _uf(record: dict) -> dict:
    # Municípios novos podem vir com "microrregiao": null; a divisão regional nova
    # ("regiao-imediata") está sempre presente.
    micro = record.get("microrregiao")
    if micro:
        return micro["mesorregiao"]["UF"]
    return record["regiao-imediata"]["regiao-intermediaria"]["UF"]


def parse_municipalities(text: str) -> pd.DataFrame:
    """Converte o JSON do IBGE em um DataFrame (geocode, name, uf, uf_name, region)."""
    rows = []
    for record in json.loads(text):
        uf = _uf(record)
        rows.append(
            {
                "geocode": int(record["id"]),
                "name": record["nome"],
                "uf": uf["sigla"],
                "uf_name": uf["nome"],
                "region": uf["regiao"]["nome"],
            }
        )
    df = pd.DataFrame(rows, columns=COLUMNS).astype({"geocode": "int64"})
    return df.drop_duplicates("geocode").sort_values("geocode").reset_index(drop=True)


def fetch_municipalities(
    *, client: httpx.Client | None = None, cache_path: Path | str | None = None
) -> pd.DataFrame:
    """Baixa a lista de municípios; se `cache_path` existir, lê dele em vez da rede."""
    cache = Path(cache_path) if cache_path is not None else None
    if cache is not None and cache.exists():
        return parse_municipalities(cache.read_text(encoding="utf-8"))

    owns_client = client is None
    client = client or httpx.Client(timeout=60.0)
    try:
        text = get_with_retry(client, MUNICIPALITIES_URL).text
    finally:
        if owns_client:
            client.close()

    df = parse_municipalities(text)  # valida antes de gravar o cache
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(text, encoding="utf-8")
    return df
