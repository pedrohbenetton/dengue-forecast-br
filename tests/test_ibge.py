import httpx
import pandas as pd

from dengue_forecast import ibge


def test_parse_columns_and_types(ibge_json):
    df = ibge.parse_municipalities(ibge_json)
    assert list(df.columns) == ["geocode", "name", "uf", "uf_name", "region"]
    assert len(df) == 7
    assert df["geocode"].dtype == "int64"
    assert df["geocode"].is_monotonic_increasing
    assert df["geocode"].is_unique


def test_parse_known_city(ibge_json):
    df = ibge.parse_municipalities(ibge_json).set_index("geocode")
    row = df.loc[3548906]
    assert row["name"] == "São Carlos"
    assert row["uf"] == "SP"
    assert row["uf_name"] == "São Paulo"
    assert row["region"] == "Sudeste"


def test_parse_falls_back_when_microrregiao_is_null(ibge_json):
    # Boa Esperança do Norte (MT, criado em 2025) vem com "microrregiao": null.
    df = ibge.parse_municipalities(ibge_json).set_index("geocode")
    row = df.loc[5101837]
    assert row["uf"] == "MT"
    assert row["region"] == "Centro-Oeste"
    assert not df.isna().any().any()


def test_fetch_uses_cache(tmp_path, ibge_json):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, text=ibge_json)

    cache = tmp_path / "ibge" / "municipios.json"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        first = ibge.fetch_municipalities(client=client, cache_path=cache)
        second = ibge.fetch_municipalities(client=client, cache_path=cache)

    assert calls == [ibge.MUNICIPALITIES_URL]
    assert cache.exists()
    pd.testing.assert_frame_equal(first, second)
