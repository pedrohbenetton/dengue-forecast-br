import json

import httpx

from dengue_forecast import download


def _handler(ibge_json, infodengue_csv, fail=()):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "servicodados.ibge.gov.br":
            return httpx.Response(200, text=ibge_json)
        if request.url.params["geocode"] in {str(g) for g in fail}:
            return httpx.Response(500)
        return httpx.Response(200, text=infodengue_csv)

    return handler


def test_run_downloads_everything_and_writes_manifest(tmp_path, ibge_json, infodengue_csv):
    handler = _handler(ibge_json, infodengue_csv, fail=[3550308])
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = download.run(
            tmp_path, start_year=2024, end_year=2024, client=client, retries=0, log=lambda s: None
        )

    assert len(result.ok) == 6
    assert list(result.failed) == [3550308]
    assert (tmp_path / "ibge" / "municipios.json").exists()
    assert len(list((tmp_path / "infodengue").glob("*.parquet"))) == 6

    manifest = json.loads((tmp_path / "infodengue" / "_manifest.json").read_text("utf-8"))
    assert manifest["start_year"] == 2024
    assert manifest["end_year"] == 2024
    assert manifest["n_municipalities"] == 7
    assert manifest["n_ok"] == 6
    assert list(manifest["failed"]) == ["3550308"]
    assert manifest["finished_at"] >= manifest["started_at"]


def test_run_limit(tmp_path, ibge_json, infodengue_csv):
    handler = _handler(ibge_json, infodengue_csv)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = download.run(
            tmp_path, start_year=2024, end_year=2024, client=client, limit=2, log=lambda s: None
        )
    assert len(result.ok) == 2


def test_parse_args_defaults():
    args = download.parse_args([])
    assert args.start_year == 2010
    assert args.workers == 4
    assert args.refresh_years is None
    assert not args.full_refresh
