from urllib.parse import parse_qs, urlparse

import httpx
import pandas as pd
import pytest

from dengue_forecast import infodengue

GEOCODE = 3548906


def test_build_url():
    url = infodengue.build_url(GEOCODE, start_year=2010, end_year=2026)
    parsed = urlparse(url)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == infodengue.BASE_URL
    assert parse_qs(parsed.query) == {
        "geocode": ["3548906"],
        "disease": ["dengue"],
        "format": ["csv"],
        "ew_start": ["1"],
        "ew_end": ["53"],
        "ey_start": ["2010"],
        "ey_end": ["2026"],
    }


def test_build_url_rejects_bad_input():
    with pytest.raises(ValueError):
        infodengue.build_url(35489, start_year=2010, end_year=2026)
    with pytest.raises(ValueError):
        infodengue.build_url(GEOCODE, start_year=2026, end_year=2010)


def test_parse_english_columns_no_portuguese_left(infodengue_csv):
    df = infodengue.parse_csv(infodengue_csv, GEOCODE)
    assert list(df.columns[:4]) == ["geocode", "epiweek", "week_start", "cases"]
    assert set(df.columns) == {"geocode", *infodengue.COLUMN_MAP.values()}
    assert len(df.columns) == 31


def test_parse_injects_geocode_and_sorts_ascending(infodengue_csv):
    df = infodengue.parse_csv(infodengue_csv, GEOCODE)
    assert (df["geocode"] == GEOCODE).all()
    assert df["epiweek"].tolist() == list(range(202401, 202409))
    assert df["week_start"].is_monotonic_increasing
    assert df.index.tolist() == list(range(8))


def test_parse_types(infodengue_csv):
    df = infodengue.parse_csv(infodengue_csv, GEOCODE)
    assert df["geocode"].dtype == "int64"
    assert df["epiweek"].dtype == "int64"
    assert pd.api.types.is_datetime64_any_dtype(df["week_start"])
    assert pd.api.types.is_datetime64_any_dtype(df["model_version"])
    for col in ["cases", "population", "cases_est_min", "alert_level", "cases_conf"]:
        assert df[col].dtype == "Int64", col
    for col in ["cases_est", "rt", "incidence_100k", "temp_min", "humid_mean", "cases_prob_est"]:
        assert df[col].dtype == "float64", col
    assert df["record_id"].dtype == "string"
    assert df["municipality_name"].dtype == "string"


def test_parse_values(infodengue_csv):
    df = infodengue.parse_csv(infodengue_csv, GEOCODE)
    last = df.iloc[-1]
    assert last["epiweek"] == 202408
    assert last["week_start"] == pd.Timestamp("2024-02-18")
    assert last["cases"] == 515
    assert last["population"] == 256898
    assert last["municipality_name"] == "São Carlos"
    assert last["record_id"] == "354890620240820145"
    assert last["rt"] == pytest.approx(1.7886969)
    assert pd.isna(last["cases_conf"])


def test_parse_drops_duplicate_weeks(infodengue_csv):
    header, *rows = infodengue_csv.strip().splitlines()
    doubled = "\n".join([header, *rows, *rows[:3]])
    df = infodengue.parse_csv(doubled, GEOCODE)
    assert len(df) == 8
    assert df["epiweek"].is_unique


def test_parse_empty_response_gives_empty_frame_with_schema(infodengue_csv):
    header = infodengue_csv.splitlines()[0]
    for text in ("", header + "\n"):
        df = infodengue.parse_csv(text, GEOCODE)
        assert df.empty
        assert set(df.columns) == {"geocode", *infodengue.COLUMN_MAP.values()}


def test_parse_rejects_unexpected_schema():
    with pytest.raises(ValueError, match="colunas"):
        infodengue.parse_csv("a,b\n1,2\n", GEOCODE)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_download_city_writes_parquet(tmp_path, infodengue_csv):
    with _client(lambda r: httpx.Response(200, text=infodengue_csv)) as client:
        path = infodengue.download_city(
            GEOCODE, tmp_path, client=client, start_year=2024, end_year=2024
        )
    assert path == tmp_path / f"{GEOCODE}.parquet"
    df = pd.read_parquet(path)
    assert len(df) == 8
    assert (df["geocode"] == GEOCODE).all()


def test_download_city_skips_existing(tmp_path, infodengue_csv):
    calls = []

    def handler(request):
        calls.append(request.url)
        return httpx.Response(200, text=infodengue_csv)

    kwargs = dict(start_year=2024, end_year=2024)
    with _client(handler) as client:
        infodengue.download_city(GEOCODE, tmp_path, client=client, **kwargs)
        infodengue.download_city(GEOCODE, tmp_path, client=client, **kwargs)
        assert len(calls) == 1
        infodengue.download_city(GEOCODE, tmp_path, client=client, overwrite=True, **kwargs)
        assert len(calls) == 2


def test_download_city_retries_then_succeeds(tmp_path, infodengue_csv):
    attempts = []
    sleeps = []

    def handler(request):
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(503)
        if len(attempts) == 2:
            raise httpx.ConnectTimeout("timeout")
        return httpx.Response(200, text=infodengue_csv)

    with _client(handler) as client:
        path = infodengue.download_city(
            GEOCODE,
            tmp_path,
            client=client,
            start_year=2024,
            end_year=2024,
            retries=3,
            backoff=0.5,
            sleep=sleeps.append,
        )
    assert path.exists()
    assert len(attempts) == 3
    assert sleeps == [0.5, 1.0]


def test_download_city_gives_up_and_leaves_no_file(tmp_path):
    with _client(lambda r: httpx.Response(500)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            infodengue.download_city(
                GEOCODE,
                tmp_path,
                client=client,
                start_year=2024,
                end_year=2024,
                retries=2,
                sleep=lambda s: None,
            )
    assert list(tmp_path.iterdir()) == []


def test_download_city_does_not_retry_client_errors(tmp_path):
    attempts = []

    def handler(request):
        attempts.append(1)
        return httpx.Response(404)

    with _client(handler) as client:
        with pytest.raises(httpx.HTTPStatusError):
            infodengue.download_city(
                GEOCODE, tmp_path, client=client, start_year=2024, end_year=2024,
                sleep=lambda s: None,
            )
    assert len(attempts) == 1


def test_download_all_collects_failures(tmp_path, infodengue_csv):
    bad = 3550308

    def handler(request):
        if request.url.params["geocode"] == str(bad):
            return httpx.Response(500)
        return httpx.Response(200, text=infodengue_csv)

    with _client(handler) as client:
        result = infodengue.download_all(
            [GEOCODE, bad, 5300108],
            tmp_path,
            client=client,
            start_year=2024,
            end_year=2024,
            retries=0,
            sleep=lambda s: None,
        )
    assert sorted(result.ok) == [GEOCODE, 5300108]
    assert list(result.failed) == [bad]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["3548906.parquet", "5300108.parquet"]


def test_download_all_caps_concurrency():
    assert infodengue.MAX_WORKERS == 4
    with pytest.raises(ValueError):
        infodengue.download_all([GEOCODE], "x", start_year=2024, end_year=2024, max_workers=5)


def test_download_city_incremental_refresh_merges_with_cache(tmp_path, infodengue_csv):
    header, *rows = infodengue_csv.strip().splitlines()  # linhas: SE 202408 ... 202401
    old_2023 = [r.replace("2024-", "2023-", 1).replace(",2024", ",2023", 1) for r in rows[:2]]
    cached_text = "\n".join([header, *rows[4:], *old_2023])  # 2024 SE 1-4 + 2023 SE 7-8
    requests = []

    def handler(request):
        requests.append(dict(request.url.params))
        text = cached_text if len(requests) == 1 else infodengue_csv
        return httpx.Response(200, text=text)

    with _client(handler) as client:
        infodengue.download_city(GEOCODE, tmp_path, client=client, start_year=2023, end_year=2024)
        path = infodengue.download_city(
            GEOCODE, tmp_path, client=client, start_year=2023, end_year=2024,
            refresh_from_year=2024,
        )

    assert requests[1]["ey_start"] == "2024"  # só a janela recente é rebaixada
    df = pd.read_parquet(path)
    assert df["epiweek"].tolist() == [202307, 202308, *range(202401, 202409)]
    assert df["epiweek"].is_unique
    assert df["cases"].dtype == "Int64"


def test_download_city_refresh_without_cache_downloads_full_range(tmp_path, infodengue_csv):
    requests = []

    def handler(request):
        requests.append(dict(request.url.params))
        return httpx.Response(200, text=infodengue_csv)

    with _client(handler) as client:
        infodengue.download_city(
            GEOCODE, tmp_path, client=client, start_year=2010, end_year=2024,
            refresh_from_year=2024,
        )
    assert requests[0]["ey_start"] == "2010"
