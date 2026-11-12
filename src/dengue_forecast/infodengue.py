"""Cliente da API `alertcity` do InfoDengue: URL, parse do CSV e download com cache."""

import io
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlencode

import httpx
import pandas as pd

from dengue_forecast._http import get_with_retry

BASE_URL = "https://info.dengue.mat.br/api/alertcity"
MAX_WORKERS = 4  # teto de concorrência: a API é pública e gratuita

# Nome original (API) -> nome em inglês, na ordem de saída.
COLUMN_MAP = {
    "SE": "epiweek",
    "data_iniSE": "week_start",
    "casos": "cases",
    "casos_est": "cases_est",
    "casos_est_min": "cases_est_min",
    "casos_est_max": "cases_est_max",
    "casprov": "cases_prob",
    "casprov_est": "cases_prob_est",
    "casprov_est_min": "cases_prob_est_min",
    "casprov_est_max": "cases_prob_est_max",
    "casconf": "cases_conf",
    "notif_accum_year": "notif_accum_year",
    "p_inc100k": "incidence_100k",
    "p_rt1": "p_rt1",
    "Rt": "rt",
    "pop": "population",
    "nivel": "alert_level",
    "nivel_inc": "incidence_level",
    "receptivo": "receptive",
    "transmissao": "transmission",
    "tempmin": "temp_min",
    "tempmed": "temp_mean",
    "tempmax": "temp_max",
    "umidmin": "humid_min",
    "umidmed": "humid_mean",
    "umidmax": "humid_max",
    "municipio_nome": "municipality_name",
    "Localidade_id": "locality_id",
    "id": "record_id",
    "versao_modelo": "model_version",
}

_INT = [
    "cases", "cases_est_min", "cases_est_max", "cases_prob", "cases_prob_est_min",
    "cases_prob_est_max", "cases_conf", "notif_accum_year", "population", "alert_level",
    "incidence_level", "receptive", "transmission", "locality_id",
]  # fmt: skip
_FLOAT = [
    "cases_est", "cases_prob_est", "incidence_100k", "p_rt1", "rt",
    "temp_min", "temp_mean", "temp_max", "humid_min", "humid_mean", "humid_max",
]  # fmt: skip
_STRING = ["municipality_name", "record_id"]
_DATE = ["week_start", "model_version"]


def build_url(
    geocode: int,
    *,
    start_year: int,
    end_year: int,
    ew_start: int = 1,
    ew_end: int = 53,
    disease: str = "dengue",
) -> str:
    """Monta a URL do `alertcity` para um município (geocode IBGE de 7 dígitos)."""
    if not 1_000_000 <= int(geocode) <= 9_999_999:
        raise ValueError(f"geocode deve ter 7 dígitos: {geocode!r}")
    if start_year > end_year:
        raise ValueError(f"start_year ({start_year}) > end_year ({end_year})")
    if not 1 <= ew_start <= 53 or not 1 <= ew_end <= 53:
        raise ValueError("semana epidemiológica fora de 1..53")
    params = {
        "geocode": int(geocode),
        "disease": disease,
        "format": "csv",
        "ew_start": ew_start,
        "ew_end": ew_end,
        "ey_start": start_year,
        "ey_end": end_year,
    }
    return f"{BASE_URL}?{urlencode(params)}"


def parse_csv(text: str, geocode: int) -> pd.DataFrame:
    """Converte o CSV do `alertcity` em um DataFrame tipado.

    Colunas em inglês (`COLUMN_MAP`), `geocode` injetado (o CSV não o traz), uma linha por
    semana epidemiológica, em ordem crescente. Em caso de semanas repetidas, fica a de
    `model_version` mais recente.
    """
    if text.strip():
        # Tudo como string: `id` tem 18 dígitos e inteiros com vazios virariam float.
        raw = pd.read_csv(io.StringIO(text), dtype="string")
    else:
        raw = pd.DataFrame(columns=list(COLUMN_MAP), dtype="string")

    if set(raw.columns) != set(COLUMN_MAP):
        missing = sorted(set(COLUMN_MAP) - set(raw.columns))
        extra = sorted(set(raw.columns) - set(COLUMN_MAP))
        raise ValueError(f"colunas inesperadas no CSV: faltando={missing}, sobrando={extra}")

    df = raw.rename(columns=COLUMN_MAP)[list(COLUMN_MAP.values())]
    df["epiweek"] = pd.to_numeric(df["epiweek"]).astype("int64")
    for col in _INT:
        # Alguns inteiros chegam como "515.0"; round() antes de Int64 evita erro de cast.
        df[col] = pd.to_numeric(df[col]).astype("Float64").round().astype("Int64")
    for col in _FLOAT:
        df[col] = pd.to_numeric(df[col]).astype("float64")
    for col in _DATE:
        df[col] = pd.to_datetime(df[col], format="%Y-%m-%d").astype("datetime64[ns]")
    for col in _STRING:
        df[col] = df[col].astype("string")
    df.insert(0, "geocode", pd.Series(int(geocode), index=df.index, dtype="int64"))

    df = (
        df.sort_values(["epiweek", "model_version"], kind="stable")
        .drop_duplicates("epiweek", keep="last")
        .reset_index(drop=True)
    )
    return df


def download_city(
    geocode: int,
    out_dir: Path | str,
    *,
    client: httpx.Client,
    start_year: int,
    end_year: int,
    overwrite: bool = False,
    refresh_from_year: int | None = None,
    retries: int = 3,
    backoff: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Baixa um município para `{out_dir}/{geocode}.parquet`.

    Se o arquivo já existir: pula (padrão); rebaixa tudo (`overwrite`); ou rebaixa só de
    `refresh_from_year` em diante e mescla com o cache (refresh incremental — as revisões por
    atraso de notificação se concentram nas semanas recentes).
    """
    path = Path(out_dir) / f"{int(geocode)}.parquet"
    cached = path.exists() and not overwrite
    if cached and refresh_from_year is None:
        return path
    incremental = cached and refresh_from_year is not None and refresh_from_year > start_year

    first_year = refresh_from_year if incremental else start_year
    url = build_url(geocode, start_year=first_year, end_year=end_year)
    response = get_with_retry(client, url, retries=retries, backoff=backoff, sleep=sleep)
    df = parse_csv(response.text, geocode)
    if incremental:
        old = pd.read_parquet(path)
        old = old[old["epiweek"] < refresh_from_year * 100]
        df = pd.concat([old, df], ignore_index=True) if len(old) else df

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")  # escrita atômica: nada de parquet pela metade
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    return path


@dataclass
class DownloadResult:
    ok: dict[int, Path] = field(default_factory=dict)
    failed: dict[int, Exception] = field(default_factory=dict)


def download_all(
    geocodes: Iterable[int],
    out_dir: Path | str,
    *,
    start_year: int,
    end_year: int,
    client: httpx.Client | None = None,
    max_workers: int = MAX_WORKERS,
    overwrite: bool = False,
    refresh_from_year: int | None = None,
    retries: int = 3,
    backoff: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    on_progress: Callable[[int, Exception | None], None] | None = None,
) -> DownloadResult:
    """Baixa vários municípios com concorrência ≤ `MAX_WORKERS`; falhas não interrompem o lote."""
    if not 1 <= max_workers <= MAX_WORKERS:
        raise ValueError(f"max_workers deve estar em 1..{MAX_WORKERS}")

    owns_client = client is None
    client = client or httpx.Client(timeout=120.0)
    result = DownloadResult()

    def one(geocode: int) -> tuple[int, Path | Exception]:
        try:
            return geocode, download_city(
                geocode,
                out_dir,
                client=client,
                start_year=start_year,
                end_year=end_year,
                overwrite=overwrite,
                refresh_from_year=refresh_from_year,
                retries=retries,
                backoff=backoff,
                sleep=sleep,
            )
        except Exception as exc:  # noqa: BLE001 - registrado em result.failed
            return geocode, exc

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for geocode, outcome in pool.map(one, [int(g) for g in geocodes]):
                if isinstance(outcome, Exception):
                    result.failed[geocode] = outcome
                else:
                    result.ok[geocode] = outcome
                if on_progress is not None:
                    on_progress(geocode, outcome if isinstance(outcome, Exception) else None)
    finally:
        if owns_client:
            client.close()
    return result
