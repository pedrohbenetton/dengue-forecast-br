"""Baixa a lista do IBGE e as séries do InfoDengue de todos os municípios.

Uso:
    uv run python -m dengue_forecast.download                    # retomável: pula o que já existe
    uv run python -m dengue_forecast.download --refresh-years 2  # rebaixa só os 2 últimos anos
    uv run python -m dengue_forecast.download --full-refresh     # rebaixa tudo
"""

import argparse
import json
import time
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

import httpx

from dengue_forecast import ibge, infodengue

DEFAULT_START_YEAR = 2010


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def run(
    raw_dir: Path | str = "data/raw",
    *,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int | None = None,
    client: httpx.Client | None = None,
    workers: int = infodengue.MAX_WORKERS,
    full_refresh: bool = False,
    refresh_years: int | None = None,
    limit: int | None = None,
    retries: int = 3,
    log: Callable[[str], None] = print,
) -> infodengue.DownloadResult:
    """Baixa tudo para `raw_dir` e grava `infodengue/_manifest.json`."""
    raw_dir = Path(raw_dir)
    out_dir = raw_dir / "infodengue"
    end_year = end_year or date.today().year
    refresh_from = end_year - refresh_years + 1 if refresh_years else None

    owns_client = client is None
    client = client or httpx.Client(timeout=120.0)
    started_at = _now()
    t0 = time.monotonic()
    try:
        municipalities = ibge.fetch_municipalities(
            client=client, cache_path=raw_dir / "ibge" / "municipios.json"
        )
        geocodes = municipalities["geocode"].tolist()[:limit]
        total = len(geocodes)
        log(f"{total} municípios, {start_year}-{end_year}, {workers} workers -> {out_dir}")

        done = 0

        def on_progress(geocode: int, error: Exception | None) -> None:
            nonlocal done
            done += 1
            if error is not None:
                log(f"  falhou {geocode}: {error!r}")
            if done % 100 == 0 or done == total:
                log(f"  {done}/{total} ({time.monotonic() - t0:.0f}s)")

        result = infodengue.download_all(
            geocodes,
            out_dir,
            start_year=start_year,
            end_year=end_year,
            client=client,
            max_workers=workers,
            overwrite=full_refresh,
            refresh_from_year=refresh_from,
            retries=retries,
            on_progress=on_progress,
        )
    finally:
        if owns_client:
            client.close()

    manifest = {
        "started_at": started_at,
        "finished_at": _now(),
        "start_year": start_year,
        "end_year": end_year,
        "refresh_from_year": refresh_from,
        "full_refresh": full_refresh,
        "n_municipalities": total,
        "n_ok": len(result.ok),
        "failed": {str(g): repr(e) for g, e in result.failed.items()},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"ok={len(result.ok)} falhas={len(result.failed)}")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--workers", type=int, default=infodengue.MAX_WORKERS)
    parser.add_argument("--limit", type=int, default=None, help="só os N primeiros (teste)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--refresh-years",
        type=int,
        default=None,
        metavar="N",
        help="rebaixa só os N últimos anos e mescla com o cache",
    )
    group.add_argument("--full-refresh", action="store_true", help="rebaixa tudo")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(
        args.raw_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        workers=args.workers,
        full_refresh=args.full_refresh,
        refresh_years=args.refresh_years,
        limit=args.limit,
    )
    return 1 if result.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
