"""Previsão semanal ao vivo: ajusta o LightGBM, calibra com o CQR por nível e grava `live.json`.

Roda depois de `download --refresh-years 1` e `panel`, no workflow semanal ou à mão:

    uv run python -m dengue_forecast.pipeline

As últimas `NOWCAST_WEEKS` semanas ainda estão incompletas (atraso de notificação). Por isso o
modelo é ajustado sem elas, e a previsão parte do nowcast do InfoDengue (`cases_est`) nessas
semanas. As margens do CQR vêm de `reports/fase3/margens_cqr_nivel.csv`, calculadas na última
temporada do backtest (`python -m dengue_forecast.pipeline --write-margins`).
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from dengue_forecast import backtest, epiweek
from dengue_forecast.baselines import HORIZONS
from dengue_forecast.conformal import StaticCQR, attach_origin_cases
from dengue_forecast.data import PanelData
from dengue_forecast.export import FAN_QUANTILES
from dengue_forecast.metrics import OUTBREAK_INCIDENCE, exceedance_probability

NOWCAST_WEEKS = 4
LIVE_HISTORY = 26  # semanas observadas enviadas ao site junto com a previsão
CQR_BY = ("horizon", "level")
MARGINS = "reports/fase3/margens_cqr_nivel.csv"


def apply_nowcast(data: PanelData, weeks: int = NOWCAST_WEEKS) -> PanelData:
    """Troca os casos brutos das últimas `weeks` semanas pelo nowcast, onde ele existe."""
    if "cases_est" not in data.extras:
        return data
    cases = data.cases.copy()
    est = data.extras["cases_est"][:, -weeks:]
    ok = np.isfinite(est) & (est >= 0)
    cases[:, -weeks:] = np.where(ok, est, cases[:, -weeks:])
    return replace(data, cases=cases)


def forecast_latest(model, data: PanelData, cqr: StaticCQR, horizons=HORIZONS) -> pd.DataFrame:
    """Previsão calibrada a partir da última semana do painel, uma linha por (município, h)."""
    model.fit(data.until(data.n_weeks - 1 - NOWCAST_WEEKS))
    live = apply_nowcast(data)
    q = model.predict(live, horizons)
    origin = int(live.epiweeks[-1])
    n_geo = len(live.geocodes)
    pop = live.population[:, -1]
    frames = []
    for j, h in enumerate(horizons):
        frame = pd.DataFrame(q[:, j, :], columns=backtest.QCOLS)
        frame.insert(0, "geocode", live.geocodes)
        frame.insert(1, "origin", origin)
        frame.insert(2, "target", int(epiweek.add(origin, h)))
        frame.insert(3, "horizon", np.full(n_geo, h))
        frame.insert(4, "population", pop)
        frames.append(frame)
    out = attach_origin_cases(pd.concat(frames, ignore_index=True), live)
    out = cqr.transform(out)
    threshold = OUTBREAK_INCIDENCE * out["population"].to_numpy(dtype=float) / 1e5
    out["p_outbreak"] = exceedance_probability(out[backtest.QCOLS].to_numpy(), threshold)
    return out


def export_live(forecasts: pd.DataFrame, data: PanelData, out_file: Path | str) -> dict:
    """`live.json`: quantis e P(surto) por horizonte, na ordem dos municípios do painel."""
    live = apply_nowcast(data)
    record: dict = {
        "origin": int(forecasts["origin"].iloc[0]),
        "geocodes": [int(g) for g in data.geocodes],
        "targets": {},
        "obs_weeks": [int(w) for w in live.epiweeks[-LIVE_HISTORY:]],
        "obs": np.round(live.cases[:, -LIVE_HISTORY:], 1).tolist(),
        "nowcast_weeks": NOWCAST_WEEKS,
    }
    for h, sub in forecasts.groupby("horizon"):
        sub = sub.set_index("geocode").loc[data.geocodes]
        record["targets"][str(h)] = int(sub["target"].iloc[0])
        record[f"h{h}"] = {q: sub[q].round(1).tolist() for q in FAN_QUANTILES}
        record[f"h{h}"]["p"] = sub["p_outbreak"].round(3).tolist()
    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, separators=(",", ":")), encoding="utf-8")
    return json.loads(out.read_text(encoding="utf-8"))


def write_margins(forecast_dir: Path | str, data: PanelData, out_file: Path | str) -> None:
    """Margens do CQR por nível calculadas na temporada mais recente do backtest."""
    season = max(backtest.TEST_SEASONS)
    cal = pd.read_parquet(Path(forecast_dir) / f"lgbm_{season}.parquet")
    cqr = StaticCQR(by=CQR_BY, name="lgbm_cqr_level").fit(attach_origin_cases(cal, data))
    cqr.to_frame().to_csv(out_file, index=False, float_format="%.6f")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--panel", default="data/processed/panel.parquet")
    parser.add_argument("--mesh", default="data/raw/ibge/malha_municipios.topo.json")
    parser.add_argument("--margins", default=MARGINS)
    parser.add_argument("--out", default="web/public/data/live.json")
    parser.add_argument(
        "--write-margins", action="store_true", help="recalcula as margens a partir do backtest"
    )
    parser.add_argument("--forecast-dir", default="data/forecasts")
    args = parser.parse_args(argv)

    from dengue_forecast.features import CLIMATE

    columns = ["geocode", "epiweek", "cases", "cases_est", "population", "pop_band", "region"]
    panel = pd.read_parquet(args.panel, columns=[*columns, *CLIMATE])
    data = PanelData.from_panel(panel, extras=(*CLIMATE, "cases_est"))
    if args.write_margins:
        write_margins(args.forecast_dir, data, args.margins)
        print(f"margens gravadas em {args.margins}")
        return 0

    cqr = StaticCQR.from_frame(pd.read_csv(args.margins), CQR_BY, "lgbm_cqr_level")
    model = backtest.make_model("lgbm", data, mesh_path=args.mesh)
    forecasts = forecast_latest(model, data, cqr)
    live = export_live(forecasts, data, args.out)
    print(f"previsão a partir de {live['origin']}: alvos {live['targets']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
