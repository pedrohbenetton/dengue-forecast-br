from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from dengue_forecast import backtest, epiweek
from dengue_forecast.baselines import Climatology, Naive, SeasonalNaive
from dengue_forecast.data import PanelData
from dengue_forecast.metrics import QUANTILES

SEASON = 2023  # alvos: SE 41/2022 a SE 40/2023


class Spy:
    """Modelo que registra o que o harness lhe mostrou e prevê a última observação."""

    name = "spy"

    def __init__(self):
        self.fit_last_weeks = []
        self.predict_calls = []

    def fit(self, data: PanelData) -> None:
        self.fit_last_weeks.append(int(data.epiweeks[-1]))

    def predict(self, data: PanelData, horizons) -> np.ndarray:
        self.predict_calls.append((int(data.epiweeks[-1]), tuple(horizons)))
        last = data.cases[:, -1]
        return np.repeat(last[:, None, None], len(horizons), 1).repeat(len(QUANTILES), 2)


def test_panel_data_from_panel_roundtrip(toy_data):
    rows = []
    for i, g in enumerate(toy_data.geocodes):
        rows.append(
            pd.DataFrame(
                {
                    "geocode": g,
                    "epiweek": toy_data.epiweeks,
                    "cases": toy_data.cases[i].astype("int64"),
                    "population": toy_data.population[i].astype("int64"),
                    "pop_band": ["<5 mil", "5–20 mil", "20–100 mil", "100–500 mil", ">500 mil"][
                        toy_data.band[i]
                    ],
                    "region": toy_data.region[i],
                }
            )
        )
    panel = pd.concat(rows).sample(frac=1.0, random_state=0)  # ordem embaralhada
    data = PanelData.from_panel(panel)
    np.testing.assert_array_equal(data.cases, toy_data.cases)
    np.testing.assert_array_equal(data.band, toy_data.band)
    np.testing.assert_array_equal(data.epiweeks, toy_data.epiweeks)
    with pytest.raises(ValueError, match="grade"):
        PanelData.from_panel(panel.iloc[1:])


def test_until_truncates(toy_data):
    cut = toy_data.until(10)
    assert cut.n_weeks == 11
    assert cut.cases.shape == (6, 11)
    assert cut.epiweeks[-1] == toy_data.epiweeks[10]


def test_season_positions(toy_data):
    pos = toy_data.season_positions(SEASON)
    assert toy_data.epiweeks[pos[0]] == 202241
    assert toy_data.epiweeks[pos[-1]] == 202340
    assert len(pos) == 52


def test_harness_shows_only_the_past(toy_data):
    spy = Spy()
    backtest.run_season(spy, toy_data, SEASON)
    assert spy.fit_last_weeks == [202240]  # treino termina na véspera da temporada
    assert spy.predict_calls[0] == (202233, (8,))  # primeira origem: 8 semanas antes do 1º alvo
    for origin, horizons in spy.predict_calls:
        for h in horizons:
            target = int(epiweek.add(np.array([origin]), h)[0])
            assert 202241 <= target <= 202340


def test_every_target_scored_for_every_horizon(toy_data):
    out = backtest.run_season(Spy(), toy_data, SEASON)
    assert set(out["horizon"]) == {1, 2, 4, 8}
    counts = out.groupby("horizon")["target"].nunique()
    assert (counts == 52).all()
    assert len(out) == 6 * 52 * 4
    assert (out["season"] == SEASON).all()
    gap = epiweek.to_index(out["target"]) - epiweek.to_index(out["origin"])
    assert (gap == out["horizon"]).all()
    assert not out.duplicated(["geocode", "origin", "horizon"]).any()


def test_truth_and_forecast_alignment(toy_data):
    out = backtest.run_season(Spy(), toy_data, SEASON)
    row = out[(out["geocode"] == 1_000_004) & (out["target"] == 202310) & (out["horizon"] == 4)]
    col = {int(e): i for i, e in enumerate(toy_data.epiweeks)}
    assert row["origin"].item() == 202306
    assert row["y"].item() == toy_data.cases[3, col[202310]]
    assert row["q0.5"].item() == toy_data.cases[3, col[202306]]  # Spy = última observação
    assert row["population"].item() == 250_000
    assert not row["crosses_year"].item()
    turn = out[(out["origin"] == 202250) & (out["horizon"] == 4)]
    assert turn["crosses_year"].all()


@pytest.mark.parametrize("model_cls", [Naive, SeasonalNaive, Climatology])
def test_no_leakage_from_the_future(toy_data, model_cls):
    """Mudar o futuro não pode mudar nenhuma previsão feita antes dele."""
    cut_week = 202310
    cut = int(np.flatnonzero(toy_data.epiweeks == cut_week)[0])
    tampered_cases = toy_data.cases.copy()
    tampered_cases[:, cut + 1 :] = tampered_cases[:, cut + 1 :] * 50 + 1000
    tampered = replace(toy_data, cases=tampered_cases)

    a = backtest.run_season(model_cls(), toy_data, SEASON)
    b = backtest.run_season(model_cls(), tampered, SEASON)
    qcols = [c for c in a.columns if c.startswith("q")]
    before = a["origin"] <= cut_week
    assert before.sum() > 0 and (~before).sum() > 0
    pd.testing.assert_frame_equal(a.loc[before, qcols], b.loc[before, qcols])
    if model_cls is Naive:  # o teste tem poder: quem usa a última observação muda depois do corte
        assert not a.loc[~before, qcols].equals(b.loc[~before, qcols])


def test_score_and_summarize(toy_data):
    forecasts = backtest.run_season(Naive(), toy_data, SEASON)
    scored = backtest.score(forecasts)
    for col in ["wis", "wis_log", "ae", "cov50", "cov90", "p_outbreak", "outbreak"]:
        assert col in scored.columns
    assert (scored["wis"] >= 0).all()
    assert scored["p_outbreak"].between(0, 1).all()
    row = scored.iloc[0]
    assert row["ae"] == abs(row["y"] - row["q0.5"])
    assert row["outbreak"] == (row["y"] / row["population"] * 1e5 > 300)

    summary = backtest.summarize(scored.assign(model="naive"), by=["horizon"])
    assert list(summary.index.names) == ["model", "horizon"]
    assert summary["n"].sum() == len(scored)
    assert summary.loc[("naive", 1), "wis"] == pytest.approx(
        scored.loc[scored["horizon"] == 1, "wis"].mean()
    )


def test_relative_skill(toy_data):
    frames = []
    for model in (Naive(), Climatology()):
        scored = backtest.score(backtest.run_season(model, toy_data, SEASON))
        frames.append(scored.assign(model=model.name))
    summary = backtest.summarize(pd.concat(frames), by=["horizon"])
    rel = backtest.relative_wis(summary, reference="naive")
    assert (rel.loc["naive"] == 1.0).all()
    expected = summary.loc[("climatology", 8), "wis"] / summary.loc[("naive", 8), "wis"]
    assert rel.loc[("climatology", 8)] == pytest.approx(expected)
