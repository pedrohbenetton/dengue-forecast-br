import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest
from matplotlib.figure import Figure

from dengue_forecast import eda, panel


@pytest.fixture
def toy() -> pd.DataFrame:
    """Dois municípios (pequeno no Norte, grande no Sudeste), 2022-2024, 52 semanas/ano."""
    rows = []
    for geocode, region, pop, level in [(1, "Norte", 4_000, 0), (2, "Sudeste", 200_000, 10)]:
        for year in (2022, 2023, 2024):
            for week in range(1, 53):
                cases = level * (2 if week == 10 else 1) * (3 if year == 2024 else 1)
                rows.append(
                    {
                        "geocode": geocode,
                        "epiweek": year * 100 + week,
                        "week_start": pd.Timestamp(f"{year}-01-01") + pd.Timedelta(weeks=week - 1),
                        "year": year,
                        "week": week,
                        "region": region,
                        "population": pop,
                        "cases": cases,
                        "cases_est": float(cases),
                    }
                )
    df = pd.DataFrame(rows)
    df["pop_band"] = panel.pop_band(df["population"])
    df["season"] = panel.season_of(df["epiweek"])
    last = df["epiweek"] == 202452
    df.loc[last, "cases_est"] = df.loc[last, "cases"] * 2.0  # nowcast acima do notificado
    return df


def test_zero_share_by_band(toy):
    out = eda.zero_share_by_band(toy)
    assert out.loc["<5 mil", "zero_share"] == 1.0
    assert out.loc["100–500 mil", "zero_share"] == 0.0
    assert out.loc["<5 mil", "n_municipalities"] == 1
    assert list(out.index) == ["<5 mil", "100–500 mil"]  # só faixas com municípios, em ordem


def test_seasonal_profile(toy):
    out = eda.seasonal_profile(toy)
    se = out[out["region"] == "Sudeste"].set_index("week")["incidence_100k"]
    assert len(se) == 52
    # semana 10: casos 20, 20, 60 -> mediana 20 por 200 mil hab. (2024 não domina)
    assert se.loc[10] == pytest.approx(20 / 200_000 * 1e5)
    assert se.idxmax() == 10
    assert (out[out["region"] == "Norte"]["incidence_100k"] == 0).all()


def test_season_position():
    weeks = pd.Series([41, 52, 1, 40])
    assert eda.season_position(weeks).tolist() == [0, 11, 12, 51]


def test_national_weekly_and_annual(toy):
    weekly = eda.national_weekly(toy)
    assert len(weekly) == 156
    assert weekly["week_start"].is_monotonic_increasing
    annual = eda.annual_totals(toy)
    assert annual.loc[2022] == 10 * 53
    assert annual.loc[2024] == 3 * 10 * 53
    assert annual.idxmax() == 2024


def test_reporting_delay(toy):
    out = eda.reporting_delay(toy, n_weeks=4)
    assert out["epiweek"].tolist() == [202449, 202450, 202451, 202452]
    assert out["completeness"].tolist() == pytest.approx([1.0, 1.0, 1.0, 0.5])


@pytest.mark.parametrize(
    "plot", [eda.plot_zeros, eda.plot_seasonality, eda.plot_record_year, eda.plot_reporting_delay]
)
def test_plots_return_figures(toy, plot):
    assert isinstance(plot(toy), Figure)


def test_make_all_writes_pngs(toy, tmp_path):
    paths = eda.make_all(toy, tmp_path)
    assert [p.name for p in paths] == [
        "eda_zeros_por_porte.png",
        "eda_sazonalidade.png",
        "eda_ano_recorde.png",
        "eda_atraso_notificacao.png",
    ]
    assert all(p.stat().st_size > 5_000 for p in paths)
