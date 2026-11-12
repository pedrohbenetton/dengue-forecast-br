import pandas as pd
import pytest

from dengue_forecast import ibge, infodengue, panel


@pytest.fixture
def municipalities(ibge_json):
    return ibge.parse_municipalities(ibge_json)


@pytest.fixture
def raw_dir(tmp_path, infodengue_csv):
    out = tmp_path / "infodengue"
    out.mkdir()
    big = infodengue.parse_csv(infodengue_csv, 3548906)
    big.to_parquet(out / "3548906.parquet", index=False)
    small = infodengue.parse_csv(infodengue_csv, 1100015).iloc[:-2].copy()  # faltam 2 semanas
    small["population"] = 4_000
    small.to_parquet(out / "1100015.parquet", index=False)
    infodengue.parse_csv("", 5300108).to_parquet(out / "5300108.parquet", index=False)  # vazio
    return out


def test_season_of():
    weeks = pd.Series([202340, 202341, 202401, 202440, 202441])
    assert panel.season_of(weeks).tolist() == [2023, 2024, 2024, 2024, 2025]


def test_pop_band():
    pop = pd.Series([4_999, 5_000, 19_999, 100_000, 499_999, 12_000_000])
    assert panel.pop_band(pop).astype(str).tolist() == [
        "<5 mil", "5–20 mil", "5–20 mil", "100–500 mil", "100–500 mil", ">500 mil",
    ]  # fmt: skip


def test_build_panel_shape_and_keys(raw_dir, municipalities):
    df = panel.build_panel(raw_dir, municipalities)
    assert len(df) == 8 + 6
    assert not df.duplicated(["geocode", "epiweek"]).any()
    assert df[["geocode", "epiweek"]].apply(tuple, axis=1).is_monotonic_increasing
    assert list(df.columns[:8]) == [
        "geocode", "epiweek", "week_start", "year", "week", "season", "name", "uf",
    ]  # fmt: skip
    for dropped in ["record_id", "locality_id", "municipality_name"]:
        assert dropped not in df.columns


def test_build_panel_joins_ibge_and_derives_columns(raw_dir, municipalities):
    df = panel.build_panel(raw_dir, municipalities)
    sc = df[df["geocode"] == 3548906]
    assert (sc["name"] == "São Carlos").all()
    assert (sc["uf"] == "SP").all()
    assert (sc["region"] == "Sudeste").all()
    assert (sc["pop_band"] == "100–500 mil").all()
    assert sc["year"].unique().tolist() == [2024]
    assert sc["week"].tolist() == list(range(1, 9))
    assert (sc["season"] == 2024).all()
    assert (df.loc[df["geocode"] == 1100015, "pop_band"] == "<5 mil").all()
    assert isinstance(df["uf"].dtype, pd.CategoricalDtype)
    assert df["pop_band"].cat.ordered


def test_validate_panel_reports_gaps_and_missing(raw_dir, municipalities):
    df = panel.build_panel(raw_dir, municipalities)
    report = panel.validate_panel(df, municipalities)
    assert report["n_rows"] == 14
    assert report["n_municipalities_expected"] == 7
    assert report["n_municipalities_with_data"] == 2
    assert 5300108 in report["municipalities_without_data"]
    assert len(report["municipalities_without_data"]) == 5
    assert report["first_epiweek"] == 202401
    assert report["last_epiweek"] == 202408
    assert report["municipalities_with_gaps"] == {1100015: 2}
    assert report["duplicate_keys"] == 0
    assert report["negative_cases"] == 0


def test_build_panel_rejects_unknown_geocode(raw_dir, municipalities):
    with pytest.raises(ValueError, match="IBGE"):
        panel.build_panel(raw_dir, municipalities[municipalities["geocode"] != 3548906])
