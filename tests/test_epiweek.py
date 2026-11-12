import numpy as np
import pandas as pd
import pytest

from dengue_forecast import epiweek, infodengue


def test_start_matches_infodengue(infodengue_csv):
    df = infodengue.parse_csv(infodengue_csv, 3548906)
    starts = [epiweek.start(int(e)) for e in df["epiweek"]]
    assert starts == [d.date() for d in df["week_start"]]


def test_week_53():
    assert epiweek.start(202053).isoformat() == "2020-12-27"
    assert epiweek.weeks_in_year(2020) == 53
    assert epiweek.weeks_in_year(2014) == 53
    assert epiweek.weeks_in_year(2024) == 52
    with pytest.raises(ValueError):
        epiweek.to_index(np.array([202453]))


def test_index_is_contiguous_across_years():
    idx = epiweek.to_index(np.array([201952, 202001, 202052, 202053, 202101]))
    assert np.diff(idx).tolist() == [1, 51, 1, 1]


def test_roundtrip():
    weeks = np.array(
        [y * 100 + w for y in range(2010, 2027) for w in range(1, epiweek.weeks_in_year(y) + 1)]
    )
    idx = epiweek.to_index(weeks)
    assert np.diff(idx).tolist() == [1] * (len(weeks) - 1)
    assert epiweek.from_index(idx).tolist() == weeks.tolist()


def test_add():
    assert epiweek.add(np.array([202452, 202050, 202401]), 4).tolist() == [202504, 202101, 202405]
    assert epiweek.add(np.array([202501]), -1).tolist() == [202452]


def test_accepts_series():
    out = epiweek.to_index(pd.Series([202401, 202402]))
    assert out.tolist() == [out[0], out[0] + 1]
