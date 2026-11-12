from pathlib import Path

import numpy as np
import pytest

from dengue_forecast import epiweek
from dengue_forecast.data import PanelData

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def infodengue_csv() -> str:
    return (FIXTURES / "infodengue_3548906_2024w01-08.csv").read_text(encoding="utf-8")


@pytest.fixture
def ibge_json() -> str:
    return (FIXTURES / "ibge_municipios_sample.json").read_text(encoding="utf-8")


@pytest.fixture
def toy_data() -> PanelData:
    """6 municípios, SE 01/2019 a SE 52/2023, sazonalidade anual + ruído de Poisson."""
    rng = np.random.default_rng(42)
    weeks = np.array(
        [y * 100 + w for y in range(2019, 2024) for w in range(1, epiweek.weeks_in_year(y) + 1)]
    )
    level = np.array([0.3, 2.0, 8.0, 30.0, 120.0, 600.0])[:, None]
    season = 1.0 + 0.9 * np.sin(2 * np.pi * (weeks % 100) / 52.0)[None, :]
    pop = np.array([3_000, 12_000, 60_000, 250_000, 900_000, 2_000_000], dtype=float)
    return PanelData(
        geocodes=np.arange(1_000_001, 1_000_007),
        epiweeks=weeks,
        cases=rng.poisson(level * season).astype(float),
        population=np.repeat(pop[:, None], len(weeks), axis=1),
        band=np.array([0, 1, 2, 3, 4, 4], dtype=np.int8),
        region=np.array(["Norte", "Norte", "Sul", "Sul", "Sudeste", "Sudeste"]),
    )
