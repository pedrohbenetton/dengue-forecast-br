from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def infodengue_csv() -> str:
    return (FIXTURES / "infodengue_3548906_2024w01-08.csv").read_text(encoding="utf-8")


@pytest.fixture
def ibge_json() -> str:
    return (FIXTURES / "ibge_municipios_sample.json").read_text(encoding="utf-8")
