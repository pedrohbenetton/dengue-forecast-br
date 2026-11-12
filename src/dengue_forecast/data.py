"""Painel em forma de matriz (município × semana), a estrutura que os modelos consomem."""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from dengue_forecast import epiweek
from dengue_forecast.panel import POP_LABELS, season_of


@dataclass(frozen=True)
class PanelData:
    """Séries alinhadas: linha = município, coluna = semana (grade completa e contígua)."""

    geocodes: np.ndarray  # [n_geo]
    epiweeks: np.ndarray  # [n_weeks], YYYYWW crescente
    cases: np.ndarray  # [n_geo, n_weeks], float
    population: np.ndarray  # [n_geo, n_weeks], float
    band: np.ndarray  # [n_geo], código da faixa de população (índice em POP_LABELS)
    region: np.ndarray  # [n_geo], str

    def __post_init__(self) -> None:
        if (np.diff(epiweek.to_index(self.epiweeks)) != 1).any():
            raise ValueError("as semanas do painel não são contíguas")
        if self.cases.shape != (len(self.geocodes), len(self.epiweeks)):
            raise ValueError("matriz de casos com forma incompatível")

    @property
    def n_weeks(self) -> int:
        return len(self.epiweeks)

    def until(self, pos: int) -> "PanelData":
        """Cópia truncada na coluna `pos` (inclusive): o que se sabia naquela semana.

        É assim que o harness garante a regra anti-vazamento: o modelo nunca recebe o futuro.
        """
        end = pos + 1
        return replace(
            self,
            epiweeks=self.epiweeks[:end],
            cases=self.cases[:, :end],
            population=self.population[:, :end],
        )

    def season_positions(self, season: int) -> np.ndarray:
        """Colunas das semanas-alvo da temporada (SE 41 de season-1 até SE 40 de season)."""
        return np.flatnonzero(season_of(pd.Series(self.epiweeks)).to_numpy() == season)

    @classmethod
    def from_panel(cls, panel: pd.DataFrame) -> "PanelData":
        df = panel.sort_values(["geocode", "epiweek"], kind="stable")
        geocodes = df["geocode"].unique()
        weeks = np.sort(df["epiweek"].unique())
        shape = (len(geocodes), len(weeks))
        if len(df) != shape[0] * shape[1] or (
            df["epiweek"].to_numpy().reshape(shape) != weeks[None, :]
        ).any():
            raise ValueError("painel não forma uma grade completa município × semana")
        static = df.drop_duplicates("geocode")
        return cls(
            geocodes=geocodes.astype(np.int64),
            epiweeks=weeks.astype(np.int64),
            cases=df["cases"].to_numpy(dtype=float).reshape(shape),
            population=df["population"].to_numpy(dtype=float).reshape(shape),
            band=pd.Categorical(static["pop_band"].astype(str), categories=POP_LABELS).codes,
            region=static["region"].astype(str).to_numpy(),
        )
