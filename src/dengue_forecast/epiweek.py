"""Aritmética de semanas epidemiológicas (padrão SINAN/MMWR: domingo a sábado).

A SE 01 é a semana que contém 4 de janeiro. `to_index` dá um inteiro contíguo, de modo que
"t + h semanas" funciona atravessando anos de 52 ou 53 semanas.
"""

from datetime import date, timedelta

import numpy as np

_EPOCH = date(1989, 12, 31)  # um domingo
_FIRST_YEAR, _LAST_YEAR = 1990, 2100


def _week1_start(year: int) -> date:
    jan4 = date(year, 1, 4)
    return jan4 - timedelta(days=(jan4.weekday() + 1) % 7)  # domingo em ou antes de 4/jan


def weeks_in_year(year: int) -> int:
    return (_week1_start(year + 1) - _week1_start(year)).days // 7


def start(epiweek: int) -> date:
    """Domingo em que começa a semana epidemiológica YYYYWW."""
    year, week = divmod(int(epiweek), 100)
    if not 1 <= week <= weeks_in_year(year):
        raise ValueError(f"semana epidemiológica inválida: {epiweek}")
    return _week1_start(year) + timedelta(weeks=week - 1)


def _build_table() -> tuple[np.ndarray, int]:
    weeks = [
        y * 100 + w
        for y in range(_FIRST_YEAR, _LAST_YEAR + 1)
        for w in range(1, weeks_in_year(y) + 1)
    ]
    return np.array(weeks, dtype=np.int64), (start(weeks[0]) - _EPOCH).days // 7


_TABLE, _OFFSET = _build_table()  # _TABLE é crescente e contíguo em semanas


def to_index(epiweeks) -> np.ndarray:
    """Índice ordinal (semanas desde a época) de cada YYYYWW."""
    values = np.asarray(epiweeks, dtype=np.int64)
    pos = np.searchsorted(_TABLE, values)
    if (pos >= len(_TABLE)).any() or (_TABLE[np.minimum(pos, len(_TABLE) - 1)] != values).any():
        raise ValueError("semana epidemiológica inválida ou fora do intervalo suportado")
    return pos + _OFFSET


def from_index(index) -> np.ndarray:
    """Inverso de `to_index`."""
    return _TABLE[np.asarray(index, dtype=np.int64) - _OFFSET]


def add(epiweeks, n_weeks: int) -> np.ndarray:
    """Soma `n_weeks` semanas a cada YYYYWW."""
    return from_index(to_index(epiweeks) + n_weeks)
