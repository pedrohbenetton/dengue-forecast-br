"""EDA da fase 0: agregações testáveis + figuras estáticas.

Uso:
    uv run python -m dengue_forecast.eda
"""

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, PercentFormatter

from dengue_forecast.viz import style

SOURCE = "Fonte: InfoDengue (api/alertcity) e IBGE. Elaboração própria."
_MONTHS = ["out", "nov", "dez", "jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set"]


# --------------------------------------------------------------------------- agregações


def zero_share_by_band(panel: pd.DataFrame) -> pd.DataFrame:
    """Fração de município-semanas com zero casos, por faixa de população."""
    grouped = panel.assign(zero=panel["cases"] == 0).groupby("pop_band", observed=True)
    out = pd.DataFrame(
        {
            "zero_share": grouped["zero"].mean().astype("float64"),
            "n_municipalities": grouped["geocode"].nunique(),
        }
    )
    return out[out["n_municipalities"] > 0]


def seasonal_profile(panel: pd.DataFrame) -> pd.DataFrame:
    """Incidência semanal típica (por 100 mil hab.) por região e SE: mediana entre os anos.

    Mediana, e não média, para que 2024 sozinho não defina o perfil.
    """
    by_week = (
        panel[panel["week"] <= 52]
        .groupby(["region", "year", "week"], observed=True)[["cases", "population"]]
        .sum()
        .astype("float64")
    )
    by_week["incidence_100k"] = by_week["cases"] / by_week["population"] * 1e5
    typical = by_week.groupby(["region", "week"], observed=True)["incidence_100k"].median()
    return typical.reset_index()


def season_position(week: pd.Series) -> pd.Series:
    """Posição da SE dentro da temporada (SE 41 -> 0, ..., SE 40 -> 51)."""
    return (week - 41) % 52


def national_weekly(panel: pd.DataFrame) -> pd.DataFrame:
    """Casos notificados no país por semana."""
    out = panel.groupby("epiweek").agg(week_start=("week_start", "min"), cases=("cases", "sum"))
    return out.reset_index().sort_values("epiweek", ignore_index=True)


def annual_totals(panel: pd.DataFrame) -> pd.Series:
    """Casos notificados no país por ano epidemiológico."""
    return panel.groupby("year")["cases"].sum().astype("int64")


def reporting_delay(panel: pd.DataFrame, n_weeks: int = 26) -> pd.DataFrame:
    """Notificado vs. nowcast (`cases_est`) nas últimas `n_weeks` semanas, soma nacional.

    `completeness` = notificado / estimado: quanto do total esperado já entrou no sistema.
    """
    weekly = panel.groupby("epiweek").agg(
        week_start=("week_start", "min"), cases=("cases", "sum"), cases_est=("cases_est", "sum")
    )
    out = weekly.sort_index().tail(n_weeks).reset_index()
    out["completeness"] = out["cases"].astype("float64") / out["cases_est"]
    return out


# ------------------------------------------------------------------------------ figuras


def _thousands(x: float, _pos=None) -> str:
    if x >= 1e6:
        return f"{x / 1e6:.1f} mi".replace(".", ",")
    if x >= 1e3:
        return f"{x / 1e3:.0f} mil"
    return f"{x:.0f}"


def _month_label(x: float, pos=None) -> str:
    date = mdates.num2date(x)
    name = _MONTHS[(date.month - 10) % 12]
    return f"{name}\n{date.year}" if date.month == 1 or pos == 0 else name


def plot_zeros(panel: pd.DataFrame) -> Figure:
    style.apply()
    data = zero_share_by_band(panel)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    y = np.arange(len(data))[::-1]
    ax.barh(y, data["zero_share"], height=0.62, color=style.BLUE)
    for yi, (band, row) in zip(y, data.iterrows(), strict=True):
        ax.text(
            row["zero_share"] + 0.012, yi, f"{row['zero_share']:.0%}",
            va="center", ha="left", fontsize=9.5, fontweight="semibold", color=style.INK,
        )  # fmt: skip
        n = f"{int(row['n_municipalities']):,}".replace(",", ".")
        ax.text(
            -0.02, yi, f"{band}\n", va="center", ha="right", fontsize=9.5, color=style.INK,
            linespacing=0.9,
        )  # fmt: skip
        ax.text(
            -0.02, yi, f"\n{n} municípios", va="center", ha="right", fontsize=7.5,
            color=style.INK_MUTED, linespacing=1.5,
        )  # fmt: skip
    ax.set_yticks([])
    ax.set_xlim(0, 1.08)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("Semanas sem nenhum caso notificado")
    first, last = int(panel["year"].min()), int(panel["year"].max())
    style.titles(
        fig,
        "Município pequeno é, quase sempre, semana com zero casos",
        f"Fração das semanas ({first}–{last}) com zero casos, por faixa de população",
        source=SOURCE,
    )
    return fig


def plot_seasonality(panel: pd.DataFrame) -> Figure:
    style.apply()
    data = seasonal_profile(panel)
    data["pos"] = season_position(data["week"])
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    regions = [r for r in style.REGION_COLORS if r in set(data["region"].astype(str))]
    for region in regions:
        d = data[data["region"] == region].sort_values("pos")
        ax.plot(d["pos"], d["incidence_100k"], color=style.REGION_COLORS[region], label=region)
    turn = 11.5  # entre a SE 52 e a SE 1
    ax.axvline(turn, color=style.AXIS, lw=0.8, ls=(0, (3, 3)), zorder=0)
    ax.annotate(
        "virada do ano:\ndegrau na fonte", (turn, 1.0), xycoords=("data", "axes fraction"),
        xytext=(-5, -2), textcoords="offset points", ha="right", va="top",
        fontsize=7.5, color=style.INK_MUTED,
    )  # fmt: skip
    ax.set_xlim(0, 51)
    ax.set_ylim(bottom=0)
    ax.set_xticks(np.arange(12) * 52 / 12)
    ax.set_xticklabels(_MONTHS)
    ax.set_xlabel("Temporada epidemiológica (SE 41 a SE 40)")
    ax.set_ylabel("Casos por 100 mil hab. por semana")
    ax.legend(loc="upper right", ncols=1, handlelength=1.4)
    first, last = int(panel["year"].min()), int(panel["year"].max())
    style.titles(
        fig,
        "A temporada começa na primavera e atinge o pico entre fevereiro e maio",
        f"Incidência semanal por região, mediana dos anos {first}–{last}",
        source=SOURCE,
    )
    return fig


def plot_record_year(panel: pd.DataFrame) -> Figure:
    style.apply()
    weekly = national_weekly(panel)
    annual = annual_totals(panel)
    record = int(annual.idxmax())
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.6, 5.6), height_ratios=[1.5, 1], gridspec_kw={"hspace": 0.42}
    )

    ax1.fill_between(weekly["week_start"], weekly["cases"], color=style.BLUE, alpha=0.12, lw=0)
    ax1.plot(weekly["week_start"], weekly["cases"], color=style.BLUE, lw=1.4)
    peak = weekly.loc[weekly["cases"].idxmax()]
    ax1.annotate(
        f"Pico: {_thousands(float(peak['cases']))} casos\nSE {int(peak['epiweek']) % 100:02d} "
        f"de {int(peak['epiweek']) // 100}",
        (peak["week_start"], peak["cases"]), xytext=(-12, -4), textcoords="offset points",
        ha="right", va="top", fontsize=8.5, color=style.INK,
    )  # fmt: skip
    ax1.set_ylim(bottom=0)
    ax1.yaxis.set_major_formatter(FuncFormatter(_thousands))
    ax1.set_title("Casos notificados por semana", color=style.INK_SECONDARY)
    ax1.margins(x=0.01)

    colors = [style.ORANGE if y == record else style.NEUTRAL_FILL for y in annual.index]
    ax2.bar(annual.index, annual.values, width=0.7, color=colors)
    ax2.text(
        record, annual[record], _thousands(float(annual[record])), ha="center", va="bottom",
        fontsize=9, fontweight="semibold", color=style.INK,
    )  # fmt: skip
    others = annual.drop(record)
    ax2.set_xticks(annual.index)
    ax2.set_xticklabels([f"{y % 100:02d}" for y in annual.index])
    ax2.yaxis.set_major_formatter(FuncFormatter(_thousands))
    ax2.set_ylim(0, float(annual.max()) * 1.15)
    ax2.set_title("Total por ano", color=style.INK_SECONDARY)
    ax2.set_xlabel(f"Ano (o último, {int(annual.index.max())}, está incompleto)")

    ratio = ""
    if len(others) and others.max() > 0:
        ratio = f" — {annual[record] / others.max():.1f}× o segundo maior ano".replace(".", ",")
    style.titles(
        fig,
        f"{record} foi o maior ano de dengue da série",
        f"Brasil, soma dos municípios{ratio}",
        source=SOURCE,
    )
    return fig


def plot_reporting_delay(panel: pd.DataFrame, n_weeks: int = 26) -> Figure:
    style.apply()
    data = reporting_delay(panel, n_weeks)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.6, 5.4), height_ratios=[1.5, 1], sharex=True, gridspec_kw={"hspace": 0.3}
    )
    x = data["week_start"]
    ax1.plot(x, data["cases_est"], color=style.ORANGE, label="Estimado (nowcast)")
    ax1.plot(x, data["cases"], color=style.BLUE, label="Notificado até agora")
    ax1.fill_between(
        x, data["cases"].astype("float64"), data["cases_est"], color=style.ORANGE, alpha=0.12, lw=0
    )
    ax1.set_ylim(bottom=0)
    ax1.yaxis.set_major_formatter(FuncFormatter(_thousands))
    ax1.set_title("Casos por semana, soma nacional", color=style.INK_SECONDARY)
    ax1.legend(loc="lower left")

    ax2.plot(x, data["completeness"], color=style.BLUE, marker="o", ms=3.5)
    ax2.axhline(1.0, color=style.AXIS, lw=0.8, zorder=0)
    last = data.iloc[-1]
    ax2.annotate(
        f"{last['completeness']:.0%}", (last["week_start"], last["completeness"]),
        xytext=(6, 0), textcoords="offset points", ha="left", va="center",
        fontsize=9, fontweight="semibold", color=style.INK,
    )  # fmt: skip
    ax2.set_ylim(0, 1.1)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.set_title("Completude: notificado ÷ estimado", color=style.INK_SECONDARY)
    ax2.margins(x=0.06)
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(FuncFormatter(_month_label))

    style.titles(
        fig,
        "As últimas semanas estão sempre incompletas",
        f"Notificado vs. nowcast do InfoDengue nas últimas {len(data)} semanas",
        source=SOURCE,
    )
    return fig


FIGURES = {
    "eda_zeros_por_porte.png": plot_zeros,
    "eda_sazonalidade.png": plot_seasonality,
    "eda_ano_recorde.png": plot_record_year,
    "eda_atraso_notificacao.png": plot_reporting_delay,
}


def make_all(panel: pd.DataFrame, out_dir: Path | str) -> list[Path]:
    """Gera todas as figuras da EDA em `out_dir` e devolve os caminhos."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, plot in FIGURES.items():
        fig = plot(panel)
        fig.savefig(out / name)
        plt.close(fig)
        paths.append(out / name)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--panel", default="data/processed/panel.parquet")
    parser.add_argument("--out-dir", default="reports/figures")
    args = parser.parse_args(argv)
    columns = [
        "geocode", "epiweek", "week_start", "year", "week", "region", "pop_band",
        "population", "cases", "cases_est",
    ]  # fmt: skip
    panel = pd.read_parquet(args.panel, columns=columns)
    for path in make_all(panel, args.out_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
