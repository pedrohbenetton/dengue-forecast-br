"""Figuras da avaliação (fase 1 em diante).

Uso:
    uv run python -m dengue_forecast.viz.evaluation
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, PercentFormatter

from dengue_forecast import backtest
from dengue_forecast.viz import style

SOURCE = "Backtest por origem móvel, temporadas 2022-23 a 2025-26. Elaboração própria."
MODEL_LABELS = {
    "naive": "Ingênuo",
    "seasonal": "Sazonal",
    "climatology": "Histórico por SE",
    "naive_level": "Ingênuo por nível",
    "lgbm": "LightGBM",
}
# Cor por modelo, fixa em todas as figuras e no site.
MODEL_COLORS = {
    "naive": style.CATEGORICAL[0],
    "seasonal": style.CATEGORICAL[1],
    "climatology": style.CATEGORICAL[2],
    "naive_level": style.CATEGORICAL[3],
    "lgbm": style.CATEGORICAL[6],
}


def _season_label(season: int) -> str:
    return f"{season - 1}-{season % 100:02d}"


def plot_skill_by_band(scored: pd.DataFrame, reference: str = backtest.REFERENCE) -> Figure:
    """WIS relativo ao modelo de referência, por horizonte, um painel por faixa de população."""
    style.apply()
    table = backtest.summarize(scored, ["band", "horizon"])
    rel = backtest.relative_wis(table, reference).reset_index()
    bands = [b for b in scored["band"].cat.categories if b in set(rel["band"].astype(str))]
    models = [m for m in rel["model"].unique() if m != reference]
    fig, axes = plt.subplots(1, len(bands), figsize=(9.6, 3.3), sharey=True)
    axes = [axes] if len(bands) == 1 else list(axes)
    horizons = sorted(rel["horizon"].unique())
    for ax, band in zip(axes, bands, strict=True):
        ax.axhline(1.0, color=style.INK_MUTED, lw=1.0, zorder=1)
        for model in models:
            d = rel[(rel["band"] == band) & (rel["model"] == model)].sort_values("horizon")
            ax.plot(
                range(len(d)), d["rel_wis"], color=MODEL_COLORS[model], marker="o", ms=4.5,
                label=MODEL_LABELS[model], zorder=3,
            )  # fmt: skip
        ax.set_yscale("log")
        ax.set_xticks(range(len(horizons)))
        ax.set_xticklabels([str(h) for h in horizons])
        ax.set_title(str(band), color=style.INK_SECONDARY)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}×".replace(".", ",")))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _p: ""))
        ax.set_yticks([0.75, 1, 2, 4, 8])
        ax.tick_params(axis="y", which="minor", length=0)
    axes[0].set_ylabel(f"WIS relativo ao {MODEL_LABELS[reference].lower()}")
    axes[len(axes) // 2].set_xlabel("Horizonte (semanas à frente)")
    axes[0].annotate(
        f"{MODEL_LABELS[reference]} = 1×", (0.03, 1.0), xycoords=("axes fraction", "data"),
        xytext=(0, -4), textcoords="offset points", va="top", fontsize=7.5, color=style.INK_MUTED,
    )  # fmt: skip
    axes[0].legend(loc="upper right")
    axes[0].set_ylim(0.65, 11)
    style.titles(
        fig,
        "Persistência é difícil de bater; o histórico só alcança em 8 semanas",
        "WIS relativo ao baseline ingênuo (abaixo de 1× é melhor), por faixa de população",
        source=SOURCE,
    )
    return fig


def plot_calibration(scored: pd.DataFrame, reference: str = backtest.REFERENCE) -> Figure:
    """Cobertura nominal vs. empírica por modelo, e cobertura de 90% do ingênuo por temporada."""
    style.apply()
    nominal = list(backtest.INTERVALS)
    cols = [f"cov{n * 100:g}" for n in nominal]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0), gridspec_kw={"wspace": 0.32})

    overall = scored.groupby("model", observed=True)[cols].mean()
    ax1.plot([0.4, 1.0], [0.4, 1.0], color=style.AXIS, lw=1.0, ls=(0, (3, 3)), zorder=1)
    for model, row in overall.iterrows():
        ax1.plot(
            nominal, row.to_numpy(), color=MODEL_COLORS[model], marker="o", ms=4.5,
            label=MODEL_LABELS[model],
        )  # fmt: skip
    ax1.set_xlim(0.45, 1.0)
    ax1.set_ylim(0.45, 1.0)
    ax1.set_xticks(nominal)
    ax1.set_aspect("equal")
    ax1.grid(axis="x")
    ax1.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax1.set_xlabel("Cobertura nominal do intervalo")
    ax1.set_ylabel("Cobertura empírica")
    ax1.set_title("Todos os horizontes (tracejado: calibração perfeita)", color=style.INK_SECONDARY)
    ax1.legend(loc="lower right")

    ref = scored[scored["model"] == reference]
    by_season = ref.groupby(["season", "horizon"], observed=True)["cov90"].mean().unstack("season")
    worst = by_season.iloc[-1].idxmin()
    x = range(len(by_season))
    ax2.axhline(0.9, color=style.AXIS, lw=1.0, ls=(0, (3, 3)), zorder=1)
    for season in by_season.columns:
        highlight = season == worst
        ax2.plot(
            x, by_season[season], marker="o", ms=4.5, zorder=3 if highlight else 2,
            color=style.ORANGE if highlight else style.NEUTRAL_FILL,
        )  # fmt: skip
        ax2.annotate(
            _season_label(int(season)), (len(by_season) - 1, by_season[season].iloc[-1]),
            xytext=(7, 0), textcoords="offset points", va="center", fontsize=8.5,
            color=style.INK if highlight else style.INK_SECONDARY,
            fontweight="semibold" if highlight else "normal",
        )  # fmt: skip
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([str(h) for h in by_season.index])
    ax2.set_xlim(-0.2, len(by_season) - 0.35)
    ax2.set_ylim(0.65, 1.0)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax2.set_xlabel("Horizonte (semanas à frente)")
    ax2.set_title(
        f"{MODEL_LABELS[reference]}: cobertura do intervalo de 90%, por temporada",
        color=style.INK_SECONDARY,
    )
    style.titles(
        fig,
        "Intervalos razoáveis na média, mas estreitos demais no ano recorde",
        "Cobertura empírica dos intervalos centrais de previsão",
        source=SOURCE,
    )
    return fig


FIGURES = {
    "fase1_skill_por_faixa.png": plot_skill_by_band,
    "fase1_calibracao.png": plot_calibration,
}


def make_all(scored: pd.DataFrame, out_dir: Path | str) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, plot in FIGURES.items():
        fig = plot(scored)
        fig.savefig(out / name)
        plt.close(fig)
        paths.append(out / name)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--forecast-dir", default="data/forecasts")
    parser.add_argument("--models", default="naive,seasonal,climatology")
    parser.add_argument("--out-dir", default="reports/figures")
    args = parser.parse_args(argv)
    scored = backtest.load_scored(args.forecast_dir, args.models.split(","))
    for path in make_all(scored, args.out_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
