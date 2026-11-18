"""Figuras da fase 4: rede em grafo espaço-temporal (ST-GNN) vs controle sem grafo e LightGBM.

Uso:
    uv run python -m dengue_forecast.viz.phase4
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from dengue_forecast import backtest
from dengue_forecast.viz import style

SOURCE = "Backtest por origem móvel (4 temporadas, 5.570 municípios). Elaboração própria."

PHASE4_LABELS = {
    "naive_level": "Ingênuo por nível",
    "lgbm": "LightGBM",
    "gnn": "Rede em grafo (ST-GNN)",
    "gnn_nograph": "Mesma rede, sem grafo",
}

PHASE4_COLORS = {
    "naive_level": style.INK_MUTED,
    "lgbm": style.CATEGORICAL[6],  # roxo
    "gnn": style.CATEGORICAL[0],  # azul
    "gnn_nograph": style.CATEGORICAL[2],  # verde-água
}


def _fmt(v: float, digits: int = 2) -> str:
    return f"{v:.{digits}f}".replace(".", ",")


def plot_graph_vs_control(scored: pd.DataFrame, reference: str = "naive_level") -> Figure:
    """WIS relativo por horizonte (esq.) e por temporada em h=8 (dir.)."""
    style.apply()
    models = [m for m in ("lgbm", "gnn", "gnn_nograph") if m in scored["model"].unique()]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8), gridspec_kw={"wspace": 0.28})

    # Painel 1: WIS relativo por horizonte
    by_h = backtest.summarize(scored, ["horizon"])
    rel_h = backtest.relative_wis(by_h, reference).reset_index()
    horizons = sorted(rel_h["horizon"].unique())
    xh = np.arange(len(horizons))
    ax1.axhline(1.0, color=style.AXIS, lw=1.0, ls=(0, (3, 3)), zorder=1)
    for model in models:
        sub = rel_h[rel_h["model"] == model].sort_values("horizon")
        ax1.plot(
            xh,
            sub["rel_wis"],
            color=PHASE4_COLORS[model],
            marker="o",
            ms=4.8,
            lw=2.0,
            label=PHASE4_LABELS[model],
            zorder=3,
        )
    ax1.set_xticks(xh)
    ax1.set_xticklabels([f"h={h}" for h in horizons])
    ax1.set_ylim(0.65, 1.03)
    ax1.set_ylabel(f"WIS relativo ao {PHASE4_LABELS[reference].lower()}")
    ax1.set_xlabel("Horizonte de previsão")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: _fmt(v, 2)))
    ax1.set_title("Agregado nas 4 temporadas", color=style.INK_SECONDARY)
    ax1.legend(loc="lower left", fontsize=8.0)

    # Painel 2: WIS relativo em h=8 por temporada
    max_h = max(horizons)
    by_s = backtest.summarize(scored, ["season", "horizon"])
    rel_s = backtest.relative_wis(by_s, reference).reset_index()
    rel_s = rel_s[rel_s["horizon"] == max_h]
    seasons = sorted(rel_s["season"].unique())
    xs = np.arange(len(seasons))
    width = 0.26
    ax2.axhline(1.0, color=style.AXIS, lw=1.0, ls=(0, (3, 3)), zorder=1)
    for idx, model in enumerate(models):
        sub = rel_s[rel_s["model"] == model].set_index("season").reindex(seasons)
        vals = sub["rel_wis"].to_numpy()
        offset = (idx - (len(models) - 1) / 2) * width
        bars = ax2.bar(
            xs + offset,
            vals,
            width=width * 0.92,
            color=PHASE4_COLORS[model],
            label=PHASE4_LABELS[model],
        )
        for b, v in zip(bars, vals, strict=True):
            if np.isfinite(v):
                ax2.text(
                    b.get_x() + b.get_width() / 2,
                    v + 0.012,
                    _fmt(v, 2),
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"{int(s) - 1}-{int(s) % 100:02d}" for s in seasons])
    ax2.set_ylim(0.50, 1.05)
    ax2.set_xlabel("Temporada de teste")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: _fmt(v, 2)))
    ax2.set_title(f"Por temporada (h = {max_h} semanas)", color=style.INK_SECONDARY)

    style.titles(
        fig,
        "A convolução no grafo não ajuda; o ganho vem da rede multi-horizonte nos anos de surto",
        "WIS relativo ao ingênuo por nível (abaixo de 1,00 é melhor)",
        source=SOURCE,
    )
    return fig


def plot_skill_by_band_phase4(scored: pd.DataFrame, reference: str = "naive_level") -> Figure:
    """WIS relativo em h=8 por faixa populacional, evidenciando o ganho nas grandes cidades."""
    style.apply()
    models = [m for m in ("lgbm", "gnn", "gnn_nograph") if m in scored["model"].unique()]
    by_b = backtest.summarize(scored, ["band", "horizon"])
    rel_b = backtest.relative_wis(by_b, reference).reset_index()
    horizons = sorted(rel_b["horizon"].unique())
    target_h = 8 if 8 in horizons else max(horizons)
    sub_h = rel_b[rel_b["horizon"] == target_h]

    if hasattr(scored["band"], "cat"):
        bands = [b for b in scored["band"].cat.categories if b in set(sub_h["band"].astype(str))]
    else:
        bands = list(sub_h["band"].unique())

    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    x = np.arange(len(bands))
    width = 0.25
    ax.axhline(1.0, color=style.AXIS, lw=1.0, ls=(0, (3, 3)), zorder=1)
    for idx, model in enumerate(models):
        m_df = sub_h[sub_h["model"] == model].set_index("band").reindex(bands)
        vals = m_df["rel_wis"].to_numpy()
        offset = (idx - (len(models) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            vals,
            width=width * 0.92,
            color=PHASE4_COLORS[model],
            label=PHASE4_LABELS[model],
        )
        for b, v in zip(bars, vals, strict=True):
            if np.isfinite(v):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    v + 0.012,
                    _fmt(v, 2),
                    ha="center",
                    va="bottom",
                    fontsize=7.8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in bands])
    ax.set_ylim(0.50, 1.05)
    ax.set_ylabel(f"WIS relativo (h={target_h})")
    ax.set_xlabel("Faixa de população municipal")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: _fmt(v, 2)))
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.94), fontsize=8.5)

    style.titles(
        fig,
        "A vantagem da rede neural cresce com o porte da cidade — sem precisar do grafo",
        f"WIS relativo ao {PHASE4_LABELS[reference].lower()} em h={target_h} semanas, por porte",
        source=SOURCE,
    )
    return fig


FIGURES = {
    "fase4_grafo_vs_controle.png": plot_graph_vs_control,
    "fase4_skill_por_porte.png": plot_skill_by_band_phase4,
}


def make_all(
    scored: pd.DataFrame, out_dir: Path | str = "reports/figures", reference: str = "naive_level"
) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, fn in FIGURES.items():
        fig = fn(scored, reference=reference)
        fig.savefig(out / name)
        plt.close(fig)
        paths.append(out / name)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--forecast-dir", default="data/forecasts")
    parser.add_argument("--models", default="naive_level,lgbm,gnn,gnn_nograph")
    parser.add_argument("--reference", default="naive_level")
    parser.add_argument("--out-dir", default="reports/figures")
    args = parser.parse_args(argv)

    scored = backtest.load_scored(args.forecast_dir, args.models.split(","))
    for p in make_all(scored, args.out_dir, reference=args.reference):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
