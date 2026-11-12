"""Figuras da fase 2: skill por UF, fan charts, ablação e importância de features.

Uso:
    uv run python -m dengue_forecast.viz.phase2
"""

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from dengue_forecast import backtest, epiweek, ibge
from dengue_forecast.viz import style
from dengue_forecast.viz.evaluation import MODEL_COLORS, MODEL_LABELS, SOURCE

REGION_ORDER = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]
DIVERGING = LinearSegmentedColormap.from_list(
    "skill", [style.CATEGORICAL[0], "#f0efec", "#e34948"]
)  # azul = melhor que a referência; cinza = igual; vermelho = pior
VARIANT_LABELS = {
    "lgbm_own": "Série própria",
    "lgbm_season": "+ sazonalidade",
    "lgbm_nb": "+ vizinhos e UF",
    "lgbm": "+ clima (completo)",
}
FEATURE_GROUP = {"own": "Série própria", "season": "Sazonalidade", "static": "Estáticas",
                 "neighbors": "Vizinhos e UF", "climate": "Clima"}  # fmt: skip


_MONTHS = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]


def _month_tick(value: float, _pos=None) -> str:
    date = mdates.num2date(value)
    name = _MONTHS[date.month - 1]
    return f"{name}\n{date.year}" if date.month <= 2 else name


def _fmt(v: float, digits: int = 2) -> str:
    return f"{v:.{digits}f}".replace(".", ",")


# ------------------------------------------------------------------------- skill por UF


def uf_skill(forecast_dir: Path | str, model: str, reference: str, seasons) -> pd.DataFrame:
    """WIS relativo (modelo / referência) por UF e horizonte, a partir das previsões salvas."""
    totals = []
    for season in seasons:
        for name in (model, reference):
            forecasts = pd.read_parquet(Path(forecast_dir) / f"{name}_{season}.parquet")
            scored = backtest.score(forecasts)
            scored["uf_code"] = scored["geocode"] // 100_000
            part = scored.groupby(["uf_code", "horizon"])["wis"].agg(["sum", "size"])
            totals.append(part.assign(model=name).reset_index())
    table = pd.concat(totals).groupby(["model", "uf_code", "horizon"])[["sum", "size"]].sum()
    mean = (table["sum"] / table["size"]).unstack("model")
    return (mean[model] / mean[reference]).unstack("horizon")


def plot_uf_heatmap(skill: pd.DataFrame, municipalities: pd.DataFrame, title: str) -> Figure:
    style.apply()
    uf = municipalities.assign(uf_code=municipalities["geocode"] // 100_000).drop_duplicates(
        "uf_code"
    )
    uf["order"] = uf["region"].map(REGION_ORDER.index)
    uf = uf.sort_values(["order", "uf"]).set_index("uf_code")
    uf = uf[uf.index.isin(skill.index)]
    values = skill.reindex(uf.index).to_numpy()
    span = max(abs(np.log(np.nanmin(values))), abs(np.log(np.nanmax(values))), 0.05)
    norm = TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span)

    fig, ax = plt.subplots(figsize=(5.2, 0.27 * len(uf) + 1.2))
    ax.imshow(np.log(values), cmap=DIVERGING, norm=norm, aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            strong = abs(np.log(values[i, j])) > 0.6 * span
            ax.text(
                j, i, _fmt(values[i, j]), ha="center", va="center", fontsize=8,
                color="#ffffff" if strong else style.INK,
            )  # fmt: skip
    ax.set_xticks(range(values.shape[1]))
    ax.set_xticklabels([f"h = {h}" for h in skill.columns])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(uf)))
    ax.set_yticklabels(uf["uf"])
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # separadores e rótulos de região
    start = 0
    for region in REGION_ORDER:
        n = int((uf["region"] == region).sum())
        if n == 0:
            continue
        if start:
            ax.axhline(start - 0.5, color=style.SURFACE, lw=3)
        ax.text(
            values.shape[1] - 0.35, start + (n - 1) / 2, region, ha="left", va="center",
            fontsize=8.5, color=style.INK_SECONDARY,
        )  # fmt: skip
        start += n
    ax.set_xticks(np.arange(values.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(uf) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=style.SURFACE, lw=1.5)
    ax.tick_params(which="minor", length=0)
    style.titles(
        fig, title, "WIS relativo por UF e horizonte (abaixo de 1 é melhor que a referência)",
        source=SOURCE, top_pad=0.28, bottom_pad=-0.35,
    )  # fmt: skip
    return fig


# ---------------------------------------------------------------------------- fan chart


def plot_fan(
    forecast_dir: Path | str,
    model: str,
    cities: dict[int, str],
    season: int,
    origins: list[int],
    title: str,
) -> Figure:
    """Previsões emitidas em algumas origens (h = 1, 2, 4, 8) sobre a série observada."""
    style.apply()
    forecasts = pd.read_parquet(Path(forecast_dir) / f"{model}_{season}.parquet")
    forecasts = forecasts[forecasts["geocode"].isin(cities)]
    color = MODEL_COLORS[model]
    fig, axes = plt.subplots(
        len(cities), 1, figsize=(7.6, 2.5 * len(cities)), sharex=True, gridspec_kw={"hspace": 0.35}
    )
    axes = np.atleast_1d(axes)
    for ax, (geocode, name) in zip(axes, cities.items(), strict=True):
        city = forecasts[forecasts["geocode"] == geocode]
        truth = city.drop_duplicates("target").sort_values("target")
        x_truth = [pd.Timestamp(epiweek.start(int(w))) for w in truth["target"]]
        ax.plot(x_truth, truth["y"], color=style.INK, lw=1.3, label="Observado", zorder=4)
        for k, origin in enumerate(origins):
            issued = city[city["origin"] == origin].sort_values("horizon")
            if issued.empty:
                continue
            x = [pd.Timestamp(epiweek.start(int(w))) for w in issued["target"]]
            ax.fill_between(
                x, issued["q0.05"], issued["q0.95"], color=color, alpha=0.18, lw=0,
                label="Intervalo de 90%" if k == 0 else None,
            )  # fmt: skip
            ax.fill_between(
                x, issued["q0.25"], issued["q0.75"], color=color, alpha=0.35, lw=0,
                label="Intervalo de 50%" if k == 0 else None,
            )  # fmt: skip
            ax.plot(
                x, issued["q0.5"], color=color, lw=1.6, marker="o", ms=3,
                label="Mediana prevista" if k == 0 else None, zorder=5,
            )  # fmt: skip
            issued_at = pd.Timestamp(epiweek.start(origin))
            ax.axvline(issued_at, color=style.AXIS, lw=0.8, ls=(0, (3, 3)), zorder=1)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:,.0f}".replace(",", ".")))
        ax.set_title(name, color=style.INK_SECONDARY)
        ax.margins(x=0.01)
    axes[0].legend(loc="upper left", ncols=1, fontsize=8)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(FuncFormatter(_month_tick))
    axes[-1].set_xlabel("Linhas tracejadas: semana em que a previsão foi emitida")
    style.titles(
        fig, title, f"Casos semanais na temporada {season - 1}-{season % 100:02d} e previsões "
        f"de {MODEL_LABELS[model]} para 1, 2, 4 e 8 semanas à frente", source=SOURCE,
        bottom_pad=0.25,
    )  # fmt: skip
    return fig


# ------------------------------------------------------------------------------- ablação


def plot_ablation(scored: pd.DataFrame, reference: str) -> Figure:
    """WIS relativo de cada variante (grupos de features cumulativos), em h = 1 e h = 8."""
    style.apply()
    table = backtest.summarize(scored, ["horizon"])
    rel = backtest.relative_wis(table, reference).unstack("horizon")
    variants = [v for v in VARIANT_LABELS if v in rel.index]
    horizons = list(rel.columns)
    fig, axes = plt.subplots(1, len(horizons), figsize=(7.6, 2.9), sharey=True)
    for ax, h in zip(np.atleast_1d(axes), horizons, strict=True):
        values = rel.loc[variants, h].to_numpy()
        y = np.arange(len(variants))[::-1]
        ax.barh(y, values, height=0.6, color=MODEL_COLORS["lgbm"])
        ax.axvline(1.0, color=style.INK_MUTED, lw=1.0)
        for yi, v in zip(y, values, strict=True):
            ax.text(
                v - 0.012, yi, _fmt(v, 3), ha="right", va="center", fontsize=8.5, color="#ffffff"
            )
        ax.set_yticks(y)
        ax.set_yticklabels([VARIANT_LABELS[v] for v in variants])
        ax.set_xlim(min(0.6, values.min() - 0.05), 1.05)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: _fmt(v)))
        ax.set_title(f"h = {h}", color=style.INK_SECONDARY)
    np.atleast_1d(axes)[0].set_xlabel(f"WIS relativo ao {MODEL_LABELS[reference].lower()}")
    style.titles(
        fig,
        "A sazonalidade traz o maior ganho; vizinhos e clima, quase nada",
        "Ablação cumulativa do LightGBM (cada linha soma um grupo à anterior)",
        source=SOURCE,
    )
    return fig


# ---------------------------------------------------------------------------- importância


def feature_group(name: str) -> str:
    if name.startswith(("nb", "uf")) and name != "uf":
        return "neighbors"
    if name.startswith(("temp_", "humid_")):
        return "climate"
    if name in {"tw_sin", "tw_cos", "seas_ref", "seas_growth", "clim", "crosses_year"}:
        return "season"
    if name in {"logpop", "band", "uf"}:
        return "static"
    return "own"


def plot_importance(importance: pd.Series, top: int = 15) -> Figure:
    style.apply()
    by_group = importance.groupby(importance.index.map(feature_group)).sum()
    by_group = by_group.sort_values(ascending=False)
    head = importance.sort_values(ascending=False).head(top)
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(9.2, 4.2), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.55}
    )
    for ax, series, labels in (
        (ax1, by_group, [FEATURE_GROUP[g] for g in by_group.index]),
        (ax2, head, list(head.index)),
    ):
        y = np.arange(len(series))[::-1]
        ax.barh(y, series.to_numpy(), height=0.62, color=MODEL_COLORS["lgbm"])
        for yi, v in zip(y, series.to_numpy(), strict=True):
            ax.text(v, yi, f" {v:.1%}".replace(".", ","), ha="left", va="center", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlim(0, float(series.max()) * 1.22)
        ax.set_xticks([])
        ax.grid(False)
        ax.tick_params(axis="y", length=0)
        ax.spines["bottom"].set_visible(False)
    ax1.set_title("Por grupo", color=style.INK_SECONDARY)
    ax2.set_title(f"As {top} features mais usadas", color=style.INK_SECONDARY)
    style.titles(
        fig,
        "De onde o modelo tira informação",
        "Participação no ganho total das árvores (36 modelos da última temporada de teste)",
        source=SOURCE,
    )
    return fig


# ------------------------------------------------------------------------------------ CLI

FAN_CITIES = {3550308: "São Paulo (SP)", 3548906: "São Carlos (SP)", 5300108: "Brasília (DF)"}
FAN_ORIGINS = [202350, 202405, 202412, 202420]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--forecast-dir", default="data/forecasts")
    parser.add_argument("--municipalities", default="data/raw/ibge/municipios.json")
    parser.add_argument("--out-dir", default="reports/figures")
    parser.add_argument("--model", default="lgbm")
    parser.add_argument("--reference", default="naive_level")
    args = parser.parse_args(argv)
    fdir, out = Path(args.forecast_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    municipalities = ibge.fetch_municipalities(cache_path=args.municipalities)
    figures: dict[str, Figure] = {}

    skill = uf_skill(fdir, args.model, args.reference, backtest.TEST_SEASONS)
    skill.to_csv(out.parent / "fase2" / "skill_por_uf.csv", float_format="%.4f")
    figures["fase2_skill_por_uf.png"] = plot_uf_heatmap(
        skill, municipalities,
        f"{MODEL_LABELS[args.model]} contra {MODEL_LABELS[args.reference].lower()}, por UF",
    )  # fmt: skip
    figures["fase2_fan_chart.png"] = plot_fan(
        fdir, args.model, FAN_CITIES, 2024, FAN_ORIGINS,
        "O que o modelo dizia durante o surto de 2024",
    )  # fmt: skip
    ablation_dir = fdir / "ablation"
    if (ablation_dir / "scored").exists():
        main_scored = backtest.load_scored(fdir, [args.reference, args.model])
        done = ablation_dir / "scored"
        variants = [v for v in VARIANT_LABELS if (done / f"{v}.parquet").exists()]
        ablation = backtest.load_scored(ablation_dir, variants)
        horizons = sorted(ablation["horizon"].unique())
        both = pd.concat([main_scored[main_scored["horizon"].isin(horizons)], ablation])
        both["model"] = both["model"].astype(str)
        figures["fase2_ablacao.png"] = plot_ablation(both, args.reference)
    importance_path = fdir / f"importance_{args.model}.csv"
    if importance_path.exists():
        importance = pd.read_csv(importance_path, index_col="feature")["gain_share"]
        figures["fase2_importancia.png"] = plot_importance(importance)

    for name, fig in figures.items():
        fig.savefig(out / name)
        plt.close(fig)
        print(out / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
