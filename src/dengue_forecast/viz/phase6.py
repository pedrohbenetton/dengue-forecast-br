"""Tabelas e figuras da fase 6: pesquisa e ablação da dívida imunológica ao DENV-3.

Uso:
    uv run python -m dengue_forecast.viz.phase6
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
from dengue_forecast.viz.evaluation import MODEL_COLORS
from dengue_forecast.viz.phase2 import feature_group

SOURCE = "Backtest por origem móvel (4 temporadas, 5.570 municípios). Elaboração própria."

PHASE6_VARIANTS = ["lgbm_own", "lgbm_season", "lgbm_nb", "lgbm", "lgbm_immunity"]
PHASE6_LABELS = {
    "lgbm_own": "Série própria + estáticas",
    "lgbm_season": "+ sazonalidade",
    "lgbm_nb": "+ vizinhos e UF",
    "lgbm": "+ clima (fase 2)",
    "lgbm_immunity": "+ dívida imunológica (DENV-3)",
}

PHASE6_COLORS = {
    "lgbm_own": style.INK_MUTED,
    "lgbm_season": style.CATEGORICAL[4],
    "lgbm_nb": style.CATEGORICAL[3],
    "lgbm": style.CATEGORICAL[6],  # roxo
    "lgbm_immunity": style.CATEGORICAL[0],  # azul
}


def _fmt(v: float, digits: int = 3) -> str:
    return f"{v:.{digits}f}".replace(".", ",")


def build_tables(
    forecast_dir: Path | str = "data/forecasts",
    ablation_dir: Path | str | None = None,
    out_dir: Path | str = "reports/fase6",
    reference: str = "naive_level",
) -> dict[str, pd.DataFrame]:
    """Gera todas as tabelas CSV da fase 6 a partir dos arquivos reais pontuados no backtest."""
    fdir = Path(forecast_dir)
    adir = Path(ablation_dir) if ablation_dir is not None else fdir / "ablation"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    main_models = [reference, "lgbm", "lgbm_immunity"]
    scored_main = backtest.load_scored(fdir, main_models)
    tables = backtest.write_tables(scored_main, out, reference=reference)

    # Ablação cumulativa em h=1 e h=8 juntando variantes intermediárias e principais
    ab_variants = [
        v
        for v in ("lgbm_own", "lgbm_season", "lgbm_nb")
        if (adir / "scored" / f"{v}.parquet").exists()
    ]
    parts = [scored_main[scored_main["horizon"].isin((1, 8))]]
    if ab_variants:
        scored_ab = backtest.load_scored(adir, ab_variants)
        parts.append(scored_ab[scored_ab["horizon"].isin((1, 8))])
    combined = pd.concat(parts, ignore_index=True)
    order = [reference, *ab_variants, "lgbm", "lgbm_immunity"]
    combined["model"] = pd.Categorical(combined["model"].astype(str), categories=order)

    ab_summary = backtest.summarize(combined, ["horizon"])
    ab_df = pd.DataFrame(
        {
            "wis": ab_summary["wis"],
            "wis_log": ab_summary["wis_log"],
            "rel": backtest.relative_wis(ab_summary, reference),
            "rel_log": backtest.relative_wis(ab_summary, reference, col="wis_log"),
        }
    ).reset_index()
    ab_df.to_csv(out / "ablacao.csv", index=False, float_format="%.4f")
    tables["ablacao"] = ab_df

    imp_path = fdir / "importance_lgbm_immunity.csv"
    if imp_path.exists():
        imp_raw = pd.read_csv(imp_path)
        col = "gain_share" if "gain_share" in imp_raw.columns else "importance"
        imp_df = pd.DataFrame(
            {
                "feature": imp_raw["feature"],
                "group": imp_raw["feature"].map(feature_group),
                "importance": imp_raw[col],
            }
        )
        imp_df.to_csv(out / "importancia.csv", index=False, float_format="%.6f")
        tables["importancia"] = imp_df

    return tables


def plot_immunity_ablation(
    ablacao: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 8),
    reference: str = "naive_level",
) -> Figure:
    """Gráfico de barras da ablação cumulativa comparando o modelo com e sem dívida imunológica."""
    style.apply()
    rel = ablacao.set_index(["model", "horizon"])["rel"].unstack("horizon")
    available_variants = [v for v in PHASE6_VARIANTS if v in rel.index]

    fig, axes = plt.subplots(
        1,
        len(horizons),
        figsize=(8.8, 3.8),
        sharey=True,
        gridspec_kw={"wspace": 0.12},
    )
    for ax, h in zip(np.atleast_1d(axes), horizons, strict=True):
        if h not in rel.columns:
            continue
        values = rel.loc[available_variants, h].to_numpy()
        y = np.arange(len(available_variants))[::-1]
        colors = [
            PHASE6_COLORS.get(v, MODEL_COLORS.get("lgbm", "#6250d6")) for v in available_variants
        ]
        ax.barh(y, values, height=0.6, color=colors)
        ax.axvline(1.0, color=style.INK_MUTED, lw=1.0, ls="--")
        for yi, v in zip(y, values, strict=True):
            ax.text(
                v - 0.012, yi, _fmt(v, 3), ha="right", va="center", fontsize=8.5, color="#ffffff"
            )
        ax.set_yticks(y)
        ax.set_yticklabels([PHASE6_LABELS.get(v, v) for v in available_variants])
        ax.set_xlim(min(0.65, float(values.min()) - 0.05), 1.05)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, _p: _fmt(val, 2)))
        ax.set_title(f"h = {h} semanas", color=style.INK_SECONDARY)

    np.atleast_1d(axes)[0].set_xlabel(f"WIS relativo ao {reference}")
    style.titles(
        fig,
        "Ablação DENV-3: efeito das features causais de dívida imunológica",
        "WIS relativo cumulativo por grupo de features em h=1 e h=8",
        source=SOURCE,
    )
    return fig


def plot_season_shock_comparison(
    por_temporada: pd.DataFrame,
    models: tuple[str, ...] = ("lgbm", "lgbm_immunity"),
    horizon: int = 8,
) -> Figure:
    """Evolução do WIS relativo temporada a temporada, destacando o choque de 2023-24."""
    style.apply()
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    seasons = sorted(por_temporada["season"].unique())
    x = np.arange(len(seasons))
    width = 0.35

    sub = por_temporada[por_temporada["horizon"] == horizon]
    for i, model in enumerate(models):
        m_data = sub[sub["model"] == model].set_index("season").reindex(seasons)
        y_vals = m_data["rel_wis"].to_numpy()
        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset,
            y_vals,
            width=width * 0.9,
            color=PHASE6_COLORS.get(model, style.CATEGORICAL[i]),
            label=PHASE6_LABELS.get(model, model),
        )
        for b, v in zip(bars, y_vals, strict=True):
            if np.isfinite(v):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    v + 0.015,
                    _fmt(v, 3),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.axhline(1.0, color=style.INK_MUTED, lw=1.0, ls="--")
    ax.set_xticks(x)
    labels = [f"{int(s) - 1}-{int(s) % 100:02d}" for s in seasons]
    labels = [lbl + "\n(choque DENV-3)" if "2023-24" in lbl else lbl for lbl in labels]
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"WIS relativo (h={horizon})")
    ax.set_ylim(0.55, 1.08)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _p: _fmt(val, 2)))
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.84))
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    style.titles(
        fig,
        "Comparação por temporada: LightGBM com e sem proxies de dívida imunológica",
        f"WIS relativo ao ingênuo por nível por temporada (horizonte {horizon} semanas)",
        source=SOURCE,
    )
    return fig


def plot_immunity_importance(importance: pd.Series, top: int = 12) -> Figure:
    """Importância relativa de features com foco no grupo de dívida imunológica."""
    style.apply()
    head = importance.sort_values(ascending=False).head(top)
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    y = np.arange(len(head))[::-1]

    immunity_set = {"immune_debt", "quiet_ratio", "reemergence_boost", "susceptible_gap"}
    colors = [
        style.CATEGORICAL[0] if feat in immunity_set else style.CATEGORICAL[6]
        for feat in head.index
    ]

    ax.barh(y, head.to_numpy(), height=0.62, color=colors)
    for yi, (feat, v) in zip(y, head.items(), strict=True):
        tag = " [imunidade]" if feat in immunity_set else ""
        ax.text(
            v + 0.002,
            yi,
            f" {v:.1%}{tag}".replace(".", ","),
            ha="left",
            va="center",
            fontsize=8,
            color=style.CATEGORICAL[0] if tag else style.INK,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(list(head.index))
    ax.set_xlim(0, float(head.max()) * 1.38)
    ax.set_xticks([])
    ax.grid(False)
    ax.tick_params(axis="y", length=0)
    ax.spines["bottom"].set_visible(False)

    style.titles(
        fig,
        "Ganho das features de dívida imunológica nas árvores",
        "Participação no ganho total (azul = proxies de dívida imunológica DENV-3)",
        source=SOURCE,
    )
    return fig


def make_all(
    ablacao: pd.DataFrame,
    por_temporada: pd.DataFrame,
    importance: pd.Series,
    out_dir: Path | str = "reports/figures",
) -> list[Path]:
    """Gera todas as figuras da fase 6 e salva em out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig_ab = plot_immunity_ablation(ablacao)
    p_ab = out / "fase6_ablacao.png"
    fig_ab.savefig(p_ab)
    plt.close(fig_ab)
    paths.append(p_ab)

    fig_sz = plot_season_shock_comparison(por_temporada)
    p_sz = out / "fase6_dinamica_choque_2024.png"
    fig_sz.savefig(p_sz)
    plt.close(fig_sz)
    paths.append(p_sz)

    fig_imp = plot_immunity_importance(importance)
    p_imp = out / "fase6_importancia_imunidade.png"
    fig_imp.savefig(p_imp)
    plt.close(fig_imp)
    paths.append(p_imp)

    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--forecast-dir", default="data/forecasts")
    parser.add_argument("--fase6-dir", default="reports/fase6")
    parser.add_argument("--out-dir", default="reports/figures")
    parser.add_argument("--reference", default="naive_level")
    args = parser.parse_args(argv)

    fdir = Path(args.forecast_dir)
    fase6_path = Path(args.fase6_dir)
    if (fdir / "scored" / "lgbm_immunity.parquet").exists():
        build_tables(fdir, out_dir=fase6_path, reference=args.reference)

    p_abl = fase6_path / "ablacao.csv"
    p_sz = fase6_path / "por_temporada.csv"
    p_imp = fase6_path / "importancia.csv"
    if not (p_abl.exists() and p_sz.exists() and p_imp.exists()):
        raise FileNotFoundError(
            f"tabelas da fase 6 ausentes em {fase6_path}; rode o backtest de lgbm_immunity antes"
        )

    ablacao = pd.read_csv(p_abl)
    por_temporada = pd.read_csv(p_sz)
    imp_df = pd.read_csv(p_imp)
    importance = imp_df.set_index("feature")["importance"]

    paths = make_all(ablacao, por_temporada, importance, out_dir=args.out_dir)
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
