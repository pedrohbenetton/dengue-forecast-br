"""Figuras da fase 3: calibração conformal (CQR e ACI).

Uso:
    uv run python -m dengue_forecast.viz.phase3
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from dengue_forecast import backtest
from dengue_forecast.viz import style

SOURCE = "Backtest por origem móvel (4 temporadas, 5.570 municípios). Elaboração própria."

PHASE3_LABELS = {
    "naive_level": "Ingênuo por nível",
    "lgbm": "LightGBM (puro)",
    "lgbm_cqr": "CQR estático",
    "lgbm_cqr_level": "CQR por nível",
    "lgbm_aci": "ACI adaptativo",
    "lgbm_aci_level": "ACI por nível",
}

PHASE3_COLORS = {
    "naive_level": style.INK_MUTED,
    "lgbm": style.CATEGORICAL[6],  # roxo
    "lgbm_cqr": style.CATEGORICAL[0],  # azul
    "lgbm_cqr_level": style.CATEGORICAL[2],  # verde água
    "lgbm_aci": style.CATEGORICAL[1],  # laranja
    "lgbm_aci_level": style.CATEGORICAL[3],  # amarelo ouro
}

NOMINAL_LEVELS = list(backtest.INTERVALS)  # [0.5, 0.8, 0.9, 0.95]


def plot_calibration_comparison(
    scored: pd.DataFrame, models: list[str] | None = None
) -> Figure:
    """Diagrama de calibração comparativo: nominal vs. empírico e cobertura de 90% por horizonte."""
    style.apply()
    if models is None:
        models = [m for m in PHASE3_LABELS if m in scored["model"].unique()]

    cols = [f"cov{n * 100:g}" for n in NOMINAL_LEVELS]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0), gridspec_kw={"wspace": 0.32})

    # Painel 1: Cobertura nominal vs. empírica geral
    ax1.plot([0.45, 1.0], [0.45, 1.0], color=style.AXIS, lw=1.0, ls=(0, (3, 3)), zorder=1)
    for model in models:
        sub = scored[scored["model"] == model]
        if sub.empty:
            continue
        means = [sub[col].mean() for col in cols]
        ax1.plot(
            NOMINAL_LEVELS,
            means,
            color=PHASE3_COLORS.get(model, style.INK),
            marker="o",
            ms=4.5,
            label=PHASE3_LABELS.get(model, model),
            zorder=3,
        )

    ax1.set_xlim(0.45, 1.0)
    ax1.set_ylim(0.45, 1.0)
    ax1.set_xticks(NOMINAL_LEVELS)
    ax1.set_aspect("equal")
    ax1.grid(axis="x")
    ax1.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax1.set_xlabel("Cobertura nominal do intervalo")
    ax1.set_ylabel("Cobertura empírica")
    ax1.set_title("Todos os horizontes (tracejado: calibração perfeita)", color=style.INK_SECONDARY)
    ax1.legend(loc="lower right", fontsize=8.0)

    # Painel 2: Cobertura de 90% por horizonte
    ax2.axhline(0.90, color=style.AXIS, lw=1.0, ls=(0, (3, 3)), zorder=1)
    horizons = sorted(scored["horizon"].unique())
    x = range(len(horizons))
    for model in models:
        sub = scored[scored["model"] == model]
        if sub.empty:
            continue
        by_h = sub.groupby("horizon", observed=True)["cov90"].mean()
        vals = [by_h.get(h, np.nan) for h in horizons]
        ax2.plot(
            x,
            vals,
            color=PHASE3_COLORS.get(model, style.INK),
            marker="o",
            ms=4.5,
            label=PHASE3_LABELS.get(model, model),
            zorder=3,
        )

    ax2.set_xticks(list(x))
    ax2.set_xticklabels([str(h) for h in horizons])
    ax2.set_xlim(-0.2, len(horizons) - 0.35)
    ax2.set_ylim(0.70, 1.0)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax2.set_xlabel("Horizonte (semanas à frente)")
    ax2.set_title("Cobertura do intervalo de 90% por horizonte", color=style.INK_SECONDARY)

    style.titles(
        fig,
        "Calibração conformal corrige a largura excessiva e recupera cobertura",
        "Cobertura empírica dos intervalos centrais antes e depois do ajuste",
        source=SOURCE,
    )
    return fig


def plot_weekly_coverage_2024(
    forecasts_2024: dict[str, pd.DataFrame] | pd.DataFrame,
    models: list[str] | None = None,
) -> Figure:
    """Evolução temporal da cobertura de 90% semana a semana na temporada recorde 2023-24."""
    style.apply()
    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    ax.axhline(0.90, color=style.AXIS, lw=1.0, ls=(0, (3, 3)), zorder=1)

    if isinstance(forecasts_2024, pd.DataFrame):
        df_dict = {
            m: forecasts_2024[forecasts_2024["model"] == m]
            for m in forecasts_2024["model"].unique()
        }
    else:
        df_dict = forecasts_2024

    if models is None:
        models = [m for m in PHASE3_LABELS if m in df_dict]

    for model in models:
        sub = df_dict.get(model)
        if sub is None or sub.empty:
            continue
        # Se 'cov90' não estiver pré-calculado, calcula com y e q
        if "cov90" not in sub.columns and "y" in sub.columns and "q0.05" in sub.columns:
            covered = (sub["q0.05"] <= sub["y"]) & (sub["y"] <= sub["q0.95"])
            sub = sub.assign(cov90=covered)

        if "cov90" in sub.columns:
            weekly = sub.groupby("origin", observed=True)["cov90"].mean()
            weeks = sorted(weekly.index)
            ax.plot(
                range(len(weeks)),
                [weekly[w] for w in weeks],
                color=PHASE3_COLORS.get(model, style.INK),
                label=PHASE3_LABELS.get(model, model),
                lw=1.8,
            )

    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0.50, 1.0)
    ax.set_ylabel("Cobertura semanal do intervalo de 90%")
    ax.set_xlabel("Semanas da temporada 2023-24 (origem da previsão)")
    ax.legend(loc="lower left", fontsize=8.5)

    style.titles(
        fig,
        "ACI reage dinamicamente durante o surto recorde de 2024",
        "Cobertura do intervalo de 90% semana a semana na temporada 2023-24",
        source=SOURCE,
    )
    return fig


def plot_alpha_trajectory_2024(aci_df: pd.DataFrame) -> Figure:
    """Trajetória do alpha efetivo de 90% na ACI durante a temporada 2023-24."""
    style.apply()
    fig, ax = plt.subplots(figsize=(9.6, 3.5))
    ax.axhline(0.10, color=style.AXIS, lw=1.0, ls=(0, (3, 3)), zorder=1)

    horizons = sorted(aci_df["horizon"].unique()) if "horizon" in aci_df.columns else [1]
    for h in horizons:
        sub = aci_df[aci_df["horizon"] == h] if "horizon" in aci_df.columns else aci_df
        if "alpha_eff_90" in sub.columns:
            weekly = sub.groupby("origin", observed=True)["alpha_eff_90"].mean()
            weeks = sorted(weekly.index)
            ax.plot(
                range(len(weeks)),
                [weekly[w] for w in weeks],
                label=f"h={h}",
                lw=1.8,
            )

    ax.set_ylabel(r"$\alpha_{\mathrm{eff}}$ para nível 90% (nominal = 0,10)")
    ax.set_xlabel("Semanas da temporada 2023-24 (origem da previsão)")
    ax.legend(loc="upper right", fontsize=8.5)

    style.titles(
        fig,
        r"$\alpha_{\mathrm{eff}}$ cai durante a subida do surto para forçar intervalos mais largos",
        "Mecanismo adaptativo da ACI ao longo das semanas de 2023-24",
        source=SOURCE,
    )
    return fig


def plot_skill_comparison(scored: pd.DataFrame, reference: str = "naive_level") -> Figure:
    """WIS relativo ao modelo de referência, por horizonte."""
    style.apply()
    table = backtest.summarize(scored, ["horizon"])
    rel = backtest.relative_wis(table, reference).reset_index()
    horizons = sorted(rel["horizon"].unique())
    models = [m for m in rel["model"].unique() if m != reference]

    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.axhline(1.0, color=style.AXIS, lw=1.0, ls=(0, (3, 3)), zorder=1)
    x = range(len(horizons))

    for model in models:
        d = rel[rel["model"] == model].sort_values("horizon")
        ax.plot(
            x,
            d["rel_wis"],
            color=PHASE3_COLORS.get(model, style.INK),
            marker="o",
            ms=4.5,
            label=PHASE3_LABELS.get(model, model),
            lw=1.8,
            zorder=3,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels([str(h) for h in horizons])
    ax.set_ylabel(f"WIS relativo ao {PHASE3_LABELS.get(reference, reference).lower()}")
    ax.set_xlabel("Horizonte (semanas à frente)")
    ax.legend(loc="upper left", fontsize=8.5)

    ref_label = PHASE3_LABELS.get(reference, reference).lower()
    style.titles(
        fig,
        "Calibração mantém ou melhora o WIS do LightGBM",
        f"WIS relativo ao {ref_label} (abaixo de 1× é melhor)",
        source=SOURCE,
    )
    return fig


FIGURES = {
    "fase3_calibracao.png": plot_calibration_comparison,
    "fase3_skill_wis.png": plot_skill_comparison,
}


def make_all(
    scored: pd.DataFrame,
    out_dir: Path | str,
    forecasts_2024: dict[str, pd.DataFrame] | pd.DataFrame | None = None,
) -> list[Path]:
    """Gera e salva todas as figuras da fase 3 no diretório de saída."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    for name, plot_fn in FIGURES.items():
        fig = plot_fn(scored)
        fig.savefig(out / name)
        plt.close(fig)
        paths.append(out / name)

    if forecasts_2024 is not None:
        fig_cov = plot_weekly_coverage_2024(forecasts_2024)
        fig_cov.savefig(out / "fase3_cobertura_semanal_2024.png")
        plt.close(fig_cov)
        paths.append(out / "fase3_cobertura_semanal_2024.png")

        # Se houver dados de ACI com alpha_eff_90
        aci_data = (
            forecasts_2024.get("lgbm_aci")
            if isinstance(forecasts_2024, dict)
            else forecasts_2024[forecasts_2024["model"] == "lgbm_aci"]
            if "model" in forecasts_2024
            else None
        )
        if aci_data is not None and "alpha_eff_90" in aci_data.columns:
            fig_alpha = plot_alpha_trajectory_2024(aci_data)
            fig_alpha.savefig(out / "fase3_alpha_eff_aci.png")
            plt.close(fig_alpha)
            paths.append(out / "fase3_alpha_eff_aci.png")

    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--forecast-dir", default="data/forecasts")
    parser.add_argument(
        "--models", default="naive_level,lgbm,lgbm_cqr,lgbm_cqr_level,lgbm_aci,lgbm_aci_level"
    )
    parser.add_argument("--out-dir", default="reports/figures")
    args = parser.parse_args(argv)

    models = args.models.split(",")
    forecast_dir = Path(args.forecast_dir)
    scored = backtest.load_scored(forecast_dir, models)

    forecasts_2024: dict[str, pd.DataFrame] = {}
    for m in models:
        p2024 = forecast_dir / f"{m}_2024.parquet"
        if p2024.exists():
            forecasts_2024[m] = pd.read_parquet(p2024)

    paths = make_all(scored, args.out_dir, forecasts_2024=forecasts_2024 or None)
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
