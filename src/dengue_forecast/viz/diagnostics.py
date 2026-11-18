"""Figuras estáticas de diagnóstico avançado: PIT aleatorizado, curvas PR e análise de erro.

Uso:
    uv run python -m dengue_forecast.viz.diagnostics
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, PercentFormatter

from dengue_forecast.viz import style

SOURCE = "Backtest por origem móvel (4 temporadas, 5.570 municípios). Elaboração própria."

DIAG_COLORS = {
    "naive_level": style.INK_MUTED,
    "lgbm": style.CATEGORICAL[6],
    "lgbm_cqr_level": style.CATEGORICAL[2],
    "lgbm_immunity": style.CATEGORICAL[0],
    "gnn_nograph": style.CATEGORICAL[5],
}

DIAG_LABELS = {
    "naive_level": "Ingênuo por nível",
    "lgbm": "LightGBM",
    "lgbm_cqr_level": "LightGBM + CQR nível",
    "lgbm_immunity": "LightGBM + Imunidade",
    "gnn_nograph": "Rede sem grafo",
}


def _fmt(v: float, digits: int = 2) -> str:
    return f"{v:.{digits}f}".replace(".", ",")


def plot_pit_and_pr(diagnostics: dict, horizon: int = 8) -> Figure:
    """Painel duplo: histograma PIT aleatorizado (esq.) e curvas Precisão-Recall em h=8 (dir.)."""
    style.apply()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 3.8), gridspec_kw={"wspace": 0.28})

    # Painel 1: PIT de lgbm_cqr_level (geral vs choque 2023-24 em h=8)
    pit_models = {m["model"]: m for m in diagnostics.get("pit", {}).get("models", [])}
    served = pit_models.get("lgbm_cqr_level") or next(iter(pit_models.values()), None)
    if served is not None:
        h_all = np.asarray(served.get(f"h{horizon}", []), dtype=float)
        h_2024 = np.asarray(served.get("h8_2024", []), dtype=float)
        bins = len(h_all) if len(h_all) else 10
        centers = (np.arange(bins) + 0.5) / bins
        w = 0.38 / bins
        ax1.axhline(1.0, color=style.AXIS, lw=1.1, ls=(0, (3, 3)), zorder=1)
        if len(h_all) == bins:
            ax1.bar(
                centers - w / 2,
                h_all,
                width=w,
                color=style.CATEGORICAL[2],
                label="4 temporadas (geral)",
            )
        if len(h_2024) == bins:
            ax1.bar(
                centers + w / 2,
                h_2024,
                width=w,
                color=style.ORANGE,
                label="Temporada 2023-24 (choque)",
            )
        top_bar = max(
            float(h_all.max()) if len(h_all) else 1.0,
            float(h_2024.max()) if len(h_2024) else 1.0,
        )
        ax1.set_xlim(0.0, 1.0)
        ax1.set_ylim(0.0, max(1.5, top_bar * 1.24))
        ax1.set_xlabel("Quantil realizado u = F(y)")
        ax1.set_ylabel("Densidade relativa (uniforme = 1,0)")
        ax1.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: _fmt(v, 1)))
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: _fmt(v, 1)))
        ax1.legend(loc="upper center", fontsize=8.0)
    ax1.set_title(f"Histograma PIT aleatorizado (h={horizon})", color=style.INK_SECONDARY)

    # Painel 2: Curvas PR do alerta de surto (> 300/100k)
    pr = diagnostics.get("pr_curves", {})
    recall = np.asarray(pr.get("recall", []), dtype=float)
    prev = pr.get("prevalence", {}).get(str(horizon), 0.0326)
    if prev is not None:
        ax2.axhline(float(prev), color=style.AXIS, lw=1.0, ls=(0, (3, 3)), zorder=1)
    for entry in pr.get("models", []):
        m = entry["model"]
        if m not in ("naive_level", "lgbm", "lgbm_cqr_level", "gnn_nograph"):
            continue
        block = entry.get("horizons", {}).get(str(horizon))
        if not block or len(recall) == 0:
            continue
        prec = np.asarray(
            [np.nan if v is None else float(v) for v in block["precision"]], dtype=float
        )
        auc_txt = _fmt(float(block["pr_auc"]), 3) if block.get("pr_auc") is not None else "—"
        ax2.plot(
            recall,
            prec,
            color=DIAG_COLORS.get(m, style.INK),
            lw=2.0,
            label=f"{DIAG_LABELS.get(m, m)} (AUC {auc_txt})",
        )
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(0.0, 1.02)
    ax2.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax2.set_xlabel("Recall (surtos > 300/100 mil detectados)")
    ax2.set_ylabel("Precisão")
    ax2.set_title(f"Alerta precoce de surto (h={horizon} semanas)", color=style.INK_SECONDARY)
    ax2.legend(loc="upper right", fontsize=7.8)

    style.titles(
        fig,
        "Calibração probabilística (PIT) e discriminação de surtos em 8 semanas",
        "Assimetria discreta em zero à esq., cauda estourada em 2024 à dir. e ganho na curva PR",
        source=SOURCE,
    )
    return fig


def plot_error_regimes(diagnostics: dict, horizon: int = 8) -> Figure:
    """Erros de cobertura das caudas e WIS log por regime epidemiológico."""
    style.apply()
    rows = diagnostics.get("error_analysis", {}).get("regimes_by_horizon", {}).get(str(horizon), [])
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(9.8, 3.6), sharey=True, gridspec_kw={"wspace": 0.14}
    )
    if not rows:
        return fig

    short_names = {
        "quiet": "Silêncio / Endemia\n(<5 casos)",
        "onset": "Explosão de surto\n(≥2× origem)",
        "active": "Platô epidêmico\nsustentado",
        "decline": "Descida pós-pico\n(≤0,5× origem)",
    }
    labels = [short_names.get(r["regime"], r["label"]) for r in rows]
    y = np.arange(len(rows))[::-1]
    under = np.array([float(r["under90"] or 0.0) for r in rows])
    over = np.array([float(r["over90"] or 0.0) for r in rows])
    wis_log = np.array([float(r["wis_log"] or 0.0) for r in rows])

    h_bar = 0.34
    ax1.axvline(0.05, color=style.AXIS, lw=1.0, ls=(0, (3, 3)), zorder=1)
    ax1.barh(
        y + h_bar / 2, under, height=h_bar, color=style.ORANGE, label="Acima de q95 (subestimou)"
    )
    ax1.barh(
        y - h_bar / 2,
        over,
        height=h_bar,
        color=style.CATEGORICAL[0],
        label="Abaixo de q05 (superestimou)",
    )
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax1.set_xlabel("Fração fora do intervalo de 90% (nominal = 5% por cauda)")
    ax1.grid(axis="x")
    ax1.grid(axis="y", visible=False)
    ax1.tick_params(axis="y", length=0)
    ax1.set_title("Violação das caudas de 90%", color=style.INK_SECONDARY)
    ax1.legend(loc="lower right", fontsize=8.0)

    bars = ax2.barh(y, wis_log, height=0.55, color=style.CATEGORICAL[2])
    for b, v in zip(bars, wis_log, strict=True):
        ax2.text(
            v + 0.008,
            b.get_y() + b.get_height() / 2,
            _fmt(v, 3),
            va="center",
            ha="left",
            fontsize=8.0,
        )
    ax2.set_xlim(0, float(wis_log.max()) * 1.25 if len(wis_log) else 1.0)
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: _fmt(v, 2)))
    ax2.set_xlabel("WIS em escala log1p")
    ax2.grid(axis="x")
    ax2.grid(axis="y", visible=False)
    ax2.tick_params(axis="y", length=0)
    ax2.set_title("Erro probabilístico (WIS log)", color=style.INK_SECONDARY)

    style.titles(
        fig,
        f"Onde o modelo servido erra (h={horizon}): subcobertura concentra-se na explosão de surto",
        "Decomposição das 1,16 milhão de previsões de h=8 por dinâmica entre origem e alvo",
        source=SOURCE,
    )
    return fig


def make_all(
    diagnostics: dict, out_dir: Path | str = "reports/figures", horizon: int = 8
) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, fn in (
        ("avaliacao_pit_pr.png", plot_pit_and_pr),
        ("erro_por_regime.png", plot_error_regimes),
    ):
        fig = fn(diagnostics, horizon=horizon)
        fig.savefig(out / name)
        plt.close(fig)
        paths.append(out / name)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--diagnostics", default="web/public/data/diagnostics.json")
    parser.add_argument("--out-dir", default="reports/figures")
    parser.add_argument("--horizon", type=int, default=8)
    args = parser.parse_args(argv)

    diag = json.loads(Path(args.diagnostics).read_text(encoding="utf-8"))
    for p in make_all(diag, args.out_dir, horizon=args.horizon):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
