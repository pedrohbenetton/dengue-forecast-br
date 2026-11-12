"""Paleta e tipografia compartilhadas entre as figuras estáticas e (depois) o site.

Os valores abaixo são a fonte única de verdade: o site da fase 5 deve ler estes mesmos hex.
"""

import matplotlib as mpl

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Categórica: ordem fixa (a ordem é o que garante separação para daltônicos); nunca ciclar.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
BLUE, ORANGE = CATEGORICAL[0], CATEGORICAL[1]
# Sequencial (um matiz, claro -> escuro) para magnitudes.
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]
NEUTRAL_FILL = "#d9d8d2"  # contexto / "os outros"

# Cor por entidade, estável em todas as figuras e no site.
REGION_COLORS = dict(
    zip(["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"], CATEGORICAL, strict=False)
)

FONT_FAMILY = ["Segoe UI", "Inter", "Helvetica Neue", "Arial", "DejaVu Sans"]


def apply() -> None:
    """Aplica o estilo do projeto ao matplotlib (idempotente)."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "font.size": 10,
            "text.color": INK,
            "figure.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 220,
            "savefig.facecolor": SURFACE,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.25,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK_SECONDARY,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "ytick.major.size": 0,
            "xtick.major.size": 3,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "legend.frameon": False,
            "legend.fontsize": 8.5,
        }
    )


def titles(fig, title: str, subtitle: str, *, source: str | None = None) -> None:
    """Título + subtítulo alinhados à esquerda e fonte no rodapé (padrão de todas as figuras)."""
    h = fig.get_figheight()  # deslocamentos em polegadas: mesmo respiro em qualquer altura
    top = max(ax.get_position().y1 for ax in fig.axes)
    bottom = min(ax.get_position().y0 for ax in fig.axes)
    base = top + (0.42 if any(ax.get_title(loc="left") for ax in fig.axes) else 0.18) / h
    fig.text(0.0, base, subtitle, fontsize=9.5, color=INK_SECONDARY, ha="left", va="bottom")
    fig.text(
        0.0, base + 0.24 / h, title, fontsize=13, fontweight="semibold", color=INK,
        ha="left", va="bottom",
    )  # fmt: skip
    if source:
        fig.text(0.0, bottom - 0.62 / h, source, fontsize=7.5, color=INK_MUTED, ha="left", va="top")
