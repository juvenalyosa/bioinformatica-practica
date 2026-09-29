"""
estilo_curso.py — Estilo gráfico común del curso *Bioinformática Práctica*.

Autor: Juvenal Yosa, PhD (juvenal.yosa@gmail.com) · Copiloto: Claude (Anthropic)
Licencia: MIT

Uso en cualquier notebook:

    import estilo_curso as ec
    ec.set_style()
    fig, ax = plt.subplots()
    ...
    ec.title(ax, "Título principal", "Subtítulo que explica qué se muestra")

La paleta categórica está validada para daltonismo (orden fijo, nunca cíclico).
Los nucleótidos usan colores convencionales (A verde, C azul, G amarillo, T rojo)
y SIEMPRE se acompañan de la letra, para no depender sólo del color.
"""

from __future__ import annotations

import platform
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Paleta
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"      # fondo de la figura
INK = "#0b0b0b"          # texto principal
INK_2 = "#52514e"        # texto secundario (subtítulos, anotaciones)
MUTED = "#898781"        # ejes, etiquetas de ticks
GRID = "#e1e0d9"         # líneas de rejilla (finas, sólidas)
BASELINE = "#c3c2b7"     # ejes / línea base

# Orden fijo de colores categóricos (serie 1, 2, 3, ...)
CATEGORICAL = [
    "#2a78d6",  # 1 azul
    "#eb6834",  # 2 naranja
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 amarillo
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 verde
    "#4a3aa7",  # 7 violeta
    "#e34948",  # 8 rojo
]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = CATEGORICAL

# Colores de nucleótidos (convención de los sequence logos)
NUC_COLORS = {"A": "#008300", "C": "#2a78d6", "G": "#eda100",
              "T": "#e34948", "U": "#e34948", "N": "#898781"}

# Rampa secuencial (magnitud): un solo tono, claro -> oscuro
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
            "#0d366b"]
CMAP_SEQ = LinearSegmentedColormap.from_list("curso_seq", SEQ_BLUE)

# Rampa divergente (polaridad): azul <- gris neutro -> rojo
CMAP_DIV = LinearSegmentedColormap.from_list(
    "curso_div", ["#104281", "#3987e5", "#9ec5f4", "#f0efec",
                  "#f3b0ae", "#e66767", "#b8302f"])

# Colores de estado (reservados: nunca se usan como "serie 5")
STATUS = {"good": "#0ca30c", "warning": "#fab219",
          "serious": "#ec835a", "critical": "#d03b3b"}

try:  # registrar los mapas de color para usarlos por nombre
    mpl.colormaps.register(CMAP_SEQ)
    mpl.colormaps.register(CMAP_DIV)
except (ValueError, AttributeError):
    pass


# ---------------------------------------------------------------------------
# Tema
# ---------------------------------------------------------------------------
def set_style(dpi: int = 110) -> None:
    """Aplica el tema del curso a matplotlib (y a Plotly si está instalado)."""
    mpl.rcParams.update({
        # Lienzo
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": dpi,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "figure.figsize": (8, 4.5),
        "figure.constrained_layout.use": True,
        # Tipografía
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "text.color": INK,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelsize": 11,
        "axes.labelcolor": INK_2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "mathtext.fontset": "dejavusans",
        # Ejes y rejilla: discretos, la información es la protagonista
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": 6,
        "ytick.major.pad": 6,
        # Marcas
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.markersize": 7,
        "patch.linewidth": 0,
        "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
        "image.cmap": "curso_seq",
        # Animaciones embebidas en el notebook
        "animation.html": "jshtml",
        "animation.embed_limit": 30,
    })
    try:  # figuras nítidas en pantallas de alta resolución
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
            ip.run_line_magic("config", "InlineBackend.figure_format = 'retina'")
    except Exception:
        pass
    _set_plotly_template()


def _set_plotly_template() -> None:
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError:
        return
    pio.templates["curso"] = go.layout.Template(layout=dict(
        font=dict(family="DejaVu Sans, Arial, sans-serif", color=INK, size=13),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, colorway=CATEGORICAL,
        title=dict(x=0.02, xanchor="left", font=dict(size=18)),
        xaxis=dict(showgrid=False, linecolor=BASELINE, ticks="",
                   tickfont=dict(color=INK_2), zeroline=False),
        yaxis=dict(gridcolor=GRID, linecolor=BASELINE, ticks="",
                   tickfont=dict(color=INK_2), zeroline=False),
        hoverlabel=dict(bgcolor="white", font=dict(color=INK)),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    ))
    pio.templates.default = "curso"


# ---------------------------------------------------------------------------
# Ayudantes de composición
# ---------------------------------------------------------------------------
def title(ax, main: str, subtitle: str | None = None) -> None:
    """Título en negrita + subtítulo gris que explica qué muestra la figura."""
    if subtitle:
        ax.set_title(main, loc="left", pad=28)
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, ha="left",
                va="bottom", fontsize=10.5, color=INK_2)
    else:
        ax.set_title(main, loc="left")


def fig_title(fig, main: str, subtitle: str | None = None) -> None:
    """Título + subtítulo sobre una figura de varios paneles (por encima del borde superior)."""
    y_main = 30 if subtitle else 10
    fig.text(0.01, 1.0, main, ha="left", va="bottom", fontsize=15, fontweight="bold", color=INK,
             transform=fig.transFigure + mpl.transforms.ScaledTranslation(0, y_main / 72, fig.dpi_scale_trans))
    if subtitle:
        fig.text(0.01, 1.0, subtitle, ha="left", va="bottom", fontsize=10.5, color=INK_2,
                 transform=fig.transFigure + mpl.transforms.ScaledTranslation(0, 8 / 72, fig.dpi_scale_trans))


def source(fig, text: str) -> None:
    """Nota de fuente de datos al pie de la figura."""
    fig.text(0.01, -0.02, text, ha="left", va="top", fontsize=8.5, color=MUTED)


def label_end(ax, x, y, text: str, color: str = INK_2, dx: float = 6) -> None:
    """Etiqueta directa al final de una línea (el texto nunca lleva el color de la serie)."""
    ax.annotate(text, (x, y), xytext=(dx, 0), textcoords="offset points",
                va="center", ha="left", fontsize=10, color=color)


def nuc_colors(seq: str) -> list[str]:
    """Lista de colores para cada base de una secuencia."""
    return [NUC_COLORS.get(b.upper(), MUTED) for b in seq]


def animate(fig, update, frames, interval: int = 120, name: str | None = None, **kwargs):
    """
    Crea una animación que se reproduce dentro del notebook (HTML + JavaScript,
    no requiere ffmpeg). Mantenga frames <= ~60 para que el notebook no pese mucho.

    Si se da `name` y existe la variable de entorno COURSE_ASSETS, además se guarda
    un GIF (vista previa que se muestra en GitHub y en las celdas de texto).
    """
    import os
    from IPython.display import HTML
    anim = animation.FuncAnimation(fig, update, frames=frames,
                                   interval=interval, **kwargs)
    assets = os.environ.get("COURSE_ASSETS")
    if name and assets:
        os.makedirs(assets, exist_ok=True)
        anim.save(os.path.join(assets, f"{name}.gif"),
                  writer=animation.PillowWriter(fps=max(1, round(1000 / interval))), dpi=72)
    with mpl.rc_context({"savefig.dpi": 96}):      # cuadros livianos: el notebook no se vuelve pesado
        html = HTML(anim.to_jshtml(default_mode="once"))
    plt.close(fig)
    return html


def environment_report() -> dict:
    """Resumen del entorno de ejecución (útil para la reproducibilidad)."""
    try:
        import google.colab  # noqa: F401
        in_colab = True
    except ImportError:
        in_colab = False
    info = {"python": sys.version.split()[0],
            "platform": platform.platform(),
            "colab": in_colab,
            "matplotlib": mpl.__version__}
    for pkg in ("numpy", "pandas", "Bio", "scipy", "plotly"):
        try:
            mod = __import__(pkg)
            info["biopython" if pkg == "Bio" else pkg] = mod.__version__
        except ImportError:
            pass
    return info
