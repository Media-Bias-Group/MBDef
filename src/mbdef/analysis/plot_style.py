"""Shared Times Roman typography and accessible blue publication palette of the tutorials' figures."""

from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

# Times Roman as the publication figures use it, else a font with the same widths (Nimbus Roman, TeX Gyre Termes), else any serif;
# chosen once here because Matplotlib warns on every text for each listed family that is not installed.
_INSTALLED = {font.name for font in font_manager.fontManager.ttflist}
FONT_FAMILY = next(
    (
        name
        for name in ("Times", "Nimbus Roman", "TeX Gyre Termes", "Liberation Serif")
        if name in _INSTALLED
    ),
    "serif",
)
FIGURE_WIDTH = 32
MANUSCRIPT_WIDTH_MM = 122
MANUSCRIPT_BODY_PT = 6.5
MANUSCRIPT_SMALL_PT = 6.0
MANUSCRIPT_EMPHASIS_PT = 6.75
# Matplotlib uses 72 points/inch; LaTeX uses 72.27 points/inch.
FONT_SIZE = MANUSCRIPT_BODY_PT * FIGURE_WIDTH * 25.4 / MANUSCRIPT_WIDTH_MM * 72 / 72.27
SMALL_FONT_SCALE = MANUSCRIPT_SMALL_PT / MANUSCRIPT_BODY_PT
EMPHASIS_FONT_SCALE = MANUSCRIPT_EMPHASIS_PT / MANUSCRIPT_BODY_PT
BLUE = "#0072B2"
BAR_ALPHA = 0.65  # Shared opacity for boxplot fills and closure-paper bars.
BLUE_LIGHT = "#E6F1F8"
INK = "#333333"
GRAY = "#999999"


def similarity_cmap():
    # Sequential tints preserve the ordering of similarities and cluster ranks.
    cmap = LinearSegmentedColormap.from_list("shared_blue", [BLUE_LIGHT, BLUE], N=256)
    cmap.set_bad("white")
    return cmap
