"""
plotpy.py — DEPRECATED, kept only for backward compatibility.
===============================================================
This module was renamed to ``tomomt.py`` (same content, plus a second
round of consolidated helpers -- see that module's docstring). This file
now just re-exports everything from ``tomomt`` under the old name, so
scripts outside this bundle that still do ``import plotpy`` -- as of this
writing, at least tacna_plot_seis.py, tacna_plot_modem_image.py, and
tacna_plot_modem_mesh.py, per this module's own original docstring --
keep working unmodified.

New code should ``import tomomt`` directly. When those three scripts are
next touched, update their import and delete this shim.
"""

from tomomt import *  # noqa: F401,F403
from tomomt import (  # noqa: F401 -- underscore-prefixed names a star import skips
    _parse_cpt_color,
    _load_cpt_colormap,
    _load_rgb_list_colormap,
    _scatter_kwargs_to_plot_kwargs,
    _to_utm,
    _to_geo,
    _VE_POS_PRESETS,
    _MARKER_KWARG_MAP,
)
