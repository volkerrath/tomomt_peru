# `cross_gradient.py` — Structural Cross-Gradient Between Field Pairs

New pipeline stage, parallel to `plot_joint.py` and `cluster.py`:

```
precompute.py -> interpolate.py -> {SITE_PREFIX}_interp_<method>.nc
                                                 |
                                    cross_gradient.py -> figures
```

It reads the same `INTERP_FILE` NetCDF as `plot_joint.py`/`cluster.py` and
is equally agnostic to which interpolation method (`krig`/`rbf`/`idw`/`nn`)
produced it. For each configured pair of fields (e.g. resistivity vs. Vp)
it computes the **cross-gradient** and plots depth-slice maps and vertical
sections of its magnitude, using the same `SITE_PREFIX`/`INTERP_TAG`
filename and title conventions as the rest of the pipeline.

## What the cross-gradient is

For two fields `m1(r)` and `m2(r)` on the same grid,

```
t(r) = grad(m1(r)) x grad(m2(r))
```

`t` is a 3-D vector, zero wherever the two fields' local gradients are
parallel or anti-parallel — i.e. wherever the two models share the same
structural orientation, regardless of their magnitudes, sign, or units —
and non-zero where they disagree structurally. It requires no assumed
petrophysical relationship between the two properties, which is why it is
the standard structural-coupling diagnostic (and inversion constraint) for
comparing independently derived geophysical models such as MT resistivity
and seismic velocity.

**References**
- Gallardo, L. A., and M. A. Meju (2003), Characterization of
  heterogeneous near-surface materials by joint 2D inversion of DC
  resistivity and seismic data, *Geophys. Res. Lett.*, 30(13), 1658,
  doi:[10.1029/2003GL017370](https://doi.org/10.1029/2003GL017370).
- Gallardo, L. A., and M. A. Meju (2004), Joint two-dimensional DC
  resistivity and seismic travel time inversion with cross-gradients
  constraints, *J. Geophys. Res.-Solid Earth*, 109, B03311,
  doi:[10.1029/2003JB002716](https://doi.org/10.1029/2003JB002716).

## Configuration

- `SITE_PREFIX` — `"tacna"`/`"saba"`, must match `interpolate.py`.
- `INTERP_FILE` — explicit path, or `None` to auto-pick the newest
  `{SITE_PREFIX}_interp_*.nc` in the current directory.
- `CROSS_GRADIENT_PAIRS` — list of `(field_a, field_b, label)` tuples;
  `field_a`/`field_b` must be variable names present in `INTERP_FILE`
  (e.g. `"rho"`, `"vp"`, `"vs"`, `"vpvs"`, `"dens"`). `label=None` falls
  back to `"{field_a}_vs_{field_b}"` in titles/filenames.
- `LOG_FIELDS` — fields gradiented in log10 space rather than linear,
  kept in sync with the `log10(rho)` convention used throughout
  `plot_joint.py`/`plot_modem_mesh.py`. Defaults to `{"rho"}`.
- `CROSS_GRADIENT_NORMALIZE` — also compute and plot the scale-free
  `|t| / (|grad m1| * |grad m2|) = sin(theta)` (0 = parallel structure,
  1 = perpendicular). Recommended when comparing across different field
  pairs, since raw `|t|` carries the product of the two fields' own units
  and isn't directly comparable pair-to-pair. `NORMALIZE_EPS` sets the
  gradient-magnitude floor below which the ratio is set to NaN (flat,
  structureless regions where the angle isn't meaningful).
- `PLOT_MAPS` / `MAP_DEPTHS` / `MAP_XLIM` / `MAP_YLIM` — depth-slice map
  controls; `MAP_DEPTHS=None` plots every depth level in the grid.
  `MAP_XLIM`/`MAP_YLIM` default to `None` (auto from the grid), matching
  the project's safe-default policy for un-verified geographic bounds.
- `PLOT_SECTIONS` / `VSLICES` / `VE` — vertical-section controls, same
  `(name, (lon1, lat1), (lon2, lat2), n_samples)` schema as
  `plot_seis.py`/`plot_joint.py`. **`VSLICES` ships empty** — profile
  endpoints are project- and site-specific and were not guessed at; see
  "Known limitation" below before populating it.
- `CMAP_CG`/`CMIN_CG`/`CMAX_CG` and `CMAP_CG_NORM`/`CMIN_CG_NORM`/
  `CMAX_CG_NORM` — colour scales for the raw and normalised cross-gradient
  respectively. The normalised scale defaults to a fixed `[0, 1]` since
  `sin(theta)` is naturally bounded; the raw scale defaults to per-panel
  auto-scaling (`None, None`).
- `ISO_LINES_MAP` / `ISO_LINES_VSLICE` / `ISO_LEVELS` — independent
  isoline toggles for maps and sections, following the same
  per-script-independent pattern as the other plot scripts.

## Grid support: `"joint"` only

`interpolate.py`'s `TARGET_GRID` can be `"joint"` (a genuinely regular
UTM-km grid) or `"seismic"` (the reused seismic-tomography native grid,
structured only in row/col index space, not orthogonal in UTM space).

**`cross_gradient.py` only supports `TARGET_GRID="joint"`.** A
coordinate-exact 3-D gradient — the standard this pipeline holds gradients
to everywhere else, via `numpy.gradient` on real, non-uniform coordinate
arrays — cannot be computed from `"seismic"`-mode index-space finite
differences without assuming near-orthogonality between grid rows/columns
and the UTM axes. Rather than silently approximating (and rather than
guessing at the seismic grid's local warp), `load_joint_grid()` raises a
clear `ValueError` if `INTERP_FILE` isn't a `"joint"`-mode grid. Re-run
`interpolate.py` with `TARGET_GRID="joint"` to produce a compatible file.

## Two known integration seams (by design, not oversight)

1. **Basemap/marker/label styling.** This script does its own minimal
   `imshow`/`pcolormesh` plotting rather than calling into `plotpy.py`'s
   `build_panel_figure`/`finish_panel_colorbar`/`draw_north_arrow`/
   `add_latlon_ticks`/`clipped_markers`/`clipped_labels`, since their exact
   current signatures weren't available to verify against here. Swapping
   these in to match `plot_joint.py`'s house style exactly is a
   straightforward follow-up — flagged rather than guessed at.
2. **Section profile projection.** `_profile_line()` is a deliberate stub
   that raises `NotImplementedError`: vertical sections need the same
   lon/lat -> UTM-km transformer `plot_seis.py`/`plot_joint.py` already
   use, and re-deriving that transform independently risked a subtle
   mismatch. Wire in the shared transformer there before populating
   `VSLICES` and enabling `PLOT_SECTIONS`. Depth-slice maps (`PLOT_MAPS`)
   do not depend on this and work as shipped.

## Output

- `PLOT_FORMATS` (default `[".pdf", ".jpg"]`) and `FIG_DPI` control which
  formats/resolution each figure is saved in — same allow-list convention
  as `plot_joint.py`'s `PLOT_FORMATS`; every stem below is written once
  per entry (set it back to `[".png"]` alone to reproduce this script's
  previous PNG-only behaviour).
- Stems: `{SITE_PREFIX}_{pair}_crossgrad_{mag|norm}_{INTERP_TAG}_map_z<depth>km`
  (maps) or `..._sec_<name>` (sections), titled with the field pair, the
  quantity plotted, and `[SITE_PREFIX, INTERP_TAG]` — same suffix
  convention as `plot_joint.py`.
- All output files for a run (every format of every stem) are bundled
  into `{SITE_PREFIX}_cross_gradient_YYYYMMDD_HHMM.zip` (Europe/Paris
  timestamp, both the zip name and each member's internal mtime),
  matching the rest of the pipeline's delivery convention.
