# `cluster.py` — Fuzzy c-means / SOM Clustering

Clusters MT resistivity/conductivity + seismic tomography properties
(Vp, Vs, Vp/Vs, density) that have already been prepared onto a common
grid by **either** `interpolate.py` (point-cloud RBF/kriging/IDW/
nearest) **or** `remapping.py` (conservative, cell-volume-weighted
regrid) — `INPUT_KIND` picks which (see "Two upstream sources" below)
— via fuzzy c-means or a self-organizing map (SOM), and plots
depth-slice maps using the same basemap engine and styling conventions
as the MT and seismic plot pipelines (`README_mt.md`, `README_seis.md`).

This is the clustering half of the pipeline; see `README_interpolate.md`
for `interpolate.py` and `README_remapping.md` for `remapping.py`,
either of which can produce this script's input.

Authors: Svetlana Byrdina (SMB) & Volker Rath (DIAS)
AI-assisted development: Claude (Anthropic)
License: GNU General Public License v3 (GPL-3.0-or-later)
AI-generated code — review before use in production.

---

## Pipeline

```
precompute.py  →  interpolate.py  →  {SITE_PREFIX}_interp_<method>.nc  ─┐
               └→  remapping.py   →  {SITE_PREFIX}_remap.npz          ─┼→  cluster.py  →  {SITE_PREFIX}_clusters.nc + figures
```

`cluster.py` reads `INTERP_FILE` or `REMAP_FILE` (`INPUT_KIND` picks
which) — no grid-building, interpolation, or remapping of its own —
and is therefore completely agnostic to which interpolation method
(`rbf`/`kriging`/`idw`/`nearest`), target-grid choice (`joint`/
`seismic`), or remapping backend (`medcoupling`/`esmpy`) produced its
input. Point `INTERP_FILE`/`REMAP_FILE` at whichever run you want to
cluster. You can also interpolate/remap a superset of variables once
and cluster on different `CLUSTER_VARS` subsets without re-running
`interpolate.py`/`remapping.py`.

## Two upstream sources: `INPUT_KIND = "interp" | "remap"`

- `"interp"` (default) — `interpolate.py`'s point-cloud RBF/kriging/
  IDW/nearest output. Grid mode (`"joint"`/`"seismic"`) is read from
  the file's own `target_grid_mode` attribute.
- `"remap"` — `remapping.py`'s conservative, cell-volume-weighted
  regrid output (see `README_remapping.md`). Always a `"joint"`
  regular UTM-km grid — there is no `"seismic"`-mode equivalent for a
  genuine conservative remap. Once loaded, this plugs into exactly the
  same `"joint"`-grid code path as an `interpolate.py` output, so
  everything downstream (feature-table build, clustering, plotting) is
  unmodified and unaware of which one actually produced its input.

Point both settings at the same `SITE_PREFIX`/`NC_DIR` outputs and
cluster each one to see directly how much `interpolate.py`'s
point-cloud cell-size bias changes a given site's clustering compared
to `remapping.py`'s conservative alternative.

`tomomt.py` must sit alongside `cluster.py` — the shared helper module
also used by the MT and seismic plot scripts.

---

## What this script does

1. Reads `INTERP_FILE` (`interpolate.py`'s output) or `REMAP_FILE`
   (`remapping.py`'s output), per `INPUT_KIND` — grid
   coords/dims, per-variable interpolated fields and their units, and
   enough attributes (`target_grid_mode`, `interp_vars`) to know how to
   plot it.
2. Picks `CLUSTER_VARS`, a subset of whatever variables are actually in
   `INTERP_FILE` (`None` = every one of them) — so you can interpolate a
   superset once and cluster on different subsets without re-running
   `interpolate.py`.
3. Builds one feature table, drops any grid cell with a NaN in a
   selected variable, optionally standardizes (`STANDARDIZE`, z-score)
   each feature, then weights (`CLUSTER_WEIGHTS`).
4. Clusters via `CLUSTERING_METHOD`:
   - **`"fcm"`** — fuzzy c-means (Bezdek, 1981), self-contained NumPy
     implementation, `N_CLUSTERS` discrete classes. Reports the fuzzy
     partition coefficient (FPC) as a quick quality check.
   - **`"som"`** — self-organizing map (Kohonen, 1982), self-contained
     NumPy implementation, `SOM_ROWS x SOM_COLS` neurons; every point is
     labelled with its best-matching unit (BMU) over the **full**
     neuron grid (`SOM_ROWS*SOM_COLS` classes, not collapsed to
     `N_CLUSTERS`), colored with a topological colormap
     (`som_grid_colormap()`) so visually similar map colors reflect
     genuinely similar feature-space neighbors. Reports mean
     quantization error and topographic error.
5. Reconstructs the hard label + membership/quantization-error back onto
   the grid and saves `{SITE_PREFIX}_clusters.nc` /
   `{SITE_PREFIX}_cluster_centers.csv`.
6. Plots horizontal cluster maps at `PLOT_DEPTHS_KM`. The cluster
   overlay switches on `target_grid_mode` (read from `INTERP_FILE`'s
   attributes): `imshow(extent=...)` for a regular `"joint"` grid,
   `pcolormesh(shading="nearest")` against the reused `"seismic"`
   grid's own 2-D `utm_easting_km`/`utm_northing_km` coordinates
   otherwise.

Writes:

| Output file                    | Contents                                   |
|----------------------------------|---------------------------------------------|
| `{SITE_PREFIX}_clusters.nc`               | Hard label + membership/quantization-error on the grid read from `INTERP_FILE` |
| `{SITE_PREFIX}_cluster_centers.csv`       | Cluster/class centers in raw (physical) units, point counts, fractions, and the `weight` row used |
| `clusters_{depth}km_{SITE_PREFIX}.{ext}`  | Plain cluster map, one per `PLOT_DEPTHS_KM` entry, one file per `PLOT_FORMATS` entry |
| `clusters_{depth}km_{SITE_PREFIX}_annotated.{ext}` | The same map, additionally annotated with seismicity/MT-sites/volcanoes/cities — produced in parallel, not instead of, the plain map; toggle with `SHOW_SPECIFIC_PLOT` |

Both `{SITE_PREFIX}_clusters.nc` and `{SITE_PREFIX}_cluster_centers.csv`
are written into `NC_DIR`, alongside `INTERP_FILE`.

---

## Per-variable weighting

`CLUSTER_WEIGHTS` lets individual variables count more or less toward
cluster/BMU assignment than a plain (unweighted) Euclidean distance
would give them. Each standardized feature is scaled by
`sqrt(weight)` before clustering — equivalent to the weighted Euclidean
distance `d² = Σⱼ weightⱼ · (xⱼ − cⱼ)²` — and the resulting centers are
divided back by the same `sqrt(weight)` afterward, before undoing
standardization, so `{SITE_PREFIX}_cluster_centers.csv` and
`{SITE_PREFIX}_clusters.nc` always report centers in true physical
units. Every variable not listed in `CLUSTER_WEIGHTS` defaults to `1.0`
(no effect).

## Gradient-based clustering features (optional)

Two independent, opt-in ways to bring spatial-derivative information
into the feature table, on top of (not instead of) the plain
property-value features above:

**(A) Per-variable gradient magnitude.** No special setting — just
list `"{key}_grad_mag"` in `CLUSTER_VARS`
(e.g. `CLUSTER_VARS = ["rho", "vps", "rho_grad_mag"]`). Only available
for `INPUT_KIND="interp"`, and only if `interpolate.py` was itself run
with `COMPUTE_GRADIENT=True` and that key in its own `GRADIENT_VARS` —
it stores `"{key}_grad_mag"` as an ordinary data variable even though
it isn't listed in `interp_vars` (see `README_interpolate.md`).
`remapping.py` doesn't compute gradients, so referencing a `"_grad_mag"`
name under `INPUT_KIND="remap"` raises a clear, specific error rather
than a generic "not found". Componentwise gradients
(`grad_easting`/`grad_northing`/`grad_depth`) are deliberately **not**
supported as direct clustering features here: they aren't
rotation-invariant (a N–S boundary and an E–W boundary look completely
different in component space despite being the same kind of feature),
so `grad_mag` — a rotation-invariant scalar "edginess" per variable —
is the one that actually belongs in a feature table.

**(B) Structural-alignment features between variable pairs**
(`STRUCTURAL_ALIGNMENT_PAIRS`) — computed here, self-contained, for
either `INPUT_KIND` (`"joint"` grid mode only), via cosine similarity
of two variables' own gradient vectors:

```
cos(theta) = (grad_A . grad_B) / (|grad_A| |grad_B|)
```

in `[-1, 1]`: `+1` = co-aligned structure (the same boundary shows up
in both models), `-1` = anti-aligned (itself often geologically
meaningful — e.g. resistive-where-fast vs. conductive-where-slow),
`0` = unrelated structure that happens to cross. This answers a
genuinely different question than clustering on property values:
"where do these two models' *structures* agree," not "what physical
unit is this."

The masking rule is the one part of this worth being deliberate about:
a value is set to NaN wherever *either* field's own local gradient
magnitude falls below *that field's own*
`STRUCTURAL_MIN_GRADIENT_PERCENTILE`-th percentile (a per-variable,
scale-free noise floor, since e.g. `"rho"` (log10 Ω·m) and `"vps"`
(dimensionless) have unrelated natural gradient scales) — **not** a
threshold on the cosine value itself. A near-zero cosine where both
gradients are genuinely large is a real anti-alignment signal and must
be kept; masking on the cosine value would throw that away along with
the genuinely-uninformative flat/textureless cells it's meant to catch.

List each pair once in `STRUCTURAL_ALIGNMENT_PAIRS`; add
`"{a}_{b}_align"` to `CLUSTER_VARS` to actually cluster on it (a listed
pair `CLUSTER_VARS` never references is never computed, and the two
base variables are loaded as needed to compute it even if you only
want the alignment score, not their raw values, in your final feature
set). `CLUSTER_VARS` must be given explicitly whenever
`STRUCTURAL_ALIGNMENT_PAIRS` is non-empty — `CLUSTER_VARS=None`'s "use
everything" can't know to include a feature this script derives rather
than one the input file itself provides.

## Specific (annotated) cluster maps

A second map per `PLOT_DEPTHS_KM` entry, in addition to (never instead
of) the plain cluster map. Reuses `_draw_cluster_overlay()` on the same
basemap, then layers `{SITE_PREFIX}_plot_seis.py`-style feature
markers/labels via `draw_specific_features()` — same CSVs, on/off
switches, and marker/label style dicts as `{SITE_PREFIX}_plot_seis.py`.

---

## Coordinate convention

**UTM Zone 19S (EPSG:32719)**, distances in km, depth in km positive
down. A `"joint"`-mode `{SITE_PREFIX}_clusters.nc` has a genuine,
uniformly-spaced regular `(depth, northing, easting)` grid; a
`"seismic"`-mode one inherits whatever native `(depth, row, col)`
resolution the reference seismic-tomography variable has, with 2-D
`utm_easting_km`/`utm_northing_km` aux coordinates rather than 1-D
regular axes — see `README_interpolate.md`.

## Verification performed (gradient-based features)

- `structural_alignment()` checked against an analytic case (two ramp
  fields at 45° to each other → cosine = 1/√2 exactly, away from grid
  edges and any flat-gradient patch), and confirmed a fully-interior
  zero-gradient cell comes back NaN rather than a spurious near-zero
  cosine value.
- The `INPUT_KIND` dispatcher's new variable-resolution logic (base
  vars needed for `STRUCTURAL_ALIGNMENT_PAIRS` loaded even when not
  directly requested; `"_grad_mag"` under `INPUT_KIND="remap"`;
  unrecognised `"*_align"` names; `STRUCTURAL_ALIGNMENT_PAIRS` with
  `CLUSTER_VARS=None`; `STRUCTURAL_ALIGNMENT_PAIRS` under a `"seismic"`
  grid) each exercised against a synthetic fake-`xarray`/fake-NPZ
  dataset, confirming both the happy path and every guard rail.
- The pre-existing base case (no gradient features requested) was
  re-checked against the same fake dataset to confirm nothing
  regressed for existing `CLUSTER_VARS`/`INPUT_KIND` usage.

## Dependencies

```
numpy, xarray, pandas, matplotlib
```
plus the local `tomomt.py` helper module. No `scikit-fuzzy`/`MiniSom`
dependency — both clustering implementations are self-contained.

## Typical run

```bash
python3 precompute.py      # must be run first (or already have been)
python3 interpolate.py     # see README_interpolate.md -- OR:
python3 remapping.py       # see README_remapping.md    -- (either, or both)
python3 cluster.py         # reads INTERP_FILE or REMAP_FILE (INPUT_KIND),
                            # clusters (fcm/som), writes {SITE_PREFIX}_clusters.nc /
                            # {SITE_PREFIX}_cluster_centers.csv and the maps
```

Re-run `cluster.py` (no need to re-run `interpolate.py`/`remapping.py`)
whenever `CLUSTER_VARS`, `CLUSTER_WEIGHTS`, `STANDARDIZE`,
`CLUSTERING_METHOD` or its `N_CLUSTERS`/`FUZZINESS`/`SOM_*` settings, or
any plotting setting change. Re-run `interpolate.py` first (see
`README_interpolate.md`) whenever the interpolated grid itself needs to
change.
