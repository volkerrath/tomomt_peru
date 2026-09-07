# `remapping.py` — Conservative (Cell-Volume-Weighted) Remapping

Sibling to `interpolate.py`: reads the same `precompute.py` outputs and
also feeds `cluster.py`, but instead of point-cloud interpolation
(RBF/kriging/IDW/nearest, each treating a native ModEM cell center or
FEMTIC tetrahedron centroid as an unweighted point) it performs a
genuine **conservative regrid** — every native cell's real volume is
geometrically intersected against the target grid, so a target cell's
value is a true volume-weighted average of whatever native volume
actually falls inside it. This removes a bias interpolate.py cannot
avoid: ModEM meshes are far finer near the surface/stations than at
depth, and FEMTIC tetrahedra vary in size by orders of magnitude across
a single mesh — none of that is visible to a point-cloud interpolator.

Authors: Svetlana Byrdina (SMB) & Volker Rath (DIAS)
AI-assisted development: Claude (Anthropic)
License: GNU General Public License v3 (GPL-3.0-or-later)
AI-generated code — review before use in production.

---

## Pipeline

```
precompute.py  →  interpolate.py  →  {SITE_PREFIX}_interp_<method>.nc  ─┐
              └→  remapping.py    →  {SITE_PREFIX}_remap.npz          ─┼→  cluster.py
                                                                        (INPUT_KIND picks which)
```

Point interpolate.py's and remapping.py's inputs at the same
`precompute.py` outputs and cluster each one's output (via
`cluster.py`'s `INPUT_KIND` setting) to see directly how much the
cell-size bias actually changes a given site's clustering.

---

## Backends: `REMAP_BACKEND = "medcoupling" | "esmpy"`

Every native source is first turned into an explicit volumetric cell
mesh in pure NumPy (nodes + connectivity + one value + one volume per
cell), independently of which library performs the actual conservative
intersection. That common representation then goes to one of two
backend adapters:

| Backend | Source kinds supported | Why |
|---|---|---|
| `"medcoupling"` (default) | `modem_cells`, `femtic_cells`, `seis_grid` | `MEDCouplingRemapper`'s `"P0P0"` intersection is MEDCoupling's core, long-standing use case — arbitrary-shaped source/target cells, exact geometric intersection, any mix of hexahedral/tetrahedral cells. The only backend used for `"femtic_cells"`. |
| `"esmpy"` | `modem_cells` **only** | ESMF's mature, well-established structured-to-structured conservative regrid (`esmpy.Grid` + `RegridMethod.CONSERVE`) — its best-supported use case, and both the ModEM cube and the target grid here are genuinely rectilinear. `"seis_grid"` (curvilinear) and `"femtic_cells"` (tetrahedral) are **not** attempted under this backend — `REMAP_BACKEND="esmpy"` raises `NotImplementedError` for both rather than guessing at ESMF APIs whose availability/correctness varies by build/version. |

Per-`VARIABLE_SOURCES`-entry `backend` overrides let you mix backends
in one run (e.g. `esmpy` for `"rho"` (ModEM), `medcoupling` for
`"rho_femtic"` (FEMTIC) in the same `REMAP_VARS` list).

Both backends are imported lazily (only inside the function that needs
them) — neither is a hard dependency for the other's sources. Install
one or both with:

```bash
pip install medcoupling      # SALOME's standalone MEDCoupling wheels
# or a system ESMF/esmpy install (conda-forge: esmf, esmpy)
```

### The math actually used (MEDCoupling backend)

Per MEDCoupling's own documentation of `NatureOfField`:

- **Value field** (e.g. resistivity): nature `IntensiveMaximum`,
  `W_ij = Vol(Ti∩Sj) / Σ_j Vol(Ti∩Sj)` — a true average over only the
  volume that actually overlaps the target cell (an intensive quantity
  like resistivity/Vp should not be diluted by an unfilled remainder).
- **Coverage fraction**: source cells' own volumes transferred with
  nature `ExtensiveConservation`,
  `W_ij = Vol(Ti∩Sj) / Vol(Sj)` — since `Σ_j W_ij · Vol(Sj) =
  Σ_j Vol(Ti∩Sj)` (the true covered volume of `Ti`), dividing by
  `Ti`'s own volume gives the covered *fraction* directly.

### The math actually used (esmpy backend)

ESMF's own `CONSERVE` weights are `W_ij = Vol(Ti∩Sj) / Vol(Ti)`
(documented as "the area of intersection... divided by the area of the
destination cell") — an extensive-style weight that *dilutes* a
partially-covered cell. Recovered via the same "regrid a ones field
too" trick: regridding a constant-1 source field with the identical
weights gives the covered fraction directly (bounded in [0, 1]
already), and dividing the raw regridded value by that fraction turns
ESMF's diluted result into the proper covered-volume average.

---

## Native sources (cell geometry, not points)

- **`"modem_cells"`** — `precompute.py`'s `modem_submesh_points.nc`
  (Part A), read as the **full structured product grid** it actually
  is (every cell present as a row, including excluded/air cells kept
  as `valid=0` NaN rows). Genuine axis-aligned cell edges are
  reconstructed from the sorted, unique easting/northing/depth values
  — exact for every interior edge; the two outermost cells along each
  axis get the same width as their single interior neighbor
  (`MODEM_EDGE_EXTRAPOLATION="mirror"`, the only option implemented) —
  flagged as a real, usually small, source of edge-volume error.
- **`"femtic_cells"`** — `precompute.py` Part C's `femtic_mesh_utm.nc`
  (full tetrahedral node/connectivity, UTM **metres**) +
  `femtic_submesh_points.nc` (one resistivity value per element, same
  row order) together give the genuine tetrahedral mesh a real
  conservative remap needs — unlike `interpolate.py`'s
  `"femtic_points"` kind, which only ever sees each element's
  centroid. The row-order match between the two files is checked (not
  assumed): a random sample of valid elements' mesh-derived centroids
  is compared against `femtic_submesh_points.nc`'s own stored
  centroids (`FEMTIC_CENTROID_CHECK_TOL_KM`) before proceeding, and
  this script raises rather than silently mis-pairing them if they
  disagree.
- **`"seis_grid"`** — one `precompute.py` Part B output
  (`{SITE_PREFIX}_vp.nc` etc.), a **curvilinear** `(depth, row, col)`
  grid. Cell corners are reconstructed from cell centers (each interior
  corner = mean of its four surrounding centers, boundary corners
  linearly extrapolated one half-cell past the outermost ones) — the
  standard "corners from centers" construction used whenever a source
  only ships cell centers. Usually the *least* affected by the
  cell-size bias `interpolate.py` has, but remapped the same way for
  consistency with the other two kinds in the same run.

Variable names inside `femtic_mesh_utm.nc` (`FEMTIC_MESH_NODE_VARS`,
`FEMTIC_MESH_CONN_VAR`) are settings, not hard-coded — adjust them if
they don't match what `precompute.py` Part C actually wrote (that
script wasn't part of this bundle, so this wasn't verified directly
against it).

---

## Target grid

Always a single freshly-built, genuinely regular UTM-km grid
(`GRID_*_KM`, same shape/meaning as `interpolate.py`'s own joint-grid
settings, auto-bounded to the tightest common overlap of every active
source's own extent unless given explicit bounds). Unlike
`interpolate.py`, there is **no** `"seismic"` (reuse one source's own
native grid) target-grid mode — every native source here already has
to become an explicit cell mesh regardless of its own resolution, so
there's no equivalent shortcut, and supporting an arbitrary curvilinear
*target* would roughly double the backend-adapter code for a case none
of this project's sites currently need.

## Coverage-fraction masking (replaces `MASK_TO_CONVEX_HULL`)

A conservative remap already knows exactly what fraction of each target
cell's volume was actually covered by valid source cells — no need for
`interpolate.py`'s separate Delaunay convex-hull check.
`MIN_COVERAGE_FRACTION` (default `0.5`) masks target cells covered less
than that fraction to NaN — a strictly more principled mask than a
convex hull, since it reflects how much real volume actually informs
that cell's value rather than just whether the cell's *center* falls
inside the source's outer hull. `APPLY_ROI_MASK`/`ROI_VERTICES_KM` work
identically to `interpolate.py`'s own, applied on top.

## Output

`{SITE_PREFIX}_remap.npz` (`NC_DIR`, override with `OUTPUT_FILE`) — an
NPZ archive (not NetCDF, since the per-variable coverage diagnostics
and metadata are simpler as a flat array set + one JSON string than
forced into NetCDF attribute conventions). Contains `depth`/
`northing`/`easting` (target grid axes, km), one `{key}` array per
`REMAP_VARS` entry (float32, dims `(depth, northing, easting)`, NaN
where masked), one `{key}_coverage` array (covered-volume fraction),
and a `meta_json` string (`json.loads()` it back) recording
`dim_names`, `remap_vars`, `units`/`label` per variable, the backend
used, `min_coverage_fraction`, and the grid/ROI settings.

`cluster.py`'s `INPUT_KIND="remap"` setting reads this back and treats
it exactly like a `"joint"`-mode `interpolate.py` output — see that
script's own README for how the two are unified downstream.

## Dependencies

```
numpy, xarray
```
plus **either** `medcoupling` **or** `esmpy` (`REMAP_BACKEND`), each
imported lazily and each optional with respect to the other.

## Typical run

```bash
python3 precompute.py      # must be run first (or already have been)
python3 remapping.py       # loads native cell geometry, builds the target
                            # grid, conservatively remaps (medcoupling/esmpy),
                            # writes {SITE_PREFIX}_remap.npz
```

Re-run whenever `REMAP_VARS`, `USE_CONDUCTIVITY`, `REMAP_BACKEND` (or a
per-source `backend` override), `GRID_*_KM`, `MIN_COVERAGE_FRACTION`,
`APPLY_ROI_MASK`/`ROI_*`, or `precompute.py`'s own output changes.

## Verification performed

- `ast.parse` on the finished file.
- Pure-NumPy geometry unit tests against hand-computed expected values:
  `cell_edges_from_centers()` (uniform and non-uniform axes),
  `hexahedron_volume()` (unit cube and an anisotropically-scaled box),
  `tetrahedron_volume()` (reference tetrahedron, volume 1/6),
  `hex_mesh_from_corner_grid()` (total volume of a small rectilinear
  mesh matches the box's own volume; individual cell node coordinates
  spot-checked), `_corners_from_centers_2d()` (exact recovery on a
  regular grid).
- A synthetic end-to-end smoke test (fake `xarray`/`medcoupling`
  modules, following this codebase's own established
  stub-unavailable-dependencies convention) exercising
  `load_modem_cells`/`load_femtic_cells`/`load_seis_cells`/
  `build_target_grid`/`remap_medcoupling`/`remap_variable` against a
  small hand-built dataset: a constant-valued ModEM cube remaps to
  exactly that constant wherever fully covered, an excluded corner
  cell correctly reduces coverage there, two differently-sized FEMTIC
  tetrahedra correctly volume-average (not naively-average) their two
  values, an unknown backend raises, and `REMAP_BACKEND="esmpy"`
  raises `NotImplementedError` for `"femtic_cells"`/`"seis_grid"` while
  correctly passing through for `"modem_cells"` (failing only on the
  real `esmpy` package being absent from this sandbox, as expected).
- `cluster.py`'s new `INPUT_KIND` dispatcher was exercised against
  fake `"interp"` and `"remap"` inputs built from the *same* underlying
  grid/values, confirming both branches populate identical
  `grid_mode`/`dim_names`/`grid_shape`/`active_cluster_vars`/loaded
  arrays before falling into the (unmodified) rest of `cluster.py`.
- **Not** verified against a real `medcoupling`/`esmpy` install or
  real `precompute.py` output (neither is available in the sandbox
  this was built in — no network access to install either library, and
  `precompute.py` itself wasn't part of this bundle). Run the
  MEDCoupling backend on a real, small `REMAP_VARS` subset first and
  sanity-check its output against `interpolate.py`'s `"nearest"`
  output for the same variable (they should be closer to each other
  than either is to, say, `"rbf"`) before trusting a full production
  run — per this codebase's own "review before use in production"
  convention.

## What's unchanged

`interpolate.py`, `tomomt.py`, and `precompute.py` are untouched by
this work. `cluster.py`'s only changes are the new `INPUT_KIND`/
`REMAP_FILE` settings and the input-reading section itself — the
feature-table build, clustering (`fcm`/`som`), plotting, and output
files are exactly as before for `INPUT_KIND="interp"`, and are
unmodified code paths (the existing `"joint"`-grid branch) for
`INPUT_KIND="remap"`.
