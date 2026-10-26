Layered mesh: 3D model output on its own mesh
==============================================

``forcingkit.layered_mesh`` holds every layer of a 3D model on the model's own horizontal mesh,
with no regridding: the triangles, a vertical coordinate, and any number of named fields. It is
the 3D companion of :doc:`mesh_frames` (``MeshFrames`` stays the single-layer surface-current
contract; ``LayeredMesh.surface()`` returns one). A viewer can draw a model's own cells from it,
cut sections through the layers, and colour several models on one shared scale.

Data contract
-------------

``LayeredMesh`` holds:

- ``times`` (UTC; naive input is taken as UTC);
- ``node_lon``, ``node_lat`` (degrees) and ``tri`` (``n_elem`` x 3, zero-based node indices);
- one vertical coordinate:

  - sigma (terrain-following): ``h`` (``n_node``, depth below the datum, positive down),
    ``siglay`` (``[n_layer, n_node]``, sigma at layer centres, surface first: 0 at the surface,
    -1 at the bed), optional ``siglev`` (``[n_layer + 1, n_node]``, at the interfaces) and
    ``zeta`` (``[t, n_node]``, surface elevation). A layer centre is at height
    ``siglay * (h + zeta) + zeta``;
  - fixed heights: ``layer_z`` (``[n_layer, n_elem]``, height of each layer centre in metres,
    positive up) and an optional ``bed_z`` (``n_elem``);
  - neither, for a single layer at the surface;

- ``fields``: ``name -> [t, n_layer, n_elem]`` or ``[t, n_layer, n_node]``, NaN where a cell has
  no value (dry, below the bed, masked);
- ``meta``: a ``FieldMeta`` per field, whose ``kind`` (``"element"`` or ``"node"``) says which;
- ``source_index`` (optional, ``n_elem``): each element's index in the source's own numbering
  (an FVCOM element id, or a flat grid-cell index).

``element_values(name)`` gives any field per element: as held for an element field, and as the
mean of the element's three nodes for a node field, which is the flat value a viewer draws for
that cell. ``speed`` is derived from ``u`` and ``v`` when the mesh holds them.

Field metadata
--------------

``FieldMeta`` carries ``name, label, units, lo, hi, log, ticks``, the entries a viewer manifest
lists for each field (``manifest_entry()`` returns exactly those), plus ``kind`` and
``symmetric`` (a signed field whose scale is centred on zero). ``encode(values)`` maps values to
one byte on the scale (1 at or below ``lo``, 255 at or above ``hi``, linear or logarithmic, 0 for
NaN) and ``decode(codes)`` maps them back, so per-field u8 codes and their legend come from one
place. Without explicit ``ticks``, legends use round values (1, 2 or 5 times a power of ten).

Sections
--------

``section(points, n=200, fields=None, node_values="element")`` samples the mesh at ``n`` points
spaced evenly by distance along a polyline ``[(lon, lat), ...]`` and returns a ``Section`` with
``distance`` (m), ``lon``, ``lat``, the host ``element`` (-1 outside the mesh), ``z`` (layer
centres, ``[t, n_layer, n]``), ``z_interfaces`` (sigma meshes with ``siglev``), ``bed_z`` and
``values[name]`` (``[t, n_layer, n]``). Element fields take the host element's value; node fields
take the element mean by default or, with ``node_values="linear"``, are interpolated within the
element. Heights on a sigma mesh are interpolated within the element and include ``zeta``. Points
outside the mesh are NaN. Point location uses matplotlib's trapezoid map, and falls back to a
search over every element for a triangulation it cannot index.

Shared colour scales
--------------------

``shared_scales(meshes, names=None, window=None)`` sets ``lo`` and ``hi`` for each field from
every element value of every mesh that holds it, at the times inside ``window``, weighted by
element area: the 1st to 99th percentiles, or for a symmetric field ``-hi`` to ``hi`` with ``hi``
the 99.5th percentile of the magnitude. Weighting by area matters on unstructured meshes, whose
rivers and harbours hold many small elements; counted per element, those elements set the scale
to their own extremes (on NECOFS around Long Island Sound on 2 April 2026, temperature reached
19 degC and salinity 5 PSU that way, against 1.7 to 8.0 degC and 28.5 to 34.2 PSU by area).
``shared_scale`` does one field and accepts other percentiles.

Particle tracks
---------------

``Tracks`` holds particle positions ``xyz`` (``[t, n_particle, 3]``: longitude, latitude, height
in metres) and an ``alive`` mask (``[t, n_particle]``; a position that is not finite is never
alive). ``local_xyz(lon0, lat0)`` gives metres about an origin, and ``quantize(lon0, lat0)``
stores them as int16 per axis with an offset and scale (error at most half a step, about 0.3 m
across 20 km; samples that are not alive get code 0). Tracking itself is the caller's: any
tracker that reports positions fits.

Adapters
--------

``necofs(bbox, start, end)`` reads NECOFS FVCOM GOM7 through
``forcingkit.fetchers.necofs.volume`` (see :doc:`necofs`) for the elements whose centres lie in
the box, hourly: 45 sigma layers with ``zeta``, element ``u``, ``v`` and ``w`` (FVCOM ``ww``,
m/s, up positive) and node ``T`` (degC) and ``S`` (PSU). ``from_necofs_volume(ds)`` converts a
dataset already read.

``structured(lon, lat, mask, z, times=..., fields=...)`` turns a regular grid (1-D axes) or a
curvilinear one (2-D centres) into two triangles per cell, both carrying the cell's own value, so
cells are drawn flat, as the model holds them. Cell corners lie midway between centres (for a
curvilinear grid, the mean of the four centres around each corner), and the outer corners half a
cell beyond the outermost centres. ``mask`` keeps water cells; ``z`` is None for one surface
layer, fixed heights ``[n_layer]`` (z-level models), or ``[n_layer, ny, nx]``; ``fields`` are
``[t, n_layer, ny, nx]`` or ``[t, ny, nx]``. This is the basis for the NYOFS and DBOFS
(curvilinear), HYCOM (regular, z levels) and HRRR (regular, surface only for now) adapters.

Agreement with an independent pipeline
--------------------------------------

Built from the same NECOFS volumes (2 April 2026, 02:00 to 21:00 UTC, four boxes around Long
Island Sound), sections along four lines reproduce byte for byte the per-field codes of a
separate pipeline written for the same data (speed, temperature, salinity and vertical velocity,
45 layers, every hourly frame: about 11.4 million codes), and its layer interfaces and bed to
within 0.1 mm. Its area-weighted shared scales agree to 0.02 percent (temperature 1.7408 to
7.9941 degC against 1.7406 to 7.9932): that pipeline measured element areas in the model's
Cartesian projection, and this module about each element's own latitude.
