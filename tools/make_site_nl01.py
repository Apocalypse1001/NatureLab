"""Build the NL01 reference plot (a real parcel in Almere) as a NatureLab world.

    python tools/make_site_nl01.py            -> data/scenario_nl01.json

Inputs are an unmodified copy of the NatureLab_Reference_Plot_NL01 package in
data/sites/NL01/ (checked against its MANIFEST.sha256 before anything is read).
The package's own rule is kept: the measured AHN DTM is the baseline and is not
edited here -- no grading, no carved ditches, no house. Everything this script
adds on top is either measured (BGT surfaces and building outlines) or derived
and labelled as such in `site` (building heights from DSM - DTM).

Grid. The DTM is 140 x 180 pixels of 0.5 m. NatureLab heights are vertices, so
the vertices sit on the pixel centres: 139 x 179 cells of 0.5 m, 69.5 x 89.5 m,
no resampling and no invented edge row. World axes: x east, y up, z SOUTH --
three.js is right-handed with y up, so with x east, north is -z. Local package
coordinates (x_local east, y_local north, origin at the SW corner) map as
    x = x_local - 35,  z = 45 - y_local.
Vertex row j is PNG/CSV row j (row 0 north).

Heights are "shifted to zero" the way the package's own CSV does (z_local): the
lowest DTM value in the window, -5.07 m NAP, becomes 0. `site.vertical_offset_m`
holds -5.07, so z_nap = height + offset everywhere.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from shapely import contains_xy
from shapely.geometry import box, shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import config  # noqa: E402
from app.persistence import save_world  # noqa: E402
from app.world_state import TerrainGrid, WorldState  # noqa: E402

SITE = ROOT / "data" / "sites" / "NL01"
NAME = "scenario_nl01"

NX, NY = 140, 180                 # DTM pixels (= vertices)
CELL = 0.5
ORIGIN_RD = (150340.0, 482880.0)  # SW corner of the working window, EPSG:28992
HALF_X, HALF_Y = 35.0, 45.0       # half the 70 x 90 m window
NODATA = 1.0e30                   # the GeoTIFFs mark NoData as float32 max
STANDING_M = 0.5                  # DSM rise below which no building stands there

# BGT "fysiek voorkomen" / type -> config.SURFACE_CLASSES code. Layers are laid
# in this order, so a later layer wins where two share a vertex (BGT is a
# planar partition, so that is only ever a boundary vertex).
CLASS_CODE = {name: code for code, (name, _) in config.SURFACE_CLASSES.items()}
LAYERS = [
    ("bgt_begroeidterreindeel", "fysiek_voorkomen",
     {"groenvoorziening": "planted"}, "grass"),
    ("bgt_onbegroeidterreindeel", "fysiek_voorkomen",
     {"erf": "yard", "onverhard": "unpaved", "half verhard": "half_paved",
      "open verharding": "open_paving", "gesloten verharding": "closed_paving"}, "yard"),
    ("bgt_wegdeel", "fysiek_voorkomen",
     {"onverhard": "unpaved", "half verhard": "half_paved",
      "open verharding": "open_paving", "gesloten verharding": "closed_paving"}, "closed_paving"),
    ("bgt_ondersteunendwegdeel", "fysiek_voorkomen",
     {"onverhard": "unpaved", "half verhard": "half_paved"}, "grass"),
    ("bgt_ondersteunendwaterdeel", "type", {}, "bank"),
    ("bgt_waterdeel", "type", {}, "water"),
]


# ---------------------------------------------------------------- inputs
def verify_manifest() -> dict:
    """Every copied file must match the package's own hash."""
    manifest = {}
    for line in (SITE / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        digest, _, name = line.strip().partition("  ")
        manifest[name.strip()] = digest
    checked = {}
    for path in sorted(SITE.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.sha256":
            continue
        rel = path.relative_to(SITE).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if manifest.get(rel) != digest:
            raise SystemExit(f"{rel}: does not match MANIFEST.sha256 -- inputs changed")
        checked[rel] = digest
    return checked


def read_grid():
    """The package's processed grid: heights (NAP, NoData nearest-filled) and
    the mask of which samples were measured, both [row from north, column]."""
    z_nap = np.zeros((NY, NX), dtype=np.float64)
    valid = np.zeros((NY, NX), dtype=bool)
    with open(SITE / "processed" / "terrain_grid_local.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            i = int(round((float(row["x_local_m"]) - CELL / 2) / CELL))
            j = int(round((2 * HALF_Y - CELL / 2 - float(row["y_local_m"])) / CELL))
            z_nap[j, i] = float(row["z_nap_m"])
            valid[j, i] = row["source_valid"] == "1"
    return z_nap, valid


def read_raster(name: str) -> np.ndarray:
    data = np.array(Image.open(SITE / "source" / name), dtype=np.float64)
    if data.shape != (NY, NX):
        raise SystemExit(f"{name}: expected {NY} x {NX}, got {data.shape}")
    data[data > NODATA] = np.nan
    return data


def current_features(layer: str):
    """BGT keeps history: only records still registered, at ground level, once
    each (the same road appears four times in the context file)."""
    data = json.loads((SITE / "source" / f"{layer}_context.geojson").read_text(encoding="utf-8"))
    window = box(ORIGIN_RD[0], ORIGIN_RD[1], ORIGIN_RD[0] + 2 * HALF_X, ORIGIN_RD[1] + 2 * HALF_Y)
    seen = set()
    for feature in data["features"]:
        props = feature["properties"]
        if props.get("eind_registratie") or props.get("relatieve_hoogteligging", 0) != 0:
            continue
        if props.get("lokaal_id") in seen:
            continue
        seen.add(props.get("lokaal_id"))
        geometry = shape(feature["geometry"])
        if geometry.intersects(window):
            yield props, geometry, window


# ---------------------------------------------------------------- helpers
def vertex_rd():
    """RD coordinates of every vertex, [row from north, column]."""
    columns = np.arange(NX)
    rows = np.arange(NY)
    x_rd = ORIGIN_RD[0] + CELL / 2 + columns * CELL
    y_rd = ORIGIN_RD[1] + 2 * HALF_Y - CELL / 2 - rows * CELL
    return np.meshgrid(x_rd, y_rd)


def rd_to_world(x_rd: float, y_rd: float):
    return x_rd - ORIGIN_RD[0] - HALF_X, HALF_Y - (y_rd - ORIGIN_RD[1])


def depressions(height: np.ndarray, inside: np.ndarray, limit: int = 4):
    """Closed hollows the rain will fill first (priority-flood from every map
    edge), largest volume first: [(row, column, depth_m, volume_m3), ...]."""
    import heapq
    filled = height.copy()
    done = np.zeros_like(inside, dtype=bool)
    heap = []
    for j in range(NY):
        for i in range(NX):
            if j in (0, NY - 1) or i in (0, NX - 1):
                heapq.heappush(heap, (filled[j, i], j, i))
                done[j, i] = True
    while heap:
        level, j, i = heapq.heappop(heap)
        for dj, di in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            b, a = j + dj, i + di
            if 0 <= b < NY and 0 <= a < NX and not done[b, a]:
                done[b, a] = True
                filled[b, a] = max(filled[b, a], level)
                heapq.heappush(heap, (filled[b, a], b, a))
    depth = filled - height
    wet = depth > 0.02
    labels = np.zeros(depth.shape, dtype=np.int32)
    found = []
    for j in range(NY):
        for i in range(NX):
            if wet[j, i] and not labels[j, i]:
                label = len(found) + 1
                stack, cells = [(j, i)], []
                labels[j, i] = label
                while stack:
                    b, a = stack.pop()
                    cells.append((b, a))
                    for db, da in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        y, x = b + db, a + da
                        if 0 <= y < NY and 0 <= x < NX and wet[y, x] and not labels[y, x]:
                            labels[y, x] = label
                            stack.append((y, x))
                deepest = max(cells, key=lambda c: depth[c])
                volume = float(sum(depth[c] for c in cells)) * CELL * CELL
                found.append((deepest[0], deepest[1], float(depth[deepest]), volume,
                              bool(inside[deepest])))
    found = [f for f in found if f[4]]
    found.sort(key=lambda f: -f[3])
    return [f[:4] for f in found[:limit]]


# ---------------------------------------------------------------- build
def build() -> WorldState:
    hashes = verify_manifest()
    meta = json.loads((SITE / "processed" / "heightmap_metadata.json").read_text(encoding="utf-8"))
    z_nap, valid = read_grid()
    offset = float(meta["z_min_nap_m"])
    height = (z_nap - offset).astype(np.float32)

    world = WorldState()
    world.terrain = TerrainGrid(width=NX - 1, height=NY - 1, cell_size=CELL)
    world.terrain.heights = height.copy()

    # surfaces
    x_rd, y_rd = vertex_rd()
    surface = np.zeros((NY, NX), dtype=np.uint8)
    unmapped = {}
    for layer, key, table, fallback in LAYERS:
        for props, geometry, _ in current_features(layer):
            value = props.get(key)
            name = table.get(value, fallback)
            if value not in table and value is not None and layer != "bgt_waterdeel" \
                    and layer != "bgt_ondersteunendwaterdeel":
                unmapped[f"{layer}:{value}"] = name
            surface[contains_xy(geometry, x_rd, y_rd)] = CLASS_CODE[name]
    world.terrain.surface = surface

    # buildings: the outline as surveyed, the height from the surface model
    dsm = read_raster("terrain_AHN_DSM_0.5m.tif")
    buildings, absent = [], []
    for props, geometry, window in current_features("bgt_pand"):
        clipped = geometry.intersection(window)
        parts = list(getattr(clipped, "geoms", [clipped]))
        for part in parts:
            if part.geom_type != "Polygon" or part.area < 1.0:
                continue
            ring = np.asarray(part.exterior.coords)[:-1]
            world_xz = np.array([rd_to_world(x, y) for x, y in ring])
            lo, hi = world_xz.min(axis=0), world_xz.max(axis=0)
            centre = (lo + hi) / 2
            inside = contains_xy(part, x_rd, y_rd)
            rise = dsm[inside] - z_nap[inside]
            rise = rise[np.isfinite(rise)]
            if rise.size == 0 or float(np.percentile(rise, 90)) < STANDING_M:
                # BGT still registers it, but the surface model sees bare
                # ground there: it was gone (or not yet built) when AHN flew
                # no BGT surface lies under a registered building; what is
                # there now is the yard around it
                surface[inside & (surface == 0)] = CLASS_CODE["yard"]
                absent.append({"bgt": props.get("lokaal_id"), "area_m2": round(part.area, 1),
                               "dsm_rise_p90_m": round(float(np.percentile(rise, 90)), 2)
                               if rise.size else None})
                continue
            roof = float(np.median(rise))
            ground = float(np.median(height[inside])) if inside.any() else float(
                world.terrain.height_at(centre[0], centre[1]))
            obj = world.add_object("BUILDING", [float(centre[0]), ground, float(centre[1])])
            obj.metadata["footprint"] = [[round(float(x - centre[0]), 3), round(float(z - centre[1]), 3)]
                                         for x, z in world_xz]
            obj.metadata["height_m"] = round(roof, 2)
            obj.metadata["source"] = f"BGT pand {props.get('lokaal_id')}"
            buildings.append({"id": obj.id, "bgt": props.get("lokaal_id"),
                              "area_m2": round(part.area, 1),
                              "height_m": obj.metadata["height_m"],
                              "height_samples": int(rise.size),
                              "clipped_by_window": bool(part.area < geometry.area - 0.5)})

    # the parcel, and gauges in its hollows
    parcel_local = json.loads((SITE / "processed" / "parcel_local.geojson").read_text(encoding="utf-8"))
    parcel = shape(parcel_local["features"][0]["geometry"])
    parcel_world = [[round(x - HALF_X, 3), round(HALF_Y - y, 3)]
                    for x, y in list(parcel.exterior.coords)[:-1]]
    parcel_rd = unary_union([shape(f["geometry"]) for f in json.loads(
        (SITE / "source" / "parcel_rd_epsg28992.geojson").read_text(encoding="utf-8"))["features"]])
    in_parcel = contains_xy(parcel_rd, x_rd, y_rd)
    gauges = []
    for j, i, depth, volume in depressions(height.astype(np.float64), in_parcel):
        x, z = (i - (NX - 1) / 2) * CELL, (j - (NY - 1) / 2) * CELL
        obj = world.add_object("GAUGE", [float(x), float(height[j, i]), float(z)])
        gauges.append({"id": obj.id, "x": x, "z": z, "hollow_depth_m": round(depth, 3),
                       "hollow_volume_m3": round(volume, 2)})

    # water: nothing enters from the edges; rain is the only source. Every
    # edge lets water out on the ground's own slope there -- the survey stops
    # at the window, the land does not (east: the "river" outlet, which is the
    # same normal-depth rule; west, north, south: open_sides)
    world.water.level = 0.0
    world.water.edge_inflow_enabled = False
    world.water.outflow_enabled = True
    world.water.outlet_kind = "river"
    world.water.open_sides = True
    world.water.rain_intensity_mm_h = 0.0

    classes = {config.SURFACE_CLASSES[c][0]: int((surface == c).sum())
               for c in np.unique(surface)}
    world.site = {
        "id": "NL01",
        "name": "NatureLab Reference Plot NL-01, Buitenplaats Oosterwold 178, Almere",
        "source_package": "NatureLab_Reference_Plot_NL01 v1.0 (data/sites/NL01)",
        "crs": "EPSG:28992", "origin_rd_m": list(ORIGIN_RD),
        "axes": "x = x_local - 35 (east), z = 45 - y_local (north is -z), y up",
        "vertical_datum": "NAP", "vertical_offset_m": offset,
        "vertical_note": "z_nap = height + vertical_offset_m; the window's lowest DTM value is 0",
        "grid": {"vertices": [NX, NY], "cell_m": CELL, "on": "DTM pixel centres"},
        "parcel": parcel_world,
        "nodata": {
            "dtm_filled_vertices": int((~valid).sum()),
            "fill": "nearest valid AHN sample (the package's processed grid)",
            "note": "open water has no DTM: the pond and the ditch read as ground at "
                    "the level of their banks, not as hollows -- their beds are unknown",
        },
        "surface_classes": classes,
        "surface_unmapped": unmapped,
        "buildings": buildings,
        "building_height": "derived: median(DSM - DTM) inside the outline",
        "buildings_absent_in_dsm": absent,
        "gauges": gauges,
        "rain_scenarios": json.loads((SITE / "simulation" / "rainfall_scenarios.json")
                                     .read_text(encoding="utf-8")),
        "soil": json.loads((SITE / "processed" / "soil_and_groundwater_summary.json")
                           .read_text(encoding="utf-8")),
        "input_sha256": hashes,
        "edges": "all four open: water leaves at normal depth on the local ground slope",
        "not_modelled": ["infiltration", "pond and ditch beds", "underground drainage",
                         "run-on from outside the window"],
    }
    return world


def main() -> int:
    world = build()
    path = save_world(world, NAME)
    site = world.site
    print(f"wrote {path}")
    print(f"  grid {world.terrain.width} x {world.terrain.height} cells of {world.terrain.cell_size} m, "
          f"heights {world.terrain.heights.min():.3f}..{world.terrain.heights.max():.3f} m "
          f"(+{site['vertical_offset_m']} = NAP)")
    print(f"  surfaces {site['surface_classes']}")
    if site["surface_unmapped"]:
        print(f"  unmapped BGT values (fallback used): {site['surface_unmapped']}")
    for b in site["buildings"]:
        print(f"  building {b['id']}: {b['area_m2']} m2, {b['height_m']} m "
              f"({b['height_samples']} DSM samples){' clipped' if b['clipped_by_window'] else ''}")
    for b in site["buildings_absent_in_dsm"]:
        print(f"  not placed: BGT {b['bgt']} ({b['area_m2']} m2) -- DSM rise p90 {b['dsm_rise_p90_m']} m")
    for g in site["gauges"]:
        print(f"  gauge {g['id']} at ({g['x']:.2f}, {g['z']:.2f}): hollow "
              f"{g['hollow_depth_m']} m deep, {g['hollow_volume_m3']} m3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
