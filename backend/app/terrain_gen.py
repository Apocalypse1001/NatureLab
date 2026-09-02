"""Procedural terrain: the river valley the flow solver needs to have a river.

Why this exists at all (docs/07_river_plan.md, item 2): terrain starts perfectly
flat -- `TerrainGrid.heights` is `np.zeros` -- and open-channel flow is driven by
the slope of the bed, not by pressure at the inlet. On a flat map no inflow mode
produces a river; it produces a spreading puddle. So a channel is a precondition
for everything else in v0.12.0, not a decoration on top of it.

Why not the brush: 200 m of channel with banks is hundreds of strokes, and the
lumps a hand-drawn bed leaves behind stand waves up in the solver. The bed here
is analytic and smooth by construction.

Shape, in one paragraph. The whole map is a floodplain falling from west to east
at a constant `slope`; a trapezoidal channel of `bed_width` is cut `incision`
metres into it along a centreline, with banks that rise over a horizontal
`bank_run` using a smoothstep rather than a straight chamfer -- a plain trapezoid
has a slope discontinuity at the toe and at the bank top, and those two kinks are
exactly where a standing wave parks itself.

Elevations are chosen so the valley sits just above y = 0: objects are placed on
that plane and the water-level control works in the same range, so a channel dug
to -18 m would be technically fine and practically unusable. Note that this puts
the world floor (`HEIGHT_MIN`) some 20 m below the bed, so it is *not* a
meaningful backstop against the long-run incision feedback measured in
docs/07_river_plan.md; `SEDIMENT_MAX_BED_CHANGE` is, and calibrating that is a
v0.13.0 item.

Default slope is the gentle end of the plan's 0.2-0.5 % range deliberately. The
probe measured 0.65 m/s at 0.5 % and h = 1 m still climbing at 20 s, and a
120 s run at that slope incised itself 2.3 m with peak speeds of 9 m/s. Until a
steady river has been observed to be stable, the default that ships should be the
one least likely to tear its own bed apart in the first minute of a demo.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from . import config

# The channel is generated to be run about a third full: `incision` is the depth
# of the cut, and the intended operating depth is `incision / 2` at most, so the
# banks contain the flow instead of the flow finding the floodplain. Item 3's
# discharge defaults are sized from this number.
OPERATING_DEPTH_FRACTION = 0.5

DEFAULTS: Dict[str, float] = {
    "slope": 0.002,             # longitudinal fall, west to east (fraction)
    "bed_width": 12.0,          # flat channel bottom (m)
    "incision": 2.0,            # depth of the cut below the floodplain (m)
    "bank_run": 4.0,            # horizontal distance the bank takes to rise (m)
    "outlet_bed": 0.5,          # channel bottom elevation at the east edge (m)
    "meander_amplitude": 0.0,   # lateral swing of the centreline (m), 0 = straight
    "meander_wavelength": 120.0,  # distance between meander crests (m)
}

_LIMITS: Dict[str, tuple] = {
    "slope": (0.0, 0.05),
    "bed_width": (1.0, config.WORLD_SIZE_M * 0.5),
    "incision": (0.1, 20.0),
    "bank_run": (0.5, 50.0),
    "outlet_bed": (config.HEIGHT_MIN + 1.0, config.HEIGHT_MAX - 20.0),
    "meander_amplitude": (0.0, config.WORLD_SIZE_M * 0.25),
    "meander_wavelength": (10.0, config.WORLD_SIZE_M * 4.0),
}


def validate(params: Dict[str, Any] | None) -> Dict[str, float]:
    """Fill in the defaults and range-check whatever the caller supplied."""
    out = dict(DEFAULTS)
    for key, value in (params or {}).items():
        if key not in DEFAULTS:
            raise ValueError(f"unknown river parameter: {key!r}")
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"river.{key} must be finite")
        low, high = _LIMITS[key]
        if not low <= number <= high:
            raise ValueError(f"river.{key} out of range [{low}, {high}]: {number}")
        out[key] = number
    if out["meander_amplitude"] > 0.0:
        # the channel plus its banks must stay on the map even at full swing
        half_top = out["bed_width"] * 0.5 + out["bank_run"]
        if out["meander_amplitude"] + half_top > config.WORLD_SIZE_M * 0.5:
            raise ValueError("river.meander_amplitude would push the banks off the map")
    return out


def river_valley(terrain, params: Dict[str, Any] | None = None) -> Dict[str, float]:
    """Write a river valley into `terrain.heights` in place.

    Returns the effective parameters, including the derived numbers a caller
    needs to fill the channel sensibly (`operating_depth`, and the bed and
    floodplain elevations at each end).
    """
    p = validate(params)
    rows, cols = terrain.heights.shape          # (z, x): x runs west -> east
    cell = terrain.cell_size

    x = np.arange(cols, dtype=np.float64) * cell             # 0 at the west edge
    z = np.arange(rows, dtype=np.float64) * cell
    length = (cols - 1) * cell

    # Longitudinal profile: the floodplain falls at `slope`, so the channel it
    # contains falls at the same rate. Anchored at the outlet, because that end
    # is where the water leaves and where the level is easiest to reason about.
    floodplain = p["outlet_bed"] + p["incision"] + (length - x) * p["slope"]

    centre_z = 0.5 * (rows - 1) * cell
    if p["meander_amplitude"] > 0.0:
        centre = centre_z + p["meander_amplitude"] * np.sin(
            2.0 * np.pi * x / p["meander_wavelength"])
    else:
        centre = np.full(cols, centre_z)

    # Cross-section: flat bottom, then a smoothstep bank. The smoothstep is the
    # whole reason this is not a plain trapezoid -- it has zero gradient at both
    # the toe and the bank top, so the bed has no kink for a standing wave to
    # anchor on, while a chamfer has two per cross-section.
    dist = np.abs(z[:, None] - centre[None, :])
    t = np.clip((dist - p["bed_width"] * 0.5) / p["bank_run"], 0.0, 1.0)
    rise = p["incision"] * (t * t * (3.0 - 2.0 * t))

    heights = floodplain[None, :] - p["incision"] + rise
    terrain.heights[:, :] = np.clip(heights, config.HEIGHT_MIN,
                                    config.HEIGHT_MAX).astype(np.float32)

    p = dict(p)
    p["operating_depth"] = p["incision"] * OPERATING_DEPTH_FRACTION
    p["inlet_bed"] = float(terrain.heights[:, 0].min())
    p["outlet_bed_actual"] = float(terrain.heights[:, -1].min())
    p["floodplain_outlet"] = float(floodplain[-1])
    # What to set the existing edge-level control to for a channel run about half
    # full: that control prescribes a surface elevation at the west columns, so
    # it is measured from the *inlet* bed, not the outlet's.
    p["inlet_level"] = p["inlet_bed"] + p["operating_depth"]
    return p


# ---------------------------------------------------------------------- dam

# A dam is terrain here, not an object, and that is a physics decision rather
# than a shortcut. `fluid_solver._is_solid` rasterizes HOUSE and BRIDGE as
# infinitely tall walls, so an object dam can never be overtopped -- and
# overtopping is the entire lesson of a dam scenario. `bed_height` (what a ROCK
# uses) would be overtoppable, but `_build_bed_offset` only knows circular domes
# of ROCK_BASE_RADIUS_M. A ridge written into `terrain.heights` needs no solver
# change at all: the reservoir fills, the crest spills when the water reaches it,
# and the user can breach it with the ordinary terrain brush -- which is the most
# valuable thing a child can do to a dam.
DAM_DEFAULTS: Dict[str, float] = {
    "dam_x": 70.0,            # distance from the WEST edge to the crest (m)
    "crest_height": 3.0,      # crest above the channel bed at that station (m)
    "crest_width": 8.0,       # thickness of the crest along the flow (m)
    "face_run": 7.0,          # horizontal run of each face, smoothstepped (m)
    "spillway_width": 6.0,    # width of the notch in the crest (m)
    "spillway_drop": 0.8,     # how far the notch sits below the crest (m)
}

_DAM_LIMITS: Dict[str, tuple] = {
    "dam_x": (10.0, config.WORLD_SIZE_M - 10.0),
    "crest_height": (0.2, 20.0),
    "crest_width": (1.0, 60.0),
    "face_run": (0.5, 60.0),
    "spillway_width": (0.0, config.WORLD_SIZE_M * 0.5),
    "spillway_drop": (0.0, 20.0),
}


def validate_dam(params: Dict[str, Any] | None) -> Dict[str, float]:
    """Fill in the dam defaults and range-check whatever the caller supplied."""
    out = dict(DAM_DEFAULTS)
    for key, value in (params or {}).items():
        if key not in DAM_DEFAULTS:
            raise ValueError(f"unknown dam parameter: {key!r}")
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"dam.{key} must be finite")
        low, high = _DAM_LIMITS[key]
        if not low <= number <= high:
            raise ValueError(f"dam.{key} out of range [{low}, {high}]: {number}")
        out[key] = number
    if out["spillway_drop"] >= out["crest_height"]:
        raise ValueError("dam.spillway_drop must stay below dam.crest_height, "
                         "otherwise the notch cuts through the whole dam")
    return out


def dam_ridge(terrain, river_params: Dict[str, Any] | None = None,
              dam_params: Dict[str, Any] | None = None) -> Dict[str, float]:
    """Cut a river valley, then lay a dam across it. Writes in place.

    The ridge is raised the same way the banks are -- smoothstep faces, no
    kinks -- for the same reason given at the top of this module: a slope
    discontinuity is where a standing wave parks itself, and a dam face is the
    last place you want one, because a standing wave at the toe is easily
    mistaken for the dam leaking.

    The crest carries a deliberate notch. A perfectly level crest has no lowest
    point, so the station where the reservoir first spills is decided by
    floating-point noise and moves from run to run -- which reads as a solver
    artefact rather than as physics. The notch is a spillway: it puts the
    overflow at a place the user can see and reason about, and it puts it over
    the channel, so the water that spills goes back where it came from.

    Returns the river's effective parameters plus the dam's, including
    `crest_elevation` and `reservoir_volume_m3` -- what the crest holds back
    when the reservoir is full to the notch, which is what sizes an inflow.
    """
    river = river_valley(terrain, river_params)
    p = validate_dam(dam_params)

    rows, cols = terrain.heights.shape          # (z, x): x runs west -> east
    cell = terrain.cell_size
    x = np.arange(cols, dtype=np.float64) * cell             # 0 at the west edge
    z = np.arange(rows, dtype=np.float64) * cell

    # Crest elevation is measured from the CHANNEL bed at the dam's station, not
    # from the floodplain: "3 m of dam" should mean 3 m of water held in the
    # channel, whatever the valley happens to be doing at that point.
    length = (cols - 1) * cell
    floodplain_at_dam = (river["outlet_bed"] + river["incision"]
                         + (length - p["dam_x"]) * river["slope"])
    channel_bed_at_dam = floodplain_at_dam - river["incision"]
    crest = channel_bed_at_dam + p["crest_height"]

    # Along-flow profile: flat over the crest width, then a smoothstep face.
    along = np.abs(x - p["dam_x"])
    t = np.clip((along - p["crest_width"] * 0.5) / p["face_run"], 0.0, 1.0)
    shape = 1.0 - t * t * (3.0 - 2.0 * t)                     # 1 on the crest

    # Across-flow profile: the crest, notched over the channel centreline.
    centre_z = 0.5 * (rows - 1) * cell
    crest_line = np.full(rows, crest)
    if p["spillway_width"] > 0.0 and p["spillway_drop"] > 0.0:
        across = np.abs(z - centre_z)
        # the notch walls get a smoothstep too, over half their own width, so
        # the spillway is a trough rather than a slot with two vertical kinks
        n = np.clip((across - p["spillway_width"] * 0.5)
                    / max(p["spillway_width"] * 0.5, 1e-6), 0.0, 1.0)
        crest_line = crest - p["spillway_drop"] * (1.0 - n * n * (3.0 - 2.0 * n))

    valley = terrain.heights.astype(np.float64)
    ridge = valley + (crest_line[:, None] - valley) * shape[None, :]
    # `maximum`, so the ridge only ever ADDS material: where the valley walls are
    # already higher than the crest the dam merges into them instead of cutting
    # a slot through the hillside.
    terrain.heights[:, :] = np.clip(np.maximum(valley, ridge),
                                    config.HEIGHT_MIN,
                                    config.HEIGHT_MAX).astype(np.float32)

    spill = float(crest_line.min())
    # What the reservoir holds when it is full to the spillway lip: every cell
    # upstream of the crest whose bed is below that level, times its area.
    upstream = terrain.heights[:, x < p["dam_x"]].astype(np.float64)
    volume = float(np.clip(spill - upstream, 0.0, None).sum() * cell * cell)

    effective = dict(river)
    effective.update(p)
    effective["crest_elevation"] = float(crest)
    effective["spill_elevation"] = spill
    effective["channel_bed_at_dam"] = float(channel_bed_at_dam)
    effective["reservoir_volume_m3"] = volume
    return effective
