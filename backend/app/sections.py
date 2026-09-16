"""Gauging lines (v0.18.0, docs/07_river_plan.md "Створ").

A SECTION is a line across the flow: a centre, a turn about the vertical, and a
length (`section_width_m`). It reports the discharge through it, and with it
what a person watching a river can actually see change -- how wide the water
is, how deep, how fast, and the Froude number of the section.

The discharge is not rebuilt from cell-centred velocities. The solver moves
water across cell FACES (`_depth_step`), carried at the upwind depth above the
higher of the two beds; a line reads those very face fluxes, from the very
arrays, in the same substep (`fluid_solver._section_flux`). So two lines on
either side of a bridge differ by exactly the water stored between them, and a
line across the inlet band reads the discharge that was set.

On a grid a straight line is a staircase of faces: every face between a cell on
one side of the line and a cell on the other, within the line's length. The
signed sum over that staircase is the net discharge from the line's back to its
front -- net, because behind a pier some faces carry water backwards.

Orientation: at rotation 0 the line runs along z and its front faces +x, the
way the rivers here flow. Turning it by `yaw` about +y turns both.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from . import config

SECTION_TYPE = "SECTION"


def section_length(obj) -> float:
    return float(obj.metadata.get("section_width_m", config.SECTION_WIDTH_M)) * float(obj.scale[2])


def validate_section_metadata(metadata: dict) -> None:
    if "section_width_m" not in metadata:
        return
    width = metadata["section_width_m"]
    if (isinstance(width, bool) or not isinstance(width, (int, float))
            or not math.isfinite(float(width))
            or not 1.0 <= float(width) <= config.SECTION_MAX_WIDTH_M):
        raise ValueError(f"section_width_m must be within [1, {config.SECTION_MAX_WIDTH_M}] m")


def section_faces(centre, yaw: float, length: float, width: int, height: int,
                  dx: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The staircase of faces a line crosses.

    Returns (cell_a, cell_b, axis, sign): the face between cells a and b (b is
    a's +x neighbour when axis is 0, its +z neighbour when axis is 1), and +1
    or -1 for whether water moving a -> b crosses the line back to front.
    Cells are on the grid's own centres, x = (i - (width-1)/2) * dx.
    """
    nx, nz = math.cos(yaw), -math.sin(yaw)          # the line's front
    tx, tz = math.sin(yaw), math.cos(yaw)           # along the line
    half = 0.5 * length
    # only the cells within reach of the line are looked at
    reach = half + 2.0 * dx
    cx, cz = float(centre[0]), float(centre[2])
    i0 = max(0, int(math.floor((cx - reach) / dx + (width - 1) * 0.5)))
    i1 = min(width - 1, int(math.ceil((cx + reach) / dx + (width - 1) * 0.5)))
    j0 = max(0, int(math.floor((cz - reach) / dx + (height - 1) * 0.5)))
    j1 = min(height - 1, int(math.ceil((cz + reach) / dx + (height - 1) * 0.5)))
    empty = (np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.int32),
             np.zeros(0, np.float32))
    if i1 <= i0 and j1 <= j0:
        return empty
    ii, jj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1))
    x = (ii - (width - 1) * 0.5) * dx - cx
    z = (jj - (height - 1) * 0.5) * dx - cz
    # A cell centred on the line itself belongs to its front, whatever the
    # rounding: at a quarter turn cos() is 6e-17, not 0, and a bare `>= 0`
    # sent such cells front or back by the sign of their x -- a step across
    # the river in the middle of a line along it, which read 0.94 m3/s.
    front = (x * nx + z * nz) >= -1.0e-6 * dx
    along = x * tx + z * tz
    cell = jj * width + ii
    parts = []
    for axis, (fa, fb, ta, tb) in enumerate((
            (front[:, :-1], front[:, 1:], along[:, :-1], along[:, 1:]),
            (front[:-1, :], front[1:, :], along[:-1, :], along[1:, :]))):
        a = cell[:, :-1] if axis == 0 else cell[:-1, :]
        b = cell[:, 1:] if axis == 0 else cell[1:, :]
        crossing = (fa != fb) & (np.abs(0.5 * (ta + tb)) <= half)
        sign = np.where(fb, 1.0, -1.0)
        parts.append((a[crossing], b[crossing], np.full(int(crossing.sum()), axis),
                      sign[crossing]))
    if not any(len(p[0]) for p in parts):
        return empty
    return (np.concatenate([p[0] for p in parts]).astype(np.int32),
            np.concatenate([p[1] for p in parts]).astype(np.int32),
            np.concatenate([p[2] for p in parts]).astype(np.int32),
            np.concatenate([p[3] for p in parts]).astype(np.float32))


def reading(flow_m3s: float, area_m2: float, wetted_width_m: float,
            level_m: float | None, gravity: float) -> dict:
    """What a line reports, from its time-averaged sums over one frame.

    `froude_section` is the SECTION's Froude number, V / sqrt(g D) with V = Q/A
    and D = A/B the hydraulic depth -- not the largest local |u|/sqrt(g h)
    anywhere along the line, which the water colouring shows instead.
    """
    wet = wetted_width_m > 1.0e-9 and area_m2 > 1.0e-9
    velocity = flow_m3s / area_m2 if wet else 0.0
    depth = area_m2 / wetted_width_m if wet else 0.0
    froude = abs(velocity) / math.sqrt(gravity * depth) if depth > 0.0 else 0.0
    return {"flow_m3s": flow_m3s, "area_m2": area_m2 if wet else 0.0,
            "wetted_width_m": wetted_width_m if wet else 0.0,
            "mean_depth_m": depth, "mean_velocity_m_s": velocity,
            "froude_section": froude, "level_m": level_m if wet else None}


def faces_for(objects: List, width: int, height: int, dx: float) -> list:
    """(object id, faces) for every SECTION, in world order."""
    out = []
    for obj in objects:
        if obj.type != SECTION_TYPE:
            continue
        out.append((obj.id, section_faces(obj.position, float(obj.rotation[1]),
                                          section_length(obj), width, height, dx)))
    return out
