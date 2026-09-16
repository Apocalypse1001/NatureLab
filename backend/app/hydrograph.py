"""Flood hydrograph Q(t) for the river inlet (v0.18.0, docs/07_river_plan.md).

A river in flood does not jump to its peak: the discharge rises over minutes or
hours, peaks, and falls back more slowly. The inlet's discharge slider stays the
BASE flow; when a hydrograph is on, the inlet is handed

    Q(t) = base                                         t < start
         = base + (peak - base) sin^2(pi/2 (t - start) / rise)       rising
         = base + (peak - base) cos^2(pi/2 (t - start - rise) / fall) falling
         = base                                         afterwards

every tick. sin^2 and cos^2 start and end with zero slope, so the inlet is never
kicked, and each averages 1/2 over its phase, so the flood's volume above the
base flow is exactly (peak - base)(rise + fall)/2 -- a number a gauging line at
the inlet can be checked against.

The times are simulation seconds, compressed the way everything on this map is:
a 200 m reach fills in minutes, so a flood that took a day would show nothing.
"""
from __future__ import annotations

import math
from typing import Any, Dict

from . import config

FIELDS = ("enabled", "peak_m3s", "start_s", "rise_s", "fall_s")


def discharge_at(t: float, base: float, peak: float, start: float,
                 rise: float, fall: float) -> float:
    tau = t - start
    if tau <= 0.0:
        return base
    if tau < rise:
        return base + (peak - base) * math.sin(0.5 * math.pi * tau / rise) ** 2
    tau -= rise
    if tau < fall:
        return base + (peak - base) * math.cos(0.5 * math.pi * tau / fall) ** 2
    return base


def flood_volume_m3(base: float, peak: float, rise: float, fall: float) -> float:
    """The volume above the base flow that one flood delivers."""
    return (peak - base) * (rise + fall) * 0.5


def validate(fields: Dict[str, Any]) -> Dict[str, Any]:
    """Check a partial hydrograph update; returns the values to store."""
    out: Dict[str, Any] = {}
    for key, value in fields.items():
        if key not in FIELDS:
            raise ValueError(f"unknown hydrograph field: {key!r}")
        if key == "enabled":
            out[key] = bool(value)
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(float(value)):
            raise ValueError(f"hydrograph.{key} must be a finite number")
        number = float(value)
        if key == "peak_m3s" and not 0.0 <= number <= config.INLET_MAX_DISCHARGE_M3S:
            raise ValueError("hydrograph.peak_m3s out of range")
        if key == "start_s" and not 0.0 <= number <= config.HYDROGRAPH_MAX_TIME_S:
            raise ValueError("hydrograph.start_s out of range")
        if key in ("rise_s", "fall_s") and not 1.0 <= number <= config.HYDROGRAPH_MAX_TIME_S:
            raise ValueError(f"hydrograph.{key} out of range")
        out[key] = number
    return out
