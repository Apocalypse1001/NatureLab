"""Versioned bulk protocol for simulation data."""
from __future__ import annotations

import struct
from enum import IntEnum

import numpy as np

MAGIC = b"NL"
VERSION = 2
HEADER = struct.Struct("<2sBBIQ")  # magic, version, kind, item count, time ms


class FrameKind(IntEnum):
    PARTICLES = 0
    WATER_HEIGHT = 1
    VELOCITY_FIELD = 2
    OBJECT_TRANSFORMS = 3
    TERRAIN_PATCH = 4
    EVENTS = 5


_COMPONENTS = {
    FrameKind.PARTICLES: 3,
    FrameKind.WATER_HEIGHT: 1,
    FrameKind.VELOCITY_FIELD: 3,
    FrameKind.OBJECT_TRANSFORMS: 10,  # position3 + rotation3 + quaternion/scale4 scaffold
    FrameKind.TERRAIN_PATCH: 3,       # grid i, grid j, height
    FrameKind.EVENTS: 1,              # numeric event records; JSON remains fallback
}


def encode_float_frame(kind: FrameKind, values: np.ndarray, sim_time: float) -> bytes:
    components = _COMPONENTS[kind]
    array = np.ascontiguousarray(values, dtype="<f4")
    if array.size % components:
        raise ValueError(f"{kind.name} payload component mismatch")
    count = array.size // components
    return HEADER.pack(MAGIC, VERSION, int(kind), count, int(sim_time * 1000)) + array.tobytes()


def encode_particles(positions: np.ndarray, sim_time: float) -> bytes:
    return encode_float_frame(FrameKind.PARTICLES, positions, sim_time)


def encode_water_height(heights: np.ndarray, sim_time: float) -> bytes:
    """Encode absolute surface elevations in terrain-vertex row-major order."""
    return encode_float_frame(FrameKind.WATER_HEIGHT, heights, sim_time)


def decode_frame(payload: bytes) -> tuple[FrameKind, int, float, np.ndarray]:
    if len(payload) < HEADER.size:
        raise ValueError("truncated frame header")
    magic, version, raw_kind, count, t_ms = HEADER.unpack_from(payload)
    if magic != MAGIC:
        raise ValueError("bad frame magic")
    if version != VERSION:
        raise ValueError(f"unsupported protocol version: {version}")
    try:
        kind = FrameKind(raw_kind)
    except ValueError as exc:
        raise ValueError(f"unknown frame kind: {raw_kind}") from exc
    components = _COMPONENTS[kind]
    expected = HEADER.size + count * components * 4
    if len(payload) != expected:
        raise ValueError(f"invalid frame length: expected {expected}, got {len(payload)}")
    array = np.frombuffer(payload, dtype="<f4", count=count * components,
                          offset=HEADER.size).reshape(count, components)
    return kind, count, t_ms / 1000.0, array
