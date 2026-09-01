"""Reproducible Foundation 0.2 backend regression tests (stdlib unittest)."""
from __future__ import annotations

import asyncio
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import protocol  # noqa: E402
from app.simulation import SimulationManager  # noqa: E402
from app.world_state import WorldState  # noqa: E402


class Foundation02Tests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.manager = SimulationManager()
        self.manager.attach(self._text, self._binary)
        self.text_frames: list[str] = []
        self.binary_frames: list[bytes] = []

    async def asyncTearDown(self) -> None:
        self.manager.stop()
        await asyncio.sleep(0)

    async def _text(self, value: str) -> None:
        self.text_frames.append(value)

    async def _binary(self, value: bytes) -> None:
        self.binary_frames.append(value)

    async def test_start_is_idempotent(self) -> None:
        self.manager.apply_object_add({"type": "HOUSE", "position": [0, 0, 0]})
        self.manager.start()
        await asyncio.sleep(0.15)
        before_time = self.manager.sim_time
        before_initial = json.dumps(self.manager.initial.to_dict(), sort_keys=True)
        self.manager.start()
        self.assertGreaterEqual(self.manager.sim_time, before_time)
        self.assertEqual(json.dumps(self.manager.initial.to_dict(), sort_keys=True), before_initial)

    async def test_running_edit_sequence_and_reset(self) -> None:
        base = self.manager.apply_object_add({"type": "HOUSE", "position": [0, 0, 0]})
        self.manager.apply_water_level(2.0)
        self.manager.start()
        initial = json.dumps(self.manager.initial.to_dict(), sort_keys=True)
        car = self.manager.apply_object_add({"type": "CAR", "position": [2, 0, 0]})
        self.manager.apply_object_update(car["id"], {"position": [3, 0, 0],
                                                     "rotation": [0, 0.3, 0]})
        self.manager._step_once()
        self.manager.apply_object_remove(car["id"])
        tree = self.manager.apply_object_add({"type": "TREE", "position": [-2, 0, 0]})
        self.manager.pause()
        paused_time = self.manager.sim_time
        self.manager.start()
        self.assertEqual(self.manager.status, self.manager.RUNNING)
        self.assertEqual(json.dumps(self.manager.initial.to_dict(), sort_keys=True), initial)
        self.manager.reset()
        self.assertEqual(self.manager.sim_time, 0.0)
        self.assertEqual(set(self.manager.world.objects), {base["id"]})
        self.assertNotIn(tree["id"], self.manager.world.objects)
        self.assertGreater(paused_time, 0.0)

    async def test_strict_transform_validation(self) -> None:
        obj = self.manager.apply_object_add({"type": "BOX", "position": [0, 0, 0]})
        for bad in ([1, 2], [1, 2, 3, 4], [1, math.nan, 3], "xyz"):
            with self.assertRaises(ValueError):
                self.manager.apply_object_update(obj["id"], {"rotation": bad})
        self.manager.apply_object_update(obj["id"], {"rotation": [0.1, 0.2, 0.3]})
        self.assertEqual(self.manager.world.objects[obj["id"]].rotation, [0.1, 0.2, 0.3])
        with self.assertRaises(ValueError):
            WorldState.from_dict({"terrain": {}, "water": {}, "environment": {"wind": [1, 2]},
                                  "objects": []})

    async def test_terrain_checksum_and_protocol_validation(self) -> None:
        patch = self.manager.apply_terrain_brush(0, 0, 6, 0.4)
        self.assertEqual(patch["checksum"], self.manager.world.terrain.checksum())
        payload = protocol.encode_particles(np.zeros((150_001, 3), dtype=np.float32), 1.0)
        kind, count, _, values = protocol.decode_frame(payload)
        self.assertEqual(kind, protocol.FrameKind.PARTICLES)
        self.assertEqual(count, 150_001)
        self.assertEqual(values.shape, (150_001, 3))
        with self.assertRaises(ValueError):
            protocol.decode_frame(payload[:-1])
        corrupt = bytearray(payload)
        corrupt[3] = 255
        with self.assertRaises(ValueError):
            protocol.decode_frame(bytes(corrupt))


if __name__ == "__main__":
    unittest.main(verbosity=2)
