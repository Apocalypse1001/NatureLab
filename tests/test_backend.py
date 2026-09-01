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
from app.fluid_solver import ShallowWaterFluidSolver  # noqa: E402
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


class ShallowWaterSolverTests(unittest.TestCase):
    """v0.3: real FluidSolver replacing the flat-level placeholder.

    See docs/04_TZ_v0.3_roadmap.md milestone v0.3. These assert the causal
    quality bar from docs/01_vision.md ("Главный критерий качества"): moving
    an obstacle must change where the water goes, water must not appear from
    nowhere, and it must not exist inside solid objects.
    """

    @staticmethod
    def _make_world(slope: float) -> WorldState:
        world = WorldState()
        w, h = world.terrain.width, world.terrain.height
        xs = np.linspace(0.0, slope, w + 1, dtype=np.float32)
        world.terrain.heights[:, :] = np.tile(xs, (h + 1, 1))
        world.water.level = 1.0
        return world

    def test_conservation_no_water_from_nowhere(self) -> None:
        world = self._make_world(slope=2.0)
        solver = ShallowWaterFluidSolver()
        solver.initialize(world)
        solver.set_boundaries(world.terrain, {})
        start_volume = solver.total_volume()
        for _ in range(30):
            solver.advance(1 / 60, 8, 1 / 120)
        self.assertAlmostEqual(solver.total_volume(), start_volume,
                                delta=max(1e-6, start_volume * 1e-3))

    def test_no_negative_and_no_phantom_water_in_obstacle(self) -> None:
        world = self._make_world(slope=3.0)
        solver = ShallowWaterFluidSolver()
        solver.initialize(world)
        obstacle = {"positions": [[0.0, 0.0, 0.0]]}
        for _ in range(40):
            solver.set_boundaries(world.terrain, obstacle)
            solver.advance(1 / 60, 8, 1 / 120)
            self.assertGreaterEqual(float(solver._depth.min()), 0.0)
            self.assertTrue(bool((solver._depth[solver._obstacle_mask] == 0).all()))

    def test_moving_obstacle_changes_flow(self) -> None:
        def run(obstacle_positions: list) -> np.ndarray:
            world = self._make_world(slope=2.5)
            solver = ShallowWaterFluidSolver()
            solver.initialize(world)
            for _ in range(50):
                solver.set_boundaries(world.terrain,
                                      {"positions": obstacle_positions} if obstacle_positions else {})
                solver.advance(1 / 60, 8, 1 / 120)
            return solver._depth.copy()

        # Obstacle positions must sit in terrain that is actually wet at t=0
        # (terrain height < water level) or a short run never reaches them.
        depth_no_obstacle = run([])
        depth_with_obstacle = run([[-30.0, 0.0, 0.0]])
        depth_moved_obstacle = run([[-45.0, 0.0, 0.0]])

        self.assertFalse(np.allclose(depth_no_obstacle, depth_with_obstacle))
        self.assertFalse(np.allclose(depth_with_obstacle, depth_moved_obstacle))

    def test_body_senses_water_around_its_own_footprint(self) -> None:
        """A registered body carves a dry hole at its own position (no phantom
        water inside a solid). sample_for_bodies must not sample exactly that
        hole, or a body could never sense water depth to float."""
        world = self._make_world(slope=0.0)  # flat terrain, uniform depth
        solver = ShallowWaterFluidSolver()
        solver.initialize(world)
        position = np.array([[0.0, 0.0, 0.0]])
        solver.set_boundaries(world.terrain, {"positions": position})
        samples = solver.sample_for_bodies(position)
        self.assertGreater(float(samples["depths"][0]), 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
