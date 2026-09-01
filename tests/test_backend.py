"""Reproducible NatureLab 0.5.0 physics and measurement regression tests."""
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

from app import config, protocol  # noqa: E402
from app.simulation import SimulationManager  # noqa: E402
from app.world_state import WorldState  # noqa: E402


class Physics04Tests(unittest.IsolatedAsyncioTestCase):
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

    async def test_warp_shallow_water_stability_and_stream(self) -> None:
        self.manager.apply_water_level(0.5)
        self.manager.start()
        initial = self.manager.fluid.diagnostics()
        self.assertEqual(initial["solver"], "warp_shallow_water")
        self.assertEqual(initial["grid"], [101, 101])
        self.assertGreater(initial["volume_m3"], 0.0)
        initial_depth = np.asarray(self.manager.fluid._h.numpy()).reshape(101, 101)
        self.assertGreater(float(initial_depth[:, :config.FLUID_SOURCE_COLUMNS].max()), 0.0)
        self.assertEqual(float(initial_depth[:, config.FLUID_SOURCE_COLUMNS:].max()), 0.0)

        upload_counts = (initial["terrain_gpu_uploads"], initial["obstacle_gpu_uploads"])
        for _ in range(600):
            self.manager._step_once()
        final = self.manager.fluid.diagnostics()
        self.assertGreater(final["volume_m3"], initial["volume_m3"])
        self.assertGreater(final["wet_cells"], initial["wet_cells"])
        self.assertGreaterEqual(final["substeps"], 1)
        self.assertGreater(final["cfl_dt"], 0.0)
        self.assertEqual((final["terrain_gpu_uploads"], final["obstacle_gpu_uploads"]),
                         upload_counts)

        surface = self.manager.fluid.get_water_height_field()
        velocity = self.manager.fluid.get_velocity_field()
        self.assertEqual(surface.shape, (101 * 101,))
        self.assertTrue(np.isfinite(surface).all())
        self.assertIsNotNone(velocity)
        self.assertTrue(np.isfinite(velocity).all())
        self.assertLess(float(np.max(np.abs(velocity[:, 2]))), 0.001)

        payload = protocol.encode_water_height(surface, self.manager.sim_time)
        kind, count, _, values = protocol.decode_frame(payload)
        self.assertEqual(kind, protocol.FrameKind.WATER_HEIGHT)
        self.assertEqual(count, 101 * 101)
        self.assertEqual(values.shape, (101 * 101, 1))

        await self.manager._stream()
        kinds = [protocol.decode_frame(frame)[0] for frame in self.binary_frames]
        self.assertIn(protocol.FrameKind.WATER_HEIGHT, kinds)
        self.assertIn(protocol.FrameKind.PARTICLES, kinds)

    async def test_closed_edge_slug_conserves_volume(self) -> None:
        self.manager.apply_water_level(0.5)
        self.manager.start()
        self.manager.fluid._source_enabled = False
        self.manager.fluid._measure()
        initial = self.manager.fluid.diagnostics()["volume_m3"]
        for _ in range(600):
            self.manager._step_once()
        self.manager.fluid._measure()
        final = self.manager.fluid.diagnostics()["volume_m3"]
        self.assertLess(abs(final - initial) / initial, 0.01)

    async def test_runtime_level_obstacle_footprint_and_flotation(self) -> None:
        self.manager.apply_water_level(1.5)
        self.manager.apply_object_add({"type": "HOUSE", "position": [-30, 0, 0]})
        car = self.manager.apply_object_add({"type": "CAR", "position": [-40, 0, 0]})
        car2 = self.manager.apply_object_add({"type": "CAR", "position": [-40, 0, 0]})
        self.manager.start()
        for _ in range(600):
            self.manager._step_once()

        self.assertGreaterEqual(int(np.count_nonzero(
            self.manager.fluid._obstacle_host)), 25)
        car_state = self.manager.world.objects[car["id"]]
        car2_state = self.manager.world.objects[car2["id"]]
        self.assertIn(car_state.state, ("MOVING", "FLOATING"))
        self.assertGreater(car_state.position[1], 0.0)
        separation = math.hypot(car_state.position[0] - car2_state.position[0],
                                car_state.position[2] - car2_state.position[2])
        self.assertGreaterEqual(separation, 4.5)

        before = self.manager.fluid.diagnostics()["volume_m3"]
        self.manager.apply_water_level(1.5)
        self.manager.fluid._measure()
        self.assertGreater(self.manager.fluid.diagnostics()["volume_m3"], before)
        self.manager._step_once()
        self.manager.fluid._measure()
        raised = self.manager.fluid.diagnostics()["volume_m3"]
        self.assertGreater(raised, before)
        self.manager.apply_water_level(0.25)
        self.manager._step_once()
        self.manager.fluid._measure()
        lowered = self.manager.fluid.diagnostics()["volume_m3"]
        self.assertLess(lowered, raised)

    async def test_edge_flow_moves_light_box(self) -> None:
        self.manager.apply_water_level(1.5)
        box = self.manager.apply_object_add({"type": "BOX", "position": [-18, 0, 0]})
        self.manager.start()
        for _ in range(600):
            self.manager._step_once()
        moved = self.manager.world.objects[box["id"]]
        self.assertEqual(moved.state, "FLOATING")
        self.assertGreater(moved.position[0], -17.5)

    async def test_rotated_house_uses_exact_obb_mask(self) -> None:
        house = self.manager.apply_object_add({"type": "HOUSE", "position": [0, 0, 0]})
        self.manager.start()
        self.manager._step_once()
        self.assertEqual(int(np.count_nonzero(self.manager.fluid._obstacle_host)), 25)
        self.manager.apply_object_update(house["id"], {"rotation": [0, math.pi / 4, 0]})
        self.manager._step_once()
        self.assertEqual(int(np.count_nonzero(self.manager.fluid._obstacle_host)), 13)

    async def test_footprint_sampling_respects_height_and_partial_wetting(self) -> None:
        self.manager.apply_water_level(0.0)
        self.manager.start()
        depth = np.zeros(101 * 101, dtype=np.float32)
        center = 50 * 101 + 50
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                if di or dj:
                    depth[center + dj * 101 + di] = 1.0
        self.manager.fluid._h.assign(depth)
        common = dict(body_velocities=np.zeros((1, 3), dtype=np.float32),
                      drag=np.ones(1, dtype=np.float32),
                      cross_area=np.ones(1, dtype=np.float32),
                      body_height=np.ones(1, dtype=np.float32),
                      rotations=np.zeros((1, 3), dtype=np.float32),
                      half_extents=np.ones((1, 2), dtype=np.float32))
        wet = self.manager.fluid.sample_for_bodies(
            np.asarray([[0, 0, 0]], dtype=np.float32), **common)
        wet_immersion = float(wet["immersions_device"].numpy()[0])
        elevated = self.manager.fluid.sample_for_bodies(
            np.asarray([[0, 2, 0]], dtype=np.float32), **common)
        self.assertGreater(wet_immersion, 0.8)
        self.assertEqual(float(elevated["immersions_device"].numpy()[0]), 0.0)

    async def test_buoyancy_schema_and_box_dimensions(self) -> None:
        migrated = WorldState.from_dict({
            "version": 1, "terrain": {}, "water": {}, "environment": {},
            "objects": [{"id": "Car_001", "type": "CAR", "buoyancy": 0.55}],
        })
        self.assertEqual(migrated.objects["Car_001"].buoyancy, 1.0)
        box = self.manager.apply_object_add({"type": "BOX", "position": [0, 0, 0]})
        self.assertAlmostEqual(box["volume_m3"], 1.2 ** 3, places=5)
        self.assertAlmostEqual(box["ground_contact_area"], 1.2 ** 2, places=5)
        with self.assertRaises(ValueError):
            self.manager.apply_object_update(box["id"], {"buoyancy": 1.1})

    async def test_gauge_point_sample_is_non_obstructing(self) -> None:
        gauge = self.manager.apply_object_add({"type": "GAUGE", "position": [0, 0, 0]})
        box = self.manager.apply_object_add({"type": "BOX", "position": [0, 0, 0]})
        self.manager.start()
        self.assertTrue(gauge["is_static"])
        self.assertEqual(int(np.count_nonzero(self.manager.fluid._obstacle_host)), 0)

        count = 101 * 101
        depth = np.zeros(count, dtype=np.float32)
        u = np.zeros(count, dtype=np.float32)
        v = np.zeros(count, dtype=np.float32)
        center = 50 * 101 + 50
        depth[center], u[center], v[center] = 1.0, 3.0, 4.0
        self.manager.fluid._h.assign(depth)
        self.manager.fluid._u.assign(u)
        self.manager.fluid._v.assign(v)
        samples = self.manager.fluid.sample_for_bodies(
            self.manager.rigid.buffer.positions, self.manager.rigid.buffer.velocities,
            self.manager.rigid.buffer.drag_coefficients, self.manager.rigid.buffer.cross_areas,
            self.manager.rigid.buffer.body_heights, self.manager.rigid.buffer.rotations,
            self.manager.rigid.buffer.half_extents)
        gauge_index = self.manager.rigid.buffer.index[gauge["id"]]
        self.assertAlmostEqual(float(samples["depths_device"].numpy()[gauge_index]), 1.0)
        velocity = samples["velocities_device"].numpy()[gauge_index]
        self.assertTrue(np.allclose(velocity, [3.0, 0.0, 4.0]))

        before = self.manager.rigid.buffer.positions.copy()
        dynamic = np.zeros(len(before), dtype=np.bool_)
        dynamic[self.manager.rigid.buffer.index[box["id"]]] = True
        self.manager.rigid._resolve_collisions(dynamic)
        self.assertTrue(np.array_equal(self.manager.rigid.buffer.positions, before))

    async def test_gauge_arrival_history_and_stream_schema(self) -> None:
        self.manager.apply_water_level(1.5)
        gauge = self.manager.apply_object_add({"type": "GAUGE", "position": [-45, 0, 0]})
        self.manager.start()
        for _ in range(4000):
            self.manager._step_once()
        runtime = self.manager._gauges[gauge["id"]]
        self.assertIsNotNone(runtime.arrival_time)
        self.assertEqual(len(runtime.history), config.GAUGE_HISTORY_CAPACITY)
        arrivals = [event for event in self.manager.events.all()
                    if event["type"] == "WATER_ENTERED_AREA"
                    and event["object_id"] == gauge["id"]]
        self.assertEqual(len(arrivals), 1)
        self.assertGreater(runtime.latest["water_depth_m"], 0.0)
        self.assertGreater(runtime.latest["speed_m_s"], 0.0)

        await self.manager._stream()
        state = json.loads(self.text_frames[-1])
        self.assertEqual(state["gauge_history_capacity"], config.GAUGE_HISTORY_CAPACITY)
        self.assertEqual(state["gauges"][0]["id"], gauge["id"])
        self.assertGreater(len(state["gauges"][0]["samples"]), 0)
        await self.manager._stream()
        state = json.loads(self.text_frames[-1])
        self.assertEqual(state["gauges"][0]["samples"], [])

        serialized = self.manager.world.to_dict()
        self.assertNotIn("arrival_time", serialized["objects"][0])
        self.manager.apply_object_update(gauge["id"], {"position": [20, 0, 0]})
        self.assertIsNone(self.manager._gauges[gauge["id"]].arrival_time)
        self.assertEqual(len(self.manager._gauges[gauge["id"]].history), 0)

    async def test_high_terrain_ridge_blocks_flux(self) -> None:
        # A ridge above the free surface spans the world and cannot be crossed.
        ridge_i = 10  # world x=-40 m, downstream of the edge inflow
        self.manager.world.terrain.heights[:, ridge_i] = 3.0
        self.manager.apply_water_level(0.5)
        self.manager.start()
        for _ in range(300):
            self.manager._step_once()
        depth = np.asarray(self.manager.fluid._h.numpy(), dtype=np.float32).reshape(101, 101)
        self.assertGreater(float(depth[:, ridge_i - 1].max()), 0.01)
        self.assertEqual(float(depth[:, ridge_i + 1:].max()), 0.0)

    async def test_house_footprint_diverts_water(self) -> None:
        self.manager.apply_object_add({"type": "HOUSE", "position": [-40, 0, 0]})
        self.manager.start()
        for _ in range(300):
            self.manager._step_once()
        depth = np.asarray(self.manager.fluid._h.numpy(), dtype=np.float32).reshape(101, 101)
        self.assertEqual(int(np.count_nonzero(self.manager.fluid._obstacle_host)), 25)
        self.assertEqual(float(depth[50, 10]), 0.0)  # house center
        self.assertGreater(float(depth[56, 10]), 0.01)  # flow around its side
        self.assertEqual(float(depth[50, 13]), 0.0)  # no direct through-flow

    async def test_obstacle_move_remove_has_no_phantom_water(self) -> None:
        house = self.manager.apply_object_add({"type": "HOUSE", "position": [-30, 0, 0]})
        self.manager.start()
        solid = self.manager.fluid._obstacle_host != 0
        depth = np.asarray(self.manager.fluid._h.numpy(), dtype=np.float32)
        self.assertEqual(float(depth[solid].max()), 0.0)
        before = self.manager.fluid.diagnostics()["volume_m3"]

        self.manager.apply_object_update(house["id"], {"position": [-25, 0, 0]})
        self.manager._obstacle_snapshot = self.manager.rigid.obstacle_snapshot()
        self.manager.fluid.set_boundaries(self.manager.world.terrain,
            self.manager._obstacle_snapshot, self.manager.terrain_revision,
            self.manager.obstacle_revision)
        self.manager.fluid._measure()
        moved = self.manager.fluid.diagnostics()["volume_m3"]
        self.assertLess(abs(moved - before) / before, 0.001)
        depth = np.asarray(self.manager.fluid._h.numpy(), dtype=np.float32)
        self.assertEqual(float(depth[self.manager.fluid._obstacle_host != 0].max()), 0.0)

        old_solid = self.manager.fluid._obstacle_host.copy()
        self.manager.apply_object_remove(house["id"])
        self.manager._obstacle_snapshot = self.manager.rigid.obstacle_snapshot()
        self.manager.fluid.set_boundaries(self.manager.world.terrain,
            self.manager._obstacle_snapshot, self.manager.terrain_revision,
            self.manager.obstacle_revision)
        self.manager.fluid._measure()
        removed = self.manager.fluid.diagnostics()["volume_m3"]
        self.assertLess(abs(removed - moved) / moved, 0.001)
        depth = np.asarray(self.manager.fluid._h.numpy(), dtype=np.float32)
        self.assertEqual(float(depth[old_solid != 0].max()), 0.0)

    async def test_lake_at_rest_and_adaptive_cfl(self) -> None:
        self.manager.start()
        count = 101 * 101
        self.manager.fluid._h.assign(np.full(count, 0.5, dtype=np.float32))
        self.manager.fluid._u.assign(np.zeros(count, dtype=np.float32))
        self.manager.fluid._v.assign(np.zeros(count, dtype=np.float32))
        for _ in range(120):
            self.manager.fluid.advance(1 / 60, 8, 1 / 120)
        h, u, v = self.manager.fluid._host_fields()
        self.assertLess(float(np.max(np.abs(h - 0.5))), 1.0e-6)
        self.assertEqual(float(np.max(np.abs(u))), 0.0)
        self.assertEqual(float(np.max(np.abs(v))), 0.0)

        self.manager.fluid._h.assign(np.full(count, 30.0, dtype=np.float32))
        self.manager.fluid._u.assign(np.full(count, 20.0, dtype=np.float32))
        substeps = self.manager.fluid.advance(1 / 60, 8, 1 / 120)
        self.assertGreater(substeps, 1)
        self.assertLess(self.manager.fluid.diagnostics()["cfl_dt"], 1 / 60)

    async def test_heavy_vs_light_and_zero_flow(self) -> None:
        box = self.manager.apply_object_add({"type": "BOX", "position": [0, 0, -10]})
        car = self.manager.apply_object_add({"type": "CAR", "position": [0, 0, 10]})
        self.manager.rigid.initialize(self.manager.world, self.manager.fluid,
                                      self.manager.events)
        samples = {"depths": np.array([0.3, 0.3], dtype=np.float32),
                   "velocities": np.tile([1.0, 0.0, 0.0], (2, 1)).astype(np.float32),
                   "forces": np.tile([1000.0, 0.0, 0.0], (2, 1)).astype(np.float32)}
        for step in range(60):
            self.manager.rigid.step(1 / 60, step / 60, samples)
        box_state = self.manager.world.objects[box["id"]]
        car_state = self.manager.world.objects[car["id"]]
        self.assertEqual(box_state.state, "FLOATING")
        self.assertEqual(car_state.state, "INTACT")
        self.assertGreater(box_state.position[0], car_state.position[0])

        still = SimulationManager()
        try:
            item = still.apply_object_add({"type": "BOX", "position": [2, 0, 3]})
            still.rigid.initialize(still.world, still.fluid, still.events)
            zero = {"depths": np.zeros(1, dtype=np.float32),
                    "velocities": np.zeros((1, 3), dtype=np.float32),
                    "forces": np.zeros((1, 3), dtype=np.float32)}
            initial = list(still.world.objects[item["id"]].position)
            for step in range(120):
                still.rigid.step(1 / 60, step / 60, zero)
            self.assertEqual(still.world.objects[item["id"]].position, initial)
            self.assertEqual(still.world.objects[item["id"]].state, "INTACT")
        finally:
            still.stop()

    async def test_running_terrain_edit_is_rejected(self) -> None:
        self.manager.start()
        with self.assertRaisesRegex(ValueError, "disabled"):
            self.manager.apply_terrain_brush(0, 0, 5, 0.2)

    async def test_dry_runtime_edge_source_creates_flow(self) -> None:
        self.manager.apply_water_level(0.0)
        self.manager.start()
        self.assertEqual(self.manager.fluid.diagnostics()["volume_m3"], 0.0)
        self.manager.apply_water_level(1.5)
        for _ in range(120):
            self.manager._step_once()
        self.manager.fluid._measure()
        diagnostics = self.manager.fluid.diagnostics()
        self.assertGreater(diagnostics["max_velocity"], 0.1)
        self.assertGreater(diagnostics["wet_cells"], config.FLUID_SOURCE_COLUMNS * 101)
        self.assertLess(diagnostics["wet_cells"], 15 * 101)
        depth = np.asarray(self.manager.fluid._h.numpy(), dtype=np.float32).reshape(101, 101)
        self.assertGreater(float(depth[:, config.FLUID_SOURCE_COLUMNS:].max()), 0.0)
        tracers_before = self.manager.fluid.get_flow_particles().copy()
        for _ in range(30):
            self.manager._step_once()
        tracers_after = self.manager.fluid.get_flow_particles()
        self.assertEqual(tracers_after.shape, (config.FLOW_TRACER_COUNT, 3))
        self.assertGreater(float(np.max(np.abs(tracers_after - tracers_before))), 1.0e-4)

    async def test_deterministic_replay(self) -> None:
        self.manager.start()
        other = type(self.manager.fluid)(self.manager.engine.device)
        other.initialize(WorldState())
        other.set_boundaries(WorldState().terrain, {}, 0, 0)
        for _ in range(120):
            self.manager.fluid.advance(1 / 60, 8, 1 / 120)
            other.advance(1 / 60, 8, 1 / 120)
        self.assertTrue(np.allclose(self.manager.fluid._h.numpy(), other._h.numpy(),
                                    rtol=0.0, atol=1.0e-6))


if __name__ == "__main__":
    unittest.main(verbosity=2)
