"""Reproducible NatureLab physics and measurement regression tests."""
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
from app.compute_engine import create_engine  # noqa: E402
from app.fluid_solver import create_fluid_solver  # noqa: E402
from app.terrain_gen import dam_ridge, river_valley  # noqa: E402

# Grid geometry is derived, never hard-coded: these tests used to spell out
# "101" and column literals like [col(0.0), 8], which silently became wrong the first
# time the world size changed. Everything below is expressed in world metres and
# converted here, so a map resize is a config edit and not a test rewrite.
N = config.TERRAIN_CELLS + 1


def col(x_m: float) -> int:
    """Grid column for a world x coordinate (rows use the same mapping for z)."""
    return int(round(x_m / config.TERRAIN_CELL_SIZE + config.TERRAIN_CELLS / 2))


def x_at(column: int) -> float:
    """World x for a grid column -- lets a test say "8 cells in from the inlet"
    and keep meaning that at any map size."""
    return (column - config.TERRAIN_CELLS / 2) * config.TERRAIN_CELL_SIZE



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
        self.assertEqual(initial["grid"], [N, N])
        self.assertGreater(initial["volume_m3"], 0.0)
        initial_depth = np.asarray(self.manager.fluid._h.numpy()).reshape(N, N)
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
        self.assertEqual(surface.shape, (N * N,))
        self.assertTrue(np.isfinite(surface).all())
        self.assertIsNotNone(velocity)
        self.assertTrue(np.isfinite(velocity).all())
        self.assertLess(float(np.max(np.abs(velocity[:, 2]))), 0.001)

        payload = protocol.encode_water_height(surface, self.manager.sim_time)
        kind, count, _, values = protocol.decode_frame(payload)
        self.assertEqual(kind, protocol.FrameKind.WATER_HEIGHT)
        self.assertEqual(count, N * N)
        self.assertEqual(values.shape, (N * N, 1))

        await self.manager._stream()
        kinds = [protocol.decode_frame(frame)[0] for frame in self.binary_frames]
        self.assertIn(protocol.FrameKind.WATER_HEIGHT, kinds)
        self.assertIn(protocol.FrameKind.PARTICLES, kinds)

    async def test_closed_edge_slug_conserves_volume(self) -> None:
        self.manager.apply_water_level(0.5)
        # "closed edge" is the point of this test: v0.8.0 opens the downstream
        # edge by default, and an open outlet legitimately loses volume
        self.manager.apply_water_outflow(False)
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
        self.manager.apply_object_add({"type": "HOUSE", "position": [x_at(20), 0, 0]})
        car = self.manager.apply_object_add({"type": "CAR", "position": [x_at(10), 0, 0]})
        car2 = self.manager.apply_object_add({"type": "CAR", "position": [x_at(10), 0, 0]})
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

        # Raising the level while RUNNING must put water in immediately, not at
        # the next tick. This used to re-apply the level it already had (1.5)
        # and pass on the diagnostics being one frame stale -- it was reading
        # "the map is still filling", not "the control did something"; since
        # v0.12.0 the diagnostics describe the frame that just ran, so the
        # assertion needs a real change to measure.
        before = self.manager.fluid.diagnostics()["volume_m3"]
        self.manager.apply_water_level(2.5)
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

    async def test_a_floating_body_does_not_flicker_out_of_floating(self) -> None:
        """A body afloat must not re-decide its buoyancy from the height that
        buoyancy itself put it at.

        Immersion measured from the body's own y is a tautology for anything
        already floating: it is held at `surface - draft`, so its immersion is
        exactly its draft and the resulting buoyancy equals its weight to
        0.000% (CAR: 14715.0 N against 14715.0 N -- draft/height reduces to
        weight/max_buoyancy for every body, by algebra, not by tuning). Such a
        body is one wave away from sinking, and once v0.11.0 gave the water
        depth-dependent friction and therefore real waves, cars began flickering:
        307 FLOATING/MOVING/SETTLED transitions in 406 frames. Measured from the
        support instead -- the bed, or a bridge deck -- the question stops
        depending on its own last answer, and the count is 0.
        """
        self.manager.apply_water_level(1.5)
        car = self.manager.apply_object_add({"type": "CAR", "position": [x_at(10), 0, 0]})
        self.manager.start()
        states: list[str] = []
        for _ in range(600):
            self.manager._step_once()
            states.append(self.manager.world.objects[car["id"]].state)
        self.assertIn("FLOATING", states, "the car never floated in 1.5 m of water")
        afloat = states[states.index("FLOATING"):]
        flips = sum(1 for a, b in zip(afloat, afloat[1:]) if a != b)
        self.assertEqual(flips, 0,
                         f"the car flickered out of FLOATING {flips} times: "
                         f"{sorted(set(afloat))}")

    async def test_edge_flow_moves_light_box(self) -> None:
        self.manager.apply_water_level(1.5)
        start_x = x_at(32)
        box = self.manager.apply_object_add({"type": "BOX", "position": [start_x, 0, 0]})
        self.manager.start()
        for _ in range(600):
            self.manager._step_once()
        moved = self.manager.world.objects[box["id"]]
        self.assertEqual(moved.state, "FLOATING")
        # downstream of where it was put -- stated as a displacement rather than
        # an absolute x, which silently stopped meaning anything when the map
        # was resized in v0.7.0
        self.assertGreater(moved.position[0], start_x + 0.5)

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
        depth = np.zeros(N * N, dtype=np.float32)
        center = col(0.0) * N + col(0.0)
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                if di or dj:
                    depth[center + dj * N + di] = 1.0
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

        count = N * N
        depth = np.zeros(count, dtype=np.float32)
        u = np.zeros(count, dtype=np.float32)
        v = np.zeros(count, dtype=np.float32)
        center = col(0.0) * N + col(0.0)
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
        ridge_i = 10  # ten cells downstream of the edge inflow, at any map size
        self.manager.world.terrain.heights[:, ridge_i] = 3.0
        self.manager.apply_water_level(0.5)
        self.manager.start()
        for _ in range(300):
            self.manager._step_once()
        depth = np.asarray(self.manager.fluid._h.numpy(), dtype=np.float32).reshape(N, N)
        self.assertGreater(float(depth[:, ridge_i - 1].max()), 0.01)
        self.assertEqual(float(depth[:, ridge_i + 1:].max()), 0.0)

    async def test_house_footprint_diverts_water(self) -> None:
        self.manager.apply_object_add({"type": "HOUSE", "position": [x_at(10), 0, 0]})
        self.manager.start()
        for _ in range(300):
            self.manager._step_once()
        depth = np.asarray(self.manager.fluid._h.numpy(), dtype=np.float32).reshape(N, N)
        self.assertEqual(int(np.count_nonzero(self.manager.fluid._obstacle_host)), 25)
        self.assertEqual(float(depth[col(0.0), 10]), 0.0)  # house center
        self.assertGreater(float(depth[col(0.0) + 6, 10]), 0.01)  # flow around its side
        self.assertEqual(float(depth[col(0.0), 13]), 0.0)  # no direct through-flow

    async def test_obstacle_move_remove_has_no_phantom_water(self) -> None:
        house = self.manager.apply_object_add({"type": "HOUSE", "position": [x_at(20), 0, 0]})
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
        count = N * N
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
        self.assertGreater(diagnostics["wet_cells"], config.FLUID_SOURCE_COLUMNS * N)
        self.assertLess(diagnostics["wet_cells"], 15 * N)
        depth = np.asarray(self.manager.fluid._h.numpy(), dtype=np.float32).reshape(N, N)
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


class RiverLabTests(unittest.IsolatedAsyncioTestCase):
    """v0.6.0: sediment transport, erosion/deposition, and ROCK as riverbed.

    Every test here follows the project's own acceptance rule (docs/04, section
    2, item 4): change a cause and show the effect changes. A test that only
    asserts "the solver did not crash" is not evidence of physics.
    """

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

    def _bed(self) -> np.ndarray:
        return self.manager.fluid.get_terrain_heights().reshape(N, N)

    def _depth(self) -> np.ndarray:
        return np.asarray(self.manager.fluid._h.numpy(), dtype=np.float32).reshape(N, N)

    def _run(self, steps: int) -> None:
        for _ in range(steps):
            self.manager._step_once()

    async def test_erosion_off_leaves_the_bed_exactly_as_built(self) -> None:
        """The control for every other test here: with the toggle off, the
        terrain the user built is bit-for-bit the terrain they get back, even
        with water running over it for ten seconds."""
        before = self.manager.world.terrain.heights.copy()
        self.manager.apply_water_level(1.0)
        self.manager.start()
        self._run(600)
        self.assertFalse(self.manager.world.water.erosion_enabled)
        np.testing.assert_array_equal(self._bed(), before)

    async def test_flowing_water_erodes_its_bed_and_carries_the_load(self) -> None:
        """Cause: erosion on. Effect: the bed under the current is cut down and
        the removed material is in suspension, not deleted."""
        self.manager.apply_water_level(1.0)
        self.manager.apply_water_erosion(True)
        self.manager.start()
        before = self._bed().copy()
        self._run(600)
        after = self._bed()
        wet = self._depth() > 0.05
        self.assertTrue(wet.any(), "no water reached the bed at all")
        self.assertLess(float(after[wet].mean()), float(before[wet].mean()) - 1e-4,
                        "wet bed was not eroded")
        # dry ground must never erode -- there is no water there to do it
        dry = self._depth() <= config.FLUID_DRY_DEPTH
        np.testing.assert_allclose(after[dry], before[dry], atol=1e-6)
        self.assertGreater(self.manager.fluid.diagnostics()["suspended_sediment"], 0.0)

    async def test_faster_water_erodes_more_than_slower_water(self) -> None:
        """The capacity law is capacity = k*|u|*h, so the same bed under a
        deeper, faster inflow must lose more material. One cause changed: the
        inflow height."""
        losses = []
        for level in (0.6, 2.0):
            manager = SimulationManager()
            manager.apply_water_level(level)
            manager.apply_water_erosion(True)
            manager.start()
            before = manager.fluid.get_terrain_heights().copy()
            for _ in range(600):
                manager._step_once()
            after = manager.fluid.get_terrain_heights()
            losses.append(float(np.sum(before - after)))
            manager.stop()
        self.assertGreater(losses[1], losses[0] * 1.2,
                           f"deeper/faster flow did not erode more: {losses}")

    async def test_rock_is_bed_not_wall_and_deflects_the_current_sideways(self) -> None:
        """A boulder must never enter the solid mask (that is a wall -- correct
        for a house, wrong for a rock). It shows up as a raised bed instead, and
        the proof it does something is lateral velocity, which is exactly zero
        in the same straight channel without it."""
        self.manager.apply_water_level(1.0)
        self.manager.start()
        self._run(600)
        straight = float(np.abs(np.asarray(self.manager.fluid._v.numpy())).max())

        rocky = SimulationManager()
        rocky.apply_object_add({"type": "ROCK", "position": [x_at(8), 0.0, 0.0]})
        rocky.apply_water_level(1.0)
        rocky.start()
        for _ in range(600):
            rocky._step_once()
        deflected = float(np.abs(np.asarray(rocky.fluid._v.numpy())).max())
        solid_cells = int(np.count_nonzero(rocky.fluid._obstacle_host))
        bed_rise = float(rocky.fluid._bed_offset_host.max())
        rocky.stop()

        self.assertEqual(solid_cells, 0, "ROCK was rasterized as a solid wall")
        self.assertGreater(bed_rise, 0.5, "ROCK did not raise the bed")
        self.assertEqual(straight, 0.0, "the straight channel already had lateral flow")
        self.assertGreater(deflected, 1e-3, "ROCK did not deflect the current")

    async def test_taller_rock_deflects_more_than_a_flatter_one(self) -> None:
        """Scale Y is already the "how much of the channel does this block"
        control: same footprint, different height, measurably different
        deflection. No extra UI knob, deliberately."""
        deflection = []
        for scale_y in (0.3, 1.0):
            manager = SimulationManager()
            rock = manager.apply_object_add({"type": "ROCK", "position": [x_at(8), 0.0, 0.0]})
            manager.apply_object_update(rock["id"], {"scale": [1.0, scale_y, 1.0]})
            manager.apply_water_level(1.0)
            manager.start()
            for _ in range(600):
                manager._step_once()
            deflection.append(float(np.abs(np.asarray(manager.fluid._v.numpy())).max()))
            manager.stop()
        self.assertGreater(deflection[1], deflection[0] * 1.2,
                           f"the taller rock did not deflect more: {deflection}")

    async def test_deep_water_flows_over_a_rock_but_never_over_a_house(self) -> None:
        """The whole reason a boulder is bed and not wall: submerge it deeply
        enough and the river runs over the top of it. A house never lets water
        through, at any depth."""
        rocky = SimulationManager()
        rocky.apply_object_add({"type": "ROCK", "position": [x_at(8), 0.0, 0.0]})
        rocky.apply_water_level(4.0)
        rocky.start()
        for _ in range(600):
            rocky._step_once()
        over_rock = float(np.asarray(rocky.fluid._h.numpy()).reshape(N, N)[col(0.0), 8])
        rocky.stop()

        walled = SimulationManager()
        walled.apply_object_add({"type": "HOUSE", "position": [x_at(8), 0.0, 0.0]})
        walled.apply_water_level(4.0)
        walled.start()
        for _ in range(600):
            walled._step_once()
        over_house = float(np.asarray(walled.fluid._h.numpy()).reshape(N, N)[col(0.0), 8])
        walled.stop()

        self.assertGreater(over_rock, 0.1, "deep water did not pass over the rock")
        self.assertEqual(over_house, 0.0, "water entered a solid house")

    async def test_moving_a_rock_leaves_no_crater_behind(self) -> None:
        """The dome is recomputed from live positions and never written into the
        world's terrain, so relocating a boulder must leave no permanent mound
        where it used to be."""
        rock = self.manager.apply_object_add({"type": "ROCK", "position": [x_at(8), 0.0, 0.0]})
        self.manager.apply_water_level(1.0)
        self.manager.start()
        self._run(120)
        raised_before = float(self.manager.fluid._bed_offset_host.reshape(N, N)[col(0.0), 8])
        self.manager.apply_object_update(rock["id"], {"position": [x_at(col(20.0)), 0.0, 0.0]})
        self._run(5)
        offset = self.manager.fluid._bed_offset_host.reshape(N, N)
        self.assertGreater(raised_before, 0.1)
        self.assertLess(float(offset[col(0.0), 8]), 1e-6, "the rock left a mound behind it")
        self.assertGreater(float(offset[col(0.0), col(20.0)]), 0.1,
                           "the rock did not raise its new bed")
        # and the world's own terrain was never touched by the dome at all
        self.assertEqual(float(self.manager.world.terrain.heights[col(0.0), 8]), 0.0)

    async def test_the_river_does_not_dig_out_from_under_a_boulder(self) -> None:
        """BED_EROSION_SHIELD: a boulder is bedrock. Without this the rock digs
        a symmetric pit under itself, and the flank/lee asymmetry that actually
        moves a channel is swamped by it."""
        self.manager.apply_object_add({"type": "ROCK", "position": [x_at(8), 0.0, 0.0]})
        self.manager.apply_water_level(1.5)
        self.manager.apply_water_erosion(True)
        self.manager.start()
        before = self._bed().copy()
        self._run(900)
        after = self._bed()
        shielded = self.manager.fluid._bed_offset_host.reshape(N, N) > config.BED_EROSION_SHIELD
        self.assertTrue(shielded.any())
        # The shield blocks erosion only. Sediment settling ON a boulder is real
        # and is deliberately still allowed, so the honest assertion is that the
        # bed under a rock never goes DOWN -- not that it never changes.
        self.assertGreaterEqual(float((after[shielded] - before[shielded]).min()), -1e-6,
                                "the river dug out from under the boulder")
        # meanwhile the unshielded channel around it did erode, or this proves nothing
        open_bed = (~shielded) & (self._depth() > 0.05)
        self.assertLess(float(after[open_bed].mean()), float(before[open_bed].mean()))

    async def test_erosion_streams_the_new_terrain_to_the_frontend(self) -> None:
        """The architectural principle from docs/04: physical state that exists
        on the backend must be visible on the frontend in the same commit. A
        correct solver whose result never reaches the screen is, to the user,
        indistinguishable from no physics at all."""
        self.manager.apply_water_level(1.5)
        self.manager.apply_water_erosion(True)
        self.manager.start()
        self._run(int(config.TERRAIN_RESYNC_INTERVAL_S * 60) + 30)
        await self.manager._stream()
        patches = [json.loads(f) for f in self.text_frames]
        terrain = [p for p in patches if p.get("type") == "terrain_patch"]
        self.assertTrue(terrain, "eroded terrain was never streamed")
        self.assertEqual(terrain[-1]["checksum"], self.manager.world.terrain.checksum())
        self.assertEqual(len(terrain[-1]["heights"]), N * N)

    async def test_erosion_does_not_re_upload_terrain_to_the_gpu(self) -> None:
        """Erosion mutates the bed on the GPU every tick, so it must NOT go
        through the host revision path -- that would re-upload the whole grid 60
        times a second and stomp the live GPU state with a stale host copy."""
        self.manager.apply_water_level(1.5)
        self.manager.apply_water_erosion(True)
        self.manager.start()
        uploads = self.manager.fluid.terrain_gpu_uploads
        before = self._bed().copy()
        self._run(600)
        self.assertEqual(self.manager.fluid.terrain_gpu_uploads, uploads)
        self.assertFalse(np.array_equal(self._bed(), before),
                         "nothing eroded, so this test proved nothing")

    async def test_a_rock_changes_where_the_river_cuts_its_bed(self) -> None:
        """The headline RiverLab acceptance test, in the River A vs River B form
        docs/01_vision.md asks for: two identical channels, one boulder the only
        difference, and the beds must end up measurably different. This is the
        closed loop the vision names -- flow -> erosion -> terrain -> flow --
        not just "a rock deflects water"."""
        def cut(with_rock: bool) -> np.ndarray:
            manager = SimulationManager()
            if with_rock:
                manager.apply_object_add({"type": "ROCK", "position": [x_at(8), 0.0, 0.0]})
            manager.apply_water_level(1.5)
            manager.apply_water_erosion(True)
            manager.start()
            before = manager.fluid.get_terrain_heights().copy()
            for _ in range(1200):
                manager._step_once()
            after = manager.fluid.get_terrain_heights()
            manager.stop()
            return (after - before).reshape(N, N)

        river_a = cut(False)
        river_b = cut(True)
        self.assertLess(float(river_a.min()), -0.01, "River A did not erode at all")
        difference = np.abs(river_b - river_a)
        near = difference[:, 6:20].max()
        far = difference[:, N - 40:].max()
        # Measured by the size of the change and by where it sits -- deliberately
        # NOT by counting cells that differ. That count is dominated by
        # SEDIMENT_MAX_BED_CHANGE, which both rivers hit: at 0.02 m/s the bed can
        # move at most 0.400 m in these 1200 steps and River A already moves
        # 0.3989 m of it, on this law and on the one before it. Wherever both
        # rivers are pegged at the same ceiling they cannot differ, so widening
        # the saturated region (v0.11.0's friction made deep water faster, and the
        # eroding area went 5628 -> 14472 cells) collapses the count -- 90 -> 17 --
        # while the boulder's actual effect GREW, 0.314 -> 0.366 m. Counting cells
        # was measuring the limiter. See docs/06_next_steps.md: the erosion
        # calibration is owed a rederivation against this friction law.
        self.assertGreater(float(near), 0.1,
                           f"the boulder barely changed the bed: {near:.3f} m")
        self.assertGreater(float(difference.sum()), 1.0,
                           f"the two beds are all but identical: {difference.sum():.3f}")
        self.assertGreater(float(near), float(far) * 3.0,
                           f"bed change was not localised to the rock: {near} vs {far}")

    async def test_rock_survives_a_save_load_round_trip(self) -> None:
        """bed_height reaches metadata via default_properties rather than a
        hand-written key list, and a world saved before the property existed
        backfills from the type defaults -- otherwise an older ROCK silently
        comes back as a flat patch of riverbed."""
        self.manager.apply_object_add({"type": "ROCK", "position": [x_at(40), 0.0, 4.0]})
        self.manager.save("rocktest")
        self.manager.load("rocktest")
        rock = next(o for o in self.manager.world.objects.values() if o.type == "ROCK")
        self.assertGreater(float(rock.metadata["bed_height"]), 0.0)

        legacy = WorldState.from_dict({
            "terrain": {"width": config.TERRAIN_CELLS,
                        "height": config.TERRAIN_CELLS,
                        "cell_size": config.TERRAIN_CELL_SIZE,
                        "heights": self.manager.world.terrain.to_list()},
            "water": {"level": 0.5, "visible": True},
            "environment": self.manager.world.environment.to_dict(),
            "objects": [{"id": "Rock_009", "type": "ROCK", "position": [0, 0, 0],
                         "rotation": [0, 0, 0], "scale": [1, 1, 1], "metadata": {}}],
        })
        self.assertGreater(
            float(legacy.objects["Rock_009"].metadata["bed_height"]), 0.0,
            "a pre-v0.6.0 ROCK came back with no bed height")


class WaterControlTests(unittest.IsolatedAsyncioTestCase):
    """v0.8.0: an open downstream edge, a placeable SOURCE, and a DRAIN whose
    vortex comes from measured circulation rather than a chosen direction."""

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

    def _depth(self) -> np.ndarray:
        return np.asarray(self.manager.fluid._h.numpy(), dtype=np.float32).reshape(N, N)

    def _volume(self) -> float:
        self.manager.fluid._measure()
        return float(self.manager.fluid.diagnostics()["volume_m3"])

    @staticmethod
    def _sloped_world(manager, grade: float = 0.02) -> None:
        """A bed that falls west to east, so water actually runs downhill."""
        xs = np.arange(N, dtype=np.float32)
        manager.world.terrain.heights = np.tile(grade * (N - 1 - xs), (N, 1)).astype(np.float32)

    # -------------------------------------------------------------- outflow
    async def test_water_leaves_through_the_open_edge_and_not_through_a_closed_one(self) -> None:
        """The defect this fixes: every outer face was no-flux, so water that
        entered could never leave and the map filled forever. Measured before
        the fix: 2290 -> 3484 -> 4311 m3, never decreasing."""
        results = {}
        for enabled in (False, True):
            manager = SimulationManager()
            self._sloped_world(manager)
            manager.apply_water_outflow(enabled)
            manager.start()
            manager.fluid._source_enabled = False
            manager.fluid._h.assign(np.full(manager.fluid._count, 1.0, dtype=np.float32))
            manager.fluid._measure()
            before = float(manager.fluid.diagnostics()["volume_m3"])
            for _ in range(1200):
                manager._step_once()
            manager.fluid._measure()
            results[enabled] = (before, float(manager.fluid.diagnostics()["volume_m3"]))
            manager.stop()
        closed_before, closed_after = results[False]
        open_before, open_after = results[True]
        self.assertAlmostEqual(closed_after / closed_before, 1.0, places=3,
                               msg="a closed domain must conserve volume exactly")
        self.assertLess(open_after, open_before * 0.999,
                        "water did not leave through the open edge")

    async def test_the_open_edge_never_pushes_water_back_in(self) -> None:
        """A transmissive outlet that copied inward velocity too would act as a
        second, accidental source. Only outward flow is allowed out."""
        self.manager.apply_water_outflow(True)
        self.manager.start()
        for _ in range(600):
            self.manager._step_once()
        edge_u = np.asarray(self.manager.fluid._u.numpy()).reshape(N, N)[:, -1]
        self.assertGreaterEqual(float(edge_u.min()), 0.0,
                                "the outlet developed inward velocity")

    # --------------------------------------------------------------- source
    async def test_a_placed_source_feeds_the_map_and_replaces_the_edge_inflow(self) -> None:
        """The user asked to be able to say where the water comes from. A SOURCE
        object is that answer, and it takes over from the edge entirely --
        otherwise the map has two sources and the question has two answers."""
        self.manager.apply_object_add({"type": "SOURCE", "position": [x_at(60), 0.0, 0.0]})
        self.manager.start()
        for _ in range(900):
            self.manager._step_once()
        depth = self._depth()
        self.assertEqual(self.manager.fluid.diagnostics()["sources"], 1)
        self.assertGreater(float(depth[col(0.0), 60]), 1.0, "the source did not fill")

        # The west edge must no longer be HELD at water.level. Water can still
        # reach it -- the source spreads in every direction, including back
        # upstream, which is correct physics -- so the honest check is against a
        # control run where the edge inflow really is doing the feeding.
        control = SimulationManager()
        control.apply_water_level(0.5)
        control.start()
        for _ in range(900):
            control._step_once()
        control_edge = float(np.asarray(control.fluid._h.numpy(),
                                        dtype=np.float32).reshape(N, N)[:, 0].max())
        control.stop()
        self.assertAlmostEqual(control_edge, 0.5, places=3,
                               msg="the control run was not edge-fed after all")
        self.assertLess(float(depth[:, 0].max()), control_edge * 0.5,
                        "the west edge kept feeding water despite a placed SOURCE")

    async def test_a_source_can_be_dragged_while_running(self) -> None:
        """Positions are read live every tick, which is the whole reason SOURCE
        is an object rather than a setting."""
        source = self.manager.apply_object_add(
            {"type": "SOURCE", "position": [x_at(60), 0.0, 0.0]})
        self.manager.start()
        for _ in range(600):
            self.manager._step_once()
        self.assertGreater(float(self._depth()[col(0.0), 60]), 1.0)
        self.manager.apply_object_update(source["id"],
                                         {"position": [x_at(60), 0.0, x_at(140)]})
        for _ in range(600):
            self.manager._step_once()
        self.assertGreater(float(self._depth()[140, 60]), 1.0,
                           "the source did not start feeding its new position")

    async def test_a_source_respects_terrain_and_does_not_drown_a_hill(self) -> None:
        """Same rule as the edge inflow: h = max(0, level - bed). A source on a
        slope fills to the height asked for instead of pouring over high ground."""
        self.manager.world.terrain.heights[col(0.0) - 3:col(0.0) + 4, 58:64] = 6.0
        source = self.manager.apply_object_add(
            {"type": "SOURCE", "position": [x_at(60), 0.0, 0.0]})
        self.manager.apply_object_update(source["id"],
                                         {"metadata": {"inflow_level": 1.5}})
        self.manager.start()
        for _ in range(300):
            self.manager._step_once()
        self.assertEqual(float(self._depth()[col(0.0), 60]), 0.0,
                         "the source flooded ground taller than its own level")

    # ---------------------------------------------------------------- drain
    async def test_a_drain_removes_water(self) -> None:
        """A drain removes water where it sits. It is not infinitely strong:
        against a continuous edge inflow it digs a depression rather than a dry
        hole, which is why this compares against the same world without one
        instead of asserting an absolute depth."""
        def field(with_drain: bool) -> np.ndarray:
            manager = SimulationManager()
            if with_drain:
                manager.apply_object_add({"type": "DRAIN", "position": [x_at(40), 0.0, 0.0]})
            manager.apply_water_level(2.0)
            manager.start()
            for _ in range(900):
                manager._step_once()
            out = np.asarray(manager.fluid._h.numpy(), dtype=np.float32).reshape(N, N).copy()
            drains = manager.fluid.diagnostics()["drains"]
            manager.stop()
            return out, drains

        plain, _ = field(False)
        drained, count = field(True)
        self.assertEqual(count, 1)
        self.assertLess(float(drained[col(0.0), 40]), float(plain[col(0.0), 40]) * 0.7,
                        "the drain did not lower the water where it sits")
        self.assertGreater(float(drained[col(0.0), 25]), 0.2,
                           "water upstream of the drain vanished too")
        self.assertLess(float(drained.sum()), float(plain.sum()),
                        "the drain removed no water overall")

    async def test_a_drain_has_no_effect_outside_its_radius(self) -> None:
        """A localized sink must be localized: identical worlds, one with a
        drain far away, must agree where the drain cannot reach."""
        def field(with_drain: bool) -> np.ndarray:
            manager = SimulationManager()
            if with_drain:
                manager.apply_object_add({"type": "DRAIN", "position": [x_at(40), 0.0, x_at(180)]})
            manager.apply_water_level(1.5)
            manager.start()
            for _ in range(600):
                manager._step_once()
            out = np.asarray(manager.fluid._h.numpy(), dtype=np.float32).reshape(N, N).copy()
            manager.stop()
            return out
        plain, drained = field(False), field(True)
        # a wide band on the far side of the map, well outside the 5 m radius
        np.testing.assert_allclose(plain[:60, :], drained[:60, :], atol=1e-5)

    async def test_the_drain_vortex_follows_the_ambient_circulation(self) -> None:
        """The headline DRAIN acceptance test, and the reason the vortex is
        physics rather than decoration.

        In a depth-averaged model a purely radial sink produces purely radial
        convergence and NO rotation -- spin has to come from angular momentum
        already present, amplified by convergence. So the assertion is that the
        rotation SIGN FOLLOWS the seeded circulation, and that a symmetric
        approach produces no rotation at all. A test that checked for a fixed
        direction would pass just as happily against a hard-coded swirl, which
        CONTINUATION.md explicitly asks not to build.
        """
        yy, xx = np.mgrid[0:N, 0:N]
        centre = col(0.0)
        dxc = (xx - centre).astype(np.float32)
        dzc = (yy - centre).astype(np.float32)
        radius = np.maximum(np.sqrt(dxc ** 2 + dzc ** 2), 1.0)
        ring = (radius > 2.0) & (radius < 4.0)

        def tangential_after(seed: float) -> float:
            manager = SimulationManager()
            manager.apply_object_add({"type": "DRAIN", "position": [0.0, 0.0, 0.0]})
            manager.apply_water_level(1.5)
            manager.start()
            manager.fluid._source_enabled = False
            manager.fluid._h.assign(np.full(manager.fluid._count, 1.5, dtype=np.float32))
            manager.fluid._u.assign((seed * -dzc / radius).astype(np.float32).ravel())
            manager.fluid._v.assign((seed * dxc / radius).astype(np.float32).ravel())
            for _ in range(180):
                manager._step_once()
            u = np.asarray(manager.fluid._u.numpy()).reshape(N, N)
            v = np.asarray(manager.fluid._v.numpy()).reshape(N, N)
            manager.stop()
            return float(((-dzc / radius) * u + (dxc / radius) * v)[ring].mean())

        clockwise = tangential_after(-0.8)
        still = tangential_after(0.0)
        counter = tangential_after(+0.8)
        self.assertGreater(counter, 0.2, "seeded counter-clockwise flow did not spin up")
        self.assertLess(clockwise, -0.2, "seeded clockwise flow did not spin up")
        self.assertAlmostEqual(still, 0.0, places=3,
                               msg="the drain invented rotation from a symmetric approach")
        self.assertAlmostEqual(counter, -clockwise, places=3,
                               msg="the vortex is not symmetric in the two directions")

    async def test_the_drain_pulls_water_inward_even_without_any_rotation(self) -> None:
        """Convergence is the sink's own behaviour and must not depend on spin:
        with zero ambient circulation there is still inflow, just no swirl."""
        yy, xx = np.mgrid[0:N, 0:N]
        centre = col(0.0)
        dxc = (xx - centre).astype(np.float32)
        dzc = (yy - centre).astype(np.float32)
        radius = np.maximum(np.sqrt(dxc ** 2 + dzc ** 2), 1.0)
        ring = (radius > 2.0) & (radius < 4.0)
        self.manager.apply_object_add({"type": "DRAIN", "position": [0.0, 0.0, 0.0]})
        self.manager.apply_water_level(1.5)
        self.manager.start()
        self.manager.fluid._source_enabled = False
        self.manager.fluid._h.assign(np.full(self.manager.fluid._count, 1.5, dtype=np.float32))
        for _ in range(180):
            self.manager._step_once()
        u = np.asarray(self.manager.fluid._u.numpy()).reshape(N, N)
        v = np.asarray(self.manager.fluid._v.numpy()).reshape(N, N)
        radial = float(((dxc / radius) * u + (dzc / radius) * v)[ring].mean())
        self.assertLess(radial, -0.2, "the drain did not pull water toward itself")

    async def test_a_stronger_drain_removes_more_water(self) -> None:
        """drain_strength is a real discharge control, not a label."""
        removed = []
        for strength in (0.4, 2.0):
            manager = SimulationManager()
            drain = manager.apply_object_add({"type": "DRAIN", "position": [0.0, 0.0, 0.0]})
            manager.apply_object_update(drain["id"],
                                        {"metadata": {"drain_strength": strength}})
            manager.apply_water_level(1.5)
            manager.start()
            manager.fluid._source_enabled = False
            manager.apply_water_outflow(False)
            manager.fluid._h.assign(np.full(manager.fluid._count, 1.5, dtype=np.float32))
            manager.fluid._measure()
            before = float(manager.fluid.diagnostics()["volume_m3"])
            for _ in range(300):
                manager._step_once()
            manager.fluid._measure()
            removed.append(before - float(manager.fluid.diagnostics()["volume_m3"]))
            manager.stop()
        self.assertGreater(removed[1], removed[0] * 1.5,
                           f"drain strength did not scale the discharge: {removed}")

    async def test_the_drain_draws_bodies_toward_itself_by_mass_and_drag(self) -> None:
        """The fifth check CONTINUATION.md asks of the drain: objects entering
        its region must be advected toward it, subject to drag, buoyancy and
        mass -- not teleported and not flung.

        Two bodies at the same distance, differing only in mass and drag: the
        light one must converge faster. Also asserts nobody is launched, because
        the drain ASSIGNS velocity inside its disc and the quadratic drag law
        would turn an unbounded assignment into a catapult.
        """
        self.manager.apply_object_add({"type": "DRAIN", "position": [0.0, 0.0, 0.0]})
        box = self.manager.apply_object_add({"type": "BOX", "position": [8.0, 0.0, 0.0]})
        person = self.manager.apply_object_add({"type": "PERSON", "position": [0.0, 0.0, 8.0]})
        self.manager.apply_water_level(1.5)
        self.manager.start()
        self.manager.fluid._h.assign(
            np.full(self.manager.fluid._count, 1.5, dtype=np.float32))

        peak = 0.0
        for _ in range(600):
            self.manager._step_once()
            peak = max(peak, float(np.abs(self.manager.rigid.buffer.velocities).max()))
        moved_box = self.manager.world.objects[box["id"]]
        moved_person = self.manager.world.objects[person["id"]]
        box_r = math.hypot(moved_box.position[0], moved_box.position[2])
        person_r = math.hypot(moved_person.position[0], moved_person.position[2])

        self.assertLess(box_r, 8.0 - 0.3, "the box was not drawn toward the drain")
        self.assertLess(person_r, 8.0 - 1.0, "the person was not drawn toward the drain")
        self.assertLess(person_r, box_r,
                        "the light, draggy body did not converge faster than the heavy one")
        self.assertLess(peak, 5.0,
                        f"a body was flung rather than drawn in (peak {peak:.2f} m/s)")

    async def test_source_and_drain_survive_a_save_load_round_trip(self) -> None:
        self.manager.apply_object_add({"type": "SOURCE", "position": [x_at(50), 0.0, 0.0]})
        self.manager.apply_object_add({"type": "DRAIN", "position": [x_at(150), 0.0, 0.0]})
        self.manager.save("watertest")
        self.manager.load("watertest")
        source = next(o for o in self.manager.world.objects.values() if o.type == "SOURCE")
        drain = next(o for o in self.manager.world.objects.values() if o.type == "DRAIN")
        self.assertGreater(float(source.metadata["inflow_radius"]), 0.0)
        self.assertGreater(float(drain.metadata["drain_strength"]), 0.0)
        self.assertTrue(self.manager.world.water.outflow_enabled)


class BridgeAndPeopleTests(unittest.IsolatedAsyncioTestCase):
    """v0.9.0: a bridge whose piers obstruct but whose deck does not, and people
    light enough that moving water actually carries them off."""

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

    def _run(self, steps: int) -> None:
        for _ in range(steps):
            self.manager._step_once()

    def _field(self, name: str) -> np.ndarray:
        array = getattr(self.manager.fluid, name)
        return np.asarray(array.numpy(), dtype=np.float32).reshape(N, N)

    # --------------------------------------------------------------- bridge
    async def test_water_flows_under_a_bridge_but_around_its_piers(self) -> None:
        """The single design decision this whole feature rests on: a bridge is
        not a wall. Rasterizing the deck would dam the river the bridge spans,
        which is the opposite of what a bridge does."""
        self.manager.apply_object_add({"type": "BRIDGE", "position": [x_at(40), 0.0, 0.0]})
        self.manager.apply_water_level(1.5)
        self.manager.start()
        self._run(1500)
        solid = self.manager.fluid._obstacle_host.reshape(N, N)
        depth = self._field("_h")
        pier_rows = sorted(set(np.nonzero(solid)[0].tolist()))
        self.assertTrue(pier_rows, "the bridge produced no piers at all")
        # three separate pier clusters across the span, not one solid block
        clusters = 1 + sum(1 for a, b in zip(pier_rows, pier_rows[1:]) if b - a > 1)
        self.assertEqual(clusters, 3, f"expected 3 piers, found {clusters}")
        centre = col(0.0)
        self.assertEqual(float(depth[centre, 40]), 0.0, "the centre pier is not solid")
        self.assertGreater(float(depth[centre - 6, 40]), 0.1,
                           "water did not pass between the piers")

    async def test_bridge_piers_speed_the_flow_up_between_them(self) -> None:
        """The educational payoff: a constriction accelerates the water. Same
        rows, with and without the bridge."""
        def speed(with_bridge: bool) -> float:
            manager = SimulationManager()
            if with_bridge:
                manager.apply_object_add({"type": "BRIDGE", "position": [x_at(40), 0.0, 0.0]})
            manager.apply_water_level(1.5)
            manager.start()
            for _ in range(1500):
                manager._step_once()
            u = np.asarray(manager.fluid._u.numpy(), dtype=np.float32).reshape(N, N)
            centre = col(0.0)
            out = float(np.abs(u[centre - 8:centre - 3, 40]).mean())
            manager.stop()
            return out
        self.assertGreater(speed(True), speed(False) * 1.2,
                           "the piers did not constrict the channel")

    async def test_a_flooded_deck_is_reported_once_and_only_when_reached(self) -> None:
        """A deck matters exactly once -- when the river reaches it. That is a
        real cause-and-effect moment, so it is an event, not a silent number."""
        def events_for(deck_height: float) -> list:
            manager = SimulationManager()
            bridge = manager.apply_object_add(
                {"type": "BRIDGE", "position": [x_at(40), 0.0, 0.0]})
            manager.apply_object_update(bridge["id"],
                                        {"metadata": {"deck_height": deck_height}})
            manager.apply_water_level(2.5)
            manager.start()
            found = []
            for _ in range(1800):
                manager._step_once()
                found += [e for e in manager.events.take_pending()
                          if e["type"] == "BRIDGE_DECK_FLOODED"]
            manager.stop()
            return found

        low = events_for(0.6)
        high = events_for(6.0)
        self.assertEqual(len(low), 1, f"expected exactly one flooding event, got {len(low)}")
        self.assertGreater(low[0]["parameters"]["water_surface_m"], 0.6)
        self.assertEqual(high, [], "a deck well above the water was reported flooded")

    # --------------------------------------------------------------- people
    async def test_moving_water_carries_a_person_away(self) -> None:
        """Light, tall for its footprint and draggy -- which is why a river moves
        a person so easily. That is the lesson, and it has to be physics."""
        person = self.manager.apply_object_add(
            {"type": "PERSON", "position": [x_at(20), 0.0, 3.0]})
        self.manager.apply_water_level(1.5)
        self.manager.start()
        start_x = self.manager.world.objects[person["id"]].position[0]
        self._run(1200)
        moved = self.manager.world.objects[person["id"]]
        self.assertGreater(moved.position[0], start_x + 2.0,
                           "the person was not carried downstream")
        self.assertIn(moved.state, {"MOVING", "FLOATING", "SETTLED"})

    async def test_a_person_moves_before_a_car_does(self) -> None:
        """Causal ordering by mass and drag: identical water, the light body
        goes first. Same shape of test as the existing BOX-before-CAR one."""
        self.manager.apply_object_add({"type": "PERSON", "position": [x_at(20), 0.0, 6.0]})
        self.manager.apply_object_add({"type": "CAR", "position": [x_at(20), 0.0, -6.0]})
        # Shallow on purpose. A CAR's draft is 0.43 m, so at the 1.2 m this test
        # used to run at the car is simply afloat, and an afloat body has no
        # friction left to hold it: both bodies leave with the arriving front
        # within 0.2 s of each other and the ordering means nothing. Measured
        # across levels: 0.25/0.35 m neither moves, 0.50 m person 10.80 s and car
        # 11.58 s (car sliding, not floating -- this is the claim), 0.80 m and
        # 1.20 m both go together with the car afloat. The lesson lives in the
        # shallow regime, which is also the one people actually drive into.
        self.manager.apply_water_level(0.5)
        self.manager.start()
        person = next(o for o in self.manager.world.objects.values() if o.type == "PERSON")
        car = next(o for o in self.manager.world.objects.values() if o.type == "CAR")
        person_start, car_start = person.position[0], car.position[0]
        person_moved_at = car_moved_at = None
        for _ in range(1800):
            self.manager._step_once()
            if person_moved_at is None and person.position[0] > person_start + 0.5:
                person_moved_at = self.manager.sim_time
            if car_moved_at is None and car.position[0] > car_start + 0.5:
                car_moved_at = self.manager.sim_time
        self.assertIsNotNone(person_moved_at, "the person never moved at all")
        if car_moved_at is not None:
            self.assertLess(person_moved_at, car_moved_at,
                            "the car moved before the person")

    async def test_a_person_on_dry_ground_stays_put(self) -> None:
        """The control: no water, no motion. Without this, "the person moved"
        proves nothing about the water."""
        person = self.manager.apply_object_add(
            {"type": "PERSON", "position": [x_at(180), 0.0, 0.0]})
        self.manager.apply_water_level(0.0)
        self.manager.start()
        self._run(900)
        moved = self.manager.world.objects[person["id"]]
        self.assertEqual(moved.state, "INTACT")
        self.assertAlmostEqual(moved.position[0], x_at(180), places=3)

    async def test_bridge_and_person_survive_a_save_load_round_trip(self) -> None:
        self.manager.apply_object_add({"type": "BRIDGE", "position": [x_at(60), 0.0, 0.0]})
        self.manager.apply_object_add({"type": "PERSON", "position": [x_at(70), 0.0, 2.0]})
        self.manager.save("bridgetest")
        self.manager.load("bridgetest")
        bridge = next(o for o in self.manager.world.objects.values() if o.type == "BRIDGE")
        person = next(o for o in self.manager.world.objects.values() if o.type == "PERSON")
        self.assertGreater(float(bridge.metadata["pier_count"]), 0.0)
        self.assertGreater(float(bridge.metadata["deck_height"]), 0.0)
        self.assertAlmostEqual(person.mass, 70.0, places=1)


class VelocityStreamTests(unittest.IsolatedAsyncioTestCase):
    """v0.10.0: the real velocity field reaches the renderer.

    FrameKind.VELOCITY_FIELD existed in the protocol from the very first version
    and was never filled. The water shader now builds its flow map from it, so
    if this stream is wrong the water animates in a direction the water is not
    going -- which is exactly the decorative behaviour docs/01_vision.md rules
    out, and is why it is tested rather than assumed.
    """

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

    def _velocity_frames(self) -> list:
        found = []
        for payload in self.binary_frames:
            kind, count, _, values = protocol.decode_frame(payload)
            if kind == protocol.FrameKind.VELOCITY_FIELD:
                found.append((count, values))
        return found

    async def test_the_velocity_field_is_streamed_and_matches_the_solver(self) -> None:
        self.manager.apply_water_level(1.5)
        self.manager.start()
        for _ in range(600):
            self.manager._step_once()
        for _ in range(config.VELOCITY_STREAM_EVERY):
            await self.manager._stream()
        frames = self._velocity_frames()
        self.assertTrue(frames, "the velocity field was never streamed")
        count, values = frames[-1]
        self.assertEqual(count, N * N)
        # the frame carries (u, 0, v) per cell, in terrain-vertex order
        streamed = values.reshape(N * N, 3)
        u = np.asarray(self.manager.fluid._u.numpy(), dtype=np.float32)
        v = np.asarray(self.manager.fluid._v.numpy(), dtype=np.float32)
        np.testing.assert_allclose(streamed[:, 0], u, atol=1e-6)
        np.testing.assert_allclose(streamed[:, 2], v, atol=1e-6)
        np.testing.assert_allclose(streamed[:, 1], 0.0, atol=1e-6)
        self.assertGreater(float(np.abs(streamed[:, 0]).max()), 0.1,
                           "a still field proves nothing about direction")

    async def test_the_streamed_direction_matches_where_the_water_goes(self) -> None:
        """Sign check, not just magnitude: water fed from the west edge must
        stream a positive u. A flow map with the sign flipped would animate the
        river backwards and every magnitude assertion would still pass."""
        self.manager.apply_water_level(1.5)
        self.manager.start()
        for _ in range(900):
            self.manager._step_once()
        for _ in range(config.VELOCITY_STREAM_EVERY):
            await self.manager._stream()
        count, values = self._velocity_frames()[-1]
        streamed = values.reshape(N, N, 3)
        depth = np.asarray(self.manager.fluid._h.numpy(), dtype=np.float32).reshape(N, N)
        wet = depth > 0.05
        self.assertTrue(wet.any())
        self.assertGreater(float(streamed[:, :, 0][wet].mean()), 0.05,
                           "the streamed current points upstream")

    async def test_the_velocity_stream_is_throttled(self) -> None:
        """It is the largest payload on the wire after the tracers, and a flow
        map is a low-frequency visual signal -- so it is deliberately not sent
        every frame."""
        self.manager.apply_water_level(1.0)
        self.manager.start()
        for _ in range(60):
            self.manager._step_once()
        for _ in range(config.VELOCITY_STREAM_EVERY * 4):
            await self.manager._stream()
        heights = sum(1 for payload in self.binary_frames
                      if protocol.decode_frame(payload)[0] == protocol.FrameKind.WATER_HEIGHT)
        velocities = len(self._velocity_frames())
        self.assertGreater(heights, velocities,
                           "the velocity field is not throttled below the height frames")
        self.assertGreaterEqual(velocities, 1)


class FrictionLawTests(unittest.IsolatedAsyncioTestCase):
    """Bed friction is what decides where water goes once a channel exists.

    These tests measure the friction law directly rather than through a
    scenario: a flat bed with a uniform depth and a uniform initial velocity
    has zero surface gradient, so gravity contributes nothing and whatever
    slows the water down is friction alone.
    """

    def _decay(self, depth: float, steps: int = 300, u0: float = 1.0) -> float:
        """Mean interior speed left after `steps` frames of friction only."""
        world = WorldState()
        world.water.level = 0.0
        solver = create_fluid_solver(create_engine().device)
        solver.initialize(world)
        solver._source_enabled = False          # no inflow: friction in isolation
        solver.set_boundaries(world.terrain, {}, 0, 0)
        solver._h.assign(np.full(solver._count, depth, dtype=np.float32))
        solver._u.assign(np.full(solver._count, u0, dtype=np.float32))
        solver._v.assign(np.zeros(solver._count, dtype=np.float32))
        for _ in range(steps):
            solver.advance(1.0 / 60.0, 8, 1.0 / 120.0)
        u = np.asarray(solver._u.numpy(), dtype=np.float32).reshape(N, N)
        return float(u[N // 4:3 * N // 4, N // 4:3 * N // 4].mean())

    async def test_friction_is_depth_dependent_not_uniform(self) -> None:
        """The one test that tells the two laws apart.

        Real bed friction is a stress on the bottom shared over the water
        column, so a deep column keeps its speed far better than a thin sheet
        over the same bed -- that is why a river stays in its channel and only
        creeps across a floodplain. A velocity damping that does not read the
        depth gives the identical answer for both, and this ratio comes out 1.
        """
        deep = self._decay(1.0)
        thin = self._decay(0.0625)          # 1/16 the depth
        self.assertGreater(deep, 0.5, "deep water was stopped by bed friction")
        self.assertGreater(
            deep / max(thin, 1.0e-6), 2.0,
            f"friction does not depend on depth: deep={deep:.3f} thin={thin:.3f} "
            f"ratio={deep / max(thin, 1.0e-6):.2f} (uniform damping gives 1.0)")

    def _terminal_speed_on_slope(self, depth: float, slope: float,
                                 seconds: float = 20.0) -> float:
        """Mean interior speed of a uniform sheet let go on a constant slope."""
        world = WorldState()
        world.water.level = 0.0
        fall = (np.arange(N, dtype=np.float32) * config.TERRAIN_CELL_SIZE * slope)
        world.terrain.heights[:, :] = (fall[-1] - fall)[None, :]
        solver = create_fluid_solver(create_engine().device)
        solver.initialize(world)
        solver._source_enabled = False
        solver.set_boundaries(world.terrain, {}, 0, 0)
        solver._h.assign(np.full(solver._count, depth, dtype=np.float32))
        for _ in range(int(seconds * 60)):
            solver.advance(1.0 / 60.0, 8, 1.0 / 120.0)
        u = np.asarray(solver._u.numpy(), dtype=np.float32).reshape(N, N)
        return float(np.abs(u[N // 4:3 * N // 4, N // 4:3 * N // 4]).mean())

    async def test_steady_flow_lands_near_the_manning_velocity(self) -> None:
        """Quantitative check that the law is Manning and not merely "something
        that reads the depth": the speed a sheet settles at on a constant slope
        should approach u = h^(2/3) * sqrt(S) / n.

        Asserted only within a factor of two, deliberately. The analytic value
        holds for steady *uniform* flow, and this is a finite map with closed
        ends where the sheet is also draining downhill, so demanding a tight
        match would be asserting the setup rather than the law. The number that
        actually discriminates the two friction laws is the depth ratio above;
        the measured-versus-analytic pair is recorded in TEST_REPORT.md.
        """
        slope, depth = 0.05, 0.5
        measured = self._terminal_speed_on_slope(depth, slope)
        analytic = depth ** (2.0 / 3.0) * math.sqrt(slope) / config.FLUID_MANNING_N
        self.assertGreater(measured, analytic * 0.5,
                           f"far below Manning: {measured:.2f} vs {analytic:.2f} m/s")
        self.assertLess(measured, analytic * 2.0,
                        f"far above Manning: {measured:.2f} vs {analytic:.2f} m/s")

    async def test_friction_can_only_slow_water_never_reverse_it(self) -> None:
        """Friction is applied by dividing, not by subtracting.

        Subtracting g*n^2*|u|*u*dt/h^(4/3) overshoots exactly where the
        coefficient is largest -- a thin fast sheet, i.e. a wetting front -- and
        turns the flow around, which shows up as a front that shudders backwards
        and needs its own timestep limit to suppress. Dividing by (1 + drag) is
        unconditionally stable and cannot change the sign, whatever the depth.
        """
        u0 = 3.0
        left = self._decay(config.FLUID_FRICTION_MIN_DEPTH, steps=60, u0=u0)
        self.assertGreater(left, 0.0, "friction reversed the flow instead of slowing it")
        self.assertLess(left, u0, "the thin sheet was not slowed at all")


class RiverValleyTests(unittest.IsolatedAsyncioTestCase):
    """v0.12.0: the generated channel the rest of the river work stands on.

    These test the *geometry*, deliberately. Whether the flow stays in the
    channel is a question about the inlet, not about the bed: the edge inflow
    wets the entire west edge including the floodplain, so a containment test
    written now would fail for the boundary condition's reasons and be weakened
    to pass. It belongs with the local inlet.
    """

    async def asyncSetUp(self) -> None:
        self.manager = SimulationManager()

    async def asyncTearDown(self) -> None:
        self.manager.stop()
        await asyncio.sleep(0)

    def _generate(self, **params) -> tuple[np.ndarray, dict]:
        patch = self.manager.apply_terrain_river(params)
        return self.manager.world.terrain.heights.copy(), patch["river"]

    async def test_the_bed_falls_downstream_at_the_slope_it_was_asked_for(self) -> None:
        """The only thing that makes water move in an open channel is the slope
        of the bed, so this is the generator's whole job. Checked cell by cell
        against the requested slope rather than end to end, because a mean fall
        can hide a staircase."""
        for slope in (0.001, 0.002, 0.005):
            heights, info = self._generate(slope=slope)
            centre = heights[N // 2, :]
            drop = -np.diff(centre)
            expected = slope * config.TERRAIN_CELL_SIZE
            np.testing.assert_allclose(drop, expected, rtol=1e-4, atol=1e-6)
            self.assertGreater(info["inlet_bed"], info["outlet_bed_actual"],
                               "the river runs uphill")

    async def test_the_cross_section_is_a_channel_and_not_a_ditch_or_a_ridge(self) -> None:
        """Bottom flat and lowest, banks strictly rising outward, floodplain
        flat at the top -- and the bank top exactly `incision` above the bed, so
        the depth the channel can hold is the depth that was asked for."""
        heights, info = self._generate(bed_width=12.0, incision=2.0, bank_run=4.0)
        column = heights[:, N // 2]
        mid = N // 2
        half_bed = int(12.0 / 2 / config.TERRAIN_CELL_SIZE)
        bed = column[mid - half_bed + 1:mid + half_bed]
        np.testing.assert_allclose(bed, bed[0], atol=1e-6)
        top = int((12.0 / 2 + 4.0) / config.TERRAIN_CELL_SIZE)
        bank = column[mid + half_bed:mid + top + 1]
        self.assertTrue(np.all(np.diff(bank) > 0.0), "the bank does not rise outward")
        self.assertAlmostEqual(float(column[mid + top] - bed[0]), 2.0, places=3)
        floodplain = column[mid + top + 1:]
        np.testing.assert_allclose(floodplain, floodplain[0], atol=1e-6)
        self.assertAlmostEqual(info["operating_depth"], 1.0, places=6)

    async def test_the_bed_is_smooth_enough_not_to_stand_waves_up(self) -> None:
        """The reason the bed is generated instead of brushed. A step in the bed
        is a step in the surface the flow has to climb; the acceptance bound is
        derived from the geometry rather than picked -- downstream, no cell may
        drop by more than the slope says, and across the channel no cell may
        rise faster than the bank's own average gradient allows."""
        heights, _ = self._generate(slope=0.002, incision=2.0, bank_run=4.0)
        along = np.abs(np.diff(heights, axis=1))
        self.assertLessEqual(float(along.max()),
                             0.002 * config.TERRAIN_CELL_SIZE + 1e-6,
                             "a step downstream: the flow would jump it")
        across = np.abs(np.diff(heights, axis=0))
        # smoothstep peaks at 1.5x the mean gradient, and never above it
        bank_mean = 2.0 / (4.0 / config.TERRAIN_CELL_SIZE)
        self.assertLessEqual(float(across.max()), bank_mean * 1.5 + 1e-6,
                             "a step across the channel steeper than the bank")

    async def test_a_meander_swings_the_channel_without_breaking_the_descent(self) -> None:
        """Optional and off by default in v0.12.0 -- the discharge counters need
        a straight reference channel first -- but when asked for it must move
        the deepest point sideways and leave the downstream fall alone."""
        straight, _ = self._generate(meander_amplitude=0.0)
        winding, _ = self._generate(meander_amplitude=20.0, meander_wavelength=120.0)
        deepest = np.argmin(winding, axis=0)
        self.assertGreater(int(deepest.max() - deepest.min()), 20,
                           "the meander did not move the channel")
        np.testing.assert_array_equal(np.argmin(straight, axis=0),
                                      np.full(N, np.argmin(straight[:, 0])))
        for grid in (straight, winding):
            centre = grid[np.argmin(grid, axis=0), np.arange(N)]
            self.assertTrue(np.all(np.diff(centre) < 0.0),
                            "the channel bottom stopped falling downstream")

    async def test_bad_parameters_are_refused_and_the_terrain_is_left_alone(self) -> None:
        before = self.manager.world.terrain.heights.copy()
        for bad in ({"slope": 0.5}, {"bed_width": 0.0}, {"incision": -1.0},
                    {"meander_amplitude": 95.0}, {"nonsense": 1.0},
                    {"slope": float("nan")}):
            with self.assertRaises(ValueError, msg=f"accepted {bad}"):
                self.manager.apply_terrain_river(bad)
        np.testing.assert_array_equal(self.manager.world.terrain.heights, before)

    async def test_generating_while_running_is_refused_like_any_terrain_edit(self) -> None:
        self.manager.start()
        with self.assertRaises(ValueError):
            self.manager.apply_terrain_river({})

    async def test_the_valley_survives_a_save_load_round_trip(self) -> None:
        """The save format already carries all 40 401 elevations, which is why
        the generator needs no file format of its own -- but that is a claim
        about the format, so it is tested and not assumed."""
        heights, info = self._generate(slope=0.003, bed_width=16.0, incision=2.5)
        self.manager.world.water.level = info["inlet_level"]
        self.manager.save("test_river_valley")
        other = SimulationManager()
        other.load("test_river_valley")
        np.testing.assert_allclose(other.world.terrain.heights, heights, atol=1e-6)
        self.assertAlmostEqual(other.world.water.level, info["inlet_level"], places=5)
        other.stop()
        (config.DATA_DIR / "test_river_valley.json").unlink()

    async def test_the_solver_sees_the_generated_bed(self) -> None:
        """A generated valley that the GPU never hears about is a picture. The
        bed the solver holds after start() must be the bed that was generated,
        and water dropped on it must run downhill rather than sit still."""
        heights, info = self._generate()
        self.manager.apply_water_level(info["inlet_level"])
        self.manager.start()
        np.testing.assert_allclose(
            self.manager.fluid.get_terrain_heights().reshape(N, N), heights, atol=1e-5)
        for _ in range(120):
            self.manager._step_once()
        u = np.asarray(self.manager.fluid._u.numpy()).reshape(N, N)
        h = np.asarray(self.manager.fluid._h.numpy()).reshape(N, N)
        wet = h > 0.05
        self.assertTrue(wet.any(), "the channel never filled")
        self.assertGreater(float(u[wet].mean()), 0.0,
                           "water on a sloped bed did not move downstream")


class RiverBoundaryTests(unittest.IsolatedAsyncioTestCase):
    """v0.12.0: the local inlet, the local outlet, and the ledger that proves them.

    A boundary condition cannot be checked by looking at the screen -- "water
    appears at the inlet" and "the right amount of water appears at the inlet"
    look identical -- so every test here reads the volume ledger or a measured
    discharge, never a picture.
    """

    async def asyncSetUp(self) -> None:
        self.manager = SimulationManager()

    async def asyncTearDown(self) -> None:
        self.manager.stop()
        await asyncio.sleep(0)

    def _valley(self, discharge: float = 12.0, width: float = 12.0,
                outlet_width: float = 20.0, **river) -> dict:
        info = self.manager.apply_terrain_river(river)["river"]
        self.manager.apply_water_level(0.0)
        self.manager.apply_river_inlet({"enabled": True, "width_m": width,
                                        "discharge_m3s": discharge})
        self.manager.apply_river_outlet({"width_m": outlet_width})
        return info

    def _prime(self, depth: float) -> None:
        """Fill the channel to `depth`, surface parallel to the bed.

        Starting a 200 m channel from dry costs two minutes of simulated time
        before the flow is developed; priming it puts the tests at the state
        they are actually asking about. It also injects water behind the
        ledger's back, so a primed run cannot be used for the balance test --
        which is why that test does not prime.
        """
        bed = self.manager.fluid.get_terrain_heights().reshape(N, N)
        surface = bed[N // 2, :] + depth
        h = np.maximum(surface[None, :] - bed, 0.0).astype(np.float32)
        self.manager.fluid._h.assign(h.ravel())

    def _run(self, seconds: float) -> None:
        for _ in range(int(seconds * 60)):
            self.manager._step_once()

    def _depth(self) -> np.ndarray:
        return np.asarray(self.manager.fluid._h.numpy(), dtype=np.float32).reshape(N, N)

    async def test_the_inlet_delivers_the_discharge_it_was_asked_for(self) -> None:
        """Q is the control, so the number that matters is the one crossing the
        face -- measured the way the depth step transports it, not as u*h at the
        edge cell, which returns the request by construction whatever happens."""
        self._valley(discharge=12.0)
        self.manager.start()
        self._prime(0.79)
        self._run(10.0)
        measured = self.manager.fluid.diagnostics()["inlet_discharge_m3s"]
        self.assertAlmostEqual(measured, 12.0, delta=0.6,
                               msg=f"asked for 12 m3/s, delivered {measured:.2f}")

    async def test_the_delivered_discharge_tracks_the_request_across_the_range(self) -> None:
        """The subtlest line in the inlet is `u = 2q/h - u[1]`, and one operating
        point does not pin it: `u = q/h` also looks right at a single Q while
        delivering roughly twice what was asked for. So the request is checked at
        four discharges spanning the control's range, on the measured face flux.

        A run from dry would not do it either -- during the fill the answer is
        dominated by the advancing front -- so each channel is primed at the
        normal depth for its own q.
        """
        for q in (4.0, 12.0, 24.0, 48.0):
            manager = SimulationManager()
            manager.apply_terrain_river({})
            manager.apply_water_level(0.0)
            manager.apply_river_inlet({"enabled": True, "width_m": 12.0,
                                       "discharge_m3s": q})
            manager.apply_river_outlet({"width_m": 20.0})
            manager.start()
            bed = manager.fluid.get_terrain_heights().reshape(N, N)
            # Manning normal depth for this discharge on this slope, the same
            # number the solver floors the inlet at
            per_width = q / 13.0
            depth = (per_width * config.FLUID_MANNING_N / math.sqrt(0.002)) ** 0.6
            surface = bed[N // 2, :] + depth
            h = np.maximum(surface[None, :] - bed, 0.0).astype(np.float32)
            manager.fluid._h.assign(h.ravel())
            for _ in range(600):
                manager._step_once()
            measured = manager.fluid.diagnostics()["inlet_discharge_m3s"]
            manager.stop()
            self.assertAlmostEqual(
                measured / q, 1.0, delta=0.15,
                msg=f"asked for {q} m3/s, delivered {measured:.2f} "
                    f"({measured / q:.2f}x)")

    async def test_a_discharge_the_channel_cannot_hold_floods_rather_than_explodes(self) -> None:
        """The top of the control's range asks for more water than the channel
        was cut for: Q = 80 m3/s over 12 m needs about 2.4 m of normal depth in a
        2.0 m channel, so the river must leave its banks. That is a flood, and a
        flood is a thing this simulator is for -- what it must not be is a
        numerical blow-up, and it must say so in `cfl_limited` if the timestep
        can no longer resolve it.

        Two minutes, not thirty seconds: the inlet delivers Q and no more, so
        the channel fills at the rate the discharge allows rather than appearing
        full. Watching it overtop is watching a flood arrive."""
        self._valley(discharge=80.0)
        self.manager.start()
        self._run(120.0)
        diag = self.manager.fluid.diagnostics()
        depth = self._depth()
        self.assertTrue(np.isfinite(depth).all(), "the flood produced NaNs")
        self.assertLess(diag["max_velocity"], config.FLUID_MAX_VELOCITY,
                        "the flood pinned the velocity clamp")
        bed = self.manager.fluid.get_terrain_heights().reshape(N, N)
        floodplain = bed >= bed.max(axis=0)[None, :] - 1.0e-3
        self.assertGreater(int((depth[floodplain] > 0.01).sum()), 500,
                           "80 m3/s did not overtop a 2 m channel")
        self.assertFalse(self.manager.fluid.diagnostics()["cfl_limited"],
                         "the flood outran the timestep and said nothing")
        self.assertLess(abs(diag["volume_error_m3"]),
                        max(1.0e-3 * diag["volume_m3"], 0.05),
                        "the ledger stopped balancing under flood conditions")

    async def test_a_dry_channel_is_actually_filled_and_not_just_reported(self) -> None:
        """The trap this boundary was most likely to fall into: `_velocity_step`
        used to zero the west column's velocity every substep, so an inlet that
        wrote its velocity anywhere else would report the requested discharge
        while no water moved at all. Both halves are asserted -- the ledger says
        water entered, and the wetted front says it went somewhere."""
        self._valley()
        self.manager.start()
        self._run(20.0)
        depth = self._depth()
        diag = self.manager.fluid.diagnostics()
        self.assertGreater(diag["added_m3"], 100.0, "no water entered the map")
        wet_columns = np.nonzero((depth > 0.05).any(axis=0))[0]
        reach_m = float(wet_columns.max()) * config.TERRAIN_CELL_SIZE
        self.assertGreater(reach_m, 20.0,
                           f"water entered but never moved downstream ({reach_m:.1f} m)")
        u = np.asarray(self.manager.fluid._u.numpy()).reshape(N, N)
        self.assertGreater(float(u[depth > 0.05].mean()), 0.1,
                           "the wetted cells are not moving downstream")

    async def test_the_river_stays_in_its_channel(self) -> None:
        """The test that could not be written against the edge inflow, because
        that one wets the whole west edge including the floodplain: with a local
        inlet, water on the floodplain can only have got there by leaving the
        channel."""
        info = self._valley(discharge=12.0)
        self.manager.start()
        self._prime(info["operating_depth"] * 0.8)
        self._run(20.0)
        depth = self._depth()
        bed = self.manager.fluid.get_terrain_heights().reshape(N, N)
        floodplain = bed >= bed.max(axis=0)[None, :] - 1.0e-3
        self.assertLess(float(depth[floodplain].max()), 0.05,
                        "the river climbed out onto the floodplain")
        mid = N // 2
        self.assertGreater(float(depth[mid, N // 2]), 0.3, "the channel ran dry")

    async def test_water_leaves_only_through_the_outlet_band(self) -> None:
        """A valley drains through its channel, not through its floodplain --
        so the outlet is a band of rows, and the rest of the edge is a wall.

        Measured as discharge leaving the map, not as depth at the edge. Depth
        is the wrong instrument here: water piling against the closed part of
        the edge spreads sideways into the open part and leaves through it, so
        within seconds the whole edge sits at one level whatever the mask says.
        That is correct physics and a useless assertion.
        """
        removed = {}
        for width_m in (0.0, 20.0):            # 0 = the whole edge, as before
            manager = SimulationManager()
            manager.apply_water_level(0.0)
            manager.apply_river_outlet({"width_m": width_m})
            manager.start()
            manager.fluid._source_enabled = False   # no inflow: drainage alone
            manager.fluid._h.assign(np.full(N * N, 1.0, dtype=np.float32))
            manager.fluid._u.assign(np.full(N * N, 1.0, dtype=np.float32))
            for _ in range(300):
                manager._step_once()
            removed[width_m] = manager.fluid.diagnostics()["removed_m3"]
            manager.stop()
        span = 20.0 / config.WORLD_SIZE_M
        self.assertGreater(removed[20.0], 0.0, "nothing left through the open band")
        self.assertLess(removed[20.0], removed[0.0] * span * 1.5,
                        f"the narrowed outlet still drains most of the edge: "
                        f"{removed[20.0]:.1f} m3 against {removed[0.0]:.1f} m3 at full width")

    async def test_the_closed_part_of_the_edge_is_a_wall(self) -> None:
        """The other half of that claim, with no lateral spreading to hide it:
        put the water only outside the band and nothing may leave at all."""
        self.manager.apply_water_level(0.0)
        self.manager.apply_river_outlet({"width_m": 20.0})
        self.manager.start()
        self.manager.fluid._source_enabled = False
        lo, hi = self.manager.fluid.diagnostics()["outflow_rows"]
        depth = np.full((N, N), 1.0, dtype=np.float32)
        depth[lo:hi + 1, :] = 0.0
        self.manager.fluid._h.assign(depth.ravel())
        self.manager.fluid._u.assign(np.full(N * N, 1.0, dtype=np.float32))
        self._run(2.0)
        self.assertEqual(self.manager.fluid.diagnostics()["removed_m3"], 0.0,
                         "water left through the closed part of the edge")

    async def test_the_volume_ledger_closes(self) -> None:
        """Inflow and outflow counters exist so the boundary can be audited, and
        an audit that does not add up is worse than none. Everything that
        injects or removes water reports it, so volume(t) - volume(0) must equal
        added - removed to within float32 reduction noise."""
        self._valley()
        self.manager.start()
        self._run(30.0)
        diag = self.manager.fluid.diagnostics()
        self.assertGreater(diag["added_m3"], 0.0)
        self.assertLess(abs(diag["volume_error_m3"]), max(1.0e-3 * diag["volume_m3"], 0.05),
                        f"the ledger does not balance: {diag['volume_error_m3']:.4f} m3 "
                        f"of {diag['volume_m3']:.1f}")

    async def test_the_outlet_carries_the_suspended_load_away(self) -> None:
        """Before this the water left and the sediment it was carrying stayed,
        so the east edge slowly grew a bar out of material the river had already
        delivered to the sea.

        Erosion is off and the sediment field is placed by hand, so nothing else
        can add to it or move it: what the domain loses must be exactly what the
        outlet reports exporting. A scenario test would have compared two runs
        whose flows differ, and a faster flow carries more load -- which would
        have confounded the very thing being measured.
        """
        results = {}
        for outflow in (False, True):
            manager = SimulationManager()
            manager.apply_water_level(0.0)
            manager.apply_water_outflow(outflow)
            manager.start()
            manager.fluid._source_enabled = False
            manager.fluid._h.assign(np.full(N * N, 1.0, dtype=np.float32))
            manager.fluid._u.assign(np.full(N * N, 1.0, dtype=np.float32))
            manager.fluid._sediment.assign(np.full(N * N, 0.02, dtype=np.float32))
            before = float(np.asarray(manager.fluid._sediment.numpy()).sum())
            for _ in range(300):
                manager._step_once()
            after = float(np.asarray(manager.fluid._sediment.numpy()).sum())
            results[outflow] = (before - after, manager.fluid.diagnostics()["sediment_out_m3"])
            manager.stop()
        self.assertEqual(results[False], (0.0, 0.0),
                         "a shut outlet exported sediment")
        lost, reported = results[True]
        area = config.TERRAIN_CELL_SIZE ** 2
        self.assertGreater(reported, 0.0, "the open outlet exported nothing")
        self.assertAlmostEqual(lost * area, reported, delta=reported * 0.01,
                               msg=f"sediment vanished unaccounted: domain lost "
                                   f"{lost * area:.4f} m3, outlet reported {reported:.4f}")

    async def test_the_inlet_arrives_loaded_and_does_not_scour_itself(self) -> None:
        """Measured in docs/07_river_plan.md: an inlet fed with clean water is at
        maximum hunger for ever and sits on SEDIMENT_MAX_BED_CHANGE for ever,
        which is where the v0.11.0 reading "erosion is limited by the clamp"
        came from. Water that arrives carrying its capacity is not hungry."""
        self._valley()
        self.manager.apply_water_erosion(True)
        self.manager.start()
        self._prime(0.79)
        before = self.manager.fluid.get_terrain_heights().reshape(N, N).copy()
        self._run(20.0)
        after = self.manager.fluid.get_terrain_heights().reshape(N, N)
        ceiling = config.SEDIMENT_MAX_BED_CHANGE * 20.0
        cut = float((before - after)[:, 0].max())
        self.assertLess(cut, ceiling * 0.25,
                        f"the inlet is scouring at the clamp: {cut:.3f} m of {ceiling:.3f}")

    async def test_the_edge_level_source_no_longer_scours_its_own_columns(self) -> None:
        """The same fix on the older boundary, which is where the artifact was
        actually measured: before it, the source columns cut 0.3989 m of a
        0.400 m ceiling in 1200 steps, and that number was being read as a
        statement about erosion physics."""
        self.manager.apply_water_level(1.0)
        self.manager.apply_water_erosion(True)
        self.manager.start()
        before = self.manager.fluid.get_terrain_heights().reshape(N, N).copy()
        self._run(20.0)
        after = self.manager.fluid.get_terrain_heights().reshape(N, N)
        ceiling = config.SEDIMENT_MAX_BED_CHANGE * 20.0
        cut = float((before - after)[:, :config.FLUID_SOURCE_COLUMNS].max())
        self.assertLess(cut, ceiling * 0.25,
                        f"the level source still scours its inlet: {cut:.3f} m")

    async def test_the_river_inlet_replaces_the_edge_level_source(self) -> None:
        """One map, one answer to "where does the water come from" -- the same
        rule a placed SOURCE already follows."""
        self._valley()
        self.manager.apply_water_level(5.0)     # would put 2 m over the floodplain
        self.manager.start()
        self._run(20.0)
        edge = self._depth()[:, 0]
        bed = self.manager.fluid.get_terrain_heights().reshape(N, N)[:, 0]
        floodplain = bed >= bed.max() - 1.0e-3
        band = self.manager.fluid._inlet_q_host > 0.0
        self.assertTrue(band.any() and floodplain.any())
        self.assertLess(float(edge[floodplain].max()), 0.05,
                        "the level source is still filling the west edge")
        self.assertGreater(float(edge[band].max()), 0.05, "the inlet band is dry")

    async def test_the_boundaries_survive_a_save_load_round_trip(self) -> None:
        """They live under `water` and not in `objects` for this reason: objects
        are restored after the water is, so a river whose inlet was an object
        would load as a dry map with a decoration on it. The erosion and outflow
        toggles are checked here too -- they were written to file and silently
        dropped on load before v0.12.0, which also meant every reset() lost
        them."""
        self._valley(discharge=9.0, width=16.0, outlet_width=24.0)
        self.manager.apply_water_erosion(True)
        self.manager.apply_water_outflow(False)
        self.manager.save("test_river_boundaries")
        other = SimulationManager()
        other.load("test_river_boundaries")
        water = other.world.water
        self.assertTrue(water.inlet_enabled)
        self.assertAlmostEqual(water.inlet_discharge_m3s, 9.0)
        self.assertAlmostEqual(water.inlet_width_m, 16.0)
        self.assertAlmostEqual(water.outlet_width_m, 24.0)
        self.assertTrue(water.erosion_enabled)
        self.assertFalse(water.outflow_enabled)
        other.stop()
        (config.DATA_DIR / "test_river_boundaries.json").unlink()

    async def test_bad_boundary_settings_are_refused(self) -> None:
        for bad in ({"width_m": 0.0}, {"discharge_m3s": -1.0},
                    {"discharge_m3s": float("inf")}, {"nonsense": 1.0}):
            with self.assertRaises(ValueError, msg=f"accepted inlet {bad}"):
                self.manager.apply_river_inlet(bad)
        for bad in ({"width_m": -1.0}, {"centre_z": float("nan")}, {"nonsense": 1}):
            with self.assertRaises(ValueError, msg=f"accepted outlet {bad}"):
                self.manager.apply_river_outlet(bad)


class DamAndScenarioTests(unittest.IsolatedAsyncioTestCase):
    """v0.12.1: the dam ridge, and the two prepared worlds built on it.

    The dam is checked as *geometry that water can get over*, because that is
    the only property that distinguishes it from an obstacle. The scenarios are
    checked for the two layout rules that fail invisibly: an object written at
    the wrong elevation is buried, not broken, and a screenshot of a buried
    house looks exactly like a screenshot of a house that was never placed.
    """

    async def asyncSetUp(self) -> None:
        self.manager = SimulationManager()

    async def asyncTearDown(self) -> None:
        self.manager.stop()
        await asyncio.sleep(0)

    async def test_the_dam_is_a_ridge_the_water_can_get_over(self) -> None:
        """A dam has to be terrain, not an obstacle.

        `fluid_solver._is_solid` rasterizes HOUSE and BRIDGE as infinitely tall
        walls, so a dam built out of objects could never be overtopped -- and
        overtopping is the entire lesson. This asserts the three properties that
        make the ridge a dam rather than a wall: it stands above the channel it
        crosses, its crest is a finite elevation reachable by water, and it has
        a low point (the spillway) so the place the reservoir first spills is
        physical rather than decided by floating-point noise.
        """
        world = WorldState()
        effective = dam_ridge(world.terrain, None, None)
        heights = world.terrain.heights
        cell = world.terrain.cell_size
        mid = world.terrain.height // 2
        crest_column = int(round(effective["dam_x"] / cell))

        self.assertGreater(effective["crest_elevation"],
                           effective["channel_bed_at_dam"] + 2.0,
                           "the ridge has to stand well clear of the channel bed")
        # finite: a wall would be config.HEIGHT_MAX or an obstacle flag
        self.assertLess(effective["crest_elevation"], config.HEIGHT_MAX)
        # the crest is genuinely there, across the channel
        self.assertAlmostEqual(float(heights[mid, crest_column]),
                               effective["spill_elevation"], places=2)
        # and it is a LOW point: the crest away from the notch is higher
        shoulder = float(heights[mid + 20, crest_column])
        self.assertGreater(shoulder, effective["spill_elevation"] + 0.3,
                           "without a notch the spill station is arbitrary")
        # upstream is a basin, downstream is not
        upstream = float(heights[mid, crest_column - 25])
        downstream = float(heights[mid, crest_column + 25])
        self.assertLess(upstream, effective["spill_elevation"])
        self.assertLess(downstream, effective["spill_elevation"])
        self.assertGreater(effective["reservoir_volume_m3"], 100.0)

    async def test_the_dam_merges_into_the_valley_instead_of_slotting_through_it(self) -> None:
        """The ridge may only add material.

        It is written with `maximum(valley, ridge)` so that where the valley
        wall is already higher than the crest the dam blends into it. Written as
        a plain assignment the same code would cut a slot through the hillside
        and the reservoir would quietly drain around the end of the dam.
        """
        plain = WorldState()
        river_valley(plain.terrain, None)
        dammed = WorldState()
        dam_ridge(dammed.terrain, None, None)
        self.assertTrue(np.all(dammed.terrain.heights >= plain.terrain.heights - 1e-4),
                        "the dam lowered the ground somewhere")

    def _scenario(self, name: str) -> WorldState:
        path = ROOT / "data" / f"{name}.json"
        self.assertTrue(path.exists(), f"{name} is missing -- run tools/make_scenarios.py")
        self.manager.load(name)
        return self.manager.world

    async def test_every_scenario_object_stands_on_the_ground(self) -> None:
        """Seating, checked for both scenarios.

        The floodplain sits near 2.5-2.9 m, so an object saved with y = 0 is
        three metres underground. It does not raise, it does not log, and from
        the camera it looks like an object that was never placed at all -- which
        is why this is a test and not a comment.
        """
        for name in ("scenario_river", "scenario_dam"):
            with self.subTest(scenario=name):
                world = self._scenario(name)
                self.assertGreaterEqual(len(world.objects), 60)
                for obj in world.objects.values():
                    ground = world.terrain.height_at(obj.position[0], obj.position[2])
                    self.assertAlmostEqual(
                        obj.position[1], ground, places=2,
                        msg=f"{obj.id} is {obj.position[1] - ground:+.2f} m off the ground")

    async def test_no_scenario_object_stands_in_the_channel(self) -> None:
        """A house in the river is a dam nobody asked for.

        The channel and its banks occupy |z| < bed_width/2 + bank_run. The two
        GAUGEs are the deliberate exception: a gauge is an instrument, it is
        excluded from the obstacle mask by its own test, and measuring the river
        is the one reason to stand in it.
        """
        from app.terrain_gen import DEFAULTS
        half_top = DEFAULTS["bed_width"] * 0.5 + DEFAULTS["bank_run"]
        for name in ("scenario_river", "scenario_dam"):
            with self.subTest(scenario=name):
                world = self._scenario(name)
                for obj in world.objects.values():
                    if obj.type == "GAUGE":
                        continue
                    self.assertGreater(
                        abs(obj.position[2]), half_top,
                        msg=f"{obj.id} sits in the channel at z={obj.position[2]:.1f}")

    async def test_a_loaded_scenario_puts_its_own_water_settings_on_the_solver(self) -> None:
        """Erosion and the open edge belong to the world, like the inlet.

        They used to be left to the first tick's live sync, so between LOAD and
        PLAY the solver reported its own constructor defaults -- erosion ON,
        outlet CLOSED, both the opposite of a fresh WaterState -- and the UI
        drew the checkboxes from that report. Nothing moved, so nothing behaved
        wrongly; the interface simply told the user the opposite of the truth
        about a world they had just loaded.
        """
        world = self._scenario("scenario_dam")
        self.assertFalse(world.water.erosion_enabled)
        self.assertTrue(world.water.outflow_enabled)
        self.assertTrue(world.water.inlet_enabled)
        diagnostics = self.manager.fluid.diagnostics()
        self.assertFalse(diagnostics["erosion"],
                         "solver still reports erosion on after loading a world with it off")
        self.assertGreater(diagnostics["outflow_columns"], 0,
                           "solver still reports the outlet closed after loading it open")
        self.assertTrue(diagnostics["inlet_enabled"])

    async def test_a_road_is_drawn_but_not_felt(self) -> None:
        """ROAD is static and must never reach the obstacle mask or the bed.

        Flat asphalt on a floodplain changes the roughness the flow feels, not
        the elevation of the bed, so water crossing a street unimpeded is the
        correct answer. The day roughness becomes a field (`FLUID_MANNING_N` as
        an array, which the volcano plan needs anyway) is the day this gets a
        second half.
        """
        from app.fluid_solver import SOLID_OBSTACLE_TYPES
        from app.world_state import default_properties
        self.assertNotIn("ROAD", SOLID_OBSTACLE_TYPES)
        self.assertEqual(default_properties("ROAD")["bed_height"], 0.0)
        self.assertTrue(default_properties("ROAD")["is_static"])

        world = self._scenario("scenario_river")
        roads = [o for o in world.objects.values() if o.type == "ROAD"]
        self.assertTrue(roads)
        # Read the arrays the flow kernels actually use. Asserting that
        # `terrain.heights` is unchanged would prove nothing: `set_boundaries`
        # never writes to it -- it reads it and builds the solid mask and the
        # raised-bed offset alongside, and `_bed` is their sum. That assertion
        # would pass just as happily with ROAD given a bed_height of 5 m.
        fluid = self.manager.fluid
        fluid.set_boundaries(self.manager.world.terrain,
                             self.manager.rigid.obstacle_snapshot(),
                             self.manager.terrain_revision + 1,
                             self.manager.obstacle_revision + 1)
        stride = self.manager.world.terrain.width + 1
        for road in roads:
            i = int(round(road.position[0] / config.TERRAIN_CELL_SIZE
                          + config.TERRAIN_CELLS / 2))
            j = int(round(road.position[2] / config.TERRAIN_CELL_SIZE
                          + config.TERRAIN_CELLS / 2))
            index = j * stride + i
            self.assertEqual(int(fluid._obstacle_host[index]), 0,
                             f"{road.id} was rasterized into the solid mask")
            self.assertAlmostEqual(float(fluid._bed_offset_host[index]), 0.0, places=6,
                                   msg=f"{road.id} raised the bed under itself")


if __name__ == "__main__":
    unittest.main(verbosity=2)
