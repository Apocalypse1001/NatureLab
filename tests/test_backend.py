"""Reproducible Foundation 0.2 backend regression tests (stdlib unittest)."""
from __future__ import annotations

import asyncio
import json
import math
import sys
import unittest
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import config, protocol  # noqa: E402
from app.events import EventLog  # noqa: E402
from app.fluid_solver import ShallowWaterFluidSolver  # noqa: E402
from app.rigid_body import ForceRigidBodySystem  # noqa: E402
from app.simulation import SimulationManager  # noqa: E402
from app.world_state import ObjectState, WorldState  # noqa: E402


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

    def test_flat_water_stays_static_when_flow_disabled(self) -> None:
        """Documents the reported symptom (docs/04_TZ_v0.3_roadmap.md v0.4
        'Важная находка'): initialize() fills depth = water_level - terrain,
        which is flat (zero gradient) from tick 1, so with flow_enabled off
        (the default) water on flat terrain never moves on its own."""
        world = self._make_world(slope=0.0)
        solver = ShallowWaterFluidSolver()
        solver.initialize(world)
        depth_before = solver._depth.copy()
        for _ in range(120):
            solver.set_boundaries(world.terrain, {})
            solver.advance(1 / 60, 8, 1 / 120)
        self.assertTrue(np.allclose(solver._depth, depth_before))

    def test_river_flow_creates_sustained_current_on_flat_terrain(self) -> None:
        """water.flow_enabled fixes the above: a west-edge source held full
        and an east-edge sink held near-empty keep a permanent height
        difference, so the same conservative interior flux scheme now
        carries a real, continuous west->east current -- not just a
        one-off transient that resettles."""
        world = self._make_world(slope=0.0)
        solver = ShallowWaterFluidSolver()
        solver.initialize(world)
        solver.set_river_flow(True)
        depth_before = solver._depth.copy()
        for _ in range(120):
            solver.set_boundaries(world.terrain, {})
            solver.advance(1 / 60, 8, 1 / 120)
        self.assertFalse(np.allclose(solver._depth, depth_before))
        self.assertGreater(float(solver._flow_x.mean()), 0.0)

        depth_mid = solver._depth.copy()
        for _ in range(120):
            solver.set_boundaries(world.terrain, {})
            solver.advance(1 / 60, 8, 1 / 120)
        self.assertFalse(np.allclose(solver._depth, depth_mid),
                         "river flow must stay continuous, not settle back to equilibrium")

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

    def test_sampling_ring_clears_bilinear_blend_with_obstacle_edge(self) -> None:
        """Regression: a ring only 0.5 cells past the obstacle disk edge
        still bilinear-interpolated with a dry boundary cell (interpolation
        reaches a full cell beyond the sample point), silently halving the
        depth reading on flat, fully wet terrain."""
        world = self._make_world(slope=0.0)
        world.water.level = 1.2
        solver = ShallowWaterFluidSolver()
        solver.initialize(world)
        position = np.array([[0.0, 0.0, 0.0]])
        radius = np.array([0.7], dtype=np.float32)
        solver.set_boundaries(world.terrain, {"positions": position, "radii": radius})
        samples = solver.sample_for_bodies(position, radius)
        self.assertAlmostEqual(float(samples["depths"][0]), 1.2, places=2)

    def test_nearby_object_does_not_contaminate_another_objects_reading(self) -> None:
        """Regression: with a HOUSE (radius 2.4 m) ~3.6 m from a BOX (radius
        0.7 m) on flat, fully wet terrain, the box's sampled depth dropped
        from 1.2 m to 0.9 m -- purely from the house's hole being close
        enough to enter the box's own sampling neighbourhood."""
        world = self._make_world(slope=0.0)
        world.water.level = 1.2
        solver = ShallowWaterFluidSolver()
        solver.initialize(world)
        positions = np.array([[-30.0, 0.0, 0.0], [-28.0, 0.0, -3.0]], dtype=np.float32)
        radii = np.array([2.4, 0.7], dtype=np.float32)
        solver.set_boundaries(world.terrain, {"positions": positions, "radii": radii})
        samples = solver.sample_for_bodies(positions, radii)
        self.assertAlmostEqual(float(samples["depths"][1]), 1.2, places=1)


class SedimentErosionTests(unittest.TestCase):
    """v0.4 RiverLab: capacity-based erosion/deposition + sediment transport
    (docs/04_TZ_v0.3_roadmap.md v0.4). Fast flow should erode terrain and
    carry sediment downstream; terrain change must feed back into the flow
    it came from (same mutate-in-place mechanism already proven for
    obstacles/terrain.brush), and total solid material (terrain + suspended
    sediment) must not appear from nowhere."""

    @staticmethod
    def _make_channel(slope: float, depth: float = 0.5) -> tuple[WorldState, ShallowWaterFluidSolver]:
        """A sloped bed with UNIFORM water depth, not solver.initialize()'s
        default hydrostatic fill. initialize() sets depth = water_level -
        terrain, which makes the water SURFACE flat by construction (total
        height = terrain + (level - terrain) = level everywhere) -- zero
        gradient, zero flow, from the very first tick. A real channel has
        roughly uniform depth following the slope, which is what actually
        drives sustained flow (found by direct measurement in this session:
        speed was exactly 0.0 at every step with the hydrostatic fill)."""
        world = WorldState()
        w, h = world.terrain.width, world.terrain.height
        xs = np.linspace(0.0, slope, w + 1, dtype=np.float32)
        world.terrain.heights[:, :] = np.tile(xs, (h + 1, 1))
        solver = ShallowWaterFluidSolver()
        solver.initialize(world)
        solver._depth[:, :] = depth
        return world, solver

    def test_fast_flow_erodes_terrain_and_carries_sediment(self) -> None:
        world, solver = self._make_channel(slope=3.0)
        start_terrain = world.terrain.heights.copy()
        for _ in range(600):
            solver.set_boundaries(world.terrain, {})
            solver.advance(1 / 60, 8, 1 / 120)
        eroded = (world.terrain.heights < start_terrain - 1e-4).any()
        self.assertTrue(eroded, "fast flow over erodible terrain should lower it somewhere")
        self.assertGreater(float(solver.get_sediment_grid().sum()), 0.0)

    def test_solid_material_conserved_by_erosion_and_deposition(self) -> None:
        world, solver = self._make_channel(slope=3.0)
        solver.set_boundaries(world.terrain, {})
        start_total = solver.total_solid_material()
        for _ in range(600):
            solver.set_boundaries(world.terrain, {})
            solver.advance(1 / 60, 8, 1 / 120)
        end_total = solver.total_solid_material()
        self.assertAlmostEqual(end_total, start_total, delta=max(1e-3, abs(start_total) * 1e-3))

    def test_terrain_change_feeds_back_into_flow(self) -> None:
        """The same mechanism already proven for obstacles/manual brushing
        (docs/04_TZ_v0.3_roadmap.md 'Главный критерий качества'): once
        erosion has measurably reshaped the bed, the flow computed next
        tick must actually reflect the new shape, not the original one."""
        world, solver = self._make_channel(slope=3.0)
        for _ in range(600):
            solver.set_boundaries(world.terrain, {})
            solver.advance(1 / 60, 8, 1 / 120)
        eroded_terrain = world.terrain.heights.copy()
        depth_on_eroded_bed = solver._depth.copy()

        # replay the same run on a solver that never erodes (frozen terrain
        # snapshot from the start) and confirm the resulting depth differs
        fresh_world, solver2 = self._make_channel(slope=3.0)
        frozen_terrain = fresh_world.terrain.heights.copy()
        for _ in range(600):
            fresh_world.terrain.heights[:, :] = frozen_terrain  # pin the bed, no erosion feedback
            solver2.set_boundaries(fresh_world.terrain, {})
            solver2.advance(1 / 60, 8, 1 / 120)

        self.assertFalse(np.array_equal(eroded_terrain, frozen_terrain))
        self.assertFalse(np.allclose(depth_on_eroded_bed, solver2._depth))

    def test_rock_changes_local_erosion_pattern(self) -> None:
        """Ties to the user's Schauberger-riverbed-rock request: a rock on
        the bed must change WHERE erosion happens, not just add a static
        hole with no consequence for the sediment mechanic."""
        def run(with_rock: bool) -> np.ndarray:
            world, solver = self._make_channel(slope=3.0)
            obstacles = {"positions": [[-20.0, 0.0, 0.0]], "radii": [3.0]} if with_rock else {}
            for _ in range(600):
                solver.set_boundaries(world.terrain, obstacles)
                solver.advance(1 / 60, 8, 1 / 120)
            return world.terrain.heights.copy()

        without_rock = run(with_rock=False)
        with_rock = run(with_rock=True)
        self.assertFalse(np.allclose(without_rock, with_rock))


class TemperatureShadeTests(unittest.TestCase):
    """v0.4 RiverLab, Schauberger hypothesis (docs/04_TZ_v0.3_roadmap.md v0.4,
    docs/01_vision.md 'Viktor Schauberger Lab'): tree shade locally cools
    water, which we treat as increasing flow energy and sediment carrying
    capacity. Implemented as a stateless per-tick multiplier recomputed from
    tree positions -- NOT a persistent/diffusing field (deferred by the
    user's own choice, 2026-09-01). These tests check the mechanism is
    real and comparable (River A vs River B), not that Schauberger was
    "right" -- see the causal- vs engineering-realism note in the roadmap."""

    @staticmethod
    def _make_channel(slope: float, depth: float = 0.5) -> tuple[WorldState, ShallowWaterFluidSolver]:
        world = WorldState()
        w, h = world.terrain.width, world.terrain.height
        xs = np.linspace(0.0, slope, w + 1, dtype=np.float32)
        world.terrain.heights[:, :] = np.tile(xs, (h + 1, 1))
        solver = ShallowWaterFluidSolver()
        solver.initialize(world)
        solver._depth[:, :] = depth
        return world, solver

    def test_no_shade_sources_keeps_factor_at_one(self) -> None:
        world, solver = self._make_channel(slope=1.0)
        solver.set_boundaries(world.terrain, {})
        solver.set_environment(15.0, {"positions": [], "radii": [], "cooling": []})
        np.testing.assert_array_equal(solver._temperature_factor, np.ones_like(solver._temperature_factor))

    def test_shade_raises_local_temperature_factor_above_one(self) -> None:
        world, solver = self._make_channel(slope=1.0)
        solver.set_boundaries(world.terrain, {})
        shade = {"positions": [[0.0, 0.0, 0.0]], "radii": [4.0], "cooling": [3.0]}
        solver.set_environment(15.0, shade)
        center = solver._temperature_factor.shape[0] // 2
        self.assertGreater(float(solver._temperature_factor[center, center]), 1.0)
        # far from the tree, no effect
        self.assertEqual(float(solver._temperature_factor[5, 5]), 1.0)

    def test_temperature_factor_is_clamped(self) -> None:
        world, solver = self._make_channel(slope=1.0)
        solver.set_boundaries(world.terrain, {})
        shade = {"positions": [[0.0, 0.0, 0.0]], "radii": [10.0], "cooling": [500.0]}
        solver.set_environment(15.0, shade)
        self.assertLessEqual(float(solver._temperature_factor.max()), config.TEMP_FACTOR_MAX)

    def test_shade_changes_downstream_flow_and_sediment_river_a_vs_b(self) -> None:
        """The actual A/B comparison the project's own methodology calls
        for: same channel, same starting conditions, only shade differs."""
        def run(with_shade: bool) -> tuple[np.ndarray, float]:
            world, solver = self._make_channel(slope=3.0)
            shade = ({"positions": [[-10.0, 0.0, 0.0]], "radii": [4.0], "cooling": [3.0]}
                     if with_shade else {"positions": [], "radii": [], "cooling": []})
            for _ in range(300):
                solver.set_boundaries(world.terrain, {})
                solver.set_environment(15.0, shade)
                solver.advance(1 / 60, 8, 1 / 120)
            return solver._depth.copy(), float(solver.get_sediment_grid().sum())

        depth_a, sediment_a = run(with_shade=False)
        depth_b, sediment_b = run(with_shade=True)
        self.assertFalse(np.allclose(depth_a, depth_b))
        self.assertNotEqual(sediment_a, sediment_b)

    def test_shade_snapshot_only_includes_shade_casting_bodies(self) -> None:
        world = WorldState()
        world.add_object("BOX", [0.0, 0.0, 0.0])
        world.add_object("TREE", [5.0, 0.0, 0.0])
        rigid = ForceRigidBodySystem()
        rigid.initialize(world, fluid=None, events=EventLog())
        shade = rigid.shade_snapshot()
        self.assertEqual(len(shade["positions"]), 1)
        self.assertAlmostEqual(float(shade["positions"][0][0]), 5.0)


class ForceRigidBodyTests(unittest.TestCase):
    """v0.3: gravity/buoyancy/drag/friction replacing the buoyancy-threshold
    placeholder. See docs/04_TZ_v0.3_roadmap.md milestone v0.3 and
    docs/01_vision.md "Поведение автомобиля" / "Эксперименты"."""

    @staticmethod
    def _make_system(obj_type: str, **overrides):
        world = WorldState()
        obj = world.add_object(obj_type, [0.0, 0.0, 0.0])
        for key, value in overrides.items():
            if key in ("mass", "friction", "buoyancy"):
                setattr(obj, key, value)
            else:
                obj.metadata[key] = value
        rigid = ForceRigidBodySystem()
        rigid.initialize(world, fluid=None, events=EventLog())
        return rigid, obj

    @staticmethod
    def _constant_flow(depth: float, speed: float):
        return {"depths": np.array([depth], dtype=np.float32),
                "velocities": np.array([[speed, 0.0, 0.0]], dtype=np.float32)}

    def test_light_box_moves_before_heavy_car_under_same_flow(self) -> None:
        rigid_box, box = self._make_system("BOX")
        rigid_car, car = self._make_system("CAR")
        samples = self._constant_flow(depth=0.5, speed=2.0)

        def steps_to_move(rigid, obj) -> Optional[int]:
            for step in range(1, 300):
                rigid.step(1 / 60, step / 60, samples)
                if obj.state != ObjectState.INTACT.value:
                    return step
            return None

        box_step = steps_to_move(rigid_box, box)
        car_step = steps_to_move(rigid_car, car)
        self.assertIsNotNone(box_step)
        self.assertTrue(car_step is None or box_step < car_step)

    def test_house_stays_intact_at_shallow_depth(self) -> None:
        rigid, house = self._make_system("HOUSE")
        samples = self._constant_flow(depth=0.3, speed=1.5)
        for step in range(1, 300):
            rigid.step(1 / 60, step / 60, samples)
        self.assertEqual(house.state, ObjectState.INTACT.value)

    def test_deep_enough_water_eventually_floats_any_positive_buoyancy_object(self) -> None:
        """Regression: buoyant force was capped at buoyancy_coeff * weight, so
        e.g. a BOX (buoyancy=0.8) could never drop below 20% ground contact
        and could never reach FLOATING no matter how deep the water got."""
        rigid, box = self._make_system("BOX")
        samples = self._constant_flow(depth=3.0, speed=0.0)
        for step in range(1, 300):
            rigid.step(1 / 60, step / 60, samples)
        self.assertEqual(box.state, ObjectState.FLOATING.value)

    def test_foundation_height_changes_flood_outcome(self) -> None:
        """Experiment A/B from docs/01_vision.md: same water, different
        foundation height must change the outcome, not be an unused field."""
        rigid_low, box_low = self._make_system("BOX", foundation_height=0.0)
        rigid_high, box_high = self._make_system("BOX", foundation_height=2.0)
        samples = self._constant_flow(depth=1.5, speed=0.0)
        for step in range(1, 300):
            rigid_low.step(1 / 60, step / 60, samples)
            rigid_high.step(1 / 60, step / 60, samples)
        self.assertNotEqual(box_low.state, box_high.state)
        self.assertEqual(box_high.state, ObjectState.INTACT.value)

    def test_bodies_do_not_pass_through_each_other(self) -> None:
        """v0.3 interim collision: two disks pushed together by water drag
        must not end up overlapping past their combined footprint radius."""
        world = WorldState()
        box_a = world.add_object("BOX", [-2.0, 0.0, 0.0])
        box_b = world.add_object("BOX", [2.0, 0.0, 0.0])
        rigid = ForceRigidBodySystem()
        rigid.initialize(world, fluid=None, events=EventLog())
        samples = {"depths": np.array([0.6, 0.6], dtype=np.float32),
                   "velocities": np.array([[3.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)}
        for step in range(1, 400):
            rigid.step(1 / 60, step / 60, samples)
        separation = abs(box_a.position[0] - box_b.position[0])
        min_separation = 2 * world.objects[box_a.id].metadata["footprint_radius"]
        self.assertGreaterEqual(separation, min_separation - 0.05)

    def test_collision_conserves_momentum_and_reduces_relative_speed(self) -> None:
        """Heavier body barely moves; lighter one bounces back -- and total
        momentum along the contact normal is conserved by construction.
        Calls _resolve_collisions directly (not step()) so ground friction
        -- a legitimate external force -- doesn't muddy a momentum check
        that is specifically about the collision response itself."""
        world = WorldState()
        world.add_object("BOX", [-1.0, 0.0, 0.0])   # mass 50
        world.add_object("HOUSE", [1.0, 0.0, 0.0])  # mass 20000
        rigid = ForceRigidBodySystem()
        rigid.initialize(world, fluid=None, events=EventLog())
        rigid.buffer.velocities[0] = [4.0, 0.0, 0.0]  # light body moving toward heavy one
        momentum_before = (rigid.buffer.masses[:, None] * rigid.buffer.velocities).sum(axis=0)
        rigid._resolve_collisions(2, 1 / 60)
        momentum_after = (rigid.buffer.masses[:, None] * rigid.buffer.velocities).sum(axis=0)
        np.testing.assert_allclose(momentum_before, momentum_after, atol=1e-3)
        self.assertLess(abs(float(rigid.buffer.velocities[1, 0])), 0.05,
                        "house should barely move from a 50kg box impact")

    def test_collision_event_fires_once_per_contact_not_every_tick(self) -> None:
        world = WorldState()
        box_a = world.add_object("BOX", [-0.5, 0.0, 0.0])
        world.add_object("BOX", [0.5, 0.0, 0.0])
        events = EventLog()
        rigid = ForceRigidBodySystem()
        rigid.initialize(world, fluid=None, events=events)
        samples = {"depths": np.zeros(2, dtype=np.float32), "velocities": np.zeros((2, 3), dtype=np.float32)}
        for step in range(1, 60):
            rigid.step(1 / 60, step / 60, samples)
        collisions = [e for e in events.all() if e["type"] == "OBJECT_COLLISION"]
        self.assertEqual(len(collisions), 1)


class TreeRootAnchorTests(unittest.TestCase):
    """v0.3: TREE gets root_strength -- extra static resistance on top of
    Coulomb friction -- so it does not slide away like a BOX under an
    ordinary flood, only uproots (BROKEN) once drag or a body impact
    exceeds that threshold. See docs/01_vision.md "root strength" /
    "break strength" and the user's questions 2026-09-01 about whether a
    tsunami-scale flow or a car impact should be able to detach a tree."""

    def test_tree_stays_rooted_under_ordinary_flood_flow(self) -> None:
        world = WorldState()
        world.add_object("TREE", [0.0, 0.0, 0.0])
        events = EventLog()
        rigid = ForceRigidBodySystem()
        rigid.initialize(world, fluid=None, events=events)
        samples = {"depths": np.array([1.0], dtype=np.float32),
                   "velocities": np.array([[2.0, 0.0, 0.0]], dtype=np.float32)}
        for step in range(1, 300):
            rigid.step(1 / 60, step / 60, samples)
        tree = next(iter(world.objects.values()))
        self.assertTrue(bool(rigid.buffer.rooted[0]))
        self.assertNotEqual(tree.state, ObjectState.BROKEN.value)

    def test_tree_uproots_under_extreme_flow(self) -> None:
        world = WorldState()
        world.add_object("TREE", [0.0, 0.0, 0.0])
        events = EventLog()
        rigid = ForceRigidBodySystem()
        rigid.initialize(world, fluid=None, events=events)
        samples = {"depths": np.array([2.0], dtype=np.float32),
                   "velocities": np.array([[12.0, 0.0, 0.0]], dtype=np.float32)}
        broke = False
        for step in range(1, 120):
            rigid.step(1 / 60, step / 60, samples)
            if not rigid.buffer.rooted[0]:
                broke = True
                break
        self.assertTrue(broke, "extreme flow should eventually uproot the tree")
        tree = next(iter(world.objects.values()))
        # BROKEN is set the instant it uproots, but the same tick can also
        # already reclassify it as FLOATING (it was already in deep water,
        # which is exactly why it broke) -- that's correct, not a bug: the
        # OBJECT_BROKEN event is the authoritative record of the moment,
        # the object's current state reflects what it's doing right now.
        self.assertIn(tree.state, (ObjectState.BROKEN.value, ObjectState.FLOATING.value,
                                    ObjectState.MOVING.value))
        broken_events = [e for e in events.all() if e["type"] == "OBJECT_BROKEN"]
        self.assertEqual(len(broken_events), 1)
        self.assertEqual(broken_events[0]["cause"], "drag_exceeded_root_strength")

    def test_tree_uproots_from_body_impact(self) -> None:
        """A fast-moving car slamming into a tree can uproot it even with
        no water at all -- root_strength must be checked against collision
        impact force, not only against water drag."""
        world = WorldState()
        world.add_object("TREE", [3.0, 0.0, 0.0])
        world.add_object("CAR", [-2.0, 0.0, 0.0])
        events = EventLog()
        rigid = ForceRigidBodySystem()
        rigid.initialize(world, fluid=None, events=events)
        car_idx = rigid.buffer.ids.index(next(o.id for o in world.objects.values() if o.type == "CAR"))
        rigid.buffer.velocities[car_idx] = [15.0, 0.0, 0.0]  # a very hard, direct hit
        no_water = {"depths": np.zeros(2, dtype=np.float32), "velocities": np.zeros((2, 3), dtype=np.float32)}
        tree_idx = rigid.buffer.ids.index(next(o.id for o in world.objects.values() if o.type == "TREE"))
        for step in range(1, 60):
            rigid.step(1 / 60, step / 60, no_water)
            if not rigid.buffer.rooted[tree_idx]:
                break
        self.assertFalse(bool(rigid.buffer.rooted[tree_idx]), "hard car impact should uproot the tree")
        broken_events = [e for e in events.all() if e["type"] == "OBJECT_BROKEN"]
        self.assertEqual(len(broken_events), 1)
        self.assertEqual(broken_events[0]["cause"], "body_impact_exceeded_root_strength")

    def test_broken_tree_still_carves_a_fluid_obstacle(self) -> None:
        """A fallen tree remains a registered rigid body, so it keeps
        blocking water exactly like before it broke -- 'дерево упало ->
        стало препятствием' from docs/01_vision.md falls out for free."""
        world = WorldState()
        tree = world.add_object("TREE", [0.0, 0.0, 0.0])
        rigid = ForceRigidBodySystem()
        rigid.initialize(world, fluid=None, events=EventLog())
        rigid.buffer.rooted[0] = False  # simulate an already-broken tree
        snapshot = rigid.obstacle_snapshot()
        self.assertIn(tree.id, snapshot["ids"])
        self.assertGreater(float(snapshot["radii"][0]), 0.0)


class RiverbedRockTests(unittest.TestCase):
    """v0.4 RiverLab (docs/04_TZ_v0.3_roadmap.md): ROCK reuses root_strength
    (tuned so high it is permanently immovable) and the existing
    footprint_radius/scale machinery, so 'effect scales with rock size'
    needs no new property -- verify that's actually true, not assumed."""

    def test_rock_never_moves_under_extreme_drag_and_impact(self) -> None:
        world = WorldState()
        world.add_object("ROCK", [0.0, 0.0, 0.0])
        world.add_object("CAR", [-2.0, 0.0, 0.0])
        events = EventLog()
        rigid = ForceRigidBodySystem()
        rigid.initialize(world, fluid=None, events=events)
        car_idx = rigid.buffer.ids.index(next(o.id for o in world.objects.values() if o.type == "CAR"))
        rigid.buffer.velocities[car_idx] = [20.0, 0.0, 0.0]  # ram it straight into the rock
        extreme_flow = {"depths": np.array([5.0, 5.0], dtype=np.float32),
                        "velocities": np.array([[20.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=np.float32)}
        rock_idx = rigid.buffer.ids.index(next(o.id for o in world.objects.values() if o.type == "ROCK"))
        start_pos = rigid.buffer.positions[rock_idx].copy()
        for step in range(1, 300):
            rigid.step(1 / 60, step / 60, extreme_flow)
        np.testing.assert_allclose(rigid.buffer.positions[rock_idx], start_pos, atol=1e-4)
        self.assertTrue(bool(rigid.buffer.rooted[rock_idx]))
        rock = next(o for o in world.objects.values() if o.type == "ROCK")
        self.assertEqual(rock.state, ObjectState.INTACT.value)

    def test_rock_footprint_scales_with_object_scale(self) -> None:
        world = WorldState()
        small = world.add_object("ROCK", [-10.0, 0.0, 0.0])
        big = world.add_object("ROCK", [10.0, 0.0, 0.0])
        big.scale = [3.0, 3.0, 3.0]
        rigid = ForceRigidBodySystem()
        rigid.initialize(world, fluid=None, events=EventLog())
        small_idx = rigid.buffer.index[small.id]
        big_idx = rigid.buffer.index[big.id]
        self.assertGreater(float(rigid.buffer.footprint_radii[big_idx]),
                            float(rigid.buffer.footprint_radii[small_idx]) * 2.5)

    def test_bigger_rock_disrupts_flow_more_than_smaller_rock(self) -> None:
        def run(scale: float) -> np.ndarray:
            world = WorldState()
            w, h = world.terrain.width, world.terrain.height
            xs = np.linspace(0.0, 2.5, w + 1, dtype=np.float32)
            world.terrain.heights[:, :] = np.tile(xs, (h + 1, 1))
            world.water.level = 1.0
            solver = ShallowWaterFluidSolver()
            solver.initialize(world)
            base_radius = 1.0
            for _ in range(60):
                solver.set_boundaries(world.terrain, {"positions": [[-30.0, 0.0, 0.0]],
                                                       "radii": [base_radius * scale]})
                solver.advance(1 / 60, 8, 1 / 120)
            return solver._depth.copy()

        baseline = run(scale=0.01)  # negligible obstacle, close to "no rock"
        small_rock = run(scale=1.0)
        big_rock = run(scale=4.0)

        disruption_small = float(np.abs(small_rock - baseline).sum())
        disruption_big = float(np.abs(big_rock - baseline).sum())
        self.assertGreater(disruption_big, disruption_small)


class RiverbedRockDeflectionTests(unittest.TestCase):
    """v0.4 RiverLab, the roadmap's "камни на дне реки меняют русло" item.

    Its explicit note is that a rock is part of the terrain/riverbed, not a
    rigid body with mass and buoyancy -- so ROCK is kept out of the binary
    obstacle mask (an infinitely tall wall) and instead raises the effective
    bed by a dome of `bed_height` (see ShallowWaterFluidSolver
    .set_bed_obstructions). These tests check the three things the roadmap
    actually asks for and that the earlier rock work did not cover: real
    lateral deflection, an effect that scales with the rock's size *and
    position*, and meandering (deposition on one side, scour on the other)
    emerging from the existing sediment mechanic rather than being scripted.
    """

    @staticmethod
    def _river(slope: float = 2.0, depth: float = 0.6,
               bank: float = 0.0) -> tuple[WorldState, ShallowWaterFluidSolver]:
        """A flowing channel: bed falls west->east, optional raised z-banks.

        Uses water.flow_enabled, otherwise the surface is flat from tick 1
        and nothing moves at all -- see the v0.4 "Важная находка" note and
        test_flat_water_stays_static_when_flow_disabled.
        """
        world = WorldState()
        w, h = world.terrain.width, world.terrain.height
        xs = np.linspace(slope, 0.0, w + 1, dtype=np.float32)
        bed = np.tile(xs, (h + 1, 1))
        zs = np.linspace(-1.0, 1.0, h + 1, dtype=np.float32)
        world.terrain.heights[:, :] = bed + bank * (zs ** 2)[:, None]
        world.water.flow_enabled = True
        solver = ShallowWaterFluidSolver()
        solver.initialize(world)
        # water surface parallels the downstream slope, so raised banks are
        # genuinely shallow/dry rather than uniformly deep
        solver._depth[:, :] = np.maximum(0.0, (bed + depth) - world.terrain.heights)
        return world, solver

    @staticmethod
    def _advance(world, solver, bed: dict, steps: int, walls: Optional[dict] = None) -> None:
        for _ in range(steps):
            solver.set_boundaries(world.terrain, walls or {})
            solver.set_bed_obstructions(bed)
            solver.advance(1 / 60, 8, 1 / 120)

    @staticmethod
    def _rock(x: float = 0.0, z: float = 0.0, radius: float = 3.0,
              height: float = 0.8) -> dict:
        return {"positions": [[x, 0.0, z]], "radii": [radius], "heights": [height]}

    def test_deep_water_flows_over_a_rock_but_never_over_a_wall(self) -> None:
        """The whole point of bed_height: the same footprint a HOUSE would
        carve as a dry, infinitely tall hole must, for a boulder, end up
        submerged and still carrying water once the river is deeper than the
        rock is tall."""
        over_rock = 0.0
        for depth in (0.4, 2.5):
            world, solver = self._river(depth=depth)
            self._advance(world, solver, self._rock(), steps=300)
            over_rock = float(solver._depth[48:53, 48:53].mean())

            wall_world, wall_solver = self._river(depth=depth)
            self._advance(wall_world, wall_solver, {}, steps=300,
                          walls={"positions": [[0.0, 0.0, 0.0]], "radii": [3.0]})
            over_wall = float(wall_solver._depth[48:53, 48:53].mean())

            self.assertEqual(over_wall, 0.0, f"a wall must stay dry at depth {depth}")
            self.assertGreater(over_rock, 0.0,
                               f"a riverbed rock must not be a dry wall at depth {depth}")
        # the deep case must be properly submerged, not merely damp
        self.assertGreater(over_rock, 0.8)

    def test_rock_deflects_the_current_sideways(self) -> None:
        """Straight channel, so any lateral (z) flow at all is deflection
        caused by the rock and nothing else."""
        world, solver = self._river()
        self._advance(world, solver, {}, steps=300)
        self.assertEqual(float(np.abs(solver._flow_z).max()), 0.0)

        world, solver = self._river()
        self._advance(world, solver, self._rock(), steps=300)
        self.assertGreater(float(np.abs(solver._flow_z).max()), 0.0)

    def test_mid_channel_rock_disturbs_flow_more_than_the_same_rock_at_the_bank(self) -> None:
        """The roadmap asks for the effect to scale with position, not only
        size: an identical rock out on the shallow bank must matter far less
        than one in the deep middle. Nothing encodes that rule -- it falls
        out of a rock displacing water only where there is water."""
        def flow_field(bed: dict) -> tuple[np.ndarray, np.ndarray]:
            world, solver = self._river(bank=1.5)
            self._advance(world, solver, bed, steps=600)
            return solver._flow_x.copy(), solver._flow_z.copy()

        base_x, base_z = flow_field({})
        mid_x, mid_z = flow_field(self._rock(z=0.0))
        bank_x, bank_z = flow_field(self._rock(z=30.0))

        mid = float(np.abs(mid_x - base_x).sum() + np.abs(mid_z - base_z).sum())
        bank = float(np.abs(bank_x - base_x).sum() + np.abs(bank_z - base_z).sum())
        self.assertGreater(mid, bank * 1.5,
                           "a mid-channel rock must dominate a bank rock, not merely beat it")

    def test_taller_rock_deflects_more_than_a_flatter_one_of_equal_footprint(self) -> None:
        """bed_height is a real degree of freedom, not decoration: same disk,
        different protrusion, different amount of deflected flow."""
        def deflection(height: float) -> float:
            world, solver = self._river()
            self._advance(world, solver, self._rock(height=height), steps=300)
            return float(np.abs(solver._flow_z).sum())

        self.assertGreater(deflection(0.8), deflection(0.2))

    def test_rock_scours_its_flanks_and_fills_its_lee(self) -> None:
        """The actual meandering claim, as a River A vs River B comparison
        (the methodology docs/01_vision.md requires): identical channels,
        only the rock differs. Downstream of the rock the flow stalls and
        drops its load (bed rises), while squeezing past the flanks speeds it
        up and scours (bed falls). That asymmetry is what bends a straight
        channel.

        Needs a long enough run: the lee first scours while the flow
        reorganises around the new obstruction and only then starts filling,
        so the sign settles at roughly 25s of sim time (measured) -- erosion
        is deliberately slow, see the config module docstring.
        """
        def bed_change(bed: dict) -> np.ndarray:
            world, solver = self._river()
            before = world.terrain.heights.copy()
            self._advance(world, solver, bed, steps=1800)
            return world.terrain.heights - before

        delta = bed_change(self._rock()) - bed_change({})
        lee = float(delta[49:52, 54:61].mean())
        flanks = float(np.r_[delta[44:47, 48:54].ravel(),
                             delta[54:57, 48:54].ravel()].mean())
        self.assertGreater(lee, 0.0, "sediment must be deposited in the rock's lee")
        self.assertLess(flanks, 0.0, "the flow squeezing past the flanks must scour them")

    def test_rock_is_never_eroded_away_by_the_river_it_deflects(self) -> None:
        """A boulder is bedrock. Unshielded, the river would just dig a
        symmetric pit under the rock and drown the flank/lee asymmetry the
        test above depends on."""
        world, solver = self._river()
        before = world.terrain.heights.copy()
        self._advance(world, solver, self._rock(), steps=900)
        under_rock = (world.terrain.heights - before)[48:53, 48:53]
        self.assertGreaterEqual(float(under_rock.min()), 0.0)

    def test_moving_a_rock_leaves_no_crater_behind(self) -> None:
        """The bed dome is recomputed from live positions every tick and
        never written into world terrain (same stateless choice as the shade
        temperature field), so dragging a rock in the editor must not leave a
        permanent bump at the old spot."""
        world, solver = self._river()
        self._advance(world, solver, self._rock(x=-20.0), steps=120)
        self.assertGreater(float(solver._bed_offset.max()), 0.0)
        solver.set_bed_obstructions({})
        self.assertEqual(float(solver._bed_offset.max()), 0.0)

    def test_rock_is_kept_out_of_the_solid_obstacle_mask(self) -> None:
        """A ROCK must reach the solver as bed, not as a wall -- otherwise the
        infinitely tall obstacle mask would win and none of the above could
        happen. A HOUSE must still be a wall."""
        world = WorldState()
        rock = world.add_object("ROCK", [0.0, 0.0, 0.0])
        house = world.add_object("HOUSE", [10.0, 0.0, 0.0])
        rigid = ForceRigidBodySystem()
        rigid.initialize(world, fluid=None, events=EventLog())

        self.assertEqual(rigid.obstacle_snapshot()["ids"], [house.id])
        bed = rigid.bed_snapshot()
        self.assertEqual(len(bed["positions"]), 1)
        self.assertGreater(float(bed["heights"][0]), 0.0)
        self.assertEqual(float(rigid.buffer.bed_heights[rigid.buffer.index[rock.id]]),
                         float(bed["heights"][0]))

    def test_bed_height_scales_with_vertical_scale_not_horizontal(self) -> None:
        """Footprint radius already scales with the horizontal scale; height
        must follow the vertical one, so a rock stretched only sideways gets
        wider without also getting taller."""
        world = WorldState()
        tall = world.add_object("ROCK", [-10.0, 0.0, 0.0])
        tall.scale = [1.0, 3.0, 1.0]
        wide = world.add_object("ROCK", [10.0, 0.0, 0.0])
        wide.scale = [3.0, 1.0, 3.0]
        rigid = ForceRigidBodySystem()
        rigid.initialize(world, fluid=None, events=EventLog())
        buf = rigid.buffer
        tall_i, wide_i = buf.index[tall.id], buf.index[wide.id]

        self.assertGreater(float(buf.bed_heights[tall_i]), float(buf.bed_heights[wide_i]))
        self.assertGreater(float(buf.footprint_radii[wide_i]), float(buf.footprint_radii[tall_i]))

    def test_rock_added_to_a_running_world_reaches_the_solver_as_bed(self) -> None:
        """End-to-end through SimulationManager, not just the solver in
        isolation: a rock placed in the editor must actually arrive as a bed
        dome (bed_snapshot -> set_bed_obstructions) and not as a wall. This is
        the wiring that the class of bug this project keeps hitting lives in --
        real physics on the backend that nothing ever connects up.
        """
        manager = SimulationManager()
        manager.world.water.flow_enabled = True
        manager.start()
        manager._step_once()
        self.assertEqual(float(manager.fluid._bed_offset.max()), 0.0)

        manager.apply_object_add({"type": "ROCK", "position": [0.0, 0.0, 0.0]})
        manager._step_once()
        self.assertGreater(float(manager.fluid._bed_offset.max()), 0.0,
                           "a ROCK must raise the effective bed")
        self.assertFalse(bool(manager.fluid._obstacle_mask.any()),
                         "a ROCK must not become a solid wall")


if __name__ == "__main__":
    unittest.main(verbosity=2)
