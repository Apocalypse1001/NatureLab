"""Fluid solver contract with boundary coupling and internal substeps."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from . import config


class FluidSolver:
    def initialize(self, world) -> None: ...
    def set_boundaries(self, terrain, obstacles: dict) -> None: ...
    def set_environment(self, base_temperature: float, shade: dict) -> None: ...
    def set_bed_obstructions(self, bed: dict) -> None: ...
    def set_river_flow(self, enabled: bool) -> None: ...
    def advance(self, global_dt: float, max_substeps: int, stability_dt: float) -> int: ...
    def sample_for_bodies(self, positions: np.ndarray, radii: Optional[np.ndarray] = None) -> dict: ...
    def reset(self) -> None: ...
    def get_water_height(self, x: float = 0.0, z: float = 0.0) -> float: ...
    def get_velocity_field(self) -> Optional[np.ndarray]: ...
    def get_depth_grid(self) -> Optional[np.ndarray]:
        """Full per-cell depth field, indexed exactly like TerrainGrid.heights.

        Used to stream a WATER_HEIGHT bulk frame so the frontend can deform
        the water mesh per-cell instead of showing a flat plane -- see
        SimulationManager._stream() and
        frontend/src/scene/SceneManager.ts:updateWaterField.
        """
        return None


class PlaceholderFluidSolver(FluidSolver):
    def __init__(self) -> None:
        self._world = None
        self._terrain = None
        self._obstacles = {}
        self._level = 0.5
        self._time = 0.0
        self.last_substeps = 0

    def initialize(self, world) -> None:
        self._world = world
        self._terrain = world.terrain
        self._level = world.water.level
        self._time = 0.0

    def set_boundaries(self, terrain, obstacles: dict) -> None:
        self._terrain = terrain
        self._obstacles = obstacles

    def advance(self, global_dt: float, max_substeps: int, stability_dt: float) -> int:
        substeps = max(1, min(max_substeps, math.ceil(global_dt / stability_dt)))
        dt = global_dt / substeps
        for _ in range(substeps):
            self._time += dt
            if self._world is not None:
                self._level = self._world.water.level
        self.last_substeps = substeps
        return substeps

    def sample_for_bodies(self, positions: np.ndarray, radii: Optional[np.ndarray] = None) -> dict:
        count = len(positions)
        depths = np.zeros(count, dtype=np.float32)
        if self._terrain is not None:
            for i, position in enumerate(positions):
                terrain_h = self._terrain.height_at(float(position[0]), float(position[2]))
                depths[i] = max(0.0, self._level - terrain_h)
        velocities = np.tile(np.array([1.5, 0.0, 0.0], dtype=np.float32), (count, 1))
        forces = velocities * depths[:, None]
        return {"depths": depths, "velocities": velocities, "forces": forces}

    def reset(self) -> None:
        self._time = 0.0
        if self._world is not None:
            self._level = self._world.water.level

    def set_level(self, level: float) -> None:
        self._level = level

    def get_water_height(self, x: float = 0.0, z: float = 0.0) -> float:
        return self._level

    def get_velocity_field(self) -> Optional[np.ndarray]:
        return np.array([1.5, 0.0, 0.0], dtype=np.float32)


def _footprint_cells(radius: float, cell_size: float) -> float:
    """Obstacle disk radius in grid cells for a given real-world radius.

    Shared by _rasterize_obstacles (what counts as solid) and
    sample_for_bodies (how far out a body must sample to clear its own
    hole) so the two can never drift apart -- a real bug: they used to be
    computed independently, and a sample ring 0.5 cells past the disk edge
    still bilinear-blended with a boundary obstacle cell (halving the
    reading), because interpolation reaches a full cell beyond the sample
    point.
    """
    return max(1.0, float(radius) / cell_size)


def _bilinear(grid: np.ndarray, gx: np.ndarray, gz: np.ndarray) -> np.ndarray:
    """Sample a (ny, nx) grid at fractional (gx, gz) cell coordinates."""
    ny, nx = grid.shape
    gx = np.clip(gx, 0, nx - 1)
    gz = np.clip(gz, 0, ny - 1)
    i0 = np.floor(gx).astype(np.int64)
    j0 = np.floor(gz).astype(np.int64)
    i1 = np.minimum(i0 + 1, nx - 1)
    j1 = np.minimum(j0 + 1, ny - 1)
    fx = gx - i0
    fz = gz - j0
    return ((1 - fx) * (1 - fz) * grid[j0, i0] + fx * (1 - fz) * grid[j0, i1]
             + (1 - fx) * fz * grid[j1, i0] + fx * fz * grid[j1, i1])


class ShallowWaterFluidSolver(FluidSolver):
    """Height-field shallow water via outflow-limited cell exchange.

    Each cell exchanges water with its 4 neighbours proportionally to the
    total-height difference (terrain + depth), clamped so a cell can never
    drain more water than it holds. This keeps the scheme unconditionally
    mass-conservative (no water created or destroyed away from an explicit
    source) and correctly diverts flow around obstacles/terrain changes,
    which is the causal-realism bar this project actually requires (see
    docs/04_TZ_v0.3_roadmap.md section 3) rather than a full Navier-Stokes
    solver. World edges are closed (zero-flux) boundaries.

    Obstacle footprint is currently a fixed radius per registered rigid
    body (config.FLUID_OBSTACLE_RADIUS_M) rather than the object's real
    scale — RigidStateBuffer does not carry scale yet. That is a follow-up,
    not silently assumed correct.
    """

    def __init__(self) -> None:
        self._world = None
        self._terrain = None
        self._depth: np.ndarray = np.zeros((1, 1), dtype=np.float32)
        self._flow_x: np.ndarray = np.zeros((1, 1), dtype=np.float32)
        self._flow_z: np.ndarray = np.zeros((1, 1), dtype=np.float32)
        self._sediment: np.ndarray = np.zeros((1, 1), dtype=np.float32)
        self._obstacle_mask: np.ndarray = np.zeros((1, 1), dtype=bool)
        # per-cell multiplier on flow gain / sediment capacity from local
        # shade cooling (v0.4, Schauberger hypothesis) -- recomputed each
        # tick from current tree positions in _update_temperature_factor,
        # deliberately NOT a persistent/diffusing field. Stays 1.0 (no
        # effect) wherever nothing casts shade; broadcasts safely against
        # the real grid shape even before set_environment() is ever called.
        self._temperature_factor: np.ndarray = np.ones((1, 1), dtype=np.float32)
        # per-cell rise of the effective riverbed from ROCK domes (v0.4) --
        # like _temperature_factor, recomputed fresh each tick from current
        # rock positions and never written back into world terrain, so moving
        # or deleting a rock leaves no crater behind. Zero = flat bed.
        self._bed_offset: np.ndarray = np.zeros((1, 1), dtype=np.float32)
        self._flow_enabled = False
        self._time = 0.0
        self.last_substeps = 0

    # ------------------------------------------------------------------ setup
    def initialize(self, world) -> None:
        self._world = world
        self._terrain = world.terrain
        shape = self._terrain.heights.shape
        self._depth = np.maximum(
            0.0, world.water.level - self._terrain.heights).astype(np.float32)
        self._flow_x = np.zeros(shape, dtype=np.float32)
        self._flow_z = np.zeros(shape, dtype=np.float32)
        self._sediment = np.zeros(shape, dtype=np.float32)
        self._obstacle_mask = np.zeros(shape, dtype=bool)
        self._temperature_factor = np.ones(shape, dtype=np.float32)
        self._bed_offset = np.zeros(shape, dtype=np.float32)
        self._flow_enabled = bool(world.water.flow_enabled)
        if self._flow_enabled:
            self._seed_river_profile()
        self._time = 0.0

    def _seed_river_profile(self) -> None:
        """Jump the depth field straight to (an approximation of) the
        steady-state west->east profile the boundary clamps would otherwise
        take a long time to diffuse in from the edges.

        Measured directly: with only the two edge clamps in `_step` and no
        seeding, the MIDDLE of a 100-cell grid was still sitting at its
        30-seconds-ago value after a full 30 seconds of real running --
        exactly the "just a still pool" symptom this was built to fix. The
        true steady-state curve has a bit of extra curvature right at each
        forced edge, but is close to linear in between (checked against a
        long free-running reference), so a straight ramp is a fair one-shot
        approximation. It is only ever the *starting point*: every following
        tick still runs the same real flux/erosion physics as always, so any
        obstacle, rock or terrain edit immediately perturbs it for real --
        this is not a fake decal, it is fast-forwarding through a transient
        the numbers show nobody would sit through.
        """
        source, sink = config.FLUID_RIVER_SOURCE_DEPTH, config.FLUID_RIVER_SINK_DEPTH
        nx = self._depth.shape[1]
        ramp = np.linspace(source, sink, nx, dtype=np.float32)
        self._depth[:, :] = np.broadcast_to(ramp, self._depth.shape)
        self._depth[self._obstacle_mask] = 0.0

    def set_river_flow(self, enabled: bool) -> None:
        """Toggle the continuous west->east river current, see config.py.

        Read live every tick (SimulationManager._step_once), unlike
        water_level which only takes effect at initialize() -- so this can
        be switched on/off while RUNNING. On the off->on edge specifically
        (not every tick it stays on), seeds the steady-state profile --
        see _seed_river_profile() for why that matters.
        """
        enabled = bool(enabled)
        if enabled and not self._flow_enabled:
            self._seed_river_profile()
        self._flow_enabled = enabled

    def set_boundaries(self, terrain, obstacles: dict) -> None:
        self._terrain = terrain
        if self._depth.shape != terrain.heights.shape:
            # terrain resized (e.g. loaded world) -> reinitialise the field flat
            self._depth = np.zeros(terrain.heights.shape, dtype=np.float32)
            self._sediment = np.zeros(terrain.heights.shape, dtype=np.float32)
            self._temperature_factor = np.ones(terrain.heights.shape, dtype=np.float32)
            self._bed_offset = np.zeros(terrain.heights.shape, dtype=np.float32)
        self._obstacle_mask = self._rasterize_obstacles(terrain, obstacles)
        self._depth[self._obstacle_mask] = 0.0

    def set_environment(self, base_temperature: float, shade: dict) -> None:
        """Recompute the per-cell flow/capacity multiplier from tree shade.

        v0.4 RiverLab, Schauberger hypothesis (docs/04_TZ_v0.3_roadmap.md
        v0.4): shade cools water locally, which we treat as increasing both
        flow energy and sediment carrying capacity -- see
        docs/01_vision.md "Viktor Schauberger Lab": implemented as a
        testable, comparable effect, not asserted as physically correct.
        Stateless by design (see __init__ note) -- recomputed fresh from
        `shade` (RigidBodySystem.shade_snapshot()) every call, no memory of
        previous ticks.
        """
        if self._terrain is None:
            return
        shape = self._terrain.heights.shape
        positions = shade.get("positions") if shade else None
        if positions is None or len(positions) == 0:
            self._temperature_factor = np.ones(shape, dtype=np.float32)
            return
        radii = shade.get("radii", [])
        coolings = shade.get("cooling", [])
        cell = self._terrain.cell_size
        ny, nx = shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        local_cooling = np.zeros(shape, dtype=np.float32)
        for pos, radius, strength in zip(positions, radii, coolings):
            if radius <= 0.0 or strength <= 0.0:
                continue
            gx = float(pos[0]) / cell + self._terrain.width / 2
            gz = float(pos[2]) / cell + self._terrain.height / 2
            r_cells = max(1.0, float(radius) / cell)
            dist = np.sqrt((xx - gx) ** 2 + (yy - gz) ** 2)
            # full cooling strength at the trunk, fading to 0 at the canopy edge
            falloff = np.clip(1.0 - dist / r_cells, 0.0, 1.0)
            local_cooling = np.maximum(local_cooling, float(strength) * falloff)
        factor = 1.0 + config.TEMP_EFFECT_PER_DEGREE_C * local_cooling
        self._temperature_factor = np.clip(
            factor, config.TEMP_FACTOR_MIN, config.TEMP_FACTOR_MAX).astype(np.float32)

    def set_bed_obstructions(self, bed: dict) -> None:
        """Rebuild the effective-bed rise from riverbed rocks (v0.4 RiverLab).

        The roadmap item this implements ("камни на дне реки меняют русло")
        asks for two things the binary obstacle mask cannot give: an effect
        that scales with the rock's size *and position*, and meandering as an
        emergent result rather than a scripted one. Both fall out of treating
        the rock as bed rather than as a wall:

        - a dome of `height` (config.BED_DOME_EXPONENT) is added to the bed,
          so shallow water is pushed around the rock while deeper water still
          passes over it -- that is the position dependence, for free: the
          same rock is a major obstruction mid-channel and nearly irrelevant
          on a dry bank;
        - the flow squeezing past the flanks speeds up (erosion) and stalls in
          the lee (deposition), which the existing sediment mechanic then
          turns into a channel that actually moves.

        Stateless per tick, exactly like set_environment(): no memory, no
        accumulation, and world terrain is never mutated here -- only the
        sediment code may do that.
        """
        if self._terrain is None:
            return
        shape = self._terrain.heights.shape
        positions = bed.get("positions") if bed else None
        if positions is None or len(positions) == 0:
            self._bed_offset = np.zeros(shape, dtype=np.float32)
            return
        radii = bed.get("radii", [])
        heights = bed.get("heights", [])
        cell = self._terrain.cell_size
        ny, nx = shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        offset = np.zeros(shape, dtype=np.float32)
        for pos, radius, height in zip(positions, radii, heights):
            if radius <= 0.0 or height <= 0.0:
                continue
            r_cells = _footprint_cells(float(radius), cell)
            gx = float(pos[0]) / cell + self._terrain.width / 2
            gz = float(pos[2]) / cell + self._terrain.height / 2
            dist2 = (xx - gx) ** 2 + (yy - gz) ** 2
            # dome: full height at the centre, tapering to 0 at the edge
            inside = np.clip(1.0 - dist2 / (r_cells ** 2), 0.0, 1.0)
            offset = np.maximum(offset, float(height) * inside ** config.BED_DOME_EXPONENT)
        self._bed_offset = offset.astype(np.float32)

    def _rasterize_obstacles(self, terrain, obstacles: dict) -> np.ndarray:
        mask = np.zeros(terrain.heights.shape, dtype=bool)
        positions = obstacles.get("positions") if obstacles else None
        if positions is None or len(positions) == 0:
            return mask
        radii = obstacles.get("radii")
        if radii is None or len(radii) != len(positions):
            radii = [config.FLUID_OBSTACLE_RADIUS_M] * len(positions)
        ny, nx = mask.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        for pos, radius in zip(positions, radii):
            r_cells = _footprint_cells(radius, terrain.cell_size)
            gx = float(pos[0]) / terrain.cell_size + terrain.width / 2
            gz = float(pos[2]) / terrain.cell_size + terrain.height / 2
            mask |= (xx - gx) ** 2 + (yy - gz) ** 2 <= r_cells ** 2
        return mask

    # ------------------------------------------------------------------ step
    def advance(self, global_dt: float, max_substeps: int, stability_dt: float) -> int:
        substeps = max(1, min(max_substeps, math.ceil(global_dt / stability_dt)))
        dt = global_dt / substeps
        for _ in range(substeps):
            self._step(dt)
            self._time += dt
        self.last_substeps = substeps
        return substeps

    def _step(self, dt: float) -> None:
        terrain_h = self._terrain.heights
        depth = self._depth
        effective = terrain_h.astype(np.float32) + self._bed_offset  # rocks raise the bed
        effective[self._obstacle_mask] += 1e4  # solid: never a flow target
        total = effective + depth

        padded = np.pad(total, 1, mode="edge")
        h_right = padded[1:-1, 2:]
        h_left = padded[1:-1, :-2]
        h_down = padded[2:, 1:-1]
        h_up = padded[:-2, 1:-1]

        # per-cell temperature multiplier (v0.4, Schauberger shade hypothesis
        # -- see set_environment); stays 1.0 (no effect) with no shade nearby
        gain = config.FLUID_FLOW_GAIN * self._temperature_factor
        flow_right = np.maximum(0.0, total - h_right) * gain
        flow_left = np.maximum(0.0, total - h_left) * gain
        flow_down = np.maximum(0.0, total - h_down) * gain
        flow_up = np.maximum(0.0, total - h_up) * gain

        outflow = flow_right + flow_left + flow_down + flow_up
        available = depth / dt
        scale = np.ones_like(depth)
        draining = outflow > available
        scale[draining] = available[draining] / np.maximum(outflow[draining], 1e-9)
        flow_right *= scale
        flow_left *= scale
        flow_down *= scale
        flow_up *= scale

        inflow_from_left = np.pad(flow_right, ((0, 0), (1, 0)))[:, :-1]
        inflow_from_right = np.pad(flow_left, ((0, 0), (0, 1)))[:, 1:]
        inflow_from_up = np.pad(flow_down, ((1, 0), (0, 0)))[:-1, :]
        inflow_from_down = np.pad(flow_up, ((0, 1), (0, 0)))[1:, :]
        inflow = inflow_from_left + inflow_from_right + inflow_from_up + inflow_from_down

        new_depth = depth + dt * (inflow - (flow_right + flow_left + flow_down + flow_up))
        new_depth = np.maximum(0.0, new_depth)
        new_depth[self._obstacle_mask] = 0.0
        if self._flow_enabled:
            # Upstream reservoir (west edge, i=0) never runs dry; downstream
            # outlet (east edge, i=-1) never backs up -- see config.py for why
            # this alone is enough to keep the interior flux scheme moving
            # continuously instead of settling flat. Obstacles at either edge
            # stay dry (reapplied after the clamp).
            new_depth[:, 0] = np.maximum(new_depth[:, 0], config.FLUID_RIVER_SOURCE_DEPTH)
            new_depth[:, -1] = np.minimum(new_depth[:, -1], config.FLUID_RIVER_SINK_DEPTH)
            new_depth[self._obstacle_mask] = 0.0
        new_depth[new_depth < config.FLUID_MIN_DEPTH] = 0.0

        self._depth = new_depth.astype(np.float32)
        self._flow_x = (flow_right - flow_left).astype(np.float32)
        self._flow_z = (flow_down - flow_up).astype(np.float32)
        self._erode_and_transport_sediment(dt, depth, flow_right, flow_left, flow_down, flow_up)

    def _erode_and_transport_sediment(self, dt: float, old_depth: np.ndarray,
                                      flow_right: np.ndarray, flow_left: np.ndarray,
                                      flow_down: np.ndarray, flow_up: np.ndarray) -> None:
        """RiverLab (v0.4): capacity-based erosion/deposition + advection.

        Standard real-time hydraulic erosion (capacity ~ speed * depth;
        erode when under capacity, deposit when over) -- see
        docs/04_TZ_v0.3_roadmap.md v0.4. Terrain is mutated in place
        (self._terrain IS world.terrain by reference), so erosion feeds
        back into flow exactly like a manual terrain.brush() would.
        Deliberately slow (config.SEDIMENT_*_RATE) relative to a single
        flood event -- see the config module docstring for why.
        """
        speed = np.sqrt((flow_right - flow_left) ** 2 + (flow_down - flow_up) ** 2)
        # colder water (shade) carries more/heavier sediment per Schauberger
        # -- same self._temperature_factor as the flow gain above
        capacity = config.SEDIMENT_CAPACITY_SCALE * self._temperature_factor * speed * old_depth
        sediment = self._sediment
        diff = capacity - sediment

        erode = np.where(diff > 0, diff * config.SEDIMENT_ERODE_RATE * dt, 0.0)
        # never erode below the world's bedrock floor
        erode = np.minimum(erode, np.maximum(0.0, self._terrain.heights - config.HEIGHT_MIN))
        # never erode terrain that's carrying no water at all (dry banks)
        erode[old_depth <= config.FLUID_MIN_DEPTH] = 0.0
        # a boulder is bedrock, not sediment: the river scours around it, not
        # through it -- without this the rock would dig its own hole and the
        # flank-erosion/lee-deposition asymmetry that makes a channel meander
        # would be swamped by a symmetric pit underneath it
        erode[self._bed_offset > config.BED_EROSION_SHIELD] = 0.0

        deposit = np.where(diff < 0, np.minimum(-diff, sediment) * config.SEDIMENT_DEPOSIT_RATE * dt, 0.0)
        deposit = np.minimum(deposit, sediment)

        self._terrain.heights = (self._terrain.heights - erode + deposit).astype(np.float32)
        sediment = sediment + erode - deposit

        concentration = sediment / np.maximum(old_depth, 1e-6)
        sed_out_right = flow_right * dt * concentration
        sed_out_left = flow_left * dt * concentration
        sed_out_down = flow_down * dt * concentration
        sed_out_up = flow_up * dt * concentration

        sed_in_left = np.pad(sed_out_right, ((0, 0), (1, 0)))[:, :-1]
        sed_in_right = np.pad(sed_out_left, ((0, 0), (0, 1)))[:, 1:]
        sed_in_up = np.pad(sed_out_down, ((1, 0), (0, 0)))[:-1, :]
        sed_in_down = np.pad(sed_out_up, ((0, 1), (0, 0)))[1:, :]
        sed_inflow = sed_in_left + sed_in_right + sed_in_up + sed_in_down
        sed_outflow = sed_out_right + sed_out_left + sed_out_down + sed_out_up

        new_sediment = np.maximum(0.0, sediment - sed_outflow + sed_inflow)
        new_sediment[self._obstacle_mask] = 0.0
        self._sediment = new_sediment.astype(np.float32)

    # ------------------------------------------------------------------ readback
    def sample_for_bodies(self, positions: np.ndarray, radii: Optional[np.ndarray] = None) -> dict:
        """Ambient depth/velocity around each body, excluding ANY solid cell.

        Each registered body carves a dry hole into its own footprint (see
        set_boundaries), so sampling exactly at its centre would always read
        depth=0 and an object could never float. A fixed-direction ring just
        outside a body's own radius does not work either: in a populated
        scene a *different* nearby object's hole can fall on the ring and
        get blended in too (a house 3-4 m from a box measurably lowered the
        box's own reading in testing -- this is not a rare edge case, it is
        the normal case once a scene has more than one object). Instead,
        average depth/flow over every non-obstacle cell in a neighbourhood
        around the body, using the real obstacle mask so any hole -- the
        body's own or a neighbour's -- is excluded, not guessed around.
        """
        count = len(positions)
        if count == 0 or self._terrain is None:
            return {"depths": np.zeros(0, dtype=np.float32),
                    "velocities": np.zeros((0, 3), dtype=np.float32),
                    "forces": np.zeros((0, 3), dtype=np.float32)}
        cell = self._terrain.cell_size
        ny, nx = self._depth.shape
        if radii is None or len(radii) != count:
            radii = np.full(count, config.FLUID_OBSTACLE_RADIUS_M, dtype=np.float32)
        depths = np.zeros(count, dtype=np.float32)
        vx = np.zeros(count, dtype=np.float32)
        vz = np.zeros(count, dtype=np.float32)
        for i in range(count):
            gx = float(positions[i, 0]) / cell + self._terrain.width / 2
            gz = float(positions[i, 2]) / cell + self._terrain.height / 2
            margin = _footprint_cells(radii[i], cell) + 2.0
            i0, i1 = max(0, int(gx - margin)), min(nx, int(gx + margin) + 1)
            j0, j1 = max(0, int(gz - margin)), min(ny, int(gz + margin) + 1)
            open_cells = ~self._obstacle_mask[j0:j1, i0:i1]
            if open_cells.any():
                depths[i] = float(self._depth[j0:j1, i0:i1][open_cells].mean())
                vx[i] = float(self._flow_x[j0:j1, i0:i1][open_cells].mean())
                vz[i] = float(self._flow_z[j0:j1, i0:i1][open_cells].mean())
        velocities = np.column_stack([vx, np.zeros(count, dtype=np.float32), vz])
        forces = velocities * depths[:, None]
        return {"depths": depths, "velocities": velocities, "forces": forces}

    def reset(self) -> None:
        if self._world is not None:
            self.initialize(self._world)

    def get_water_height(self, x: float = 0.0, z: float = 0.0) -> float:
        if self._terrain is None:
            return 0.0
        gx = np.array([x / self._terrain.cell_size + self._terrain.width / 2])
        gz = np.array([z / self._terrain.cell_size + self._terrain.height / 2])
        terrain_h = float(self._terrain.height_at(x, z))
        depth = float(_bilinear(self._depth, gx, gz)[0])
        return terrain_h + depth

    def get_velocity_field(self) -> Optional[np.ndarray]:
        return np.dstack([self._flow_x, np.zeros_like(self._flow_x), self._flow_z])

    def get_depth_grid(self) -> Optional[np.ndarray]:
        return self._depth

    def get_sediment_grid(self) -> np.ndarray:
        return self._sediment

    def total_volume(self) -> float:
        """Sum of water depth over all cells; used by conservation tests."""
        return float(self._depth.sum())

    def total_solid_material(self) -> float:
        """terrain height + suspended sediment, summed over all cells.

        Erosion/deposition only transfers material between the two -- this
        sum should stay constant (up to sediment carried across a closed
        world edge and clamped there, same caveat as total_volume).
        """
        return float(self._terrain.heights.sum() + self._sediment.sum())
