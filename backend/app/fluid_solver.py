"""Warp shallow-water solver with revision-driven GPU coupling."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from . import config
from .compute_engine import WARP_IMPORTED, wp


if WARP_IMPORTED:
    @wp.kernel
    def _velocity_step(h: wp.array(dtype=float), u: wp.array(dtype=float),
                       v: wp.array(dtype=float), bed: wp.array(dtype=float),
                       solid: wp.array(dtype=wp.int32),
                       next_u: wp.array(dtype=float), next_v: wp.array(dtype=float),
                       width: int, height: int, dx: float, dt: float,
                       gravity: float, dry: float, damping: float,
                       max_velocity: float):
        idx = wp.tid()
        i = idx % width
        j = idx // width
        if solid[idx] != 0 or h[idx] <= dry:
            next_u[idx] = 0.0
            next_v[idx] = 0.0
        else:
            eta = bed[idx] + h[idx]
            eta_l = eta
            eta_r = eta
            eta_d = eta
            eta_u = eta
            if i > 0 and solid[idx - 1] == 0:
                eta_l = bed[idx - 1] + h[idx - 1]
                if h[idx - 1] <= dry and bed[idx - 1] >= eta:
                    eta_l = eta
            if i < width - 1 and solid[idx + 1] == 0:
                eta_r = bed[idx + 1] + h[idx + 1]
                if h[idx + 1] <= dry and bed[idx + 1] >= eta:
                    eta_r = eta
            if j > 0 and solid[idx - width] == 0:
                eta_d = bed[idx - width] + h[idx - width]
                if h[idx - width] <= dry and bed[idx - width] >= eta:
                    eta_d = eta
            if j < height - 1 and solid[idx + width] == 0:
                eta_u = bed[idx + width] + h[idx + width]
                if h[idx + width] <= dry and bed[idx + width] >= eta:
                    eta_u = eta

            decay = wp.max(0.0, 1.0 - damping * dt)
            ux = (u[idx] - gravity * dt * (eta_r - eta_l) / (2.0 * dx)) * decay
            vz = (v[idx] - gravity * dt * (eta_u - eta_d) / (2.0 * dx)) * decay
            if i == 0 or i == width - 1:
                ux = 0.0
            if j == 0 or j == height - 1:
                vz = 0.0
            next_u[idx] = wp.clamp(ux, -max_velocity, max_velocity)
            next_v[idx] = wp.clamp(vz, -max_velocity, max_velocity)


    @wp.func
    def _face_flux(velocity: float, h_from: float, bed_from: float,
                   bed_other: float) -> float:
        available = wp.max(0.0, bed_from + h_from - wp.max(bed_from, bed_other))
        return velocity * available


    @wp.kernel
    def _depth_step(h: wp.array(dtype=float), u: wp.array(dtype=float),
                    v: wp.array(dtype=float), bed: wp.array(dtype=float),
                    solid: wp.array(dtype=wp.int32), next_h: wp.array(dtype=float),
                    width: int, height: int, dx: float, dt: float):
        idx = wp.tid()
        i = idx % width
        j = idx // width
        if solid[idx] != 0:
            next_h[idx] = 0.0
        else:
            q_right = float(0.0)
            q_left = float(0.0)
            q_up = float(0.0)
            q_down = float(0.0)
            if i < width - 1 and solid[idx + 1] == 0:
                face = 0.5 * (u[idx] + u[idx + 1])
                if face >= 0.0:
                    q_right = _face_flux(face, h[idx], bed[idx], bed[idx + 1])
                else:
                    q_right = _face_flux(face, h[idx + 1], bed[idx + 1], bed[idx])
            if i > 0 and solid[idx - 1] == 0:
                face = 0.5 * (u[idx - 1] + u[idx])
                if face >= 0.0:
                    q_left = _face_flux(face, h[idx - 1], bed[idx - 1], bed[idx])
                else:
                    q_left = _face_flux(face, h[idx], bed[idx], bed[idx - 1])
            if j < height - 1 and solid[idx + width] == 0:
                face = 0.5 * (v[idx] + v[idx + width])
                if face >= 0.0:
                    q_up = _face_flux(face, h[idx], bed[idx], bed[idx + width])
                else:
                    q_up = _face_flux(face, h[idx + width], bed[idx + width], bed[idx])
            if j > 0 and solid[idx - width] == 0:
                face = 0.5 * (v[idx - width] + v[idx])
                if face >= 0.0:
                    q_down = _face_flux(face, h[idx - width], bed[idx - width], bed[idx])
                else:
                    q_down = _face_flux(face, h[idx], bed[idx], bed[idx - width])
            value = h[idx] - dt * ((q_right - q_left) + (q_up - q_down)) / dx
            next_h[idx] = wp.max(0.0, value)


    @wp.kernel
    def _apply_source(h: wp.array(dtype=float), bed: wp.array(dtype=float),
                      solid: wp.array(dtype=wp.int32), width: int, height: int,
                      source_columns: int, level: float):
        idx = wp.tid()
        if idx < width * height:
            i = idx % width
            if i < source_columns and solid[idx] == 0:
                h[idx] = wp.max(0.0, level - bed[idx])


    @wp.kernel
    def _apply_outflow(h: wp.array(dtype=float), u: wp.array(dtype=float),
                       v: wp.array(dtype=float), solid: wp.array(dtype=wp.int32),
                       width: int, height: int, columns: int):
        """Zero-gradient (transmissive) outlet on the east edge.

        Without it every outer face is no-flux, so water that enters can never
        leave and the map fills forever -- measured before this existed: volume
        climbing 2290 -> 3484 -> 4311 m3 with no outflow at all. A river that
        runs off the edge of the domain is exactly this boundary condition: copy
        the interior state outward and let the flux carry water out.

        Only positive (outward) velocity is copied, so the outlet can never suck
        water back in and become a second, accidental source.
        """
        idx = wp.tid()
        i = idx % width
        j = idx // width
        if i < width - columns or solid[idx] != 0:
            return
        inner = j * width + (width - columns - 1)
        if solid[inner] != 0:
            return
        h[idx] = h[inner]
        u[idx] = wp.max(0.0, u[inner])
        v[idx] = 0.0


    @wp.kernel
    def _apply_point_sources(h: wp.array(dtype=float), bed: wp.array(dtype=float),
                             solid: wp.array(dtype=wp.int32),
                             centres: wp.array(dtype=wp.vec3),
                             radii: wp.array(dtype=float),
                             levels: wp.array(dtype=float),
                             count: int, width: int, height: int, dx: float):
        """Placeable inflow: hold water at `level` inside a disc of `radius`.

        Same rule as the edge inflow (`h = max(0, level - bed)`), so a source
        dropped on high ground fills only to the height the user asked for and a
        hill inside its radius stays proud of the water rather than being
        flooded from above. Everything downstream of the disc is ordinary flux
        physics -- the source sets a boundary, it does not paint a river.
        """
        idx = wp.tid()
        if solid[idx] != 0:
            return
        i = idx % width
        j = idx // width
        x = (float(i) - float(width - 1) * 0.5) * dx
        z = (float(j) - float(height - 1) * 0.5) * dx
        for n in range(count):
            centre = centres[n]
            radius = radii[n]
            if radius <= 0.0:
                continue
            dxc = x - centre[0]
            dzc = z - centre[2]
            if dxc * dxc + dzc * dzc <= radius * radius:
                h[idx] = wp.max(h[idx], wp.max(0.0, levels[n] - bed[idx]))


    @wp.kernel
    def _measure_drain_circulation(u: wp.array(dtype=float), v: wp.array(dtype=float),
                                   h: wp.array(dtype=float),
                                   solid: wp.array(dtype=wp.int32),
                                   centres: wp.array(dtype=wp.vec3),
                                   radii: wp.array(dtype=float),
                                   circulation: wp.array(dtype=float),
                                   count: int, width: int, height: int,
                                   dx: float, dry: float):
        """Sum the tangential velocity in the annulus around each drain.

        This is the whole reason the vortex is physics and not decoration. In a
        depth-averaged shallow-water field a purely radial sink produces purely
        radial convergence and NO rotation -- spin has to come from angular
        momentum that is already present, which converging flow then amplifies.
        So the drain measures the ambient circulation it actually finds and
        conserves it; it never imposes a direction of its own. A perfectly
        symmetric approach flow therefore yields a drain that does not spin,
        which is correct, and the acceptance test asserts that the rotation sign
        FOLLOWS the seeded circulation rather than matching a chosen constant.
        """
        idx = wp.tid()
        if solid[idx] != 0 or h[idx] <= dry:
            return
        i = idx % width
        j = idx // width
        x = (float(i) - float(width - 1) * 0.5) * dx
        z = (float(j) - float(height - 1) * 0.5) * dx
        for n in range(count):
            radius = radii[n]
            if radius <= 0.0:
                continue
            centre = centres[n]
            dxc = x - centre[0]
            dzc = z - centre[2]
            r = wp.sqrt(dxc * dxc + dzc * dzc)
            # an annulus just outside the sink itself: inside it the flow is
            # dominated by the sink we are about to apply, which would make the
            # measurement circular in the bad sense
            if r < radius or r > radius * 2.0:
                continue
            # tangential unit vector (counter-clockwise positive about +y)
            tx = -dzc / r
            tz = dxc / r
            wp.atomic_add(circulation, n, (u[idx] * tx + v[idx] * tz) * h[idx])


    @wp.kernel
    def _apply_drains(h: wp.array(dtype=float), u: wp.array(dtype=float),
                      v: wp.array(dtype=float), solid: wp.array(dtype=wp.int32),
                      centres: wp.array(dtype=wp.vec3), radii: wp.array(dtype=float),
                      strengths: wp.array(dtype=float),
                      circulation: wp.array(dtype=float),
                      count: int, width: int, height: int, dx: float, dt: float,
                      dry: float, swirl_gain: float, max_velocity: float):
        """Remove water through a localized sink and spin up the flow around it.

        Removal uses a smooth radial profile and is capped by the water actually
        present, so a drain can never pull a cell below zero or invent negative
        depth. The velocity it imposes has two parts:

        - radial, toward the centre: this is the sink's own convergence;
        - tangential, scaled as 1/r from the measured ambient circulation: this
          is conservation of angular momentum, which is what makes a real
          bathtub vortex. Sign and magnitude both come from
          `_measure_drain_circulation`, never from a constant.

        Honest scope boundary: a depth-averaged model cannot represent the
        vertical core of a vortex or the true free-surface funnel -- the vertical
        coordinate is integrated out. What it does give is a rotating, converging
        surface depression whose direction, strength and dependence on discharge
        and radius are all real. That is the "causal realism, not engineering
        realism" bar this project set for itself.
        """
        idx = wp.tid()
        if solid[idx] != 0:
            return
        i = idx % width
        j = idx // width
        x = (float(i) - float(width - 1) * 0.5) * dx
        z = (float(j) - float(height - 1) * 0.5) * dx
        for n in range(count):
            radius = radii[n]
            if radius <= 0.0 or strengths[n] <= 0.0:
                continue
            centre = centres[n]
            dxc = x - centre[0]
            dzc = z - centre[2]
            r = wp.sqrt(dxc * dxc + dzc * dzc)
            if r > radius:
                continue
            # smooth bell so the sink has no hard rim to ring against
            falloff = 1.0 - (r / radius) * (r / radius)
            removed = wp.min(h[idx], strengths[n] * falloff * dt)
            h[idx] = h[idx] - removed
            if h[idx] <= dry:
                h[idx] = 0.0
                u[idx] = 0.0
                v[idx] = 0.0
                continue
            r_safe = wp.max(r, dx * 0.5)
            nx = dxc / r_safe
            nz = dzc / r_safe
            tx = -nz
            tz = nx
            # angular momentum: u_theta * r stays constant as fluid converges,
            # so tangential speed grows as 1/r toward the centre
            spin = swirl_gain * circulation[n] / r_safe
            radial = -strengths[n] * falloff
            u[idx] = wp.clamp(u[idx] + (nx * radial + tx * spin) * dt,
                              -max_velocity, max_velocity)
            v[idx] = wp.clamp(v[idx] + (nz * radial + tz * spin) * dt,
                              -max_velocity, max_velocity)


    @wp.kernel
    def _combine_bed(bed_terrain: wp.array(dtype=float),
                     bed_offset: wp.array(dtype=float),
                     bed: wp.array(dtype=float)):
        """Effective bed = erodible terrain + rock domes.

        Kept as a separate array so the flow kernels keep taking a single `bed`
        (they are unchanged from the verified 0.5.1 versions), while erosion may
        mutate `bed_terrain` freely and rock domes stay a transient overlay that
        is rebuilt from live positions. That separation is what makes a moved or
        deleted rock leave no crater behind.
        """
        idx = wp.tid()
        bed[idx] = bed_terrain[idx] + bed_offset[idx]


    @wp.kernel
    def _erode_deposit(h: wp.array(dtype=float), u: wp.array(dtype=float),
                       v: wp.array(dtype=float), bed_terrain: wp.array(dtype=float),
                       bed_offset: wp.array(dtype=float),
                       sediment: wp.array(dtype=float),
                       solid: wp.array(dtype=wp.int32),
                       next_sediment: wp.array(dtype=float),
                       dt: float, capacity_scale: float, erode_rate: float,
                       deposit_rate: float, max_change: float, shield: float,
                       height_min: float, dry: float):
        """Capacity-based exchange between the bed and suspended sediment.

        capacity = scale * |velocity| * depth. Under capacity the flow picks
        material up, over capacity it drops it. Water depth is deliberately NOT
        adjusted for the material moved (the standard treatment in this
        algorithm class): sediment volume is small next to water volume, and
        leaving `h` alone keeps water mass conservation exactly intact, which
        several existing tests rely on.
        """
        idx = wp.tid()
        if solid[idx] != 0 or h[idx] <= dry:
            next_sediment[idx] = sediment[idx]
            return
        speed = wp.sqrt(u[idx] * u[idx] + v[idx] * v[idx])
        capacity = capacity_scale * speed * h[idx]
        gap = capacity - sediment[idx]
        limit = max_change * dt
        if gap > 0.0:
            amount = wp.min(gap * erode_rate * dt, limit)
            # never dig through the world floor
            amount = wp.min(amount, wp.max(0.0, bed_terrain[idx] - height_min))
            # a boulder is bedrock: the river scours around it, not through it.
            # Without this the rock digs a symmetric pit under itself and the
            # flank-scour / lee-fill asymmetry that moves a channel is swamped.
            if bed_offset[idx] > shield:
                amount = 0.0
            bed_terrain[idx] = bed_terrain[idx] - amount
            next_sediment[idx] = sediment[idx] + amount
        else:
            amount = wp.min(wp.min(-gap, sediment[idx]) * deposit_rate * dt, limit)
            bed_terrain[idx] = bed_terrain[idx] + amount
            next_sediment[idx] = sediment[idx] - amount


    @wp.kernel
    def _advect_sediment(sediment: wp.array(dtype=float),
                         next_sediment: wp.array(dtype=float),
                         u: wp.array(dtype=float), v: wp.array(dtype=float),
                         h: wp.array(dtype=float), solid: wp.array(dtype=wp.int32),
                         width: int, height: int, dx: float, dt: float, dry: float):
        """Semi-Lagrangian transport of suspended load by the real velocity field.

        Back-traces one step and bilinearly samples, skipping solid/dry cells so
        material is never drawn out of a wall or a dry bank. Semi-Lagrangian is
        stable at any timestep but is not strictly conservative -- acceptable and
        standard here, and stated rather than assumed: the acceptance tests below
        check causal behaviour (where material is picked up and dropped), not a
        conservation identity this scheme cannot provide.
        """
        idx = wp.tid()
        i = idx % width
        j = idx // width
        if solid[idx] != 0 or h[idx] <= dry:
            next_sediment[idx] = sediment[idx]
            return
        x = float(i) - u[idx] * dt / dx
        z = float(j) - v[idx] * dt / dx
        x = wp.clamp(x, 0.0, float(width - 1))
        z = wp.clamp(z, 0.0, float(height - 1))
        i0 = int(wp.floor(x))
        j0 = int(wp.floor(z))
        i1 = wp.min(i0 + 1, width - 1)
        j1 = wp.min(j0 + 1, height - 1)
        fx = x - float(i0)
        fz = z - float(j0)
        total = float(0.0)
        weight = float(0.0)
        for corner in range(4):
            ci = i0
            cj = j0
            w = (1.0 - fx) * (1.0 - fz)
            if corner == 1:
                ci = i1
                w = fx * (1.0 - fz)
            if corner == 2:
                cj = j1
                w = (1.0 - fx) * fz
            if corner == 3:
                ci = i1
                cj = j1
                w = fx * fz
            cidx = cj * width + ci
            if solid[cidx] == 0 and h[cidx] > dry:
                total = total + w * sediment[cidx]
                weight = weight + w
        if weight > 1.0e-6:
            next_sediment[idx] = total / weight
        else:
            next_sediment[idx] = sediment[idx]


    @wp.kernel
    def _advect_flow_tracers(particles: wp.array(dtype=wp.vec3),
                             h: wp.array(dtype=float), u: wp.array(dtype=float),
                             v: wp.array(dtype=float), bed: wp.array(dtype=float),
                             solid: wp.array(dtype=wp.int32), width: int, height: int,
                             source_columns: int, dx: float, dt: float, dry: float):
        n = wp.tid()
        p = particles[n]
        i = int(wp.floor(p.x / dx + float(width - 1) * 0.5 + 0.5))
        j = int(wp.floor(p.z / dx + float(height - 1) * 0.5 + 0.5))
        valid = i > 0 and i < width - 1 and j > 0 and j < height - 1
        if valid:
            idx = j * width + i
            valid = solid[idx] == 0 and h[idx] > dry
        if valid:
            nx = p.x + u[idx] * dt
            nz = p.z + v[idx] * dt
            ni = int(wp.floor(nx / dx + float(width - 1) * 0.5 + 0.5))
            nj = int(wp.floor(nz / dx + float(height - 1) * 0.5 + 0.5))
            if ni > 0 and ni < width - 1 and nj > 0 and nj < height - 1:
                next_idx = nj * width + ni
                if solid[next_idx] == 0 and h[next_idx] > dry:
                    particles[n] = wp.vec3(nx, bed[next_idx] + h[next_idx] + 0.08, nz)
                else:
                    particles[n] = wp.vec3(p.x, bed[idx] + h[idx] + 0.08, p.z)
        else:
            rows = wp.max(1, height - 2)
            si = wp.min(width - 2, wp.max(1, source_columns - 1))
            sj = 1 + n % rows
            subcell = float((n // rows) % 64) / 64.0 - 0.5
            spawn_idx = sj * width + si
            sx = (float(si) - float(width - 1) * 0.5) * dx
            sz = (float(sj) + subcell - float(height - 1) * 0.5) * dx
            sy = -100.0
            if solid[spawn_idx] == 0 and h[spawn_idx] > dry:
                sy = bed[spawn_idx] + h[spawn_idx] + 0.08
            particles[n] = wp.vec3(sx, sy, sz)


    @wp.kernel
    def _clear_diagnostics(max_depth: wp.array(dtype=float),
                           max_speed: wp.array(dtype=float),
                           max_wave: wp.array(dtype=float),
                           volume: wp.array(dtype=float), wet: wp.array(dtype=wp.int32)):
        max_depth[0] = 0.0
        max_speed[0] = 0.0
        max_wave[0] = 0.0
        volume[0] = 0.0
        wet[0] = 0


    @wp.kernel
    def _reduce_diagnostics(h: wp.array(dtype=float), u: wp.array(dtype=float),
                            v: wp.array(dtype=float), solid: wp.array(dtype=wp.int32),
                            gravity: float, area: float, dry: float,
                            max_depth: wp.array(dtype=float),
                            max_speed: wp.array(dtype=float),
                            max_wave: wp.array(dtype=float),
                            volume: wp.array(dtype=float), wet: wp.array(dtype=wp.int32)):
        idx = wp.tid()
        if solid[idx] == 0:
            depth = wp.max(0.0, h[idx])
            speed = wp.sqrt(u[idx] * u[idx] + v[idx] * v[idx])
            wave = wp.max(wp.abs(u[idx]), wp.abs(v[idx])) + wp.sqrt(gravity * depth)
            wp.atomic_max(max_depth, 0, depth)
            wp.atomic_max(max_speed, 0, speed)
            wp.atomic_max(max_wave, 0, wave)
            wp.atomic_add(volume, 0, depth * area)
            if depth > dry:
                wp.atomic_add(wet, 0, 1)


    @wp.kernel
    def _sample_bodies(h: wp.array(dtype=float), u: wp.array(dtype=float),
                       v: wp.array(dtype=float), bed: wp.array(dtype=float),
                       solid: wp.array(dtype=wp.int32),
                       positions: wp.array(dtype=wp.vec3),
                       rotations: wp.array(dtype=wp.vec3),
                       half_extents: wp.array(dtype=wp.vec2),
                       body_velocities: wp.array(dtype=wp.vec3),
                       drag: wp.array(dtype=float), cross_area: wp.array(dtype=float),
                       body_height: wp.array(dtype=float),
                       sample_depth: wp.array(dtype=float),
                       sample_immersion: wp.array(dtype=float),
                       sample_surface: wp.array(dtype=float),
                       sample_support: wp.array(dtype=float),
                       sample_velocity: wp.array(dtype=wp.vec3),
                       sample_force: wp.array(dtype=wp.vec3),
                       width: int, height: int, dx: float, rho: float):
        n = wp.tid()
        p = positions[n]
        extent = half_extents[n]
        yaw = rotations[n].y
        c = wp.cos(yaw)
        s = wp.sin(yaw)
        support = -1.0e20
        for sample in range(9):
            sx = float(sample % 3 - 1) * extent.x
            sz = float(sample // 3 - 1) * extent.y
            x = p.x + c * sx + s * sz
            z = p.z - s * sx + c * sz
            i = int(wp.floor(x / dx + float(width - 1) * 0.5 + 0.5))
            j = int(wp.floor(z / dx + float(height - 1) * 0.5 + 0.5))
            if i >= 0 and i < width and j >= 0 and j < height:
                support = wp.max(support, bed[j * width + i])
        if support < -1.0e10:
            support = p.y
        base_y = wp.max(p.y, support)
        total_depth = float(0.0)
        total_immersion = float(0.0)
        total_surface = float(0.0)
        wet_samples = int(0)
        total_flow = wp.vec3(0.0, 0.0, 0.0)
        total_force = wp.vec3(0.0, 0.0, 0.0)
        for sample in range(9):
            sx = float(sample % 3 - 1) * extent.x
            sz = float(sample // 3 - 1) * extent.y
            x = p.x + c * sx + s * sz
            z = p.z - s * sx + c * sz
            i = int(wp.floor(x / dx + float(width - 1) * 0.5 + 0.5))
            j = int(wp.floor(z / dx + float(height - 1) * 0.5 + 0.5))
            if i >= 0 and i < width and j >= 0 and j < height:
                idx = j * width + i
                if solid[idx] == 0 and h[idx] > 0.0:
                    depth = wp.max(0.0, h[idx])
                    surface = bed[idx] + depth
                    immersion = wp.clamp(surface - base_y, 0.0, body_height[n])
                    flow = wp.vec3(u[idx], 0.0, v[idx])
                    relative = flow - body_velocities[n]
                    speed = wp.length(relative)
                    submerged = immersion / wp.max(body_height[n], 1.0e-4)
                    total_depth = total_depth + depth
                    total_immersion = total_immersion + immersion
                    total_surface = total_surface + surface
                    total_flow = total_flow + flow
                    total_force = total_force + relative * (
                        0.5 * rho * drag[n] * cross_area[n] * submerged * speed)
                    wet_samples = wet_samples + 1
        sample_depth[n] = total_depth / 9.0
        sample_immersion[n] = total_immersion / 9.0
        sample_support[n] = support
        sample_force[n] = total_force / 9.0
        if wet_samples > 0:
            sample_surface[n] = total_surface / float(wet_samples)
            sample_velocity[n] = total_flow / float(wet_samples)
        else:
            sample_surface[n] = support
            sample_velocity[n] = wp.vec3(0.0, 0.0, 0.0)


# Body types that act as solid walls for the flow. A body carrying a positive
# `bed_height` is riverbed instead and is excluded regardless of this set --
# see WarpShallowWaterSolver._is_solid.
SOLID_OBSTACLE_TYPES = frozenset({"HOUSE"})

# A ROCK of scale 1 raises the bed over this radius, in metres. Matches the
# radius of the ROCK mesh in frontend/src/world/ObjectFactory.ts so what the
# user sees is what the water feels.
ROCK_BASE_RADIUS_M = 1.5


class FluidSolver:
    def initialize(self, world) -> None: ...
    def set_boundaries(self, terrain, obstacles: dict, terrain_revision=0,
                       obstacle_revision=0) -> None: ...
    def advance(self, global_dt: float, max_substeps: int, stability_dt: float) -> int: ...
    def sample_for_bodies(self, positions: np.ndarray, body_velocities=None,
                           drag=None, cross_area=None, body_height=None,
                           rotations=None, half_extents=None) -> dict: ...
    def reset(self) -> None: ...
    def get_water_height(self, x: float = 0.0, z: float = 0.0) -> float: ...
    def get_velocity_field(self) -> Optional[np.ndarray]: ...
    def get_water_height_field(self) -> np.ndarray: ...
    def get_flow_particles(self) -> np.ndarray: ...
    def diagnostics(self) -> dict: ...


class PlaceholderFluidSolver(FluidSolver):
    def __init__(self) -> None:
        self._world = self._terrain = None
        self._level = 0.5
        self.last_substeps = 0

    def initialize(self, world) -> None:
        self._world, self._terrain = world, world.terrain
        self._level = world.water.level

    def set_boundaries(self, terrain, obstacles: dict, terrain_revision=0,
                       obstacle_revision=0) -> None:
        self._terrain = terrain

    def advance(self, global_dt: float, max_substeps: int, stability_dt: float) -> int:
        self.last_substeps = max(1, min(max_substeps, math.ceil(global_dt / stability_dt)))
        return self.last_substeps

    def sample_for_bodies(self, positions: np.ndarray, body_velocities=None,
                           drag=None, cross_area=None, body_height=None,
                           rotations=None, half_extents=None) -> dict:
        count = len(positions)
        depths = np.zeros(count, dtype=np.float32)
        if self._terrain is not None:
            for i, p in enumerate(positions):
                depths[i] = max(0.0, self._level - self._terrain.height_at(p[0], p[2]))
        velocities = np.zeros((count, 3), dtype=np.float32)
        heights = np.ones(count, dtype=np.float32) if body_height is None \
            else np.asarray(body_height, dtype=np.float32)
        surfaces = depths + np.asarray([
            self._terrain.height_at(float(p[0]), float(p[2])) for p in positions],
            dtype=np.float32)
        immersions = np.clip(surfaces - np.asarray(positions, dtype=np.float32)[:, 1],
                             0.0, heights)
        return {"depths": depths, "immersions": immersions,
                "surface_elevations": surfaces,
                "support_elevations": surfaces - depths, "velocities": velocities,
                "forces": np.zeros_like(velocities)}

    def reset(self) -> None: pass
    def set_level(self, level: float) -> None: self._level = float(level)
    def get_water_height(self, x=0.0, z=0.0) -> float: return self._level
    def get_velocity_field(self) -> Optional[np.ndarray]: return None
    def get_water_height_field(self) -> np.ndarray:
        return (np.full(self._terrain.heights.size, self._level, dtype=np.float32)
                 if self._terrain is not None else np.zeros(0, dtype=np.float32))
    def get_flow_particles(self) -> np.ndarray: return np.zeros((0, 3), dtype=np.float32)
    def diagnostics(self) -> dict:
        return {"solver": "placeholder", "substeps": self.last_substeps}


class WarpShallowWaterSolver(FluidSolver):
    def __init__(self, device: str) -> None:
        if not WARP_IMPORTED:
            raise RuntimeError("NVIDIA Warp is unavailable")
        self.device = device
        self._world = self._terrain = None
        self._width = self._height = self._count = 0
        self._h = self._u = self._v = None
        self._next_h = self._next_u = self._next_v = None
        self._bed = self._obstacles = None
        # v0.6.0 RiverLab: the bed is split in two. `_bed_terrain` is the real,
        # erodible ground the solver now OWNS (erosion mutates it every tick, so
        # it can no longer be a straight copy of world.terrain uploaded on a
        # revision bump); `_bed_offset` holds ROCK domes, rebuilt from live
        # positions whenever the obstacle set changes and never written back into
        # the world. `_bed` is their sum and is what every flow kernel reads.
        self._bed_terrain = self._bed_offset = None
        self._sediment = self._next_sediment = None
        self._obstacle_host = np.zeros(0, dtype=np.int32)
        self._bed_offset_host = np.zeros(0, dtype=np.float32)
        self._bed_host = np.zeros(0, dtype=np.float32)
        self._erosion_enabled = True
        # v0.8.0: placeable inflow / outlet. Empty arrays mean "no placed
        # SOURCE", in which case the edge columns keep feeding the map exactly
        # as they did in 0.7.0 -- a world with no SOURCE object behaves
        # identically, which is what keeps the whole 0.7.0 suite valid.
        self._outflow_columns = 0
        self._source_count = 0
        self._drain_count = 0
        self._src_centres = self._src_radii = self._src_levels = None
        self._drain_centres = self._drain_radii = None
        self._drain_strengths = self._drain_circulation = None
        self._level = 0.5
        self._source_enabled = True
        self._seen_terrain_revision = -1
        self._seen_obstacle_revision = -1
        self.terrain_gpu_uploads = 0
        self.obstacle_gpu_uploads = 0
        self.last_substeps = 0
        self._time = 0.0
        self._diag = {"cfl_dt": config.FIXED_DT, "max_depth": 0.0,
                      "max_velocity": 0.0, "wet_cells": 0, "volume_m3": 0.0,
                      "cfl_limited": False}
        self._body_count = 0
        self._flow_particles = None

    def initialize(self, world) -> None:
        self._world, self._terrain = world, world.terrain
        self._level = float(world.water.level)
        self._source_enabled = True
        self._width, self._height = world.terrain.width + 1, world.terrain.height + 1
        self._count = self._width * self._height
        bed_grid = np.ascontiguousarray(world.terrain.heights, dtype=np.float32)
        self._bed_host = bed_grid.ravel().copy()
        depth_grid = np.zeros_like(bed_grid, dtype=np.float32)
        source_columns = min(config.FLUID_SOURCE_COLUMNS, self._width)
        depth_grid[:, :source_columns] = np.maximum(
            self._level - bed_grid[:, :source_columns], 0.0)
        depth = depth_grid.ravel()
        zeros = np.zeros(self._count, dtype=np.float32)
        self._obstacle_host = np.zeros(self._count, dtype=np.int32)
        self._h = wp.array(depth, dtype=float, device=self.device)
        self._u = wp.array(zeros, dtype=float, device=self.device)
        self._v = wp.array(zeros, dtype=float, device=self.device)
        self._next_h = wp.empty(self._count, dtype=float, device=self.device)
        self._next_u = wp.empty(self._count, dtype=float, device=self.device)
        self._next_v = wp.empty(self._count, dtype=float, device=self.device)
        self._bed_terrain = wp.array(self._bed_host, dtype=float, device=self.device)
        self._bed_offset_host = np.zeros(self._count, dtype=np.float32)
        self._bed_offset = wp.array(self._bed_offset_host, dtype=float, device=self.device)
        self._bed = wp.array(self._bed_host.copy(), dtype=float, device=self.device)
        self._sediment = wp.zeros(self._count, dtype=float, device=self.device)
        self._next_sediment = wp.empty(self._count, dtype=float, device=self.device)
        self._obstacles = wp.array(self._obstacle_host, dtype=wp.int32, device=self.device)
        self._diag_max_depth = wp.zeros(1, dtype=float, device=self.device)
        self._diag_max_speed = wp.zeros(1, dtype=float, device=self.device)
        self._diag_max_wave = wp.zeros(1, dtype=float, device=self.device)
        self._diag_volume = wp.zeros(1, dtype=float, device=self.device)
        self._diag_wet = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._seen_terrain_revision = -1
        self._seen_obstacle_revision = -1
        self.terrain_gpu_uploads = 1
        self.obstacle_gpu_uploads = 1
        self.last_substeps = 0
        self._time = 0.0
        self._body_count = 0
        tracer_count = config.FLOW_TRACER_COUNT
        rows = max(1, self._height - 2)
        tracer_ids = np.arange(tracer_count, dtype=np.int32)
        ti = np.full(tracer_count, min(self._width - 2,
                                      max(1, source_columns - 1)), dtype=np.int32)
        tj = 1 + tracer_ids % rows
        tracer_idx = tj * self._width + ti
        tracer_x = (ti - world.terrain.width / 2) * world.terrain.cell_size
        subcell = ((tracer_ids // rows) % 64).astype(np.float32) / 64.0 - 0.5
        tracer_z = (tj + subcell - world.terrain.height / 2) * world.terrain.cell_size
        tracer_y = self._bed_host[tracer_idx] + depth[tracer_idx] + 0.08
        tracers = np.column_stack((tracer_x, tracer_y, tracer_z)).astype(np.float32)
        tracers[depth[tracer_idx] <= config.FLUID_DRY_DEPTH, 1] = -100.0
        self._flow_particles = wp.array(tracers, dtype=wp.vec3, device=self.device)
        self._recombine_bed()
        self._measure()

    def _build_obstacle_mask(self, terrain, obstacles: dict) -> np.ndarray:
        mask = np.zeros(self._count, dtype=np.int32)
        positions = obstacles.get("positions", [])
        rotations = obstacles.get("rotations", [])
        scales = obstacles.get("scales", [])
        types = obstacles.get("types", [])
        bed_heights = obstacles.get("bed_heights", [])
        for n, position in enumerate(positions):
            if n >= len(types) or not self._is_solid(types[n], bed_heights, n):
                continue
            scale = scales[n] if n < len(scales) else [1.0, 1.0, 1.0]
            yaw = float(rotations[n][1]) if n < len(rotations) else 0.0
            cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
            half_x = 2.0 * float(scale[0])
            half_z = 2.0 * float(scale[2])
            bound_x = abs(cos_yaw) * half_x + abs(sin_yaw) * half_z
            bound_z = abs(sin_yaw) * half_x + abs(cos_yaw) * half_z
            center_x, center_z = float(position[0]), float(position[2])
            lo_i = max(1, int(math.floor((center_x - bound_x) / terrain.cell_size
                                         + terrain.width / 2)))
            hi_i = min(self._width - 2, int(math.ceil((center_x + bound_x)
                                                       / terrain.cell_size
                                                       + terrain.width / 2)))
            lo_j = max(1, int(math.floor((center_z - bound_z) / terrain.cell_size
                                         + terrain.height / 2)))
            hi_j = min(self._height - 2, int(math.ceil((center_z + bound_z)
                                                        / terrain.cell_size
                                                        + terrain.height / 2)))
            epsilon = 1.0e-6 * max(1.0, half_x, half_z, terrain.cell_size)
            for row in range(lo_j, hi_j + 1):
                world_z = (row - terrain.height / 2) * terrain.cell_size
                for column in range(lo_i, hi_i + 1):
                    world_x = (column - terrain.width / 2) * terrain.cell_size
                    dx, dz = world_x - center_x, world_z - center_z
                    local_x = cos_yaw * dx + sin_yaw * dz
                    local_z = -sin_yaw * dx + cos_yaw * dz
                    if (abs(local_x) <= half_x + epsilon
                            and abs(local_z) <= half_z + epsilon):
                        mask[row * self._width + column] = 1
        return mask

    @staticmethod
    def _is_solid(obj_type: str, bed_heights, index: int) -> bool:
        """Does this body act as an infinitely tall wall for the flow?

        Expressed as data rather than a hard-coded type check (it used to be
        literally `types[n] != "HOUSE"`), because v0.6.0 introduces the opposite
        case: a body with `bed_height > 0` is riverbed, not wall, and must never
        be rasterized into the solid mask however tall it looks. GAUGE stays out
        of the mask too -- a measuring stick must not divert the water it
        measures, which is asserted by its own test.
        """
        if index < len(bed_heights) and float(bed_heights[index]) > 0.0:
            return False
        return obj_type in SOLID_OBSTACLE_TYPES

    def _build_bed_offset(self, terrain, obstacles: dict) -> np.ndarray:
        """Raised-bed domes for riverbed bodies (ROCK), in grid order.

        Radius comes from the body's horizontal scale and height from its
        vertical one, so `Scale Y` in the properties panel is already the
        "how much of the channel does this block" control -- no extra UI knob,
        deliberately. The dome tapers as (1 - r^2/R^2)^BED_DOME_EXPONENT, i.e.
        hemispherical at the default exponent.
        """
        offset = np.zeros(self._count, dtype=np.float32)
        positions = obstacles.get("positions", []) if obstacles else []
        if len(positions) == 0:
            return offset
        scales = obstacles.get("scales", [])
        bed_heights = obstacles.get("bed_heights", [])
        grid = offset.reshape(self._height, self._width)
        yy, xx = np.mgrid[0:self._height, 0:self._width]
        for n, position in enumerate(positions):
            height = float(bed_heights[n]) if n < len(bed_heights) else 0.0
            if height <= 0.0:
                continue
            scale = scales[n] if n < len(scales) else (1.0, 1.0, 1.0)
            radius_m = ROCK_BASE_RADIUS_M * max(float(scale[0]), float(scale[2]))
            radius_cells = max(1.0, radius_m / terrain.cell_size)
            gx = float(position[0]) / terrain.cell_size + terrain.width / 2
            gz = float(position[2]) / terrain.cell_size + terrain.height / 2
            inside = np.clip(1.0 - ((xx - gx) ** 2 + (yy - gz) ** 2)
                             / radius_cells ** 2, 0.0, 1.0)
            dome = (height * float(scale[1])) * inside ** config.BED_DOME_EXPONENT
            np.maximum(grid, dome.astype(np.float32), out=grid)
        return offset

    def _recombine_bed(self) -> None:
        wp.launch(_combine_bed, dim=self._count,
                  inputs=[self._bed_terrain, self._bed_offset, self._bed],
                  device=self.device)

    def _remap_obstacles(self, new_mask: np.ndarray) -> None:
        h, u, v = self._host_fields()
        old_mask = self._obstacle_host
        newly_solid = np.flatnonzero((old_mask == 0) & (new_mask != 0))
        freed = (old_mask != 0) & (new_mask == 0)
        h[freed] = 0.0
        u[freed] = 0.0
        v[freed] = 0.0
        fluid_cells = np.flatnonzero((new_mask == 0) & (old_mask == 0))
        for source in newly_solid:
            depth = float(h[source])
            if depth <= 0.0 or not len(fluid_cells):
                continue
            sj, si = divmod(int(source), self._width)
            fj, fi = np.divmod(fluid_cells, self._width)
            distance = (fi - si) ** 2 + (fj - sj) ** 2
            target = int(fluid_cells[int(np.argmin(distance))])
            old_depth = float(h[target])
            total = old_depth + depth
            if total > config.FLUID_DRY_DEPTH:
                u[target] = (u[target] * old_depth + u[source] * depth) / total
                v[target] = (v[target] * old_depth + v[source] * depth) / total
            h[target] = total
        h[new_mask != 0] = 0.0
        u[new_mask != 0] = 0.0
        v[new_mask != 0] = 0.0
        self._h = wp.array(h, dtype=float, device=self.device)
        self._u = wp.array(u, dtype=float, device=self.device)
        self._v = wp.array(v, dtype=float, device=self.device)

    def set_boundaries(self, terrain, obstacles: dict, terrain_revision=0,
                       obstacle_revision=0) -> None:
        """Re-upload only what a revision bump says actually changed.

        A revision bump means the HOST changed the terrain -- a brush stroke, a
        loaded world, a reset. It deliberately does NOT cover erosion: the
        solver owns `_bed_terrain` while RUNNING and mutates it on the GPU every
        tick, so bumping a revision for erosion would both re-upload needlessly
        and stomp the GPU-side state with a stale host copy. That is also why
        the "600 unchanged steps do not increase the upload counters" test keeps
        holding with erosion switched on.
        """
        self._terrain = terrain
        changed = False
        if terrain_revision != self._seen_terrain_revision:
            self._bed_host = np.ascontiguousarray(terrain.heights, dtype=np.float32).ravel().copy()
            self._bed_terrain = wp.array(self._bed_host, dtype=float, device=self.device)
            self._seen_terrain_revision = terrain_revision
            self.terrain_gpu_uploads += 1
            changed = True
        if obstacle_revision != self._seen_obstacle_revision:
            new_mask = self._build_obstacle_mask(terrain, obstacles)
            if not np.array_equal(new_mask, self._obstacle_host):
                self._remap_obstacles(new_mask)
                self._obstacle_host = new_mask
                self._obstacles = wp.array(new_mask, dtype=wp.int32, device=self.device)
            new_offset = self._build_bed_offset(terrain, obstacles)
            if not np.array_equal(new_offset, self._bed_offset_host):
                self._bed_offset_host = new_offset
                self._bed_offset = wp.array(new_offset, dtype=float, device=self.device)
            self._seen_obstacle_revision = obstacle_revision
            self.obstacle_gpu_uploads += 1
            changed = True
        if changed:
            self._recombine_bed()
            self._measure()

    def set_outflow(self, columns: int) -> None:
        """Width, in cells, of the transmissive outlet on the east edge.

        Zero restores the fully closed domain of 0.7.0 -- which is what the
        volume-conservation test wants, and it says so in its own name.
        """
        self._outflow_columns = max(0, int(columns))

    def set_water_features(self, sources: list, drains: list) -> None:
        """Upload placeable SOURCE and DRAIN objects, in world coordinates.

        Called every tick from SimulationManager with live positions, so either
        can be dragged while the simulation is RUNNING and the water responds
        immediately -- that is the point of making them objects rather than
        settings. Arrays are reallocated only when the count changes, so
        dragging one does not churn GPU memory.
        """
        self._source_count = len(sources)
        self._drain_count = len(drains)
        if sources:
            self._src_centres = wp.array(
                np.array([item[0] for item in sources], dtype=np.float32),
                dtype=wp.vec3, device=self.device)
            self._src_radii = wp.array(
                np.array([item[1] for item in sources], dtype=np.float32),
                dtype=float, device=self.device)
            self._src_levels = wp.array(
                np.array([item[2] for item in sources], dtype=np.float32),
                dtype=float, device=self.device)
        if drains:
            self._drain_centres = wp.array(
                np.array([item[0] for item in drains], dtype=np.float32),
                dtype=wp.vec3, device=self.device)
            self._drain_radii = wp.array(
                np.array([item[1] for item in drains], dtype=np.float32),
                dtype=float, device=self.device)
            self._drain_strengths = wp.array(
                np.array([item[2] for item in drains], dtype=np.float32),
                dtype=float, device=self.device)
            if (self._drain_circulation is None
                    or len(self._drain_circulation) != len(drains)):
                self._drain_circulation = wp.zeros(len(drains), dtype=float,
                                                   device=self.device)

    def set_erosion(self, enabled: bool) -> None:
        """RiverLab erosion on/off, read live each tick by SimulationManager.

        Off leaves the bed exactly as the user built it, which is what a short
        FloodLab experiment wants; on lets the river cut its own channel.
        """
        self._erosion_enabled = bool(enabled)

    def get_terrain_heights(self) -> np.ndarray:
        """Current erodible bed, in TerrainGrid order, read back from the GPU.

        Rock domes are excluded on purpose -- they are not terrain, and writing
        them back would leave a permanent crater-and-mound the moment the rock
        moves. Called on a throttle (config.TERRAIN_RESYNC_INTERVAL_S), not per
        tick: it is a device-to-host copy of the whole grid.
        """
        if self._bed_terrain is None:
            return np.zeros(0, dtype=np.float32)
        return np.asarray(self._bed_terrain.numpy(), dtype=np.float32)

    def _measure(self) -> None:
        if self._h is None:
            return
        wp.launch(_clear_diagnostics, dim=1, inputs=[self._diag_max_depth,
                  self._diag_max_speed, self._diag_max_wave, self._diag_volume,
                  self._diag_wet], device=self.device)
        wp.launch(_reduce_diagnostics, dim=self._count, inputs=[self._h, self._u,
                  self._v, self._obstacles, float(self._world.environment.gravity),
                  float(self._terrain.cell_size ** 2), config.FLUID_DRY_DEPTH,
                  self._diag_max_depth, self._diag_max_speed, self._diag_max_wave,
                  self._diag_volume, self._diag_wet], device=self.device)
        max_wave = float(self._diag_max_wave.numpy()[0])
        cfl_dt = (config.FIXED_DT if max_wave <= 1.0e-8 else
                  config.FLUID_CFL * self._terrain.cell_size / max_wave)
        self._diag.update({
            "cfl_dt": cfl_dt,
            "max_depth": float(self._diag_max_depth.numpy()[0]),
            "max_velocity": float(self._diag_max_speed.numpy()[0]),
            "wet_cells": int(self._diag_wet.numpy()[0]),
            "volume_m3": float(self._diag_volume.numpy()[0]),
        })

    def advance(self, global_dt: float, max_substeps: int, stability_dt: float) -> int:
        if self._h is None:
            return 0
        self._measure()
        required = max(1, math.ceil(global_dt / max(self._diag["cfl_dt"], 1.0e-8)))
        substeps = min(max_substeps, required)
        dt = global_dt / substeps
        self._diag["cfl_limited"] = required > max_substeps
        gravity = float(self._world.environment.gravity)
        for _ in range(substeps):
            wp.launch(_velocity_step, dim=self._count, inputs=[self._h, self._u,
                      self._v, self._bed, self._obstacles, self._next_u,
                      self._next_v, self._width, self._height,
                      float(self._terrain.cell_size), dt, gravity,
                      config.FLUID_DRY_DEPTH, config.FLUID_DAMPING,
                      config.FLUID_MAX_VELOCITY], device=self.device)
            self._u, self._next_u = self._next_u, self._u
            self._v, self._next_v = self._next_v, self._v
            wp.launch(_depth_step, dim=self._count, inputs=[self._h, self._u,
                      self._v, self._bed, self._obstacles, self._next_h,
                      self._width, self._height, float(self._terrain.cell_size), dt],
                      device=self.device)
            self._h, self._next_h = self._next_h, self._h
            if self._source_enabled and not self._source_count:
                # a placed SOURCE takes over from the edge inflow entirely --
                # otherwise the map has two sources and "where does the water
                # come from" stops having a single answer
                wp.launch(_apply_source, dim=self._count, inputs=[self._h,
                          self._bed, self._obstacles, self._width,
                          self._height, config.FLUID_SOURCE_COLUMNS,
                          self._level], device=self.device)
            if self._source_count:
                wp.launch(_apply_point_sources, dim=self._count,
                          inputs=[self._h, self._bed, self._obstacles,
                                  self._src_centres, self._src_radii,
                                  self._src_levels, self._source_count,
                                  self._width, self._height,
                                  float(self._terrain.cell_size)],
                          device=self.device)
            if self._drain_count:
                self._drain_circulation.zero_()
                wp.launch(_measure_drain_circulation, dim=self._count,
                          inputs=[self._u, self._v, self._h, self._obstacles,
                                  self._drain_centres, self._drain_radii,
                                  self._drain_circulation, self._drain_count,
                                  self._width, self._height,
                                  float(self._terrain.cell_size),
                                  config.FLUID_DRY_DEPTH], device=self.device)
                wp.launch(_apply_drains, dim=self._count,
                          inputs=[self._h, self._u, self._v, self._obstacles,
                                  self._drain_centres, self._drain_radii,
                                  self._drain_strengths, self._drain_circulation,
                                  self._drain_count, self._width, self._height,
                                  float(self._terrain.cell_size), dt,
                                  config.FLUID_DRY_DEPTH, config.DRAIN_SWIRL_GAIN,
                                  config.FLUID_MAX_VELOCITY], device=self.device)
            if self._outflow_columns:
                wp.launch(_apply_outflow, dim=self._count,
                          inputs=[self._h, self._u, self._v, self._obstacles,
                                  self._width, self._height,
                                  self._outflow_columns], device=self.device)
            if self._erosion_enabled:
                # RiverLab (v0.6.0): pick material up where the flow is fast and
                # deep, drop it where it slows, then carry the suspended load
                # with the same velocity field. Runs AFTER the depth/source
                # update so it sees this substep's real h/u/v, and the bed is
                # recombined immediately so the next substep's velocity step
                # already feels the freshly cut channel -- that closed loop
                # (flow -> erosion -> terrain -> flow) is the whole point.
                wp.launch(_erode_deposit, dim=self._count,
                          inputs=[self._h, self._u, self._v, self._bed_terrain,
                                  self._bed_offset, self._sediment, self._obstacles,
                                  self._next_sediment, dt,
                                  config.SEDIMENT_CAPACITY_SCALE,
                                  config.SEDIMENT_ERODE_RATE,
                                  config.SEDIMENT_DEPOSIT_RATE,
                                  config.SEDIMENT_MAX_BED_CHANGE,
                                  config.BED_EROSION_SHIELD,
                                  config.HEIGHT_MIN, config.FLUID_DRY_DEPTH],
                          device=self.device)
                self._sediment, self._next_sediment = self._next_sediment, self._sediment
                wp.launch(_advect_sediment, dim=self._count,
                          inputs=[self._sediment, self._next_sediment, self._u,
                                  self._v, self._h, self._obstacles, self._width,
                                  self._height, float(self._terrain.cell_size), dt,
                                  config.FLUID_DRY_DEPTH], device=self.device)
                self._sediment, self._next_sediment = self._next_sediment, self._sediment
                self._recombine_bed()
            wp.launch(_advect_flow_tracers, dim=config.FLOW_TRACER_COUNT,
                      inputs=[self._flow_particles, self._h, self._u, self._v,
                              self._bed, self._obstacles, self._width, self._height,
                              config.FLUID_SOURCE_COLUMNS,
                              float(self._terrain.cell_size), dt,
                              config.FLUID_DRY_DEPTH], device=self.device)
            self._time += dt
        self.last_substeps = substeps
        return substeps

    def _host_fields(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._h is None:
            empty = np.zeros(0, dtype=np.float32)
            return empty, empty, empty
        return (np.asarray(self._h.numpy(), dtype=np.float32),
                np.asarray(self._u.numpy(), dtype=np.float32),
                np.asarray(self._v.numpy(), dtype=np.float32))

    def sample_for_bodies(self, positions: np.ndarray, body_velocities=None,
                           drag=None, cross_area=None, body_height=None,
                           rotations=None, half_extents=None) -> dict:
        count = len(positions)
        if not count:
            empty3 = np.zeros((0, 3), dtype=np.float32)
            return {"depths": np.zeros(0, dtype=np.float32), "velocities": empty3,
                    "forces": empty3.copy()}
        body_velocities = np.zeros((count, 3), dtype=np.float32) if body_velocities is None \
            else np.asarray(body_velocities, dtype=np.float32)
        drag = np.ones(count, dtype=np.float32) if drag is None else np.asarray(drag, dtype=np.float32)
        cross_area = np.ones(count, dtype=np.float32) if cross_area is None \
            else np.asarray(cross_area, dtype=np.float32)
        body_height = np.ones(count, dtype=np.float32) if body_height is None \
            else np.asarray(body_height, dtype=np.float32)
        rotations = np.zeros((count, 3), dtype=np.float32) if rotations is None \
            else np.asarray(rotations, dtype=np.float32)
        half_extents = np.zeros((count, 2), dtype=np.float32) if half_extents is None \
            else np.asarray(half_extents, dtype=np.float32)
        if count != self._body_count:
            self._body_positions = wp.empty(count, dtype=wp.vec3, device=self.device)
            self._body_rotations = wp.empty(count, dtype=wp.vec3, device=self.device)
            self._body_extents = wp.empty(count, dtype=wp.vec2, device=self.device)
            self._body_velocities = wp.empty(count, dtype=wp.vec3, device=self.device)
            self._body_drag = wp.empty(count, dtype=float, device=self.device)
            self._body_area = wp.empty(count, dtype=float, device=self.device)
            self._body_height = wp.empty(count, dtype=float, device=self.device)
            self._sample_depth = wp.empty(count, dtype=float, device=self.device)
            self._sample_immersion = wp.empty(count, dtype=float, device=self.device)
            self._sample_surface = wp.empty(count, dtype=float, device=self.device)
            self._sample_support = wp.empty(count, dtype=float, device=self.device)
            self._sample_velocity = wp.empty(count, dtype=wp.vec3, device=self.device)
            self._sample_force = wp.empty(count, dtype=wp.vec3, device=self.device)
            self._body_count = count
        self._body_positions.assign(np.asarray(positions, dtype=np.float32))
        self._body_rotations.assign(rotations)
        self._body_extents.assign(half_extents)
        self._body_velocities.assign(body_velocities)
        self._body_drag.assign(drag)
        self._body_area.assign(cross_area)
        self._body_height.assign(body_height)
        wp.launch(_sample_bodies, dim=count, inputs=[self._h, self._u, self._v,
                   self._bed, self._obstacles, self._body_positions,
                   self._body_rotations, self._body_extents, self._body_velocities,
                   self._body_drag, self._body_area, self._body_height,
                   self._sample_depth, self._sample_immersion, self._sample_surface,
                   self._sample_support, self._sample_velocity, self._sample_force,
                  self._width, self._height,
                  float(self._terrain.cell_size), config.WATER_DENSITY], device=self.device)
        return {"device": self.device, "count": count,
                 "depths_device": self._sample_depth,
                 "immersions_device": self._sample_immersion,
                 "surface_elevations_device": self._sample_surface,
                 "support_elevations_device": self._sample_support,
                "velocities_device": self._sample_velocity,
                "forces_device": self._sample_force}

    def reset(self) -> None:
        if self._world is not None:
            self.initialize(self._world)

    def set_level(self, level: float) -> None:
        level = float(level)
        if abs(level - self._level) > 1.0e-9 and self._h is not None:
            self._source_enabled = True
            wp.launch(_apply_source, dim=self._count, inputs=[self._h, self._bed,
                      self._obstacles, self._width, self._height,
                      config.FLUID_SOURCE_COLUMNS, level], device=self.device)
        self._level = level

    def get_water_height(self, x=0.0, z=0.0) -> float:
        field = self.get_water_height_field()
        if not len(field):
            return 0.0
        i = int(np.clip(round(x / self._terrain.cell_size + self._terrain.width / 2),
                        0, self._width - 1))
        j = int(np.clip(round(z / self._terrain.cell_size + self._terrain.height / 2),
                        0, self._height - 1))
        return float(field[j * self._width + i])

    def get_water_height_field(self) -> np.ndarray:
        if self._h is None:
            return np.zeros(0, dtype=np.float32)
        depth = np.asarray(self._h.numpy(), dtype=np.float32)
        return np.where(depth > config.FLUID_DRY_DEPTH, self._bed_host + depth,
                        self._bed_host - 0.05).astype(np.float32)

    def get_velocity_field(self) -> Optional[np.ndarray]:
        _, u, v = self._host_fields()
        return (np.column_stack((u, np.zeros_like(u), v)).astype(np.float32)
                 if len(u) else None)

    def get_flow_particles(self) -> np.ndarray:
        if self._flow_particles is None:
            return np.zeros((0, 3), dtype=np.float32)
        return np.asarray(self._flow_particles.numpy(), dtype=np.float32)

    def diagnostics(self) -> dict:
        suspended = (float(np.asarray(self._sediment.numpy()).sum())
                     if self._sediment is not None else 0.0)
        return {"solver": "warp_shallow_water", "device": self.device,
                "erosion": self._erosion_enabled,
                "outflow_columns": self._outflow_columns,
                "sources": self._source_count,
                "drains": self._drain_count,
                "drain_circulation": (
                    [float(x) for x in self._drain_circulation.numpy()]
                    if self._drain_count else []),
                "suspended_sediment": suspended,
                "grid": [self._width, self._height], "substeps": self.last_substeps,
                "terrain_gpu_uploads": self.terrain_gpu_uploads,
                "obstacle_gpu_uploads": self.obstacle_gpu_uploads,
                **self._diag}


def create_fluid_solver(device: str) -> FluidSolver:
    if WARP_IMPORTED:
        try:
            return WarpShallowWaterSolver(device)
        except Exception as exc:  # pragma: no cover
            print(f"[naturelab] FloodSolver unavailable: {exc}; using placeholder")
    return PlaceholderFluidSolver()
