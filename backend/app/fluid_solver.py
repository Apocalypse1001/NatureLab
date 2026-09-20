"""Warp shallow-water solver with revision-driven GPU coupling."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from . import config
from .compute_engine import WARP_IMPORTED, wp


if WARP_IMPORTED:
    @wp.func
    def _face_flux(velocity: float, h_from: float, bed_from: float,
                   bed_other: float) -> float:
        available = wp.max(0.0, bed_from + h_from - wp.max(bed_from, bed_other))
        return velocity * available


    @wp.func
    def _upwind_flux(face: float, h_a: float, bed_a: float, h_b: float,
                     bed_b: float) -> float:
        """Discharge per unit width from cell a to cell b across their shared
        face, carried at the upwind cell's depth above the higher of the two
        beds -- water does not climb out over a higher dry bed."""
        if face >= 0.0:
            return _face_flux(face, h_a, bed_a, bed_b)
        return _face_flux(face, h_b, bed_b, bed_a)


    @wp.kernel
    def _velocity_step(h: wp.array(dtype=float), u: wp.array(dtype=float),
                       v: wp.array(dtype=float), bed: wp.array(dtype=float),
                       solid: wp.array(dtype=wp.int32),
                       next_u: wp.array(dtype=float), next_v: wp.array(dtype=float),
                       inlet_q: wp.array(dtype=float),
                       width: int, height: int, dx: float, dt: float,
                       gravity: float, dry: float, manning: wp.array(dtype=float),
                       friction_min_depth: float,
                       max_velocity: float, outflow_columns: int,
                       outflow_row_lo: int, outflow_row_hi: int,
                       outlet_normal: int, outlet_slope: wp.array(dtype=float)):
        """Local-inertial momentum at cell FACES (v0.15.0, docs/15_cgrid_plan.md).

        `u[idx]` is the velocity on the face between cell idx and idx+1, and
        `v[idx]` on the face between idx and idx+width. The slot in the last
        column (u) or row (v) is the outer edge face. Cell-centred velocity is
        derived from these by `_centre_velocity` for everything that wants a
        velocity AT a cell.

        Until v0.14.3 u, v and h shared one point, the surface gradient was the
        central (eta[i+1] - eta[i-1]) / 2dx, and `_depth_step` carried water at
        the average of two cell velocities. Both operators are blind to a
        pattern that alternates cell to cell, so that pattern neither grew nor
        decayed: it scoured alternate cells under erosion (v0.12.0) and grew to
        56% of the depth in a rain film running into a channel (v0.14.3 rain,
        measured in docs/probe_2dx_gauge_v1.py). On a face the gradient is the
        compact (eta_b - eta_a) / dx, which such a pattern cannot hide from:
        it becomes an ordinary short gravity wave that runs off and is damped.
        This is also where Bates, Horritt & Fewtrell (2010) put the discharge
        in the first place.

        The face is wet when the higher surface stands above the higher bed --
        the same test `_face_flux` already applied to transport, so a dry bank
        higher than the water beside it feels no gradient and takes no flow.
        ANY positive depth counts, not only depth above FLUID_DRY_DEPTH: with
        the threshold here a draining film decays toward the threshold and
        stops exactly on it -- measured on the tsunami beach, a strip of cells
        left at 0.100 mm that the waterline (and every "is it wet" reader)
        counted as sea, so the sea "retreated" 52 m while at a 1 mm threshold it
        had gone 102 m. The dry threshold decides what is reported as wet and
        whether a cell has a velocity; it must not decide whether water can
        leave.

        Manning bed friction, tau/rho = g*n^2*|u|*u / h^(4/3), is applied
        semi-implicitly (divide by 1 + drag) rather than subtracted: the
        explicit form overshoots and reverses the flow exactly where the
        coefficient is largest, a thin fast wetting front, and would need its
        own timestep limit. Dividing can only slow water, never turn it round
        (FrictionLawTests). One denominator from the full speed -- this face's
        component and the mean of the four faces of the other kind around it
        -- so diagonal flow is not dragged less than axis-aligned flow.

        Boundaries live on faces too:
        - west inlet (a row with inlet_q > 0): the face 0->1 carries the
          arriving velocity q/h, capped at twice critical so a nearly dry edge
          cell does not turn q/h into a jet. The discharge itself is written by
          `_depth_step`; this value is what the cells around it, the sediment
          capacity and the CFL limit read. v0.12.0 learned the hard way that a
          flux coaxed out of an averaged cell velocity comes out doubled, and
          that the "exact" correction u[0] = 2q/h - u[1] generates the very
          odd-even mode this grid removes;
        - east outlet band: the edge face copies the last interior face's
          outward velocity (zero gradient), and every face in the band may only
          point outward, so the boundary can never become a second source.
          `_apply_outflow` removes what crosses the edge face;
        - every other outer face is a wall, u = 0.
        """
        idx = wp.tid()
        i = idx % width
        j = idx // width
        in_band = (outflow_columns > 0 and j >= outflow_row_lo
                   and j <= outflow_row_hi and i >= width - 1 - outflow_columns)

        ux = float(0.0)
        if i == width - 1:
            if in_band and solid[idx] == 0 and h[idx] > dry and solid[idx - 1] == 0:
                # The open edge: a ghost cell beyond the map continues the
                # surface slope of the last two cells, and the edge face feels
                # that slope and friction like any other face. A level pool
                # feels nothing and stays; a wave or a river running at the
                # edge draws its own surface down and leaves.
                #
                # Capped at critical, sqrt(g*h): the most a free edge can pass is
                # critical flow at its brink, and a local-inertial model has no
                # advection to stop it running supercritical on its own. For a
                # river this makes the edge a FREE OVERFALL -- measured, 80 m3/s
                # settles at Froude 1.00 at the edge with a drawdown over the
                # last ~100 m -- not a normal-depth boundary. A normal-depth
                # edge would need to know what lies beyond the map: a river
                # continuing at its bed slope, or a sea it would drain.
                # Measured before this, with the edge copying the last interior
                # face's velocity: 80 m3/s drew the last cell down to 0.61 m at
                # 8.1 m/s, Froude 3.3, and the whole 200 m channel sat in the
                # drawdown. The collocated outlet this replaced had the opposite
                # fault -- it held 4.3 m of still water in the last two columns,
                # a dam at the map edge (docs/15_cgrid_plan.md).
                #
                # v0.16.0, outlet_kind "river": the ghost cell instead holds the
                # last cell's DEPTH on a bed that keeps falling at the channel's
                # own slope (fitted over config.OUTLET_SLOPE_REACH_M), so the
                # face feels exactly that slope. Gravity along the bed against
                # Manning friction is uniform flow, and the depth at the edge
                # settles at normal depth instead of drawing down to critical.
                # A row whose bed does not fall toward the edge has no normal
                # depth to run at and keeps the overfall.
                eta_last = bed[idx] + h[idx]
                eta_prev = bed[idx - 1] + h[idx - 1]
                drive = float(0.0)
                if outlet_normal != 0 and outlet_slope[j] > 0.0:
                    drive = outlet_slope[j]
                else:
                    drive = (eta_prev - eta_last) / dx
                ux = u[idx] + gravity * dt * drive
                cross = v[idx]
                if j > 0:
                    cross = 0.5 * (cross + v[idx - width])
                speed = wp.sqrt(ux * ux + cross * cross)
                n = manning[idx]
                hf = wp.max(h[idx], friction_min_depth)
                ux = ux / (1.0 + gravity * n * n * speed * dt / wp.pow(hf, 1.3333333))
                ux = wp.clamp(ux, 0.0, wp.min(wp.sqrt(gravity * h[idx]), max_velocity))
        elif i == 0 and inlet_q[j] > 0.0:
            if solid[idx] == 0 and h[idx] > dry:
                ux = wp.min(inlet_q[j] / h[idx], 2.0 * wp.sqrt(gravity * h[idx]))
        elif solid[idx] == 0 and solid[idx + 1] == 0:
            eta_a = bed[idx] + h[idx]
            eta_b = bed[idx + 1] + h[idx + 1]
            flow = wp.max(eta_a, eta_b) - wp.max(bed[idx], bed[idx + 1])
            if flow > 0.0:
                ux = u[idx] - gravity * dt * (eta_b - eta_a) / dx
                cross = v[idx] + v[idx + 1]
                if j > 0:
                    cross = cross + v[idx - width] + v[idx - width + 1]
                cross = 0.25 * cross
                speed = wp.sqrt(ux * ux + cross * cross)
                n = 0.5 * (manning[idx] + manning[idx + 1])
                hf = wp.max(flow, friction_min_depth)
                ux = ux / (1.0 + gravity * n * n * speed * dt / wp.pow(hf, 1.3333333))
                if i == 1 and inlet_q[j] > 0.0 and h[idx - 1] > dry:
                    # The arriving water brings its momentum with it -- the one
                    # place the model carries momentum flux at all, because a
                    # prescribed discharge is momentum entering the map. Without
                    # it the inlet hands cell 1 mass and nothing else, and a
                    # river primed at rest has to accelerate from gravity alone:
                    # measured on this grid before the term was restored, 10.8
                    # of a requested 12 m3/s crossed this face after 10 s, 11.3
                    # after 60. Mixing, not forcing: the fraction of cell 1's
                    # column replaced this substep arrives at the inlet
                    # velocity, the rest keeps what it had, and at equilibrium
                    # the two are the same number so the term does nothing.
                    arriving = wp.min(inlet_q[j] / h[idx - 1],
                                      2.0 * wp.sqrt(gravity * h[idx - 1]))
                    fraction = wp.min(1.0, inlet_q[j] * dt / (dx * h[idx]))
                    ux = ux + fraction * (arriving - ux)
                if in_band:
                    ux = wp.max(0.0, ux)
                ux = wp.clamp(ux, -max_velocity, max_velocity)
        next_u[idx] = ux

        vz = float(0.0)
        if j < height - 1 and solid[idx] == 0 and solid[idx + width] == 0:
            eta_a = bed[idx] + h[idx]
            eta_b = bed[idx + width] + h[idx + width]
            flow = wp.max(eta_a, eta_b) - wp.max(bed[idx], bed[idx + width])
            if flow > 0.0:
                vz = v[idx] - gravity * dt * (eta_b - eta_a) / dx
                cross = u[idx] + u[idx + width]
                if i > 0:
                    cross = cross + u[idx - 1] + u[idx + width - 1]
                cross = 0.25 * cross
                speed = wp.sqrt(vz * vz + cross * cross)
                n = 0.5 * (manning[idx] + manning[idx + width])
                hf = wp.max(flow, friction_min_depth)
                vz = vz / (1.0 + gravity * n * n * speed * dt / wp.pow(hf, 1.3333333))
                vz = wp.clamp(vz, -max_velocity, max_velocity)
        next_v[idx] = vz


    @wp.kernel
    def _depth_step(h: wp.array(dtype=float), u: wp.array(dtype=float),
                    v: wp.array(dtype=float), bed: wp.array(dtype=float),
                    solid: wp.array(dtype=wp.int32), next_h: wp.array(dtype=float),
                    inlet_q: wp.array(dtype=float),
                    width: int, height: int, dx: float, dt: float):
        """Continuity: each cell gains what its four faces carry in.

        Every face is read from one stored velocity, so the water leaving one
        cell across a face is the same number as the water arriving in its
        neighbour -- conservation holds face by face, not on average. The outer
        faces carry nothing here; the outlet's edge face is emptied by
        `_apply_outflow`, which books it to the ledger.
        """
        idx = wp.tid()
        i = idx % width
        j = idx // width
        if solid[idx] != 0:
            next_h[idx] = 0.0
            return
        q_right = float(0.0)
        q_left = float(0.0)
        q_up = float(0.0)
        q_down = float(0.0)
        if i < width - 1 and solid[idx + 1] == 0:
            q_right = _upwind_flux(u[idx], h[idx], bed[idx], h[idx + 1], bed[idx + 1])
        if i > 0 and solid[idx - 1] == 0:
            q_left = _upwind_flux(u[idx - 1], h[idx - 1], bed[idx - 1], h[idx], bed[idx])
        if j < height - 1 and solid[idx + width] == 0:
            q_up = _upwind_flux(v[idx], h[idx], bed[idx], h[idx + width], bed[idx + width])
        if j > 0 and solid[idx - width] == 0:
            q_down = _upwind_flux(v[idx - width], h[idx - width], bed[idx - width],
                                  h[idx], bed[idx])
        if inlet_q[j] > 0.0 and i <= 1:
            # The river inlet is a flux boundary, so its discharge is written on
            # the face between columns 0 and 1 rather than coaxed out of a
            # velocity. Both cells compute the same number from the same depth,
            # so the water that leaves cell 0 is exactly the water that arrives
            # in cell 1 -- prescribing it independently on each side would leak
            # or manufacture volume at the boundary.
            #
            # Capped by what the edge cell actually holds: a dry inlet cannot
            # deliver its discharge, and taking more than is there would push
            # the cell negative, where `wp.max(0.0, ...)` would quietly create
            # the difference and put the volume ledger out.
            crossing = wp.min(inlet_q[j], h[idx - i] * dx / dt)
            if i == 0:
                q_right = crossing
            else:
                q_left = crossing
        next_h[idx] = wp.max(0.0, h[idx] - dt * ((q_right - q_left) + (q_up - q_down)) / dx)


    @wp.kernel
    def _centre_velocity(h: wp.array(dtype=float), u: wp.array(dtype=float),
                         v: wp.array(dtype=float), bed: wp.array(dtype=float),
                         solid: wp.array(dtype=wp.int32),
                         inlet_q: wp.array(dtype=float),
                         uc: wp.array(dtype=float), vc: wp.array(dtype=float),
                         width: int, height: int, dry: float, max_velocity: float):
        """Cell-centred velocity: the discharge through the cell over its depth.

        Everything that asks "how fast is the water HERE" -- sediment capacity
        and transport, flow tracers, floating bodies, gauges, the drain's
        circulation measurement, the velocity stream to the browser -- reads
        this rather than a face.

        Built from the face DISCHARGES `_depth_step` actually moves, averaged
        and divided by this cell's own depth, not from the mean of the two
        face velocities. The two agree in uniform flow and part company exactly
        where it matters: a face velocity belongs to the depth it was carried
        at. Measured on the first version of this kernel (plain velocity mean):
        the cell after the river inlet averaged in q/h of the SHALLOW inlet
        cell, read 1.4 m/s where its own depth carried the same q at 0.85, took
        that as sediment capacity, scoured, got deeper -- and a deeper cell with
        the same borrowed speed has even more capacity. 1.4 m of trench in
        150 s, across the whole inlet band.

        A dry cell has no velocity, and the result is clamped to the faster of
        its two faces: a film receiving the flux of a deep neighbour would
        otherwise divide a large q by a small h and report a speed no face has.
        In an inlet row the west edge delivers the inlet's q, so column 0 is not
        averaged against a wall -- a half-loaded inlet is a hungry one.
        """
        idx = wp.tid()
        if solid[idx] != 0 or h[idx] <= dry:
            uc[idx] = 0.0
            vc[idx] = 0.0
            return
        i = idx % width
        j = idx // width
        inlet_row = inlet_q[j] > 0.0 and i <= 1

        q_left = float(0.0)
        u_left = float(0.0)
        if i == 0 and inlet_row:
            q_left = inlet_q[j]
            u_left = u[idx]
        elif i == 1 and inlet_row:
            q_left = inlet_q[j]
            u_left = u[idx - 1]
        elif i > 0 and solid[idx - 1] == 0:
            q_left = _upwind_flux(u[idx - 1], h[idx - 1], bed[idx - 1], h[idx], bed[idx])
            u_left = u[idx - 1]
        q_right = float(0.0)
        if i == 0 and inlet_row:
            q_right = inlet_q[j]
        elif i == width - 1:
            q_right = u[idx] * h[idx]           # the outlet's edge face
        elif solid[idx + 1] == 0:
            q_right = _upwind_flux(u[idx], h[idx], bed[idx], h[idx + 1], bed[idx + 1])
        limit_x = wp.min(wp.max(wp.abs(u_left), wp.abs(u[idx])), max_velocity)
        uc[idx] = wp.clamp(0.5 * (q_left + q_right) / h[idx], -limit_x, limit_x)

        q_down = float(0.0)
        v_down = float(0.0)
        if j > 0 and solid[idx - width] == 0:
            q_down = _upwind_flux(v[idx - width], h[idx - width], bed[idx - width],
                                  h[idx], bed[idx])
            v_down = v[idx - width]
        q_up = float(0.0)
        if j < height - 1 and solid[idx + width] == 0:
            q_up = _upwind_flux(v[idx], h[idx], bed[idx], h[idx + width], bed[idx + width])
        limit_z = wp.min(wp.max(wp.abs(v_down), wp.abs(v[idx])), max_velocity)
        vc[idx] = wp.clamp(0.5 * (q_down + q_up) / h[idx], -limit_z, limit_z)


    @wp.kernel
    def _apply_source(h: wp.array(dtype=float), bed: wp.array(dtype=float),
                      u: wp.array(dtype=float), v: wp.array(dtype=float),
                      sediment: wp.array(dtype=float),
                      solid: wp.array(dtype=wp.int32), width: int, height: int,
                      source_columns: int, level: float, area: float,
                      capacity_scale: float, added: wp.array(dtype=float),
                      sediment_in: wp.array(dtype=float)):
        idx = wp.tid()
        if idx < width * height:
            i = idx % width
            if i < source_columns and solid[idx] == 0:
                target = wp.max(0.0, level - bed[idx])
                wp.atomic_add(added, 0, (target - h[idx]) * area)
                h[idx] = target
                # v0.12.0: the arriving water carries what it can carry.
                # Without this the source columns see clean water every substep
                # for ever, are therefore at maximum hunger for ever, and sit on
                # SEDIMENT_MAX_BED_CHANGE for ever -- which is where the v0.11.0
                # "erosion is limited by the clamp" reading came from: 29 of the
                # 36 clamped cells were these. Measured in
                # docs/07_river_plan.md. A river arriving at capacity is not
                # hungry and does not scour its own inlet.
                speed = wp.sqrt(u[idx] * u[idx] + v[idx] * v[idx])
                arriving = capacity_scale * speed * target
                # v0.18.0: booked, so the sediment ledger can close
                wp.atomic_add(sediment_in, 0, (arriving - sediment[idx]) * area)
                sediment[idx] = arriving


    @wp.kernel
    def _apply_tsunami_edge(h: wp.array(dtype=float), bed: wp.array(dtype=float),
                            solid: wp.array(dtype=wp.int32),
                            width: int, height: int, columns: int,
                            level: float, area: float,
                            added: wp.array(dtype=float)):
        """Hold the EAST columns at the sea level outside the map.

        The open ocean beyond the domain draws down and then surges; this is
        the window onto a coast, not a container for a whole wave. Mechanically
        identical to `_apply_source` on the west edge -- target depth is
        `max(0, level - bed)` and the change is booked to the volume ledger --
        except that `level` is a function of time, supplied per substep by
        `WarpShallowWaterSolver.advance` from `WaterState.tsunami_*`.

        Why this rather than seeding a wave inside the grid, which is what
        v0.14.0 did and what docs/probe_tsunami_v2.py measured as wrong: a wave
        long enough to draw the sea back does not FIT in the domain alongside
        the shore and the land behind it. Driving the boundary moves the wave
        period into TIME, where the domain size stops constraining it.

        Deliberately a hard write, not a soft pull toward `level`: a "blend
        continuously, leave the edge's outward velocity always free" version
        was tried (v0.14.3's first attempt) and measured wrong -- with
        `set_outflow`'s wavemaker override removed, `_velocity_step`'s outlet
        branch clamps ANY negative (inward) edge velocity to zero the instant
        `outflow_columns > 0`, so the crest could never push water into the
        domain no matter how hard this kernel pulled `h`: flood measured at
        0 m instead of 258 m, reproducing the exact regression
        docs/13_tsunami2_plan.md already recorded once for v0.14.1. Velocity
        MUST be fully closed while a pulse is actually arriving; see
        `WarpShallowWaterSolver.advance` for the (now discrete, not
        continuous) on/off gate this settled on instead, and
        config.TSUNAMI_ACTIVE_WINDOW_PERIODS for why the reflection problem
        is solved between pulses rather than during one.
        """
        idx = wp.tid()
        if idx < width * height:
            i = idx % width
            if i >= width - columns and solid[idx] == 0:
                target = wp.max(0.0, level - bed[idx])
                wp.atomic_add(added, 0, (target - h[idx]) * area)
                h[idx] = target


    @wp.kernel
    def _apply_river_inlet(h: wp.array(dtype=float), u: wp.array(dtype=float),
                           v: wp.array(dtype=float),
                           sediment: wp.array(dtype=float),
                           solid: wp.array(dtype=wp.int32),
                           inlet_q: wp.array(dtype=float),
                           normal_depth: wp.array(dtype=float),
                           width: int, height: int, dx: float, dt: float,
                           area: float,
                           capacity_scale: float, added: wp.array(dtype=float),
                           sediment_in: wp.array(dtype=float)):
        """Local inlet on the west edge, prescribing discharge rather than level.

        A river is delivered as a discharge Q; a level is what the channel
        answers with. So this kernel owns `h` only, and `_velocity_step` turns
        the same q into u = q/h -- the two halves of one boundary condition,
        deliberately not both written here (see the comment at `i == 0`).

        Depth is the interior value carried outward (zero-gradient), floored at
        the normal depth for the requested q on this bed slope. Zero-gradient
        alone cannot start a dry channel -- h[0] = h[1] = 0 for ever, and the
        inlet is silently inert; the floor is what lets the flow arrive at the
        depth Manning says it should have. Where the channel is already deeper
        than that, backwater from downstream wins, which is the physically right
        way round for subcritical flow.

        NOT "add Q*dt of volume to the edge cells": volume with no momentum to
        go with it makes a mound that spreads radially, not a river.
        """
        idx = wp.tid()
        i = idx % width
        j = idx // width
        if i != 0 or solid[idx] != 0:
            return
        q = inlet_q[j]
        if q <= 0.0:
            return
        target = wp.max(h[idx + 1], normal_depth[j])
        # A discharge boundary delivers Q and no more. Holding the edge cell at
        # its target depth unconditionally makes it an infinite reservoir
        # instead: whatever the interior draws, the boundary refills, and with
        # erosion on that closes a loop -- the flow scours a hollow, the hollow
        # accelerates the flow, the faster flow draws harder, and the inlet
        # obliges. Measured before this cap: a requested 12 m3/s delivering 700,
        # 34 000 m3 on a map that should have held 3 000, and the velocity clamp
        # pinned at 20 m/s.
        #
        # In steady state a cell of width dx loses exactly q*dt/dx of depth per
        # substep to the face, so this allowance replaces the water that leaves
        # and nothing else. If the interior pulls harder than that, the edge
        # depth falls -- which is the honest answer: a pipe delivering Q cannot
        # be made to deliver more by pulling on it.
        allowance = q * dt / dx
        gain = wp.min(wp.max(0.0, target - h[idx]), allowance)
        depth = h[idx] + gain
        wp.atomic_add(added, 0, gain * area)
        h[idx] = depth
        # Arriving loaded to capacity, computed from the velocity the erosion
        # kernel will use a few lines later in this same substep rather than
        # from the requested q -- the two differ while the flow is developing,
        # and the difference is a gap the inlet would erode its own bed to
        # close. Equal capacity, zero gap, no self-scour.
        speed = wp.sqrt(u[idx] * u[idx] + v[idx] * v[idx])
        arriving = capacity_scale * speed * depth
        # v0.18.0: the load the river brings in is assigned, not transported --
        # booked here so the sediment ledger has a source term to close against
        wp.atomic_add(sediment_in, 0, (arriving - sediment[idx]) * area)
        sediment[idx] = arriving


    @wp.kernel
    def _apply_outflow(h: wp.array(dtype=float), u: wp.array(dtype=float),
                       sediment: wp.array(dtype=float),
                       solid: wp.array(dtype=wp.int32), width: int, height: int,
                       columns: int, row_lo: int, row_hi: int,
                       dx: float, dt: float, area: float,
                       removed: wp.array(dtype=float),
                       sediment_out: wp.array(dtype=float)):
        """Let water leave through the east edge instead of piling against it.

        Every outer face is otherwise no-flux, so water that enters the map can
        never leave and the domain fills forever -- measured before this
        existed: volume climbing 2290 -> 3484 -> 4311 m3, never decreasing. A
        river that runs off the edge of the domain is exactly this boundary.

        `_depth_step` cannot carry flux across the outer face (the last column
        has no right neighbour to exchange with), so the discharge that would
        have crossed it is removed here explicitly: q = u*h per unit width, over
        one cell, in one timestep. Only outward velocity counts, and removal is
        capped by the water present, so the outlet can neither inject water nor
        drive a cell negative. `_velocity_step` is what allows the edge velocity
        to be non-zero in the first place -- before that change this kernel was
        inert, because the edge column's u was clamped to 0 every substep before
        anything could use it.

        v0.12.0 adds two things. The outlet can be a band of rows rather than
        the whole edge, so a valley can drain through its channel and not
        through its floodplain. And the departing water takes its suspended load
        with it: before this, `h` left and the sediment it was carrying stayed,
        so the east edge slowly turned into a bar made of material the river had
        already delivered to the sea.
        """
        idx = wp.tid()
        i = idx % width
        j = idx // width
        # v0.15.0: water leaves across the one outer face, from the last
        # column, at that face's velocity (`_velocity_step` gives the edge face
        # the last interior face's outward speed). Before, u lived at centres
        # and each of the last `columns` columns drained on its own velocity;
        # on faces the interior columns already pass their water on, so
        # draining them too would take it twice.
        if i != width - 1 or solid[idx] != 0:
            return
        if j < row_lo or j > row_hi:
            return
        outward = wp.max(0.0, u[idx])
        fraction = wp.min(1.0, outward * dt / dx)
        loss = h[idx] * fraction
        h[idx] = wp.max(0.0, h[idx] - loss)
        wp.atomic_add(removed, 0, loss * area)
        # the same fraction of the column leaves, so the same fraction of what
        # that column was carrying leaves with it
        wp.atomic_add(sediment_out, 0, sediment[idx] * fraction * area)
        sediment[idx] = sediment[idx] * (1.0 - fraction)


    @wp.kernel
    def _apply_point_sources(h: wp.array(dtype=float), bed: wp.array(dtype=float),
                             solid: wp.array(dtype=wp.int32),
                             centres: wp.array(dtype=wp.vec3),
                             radii: wp.array(dtype=float),
                             levels: wp.array(dtype=float),
                             count: int, width: int, height: int, dx: float,
                             area: float, added: wp.array(dtype=float)):
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
                target = wp.max(h[idx], wp.max(0.0, levels[n] - bed[idx]))
                wp.atomic_add(added, 0, (target - h[idx]) * area)
                h[idx] = target


    @wp.kernel
    def _apply_rain(h: wp.array(dtype=float), solid: wp.array(dtype=wp.int32),
                    depth: float, area: float, added: wp.array(dtype=float)):
        """RainLab-1: uniform rain, one poured portion `depth` on every open cell.

        An AREAL source (docs/14_rain_plan.md): it adds to whichever boundary is
        active rather than replacing it, which is why it is not gated with the
        edge/SOURCE/inlet "one answer to where the water comes from" rule.

        The ledger books what `h` actually gained, not `depth * area`: `h` is
        float32, and on a deep cell a small add is rounded. Booking the request
        would turn that rounding into an unexplained conservation error; booking
        the increment keeps `volume_error_m3` honest about the water that is
        really there. Solid cells receive nothing -- until roofs exist
        (RainLab-2) rain on a building is not delivered at all.
        """
        idx = wp.tid()
        if solid[idx] != 0:
            return
        before = h[idx]
        h[idx] = before + depth
        wp.atomic_add(added, 0, (h[idx] - before) * area)


    @wp.kernel
    def _measure_drain_circulation(u: wp.array(dtype=float), v: wp.array(dtype=float),
                                   h: wp.array(dtype=float),
                                   solid: wp.array(dtype=wp.int32),
                                   centres: wp.array(dtype=wp.vec3),
                                   radii: wp.array(dtype=float),
                                   circulation: wp.array(dtype=float),
                                   samples: wp.array(dtype=float),
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

        `samples` is accumulated alongside so the caller can turn the sum into a
        MEAN tangential speed. Using the raw sum would make the vortex strength
        scale with how many cells happen to fall in the annulus -- i.e. with grid
        resolution and drain radius -- so doubling the map would have doubled the
        spin. That is a physics bug, not a tuning issue.
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
            wp.atomic_add(circulation, n, u[idx] * tx + v[idx] * tz)
            wp.atomic_add(samples, n, 1.0)


    @wp.func
    def _sink_velocity(dxc: float, dzc: float, radius: float, strength: float,
                       depth: float, dry: float, mean_tangential: float,
                       swirl_gain: float) -> wp.vec2:
        """The drain's imposed (u, v) at offset (dxc, dzc) from its centre."""
        r = wp.sqrt(dxc * dxc + dzc * dzc)
        # keep the core finite: 1/r blows up at the exact centre, and a real
        # vortex has a rotational core of finite size anyway
        r_safe = wp.max(r, radius * 0.25)
        nx = dxc / r_safe
        nz = dzc / r_safe
        tx = -nz
        tz = nx
        # radial speed straight from continuity with the removal
        enclosed = strength * 3.14159265 * r_safe * r_safe * (
            1.0 - r_safe * r_safe / (2.0 * radius * radius))
        radial = -enclosed / (2.0 * 3.14159265 * r_safe * wp.max(depth, dry))
        # Mean ambient tangential speed measured in the annulus at ~1.5R.
        # Angular momentum conservation, v_theta * r = const, carries it
        # inward: v_theta(r) = v_mean * r_ref / r. Sign and magnitude both
        # come from the measurement -- nothing here picks a direction.
        spin = swirl_gain * mean_tangential * (radius * 1.5) / r_safe
        return wp.vec2(nx * radial + tx * spin, nz * radial + tz * spin)


    @wp.func
    def _vent_velocity(dxc: float, dzc: float, radius: float, strength: float,
                       depth: float, dry: float) -> wp.vec2:
        """A vent's outward (u, v) at offset (dxc, dzc): the drain's continuity
        relation run backwards, with no swirl."""
        r = wp.sqrt(dxc * dxc + dzc * dzc)
        r_safe = wp.max(r, radius * 0.25)
        enclosed = strength * 3.14159265 * r_safe * r_safe * (
            1.0 - r_safe * r_safe / (2.0 * radius * radius))
        radial = enclosed / (2.0 * 3.14159265 * r_safe * wp.max(depth, dry))
        return wp.vec2(dxc / r_safe * radial, dzc / r_safe * radial)


    @wp.kernel
    def _apply_drains(h: wp.array(dtype=float), u: wp.array(dtype=float),
                      v: wp.array(dtype=float), solid: wp.array(dtype=wp.int32),
                      centres: wp.array(dtype=wp.vec3), radii: wp.array(dtype=float),
                      strengths: wp.array(dtype=float),
                      circulation: wp.array(dtype=float),
                      samples: wp.array(dtype=float),
                      count: int, width: int, height: int, dx: float, dt: float,
                      dry: float, swirl_gain: float, max_velocity: float,
                      area: float, removed_total: wp.array(dtype=float),
                      removed_each: wp.array(dtype=float), take_dregs: int,
                      impose_flow: int, measure_only: int):
        """Remove water through a localized sink and spin up the flow around it.

        v0.17.0: the same kernel runs the storm sewer's inlets. `removed_each`
        books what every sink took, so a pipe can hand exactly that volume to
        its outfall; `take_dregs` = 0 leaves a film below the dry threshold
        where it is, so an inlet on a wet street cannot take more than its pipe
        carries by sweeping up films (at 60 substeps a second, the dregs of a
        few cells alone come to tens of litres a second).

        `impose_flow` = 0 (storm inlets) removes water and nothing else. The
        imposed field below is an ideal funnel centred on the grate; on a real
        pond it overrode the surface slope face by face. Measured on the Sewer
        scenario at 360 s: the cell east of the west grate, 0.2 cm LOWER than
        the grate's own cell, stayed dry beside a 15.9 cm column, because the
        funnel set the face between them to ~0 every substep -- and the grate
        took 13.1 L/s through a pipe rated 20.8. Without it, the depression the
        removal digs is what draws the water in, down the real slope.

        `measure_only` = 1 books into `removed_each` what each sink WOULD take
        at these strengths and changes nothing. v0.18.0 runs it with every
        grate at its whole path's capacity, to learn what the grates want
        before a shared pipe is split between them: splitting by what they
        took instead feeds back on itself, because a sink allowed less takes
        less at the same depth (measured: the west grate held 5 cm and took
        15.8 of 20.8 L/s).

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

        The radial velocity is DERIVED from the removal rate rather than being a
        second free knob. An earlier version set the two independently, and the
        convergence then out-ran the sink: water piled up at the centre and the
        drain cell ended up DEEPER than the same spot with no drain at all
        (measured: 1.254 m vs 1.219 m). Continuity fixes it -- the flux crossing
        a circle of radius r must equal the volume removed inside it:

            Q(r) = strength * pi * r^2 * (1 - r^2 / (2 R^2))
            v_r  = -Q(r) / (2 pi r h)

        so convergence and removal can never disagree again.
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
            strength = strengths[n]
            if radius <= 0.0 or strength <= 0.0:
                continue
            centre = centres[n]
            dxc = x - centre[0]
            dzc = z - centre[2]
            r = wp.sqrt(dxc * dxc + dzc * dzc)
            if r > radius:
                continue
            # smooth bell, so the sink has no hard rim for the scheme to ring on
            ratio = r / radius
            falloff = 1.0 - ratio * ratio
            if measure_only != 0:
                wp.atomic_add(removed_each, n, wp.min(h[idx], strength * falloff * dt) * area)
                continue
            removed = wp.min(h[idx], strength * falloff * dt)
            h[idx] = h[idx] - removed
            wp.atomic_add(removed_total, 0, removed * area)
            wp.atomic_add(removed_each, n, removed * area)
            if take_dregs != 0 and h[idx] <= dry:
                # the dregs are removed too, so the volume ledger stays exact
                # rather than exact-to-within-a-film
                wp.atomic_add(removed_total, 0, h[idx] * area)
                wp.atomic_add(removed_each, n, h[idx] * area)
                h[idx] = 0.0
                u[idx] = 0.0
                v[idx] = 0.0
                continue
            if impose_flow == 0:
                continue
            mean_tangential = circulation[n] / wp.max(1.0, samples[n])
            # Inside the sink the drain owns the flow, so this is assigned, not
            # accumulated -- accumulating let the two terms drift apart.
            # v0.15.0: velocity lives on faces, so the field is evaluated where
            # each face actually is (half a cell east of the centre for u, half
            # a cell north for v) instead of being written at the centre and
            # silently shifted half a cell by the grid.
            if i < width - 1:
                face_u = _sink_velocity(dxc + 0.5 * dx, dzc, radius, strength,
                                        h[idx], dry, mean_tangential, swirl_gain)
                u[idx] = wp.clamp(face_u[0], -max_velocity, max_velocity)
            if j < height - 1:
                face_v = _sink_velocity(dxc, dzc + 0.5 * dx, radius, strength,
                                        h[idx], dry, mean_tangential, swirl_gain)
                v[idx] = wp.clamp(face_v[1], -max_velocity, max_velocity)


    @wp.kernel
    def _section_flux(h: wp.array(dtype=float), u: wp.array(dtype=float),
                      v: wp.array(dtype=float), bed: wp.array(dtype=float),
                      solid: wp.array(dtype=wp.int32), inlet_q: wp.array(dtype=float),
                      cell_a: wp.array(dtype=wp.int32), cell_b: wp.array(dtype=wp.int32),
                      axis: wp.array(dtype=wp.int32), sign: wp.array(dtype=float),
                      owner: wp.array(dtype=wp.int32),
                      volume: wp.array(dtype=float), area: wp.array(dtype=float),
                      wet: wp.array(dtype=float), level: wp.array(dtype=float),
                      width: int, dx: float, dt: float, dry: float):
        """v0.18.0 gauging lines (backend/app/sections.py): book, per line, the
        water its faces carry this substep.

        Launched immediately before `_depth_step`, on the same `h`, `u`, `v`,
        `bed` it reads, through the same `_upwind_flux` and the same inlet-band
        branch -- so what a line reports is the discharge continuity moves, not
        a reconstruction of it. Everything is booked as time integrals (m3,
        m2*s, m*s) and divided by the frame length when folded.
        """
        k = wp.tid()
        a = cell_a[k]
        b = cell_b[k]
        if solid[a] != 0 or solid[b] != 0:
            return
        q = float(0.0)
        if axis[k] == 0:
            q = _upwind_flux(u[a], h[a], bed[a], h[b], bed[b])
            j = a // width
            if a % width == 0 and inlet_q[j] > 0.0:
                q = wp.min(inlet_q[j], h[a] * dx / dt)
        else:
            q = _upwind_flux(v[a], h[a], bed[a], h[b], bed[b])
        s = owner[k]
        wp.atomic_add(volume, s, sign[k] * q * dx * dt)
        depth = 0.5 * (h[a] + h[b])
        if depth > dry:
            wp.atomic_add(area, s, depth * dx * dt)
            wp.atomic_add(wet, s, dx * dt)
            wp.atomic_add(level, s, 0.5 * (h[a] + bed[a] + h[b] + bed[b]) * dx * dt)


    @wp.kernel
    def _sewer_route(step: wp.array(dtype=float), outfall_of: wp.array(dtype=wp.int32),
                     outfall_volume: wp.array(dtype=float), frame: wp.array(dtype=float)):
        """Hand what every storm inlet took this substep to its pipe's outfall.

        In the SAME substep, on the GPU: `_apply_drains` has already written
        `step`, so there is no delay and no host readback. The pipe holds no
        water (backend/app/sewer.py), so this is the whole of the pipe.
        """
        n = wp.tid()
        taken = step[n]
        frame[n] = frame[n] + taken
        target = outfall_of[n]
        if target >= 0:
            wp.atomic_add(outfall_volume, target, taken)


    @wp.kernel
    def _count_outfall_cells(solid: wp.array(dtype=wp.int32),
                             centres: wp.array(dtype=wp.vec3),
                             radii: wp.array(dtype=float),
                             count: int, width: int, height: int, dx: float,
                             cells: wp.array(dtype=float)):
        """How many open cells each outfall disc covers -- with the very same
        disc test `_apply_outfalls` uses, so the volume spread over them is
        exactly the volume that arrived."""
        idx = wp.tid()
        if solid[idx] != 0:
            return
        i = idx % width
        j = idx // width
        x = (float(i) - float(width - 1) * 0.5) * dx
        z = (float(j) - float(height - 1) * 0.5) * dx
        for n in range(count):
            if radii[n] <= 0.0:
                continue
            centre = centres[n]
            dxc = x - centre[0]
            dzc = z - centre[2]
            if wp.sqrt(dxc * dxc + dzc * dzc) <= radii[n]:
                wp.atomic_add(cells, n, 1.0)


    @wp.kernel
    def _apply_outfalls(h: wp.array(dtype=float), solid: wp.array(dtype=wp.int32),
                        centres: wp.array(dtype=wp.vec3), radii: wp.array(dtype=float),
                        volumes: wp.array(dtype=float), cells: wp.array(dtype=float),
                        count: int, width: int, height: int, dx: float, area: float,
                        added: wp.array(dtype=float)):
        """Pour what the pipes delivered this substep out over each outfall disc.

        Evenly over the disc's open cells, and mass only: the water arrives at
        rest and runs off down the surface slope. A jet with the pipe's own
        velocity would need the pipe's flow state, which a primitive pipe does
        not have (docs/16_sewer_plan.md).
        """
        idx = wp.tid()
        if solid[idx] != 0:
            return
        i = idx % width
        j = idx // width
        x = (float(i) - float(width - 1) * 0.5) * dx
        z = (float(j) - float(height - 1) * 0.5) * dx
        for n in range(count):
            volume = volumes[n]
            if volume <= 0.0 or cells[n] <= 0.0 or radii[n] <= 0.0:
                continue
            centre = centres[n]
            dxc = x - centre[0]
            dzc = z - centre[2]
            if wp.sqrt(dxc * dxc + dzc * dzc) <= radii[n]:
                gain = volume / (cells[n] * area)
                h[idx] = h[idx] + gain
                wp.atomic_add(added, 0, gain * area)


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


    # ------------------------------------------------------------ VolcanoLab (v0.13.0)
    # Physical constants (basalt-order), not tuning knobs -- the tunable part of
    # the cooling law lives in config.py as LAVA_EMISSIVITY / LAVA_COOLING_ENHANCEMENT.
    _LAVA_SIGMA = 5.670374419e-8   # Stefan-Boltzmann, W/(m^2 K^4)
    _LAVA_RHO = 2700.0             # kg/m^3
    _LAVA_CP = 1200.0              # J/(kg K)

    @wp.kernel
    def _advect_lava_energy(old_h: wp.array(dtype=float), new_h: wp.array(dtype=float),
                            temperature: wp.array(dtype=float),
                            next_temperature: wp.array(dtype=float),
                            u: wp.array(dtype=float), v: wp.array(dtype=float),
                            bed: wp.array(dtype=float), solid: wp.array(dtype=wp.int32),
                            width: int, height: int, dx: float, dt: float, dry: float,
                            ambient_k: float):
        """Temperature transport by face flux, not by `_advect_sediment`'s back-trace.

        The plan's item 2 asked for the sediment kernel reused as-is ("the same
        kernel, a different field"), and that was tried first. It fails
        specifically at a wetting front: `_advect_sediment` back-traces using
        the DESTINATION cell's own u/v, and a cell that is dry at the start of
        a substep has u=v=0 there (`_velocity_step` zeroes a dry cell before
        `_depth_step` gives it any water) -- so the instant it receives its
        first inflow this substep, the back-trace distance is exactly zero and
        it resamples itself, keeping its old (ambient) temperature instead of
        the hot value flowing in. Measured directly (docs/08_volcano_plan.md,
        the vent-flow measurement): every newly-wetted cell along the front
        read exactly ambient every substep, `_lava_manning` read it as already
        below the solidus, and `_solidify_lava` converted the first trickle to
        reach it straight into bed -- forever, at the very edge of the vent,
        which is why nothing ever got further than a few metres.

        This kernel instead moves T*h (a depth-integrated heat proxy) across
        the SAME faces with the SAME upwind face velocities `_depth_step`
        already used to move `h` for this substep -- literally the flux that
        just carried water into the cell also carries the temperature that
        water had. `old_h` is the depth `_depth_step` computed FROM (its
        `_face_flux` inputs), `new_h` is what it produced; both are cheaply
        available as the two sides of the swap `advance()` just performed.
        """
        idx = wp.tid()
        if solid[idx] != 0:
            next_temperature[idx] = temperature[idx]
            return
        i = idx % width
        j = idx // width
        q_right = float(0.0)
        q_left = float(0.0)
        q_up = float(0.0)
        q_down = float(0.0)
        t_right = temperature[idx]
        t_left = temperature[idx]
        t_up = temperature[idx]
        t_down = temperature[idx]
        if i < width - 1 and solid[idx + 1] == 0:
            face = u[idx]
            if face >= 0.0:
                q_right = _face_flux(face, old_h[idx], bed[idx], bed[idx + 1])
            else:
                q_right = _face_flux(face, old_h[idx + 1], bed[idx + 1], bed[idx])
                t_right = temperature[idx + 1]
        if i > 0 and solid[idx - 1] == 0:
            face = u[idx - 1]
            if face >= 0.0:
                q_left = _face_flux(face, old_h[idx - 1], bed[idx - 1], bed[idx])
                t_left = temperature[idx - 1]
            else:
                q_left = _face_flux(face, old_h[idx], bed[idx], bed[idx - 1])
        if j < height - 1 and solid[idx + width] == 0:
            face = v[idx]
            if face >= 0.0:
                q_up = _face_flux(face, old_h[idx], bed[idx], bed[idx + width])
            else:
                q_up = _face_flux(face, old_h[idx + width], bed[idx + width], bed[idx])
                t_up = temperature[idx + width]
        if j > 0 and solid[idx - width] == 0:
            face = v[idx - width]
            if face >= 0.0:
                q_down = _face_flux(face, old_h[idx - width], bed[idx - width], bed[idx])
                t_down = temperature[idx - width]
            else:
                q_down = _face_flux(face, old_h[idx], bed[idx], bed[idx - width])
        energy_old = old_h[idx] * temperature[idx]
        energy_new = energy_old - dt * ((q_right * t_right - q_left * t_left)
                                        + (q_up * t_up - q_down * t_down)) / dx
        hn = new_h[idx]
        if hn > dry:
            next_temperature[idx] = wp.max(ambient_k, energy_new / hn)
        else:
            next_temperature[idx] = temperature[idx]


    @wp.kernel
    def _apply_lava_vents(h: wp.array(dtype=float), u: wp.array(dtype=float),
                          v: wp.array(dtype=float), temperature: wp.array(dtype=float),
                          solid: wp.array(dtype=wp.int32),
                          centres: wp.array(dtype=wp.vec3),
                          radii: wp.array(dtype=float),
                          discharges: wp.array(dtype=float),
                          temps: wp.array(dtype=float),
                          count: int, width: int, height: int, dx: float, dt: float,
                          dry: float, max_velocity: float,
                          area: float, added: wp.array(dtype=float)):
        """A vent: a prescribed-discharge point source, not a level-held one.

        `_apply_point_sources` (`SOURCE`) computes `target = level - bed`, which
        would starve a lava vent as solidified flow raises the bed around it --
        the vent burying itself, at which point "how much has erupted" stops
        being a control at all. A prescribed Q cannot be starved that way, so
        this is modelled on `_apply_drains` instead (a sink run backwards):
        volume is added over a smooth disc profile, and the radial velocity
        needed to carry exactly that volume outward past the disc's rim is
        derived from continuity -- `strength` is picked so that the enclosed
        flow at r = radius equals the requested Q, the same relation
        `_apply_drains` already proved for a sink. Without an imposed outward
        velocity the fresh volume would sit as a mound with no momentum, ring,
        and read as a reservoir rather than an eruption -- the same failure
        `_apply_river_inlet`'s own docstring warns about.

        Writes `temperature` unconditionally inside the disc, every call: a
        vent that wrote `h` and not `T` would erupt lava at ambient temperature
        forever, which is the 0.3989 bug from v0.11.0 reproduced exactly
        (docs/07_river_plan.md) -- a source must fill every field it creates.
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
            discharge = discharges[n]
            if radius <= 0.0 or discharge <= 0.0:
                continue
            centre = centres[n]
            dxc = x - centre[0]
            dzc = z - centre[2]
            r = wp.sqrt(dxc * dxc + dzc * dzc)
            if r > radius:
                continue
            ratio = r / radius
            falloff = 1.0 - ratio * ratio
            # strength (m/s of depth, disc-wide) such that the enclosed flow at
            # the rim (r=radius) equals the requested Q: Q = strength*pi*R^2*0.5
            strength = 2.0 * discharge / (3.14159265 * radius * radius)
            gain = strength * falloff * dt
            wp.atomic_add(added, 0, gain * area)
            h[idx] = h[idx] + gain
            temperature[idx] = temps[n]
            # Deliberately NOT Froude-capped the way `_apply_river_inlet` caps
            # its edge velocity: that cap desynchronises mass from momentum here
            # -- `gain` above already added the full requested volume this
            # substep regardless of the cap, so capping only the velocity that
            # is supposed to carry it back out leaves mass arriving faster than
            # it can leave, and the vent pools instead of flowing. Measured
            # directly: with the cap in place a 20 m3/s vent reached 15.6 m
            # deep and 0.10 m/s in 30 s (a lake), not a flow. `max_velocity`
            # below is the same hard ceiling every other kernel in this file
            # already clamps to, which is what actually bounds a near-dry disc.
            # v0.15.0: evaluated at each face's own position, as in `_apply_drains`
            if i < width - 1:
                face_u = _vent_velocity(dxc + 0.5 * dx, dzc, radius, strength, h[idx], dry)
                u[idx] = wp.clamp(face_u[0], -max_velocity, max_velocity)
            if j < height - 1:
                face_v = _vent_velocity(dxc, dzc + 0.5 * dx, radius, strength, h[idx], dry)
                v[idx] = wp.clamp(face_v[1], -max_velocity, max_velocity)


    @wp.kernel
    def _lava_manning(temperature: wp.array(dtype=float), h: wp.array(dtype=float),
                      solid: wp.array(dtype=wp.int32), manning: wp.array(dtype=float),
                      min_n: float, max_n: float, solidus_k: float, erupt_k: float,
                      dry: float):
        """mu(T), mapped onto the existing Manning-friction slot in `_velocity_step`.

        Linear between the two calibration points the plan asks for -- fluid at
        eruption temperature (`min_n`), almost stopped at the solidus (`max_n`)
        -- rather than a real viscosity law, for the same reason the plan gives
        for choosing (a) over (b) in docs/08_volcano_plan.md: it reuses the
        proven SWE solver and is honest about what it is not.

        Dry or solid cells are left at `min_n`, not `max_n`: `_velocity_step`
        already zeroes their velocity outright, so the value is inert there, but
        a cell that is about to be wetted by an advancing front should start
        fluid rather than pre-frozen by whatever this array last held.
        """
        idx = wp.tid()
        if solid[idx] != 0 or h[idx] <= dry:
            manning[idx] = min_n
            return
        span = wp.max(erupt_k - solidus_k, 1.0)
        frac = wp.clamp((temperature[idx] - solidus_k) / span, 0.0, 1.0)
        manning[idx] = max_n + frac * (min_n - max_n)


    @wp.kernel
    def _cool_lava(temperature: wp.array(dtype=float),
                   next_temperature: wp.array(dtype=float),
                   h: wp.array(dtype=float), solid: wp.array(dtype=wp.int32),
                   dt: float, k0: float, ambient_k: float, min_depth: float,
                   dry: float):
        """Radiative cooling of a well-mixed column: dT/dt = -(k0/h)*T^4.

        Closed-form (not integrated step by step), so it is unconditionally
        stable and cannot overshoot past ambient the way an explicit subtraction
        would -- see docs/08_volcano_plan.md's trap 2, and compare with how
        `_velocity_step` treats friction semi-implicitly for the same reason.
        Ambient's own T^4 is dropped from the balance (under 0.2% of erupting
        T^4, and this is a bulk-column model already, not a calibrated one);
        the `wp.max(ambient_k, ...)` floor below is what puts back the one
        physical fact that omission would otherwise lose -- radiative cooling
        approaches ambient asymptotically, it does not run past it.

        `min_depth` is `LAVA_COOLING_MIN_DEPTH`, a physics knob and not a
        divide-by-zero guard: `k0/h` is the entire "thin cools faster than
        thick" mechanism (h in the denominator), so this floor alone decides how
        fast the leading edge of a flow can freeze relative to its interior.
        """
        idx = wp.tid()
        if solid[idx] != 0 or h[idx] <= dry:
            next_temperature[idx] = temperature[idx]
            return
        hh = wp.max(h[idx], min_depth)
        t0 = temperature[idx]
        growth = 1.0 + 3.0 * k0 * t0 * t0 * t0 * dt / hh
        cooled = t0 / wp.pow(growth, 0.3333333)
        next_temperature[idx] = wp.max(ambient_k, cooled)


    @wp.kernel
    def _solidify_lava(h: wp.array(dtype=float), temperature: wp.array(dtype=float),
                       bed_terrain: wp.array(dtype=float),
                       solid: wp.array(dtype=wp.int32), solidus_k: float,
                       dry: float, area: float, solidified: wp.array(dtype=float)):
        """Below the solidus, lava stops being flow and becomes ground.

        Literal item 5 of the plan: `bed += h; h = 0`, unconditional and
        irreversible in one substep, with NO limiter pre-applied -- the
        v0.11.0 lesson (docs/07_river_plan.md) was that a clamp shipped before
        anything measured what it binds against gets read back as physics.
        `solidified` accumulates the volume this removes from the fluid, so the
        volume ledger (`diagnostics()`) can account for where it went instead
        of reporting it as an unexplained conservation error.

        Freezing into `bed_terrain` and not the rock-dome mask keeps solidified
        lava overflowable by the next hot pulse, for the same reason a ROCK is
        never rasterized as a solid wall: the mask is an infinitely tall wall,
        right for a house and wrong for terrain, and a lava levee is terrain.
        """
        idx = wp.tid()
        if solid[idx] != 0 or h[idx] <= dry:
            return
        if temperature[idx] < solidus_k:
            amount = h[idx]
            bed_terrain[idx] = bed_terrain[idx] + amount
            wp.atomic_add(solidified, 0, amount * area)
            h[idx] = 0.0


    @wp.kernel
    def _reduce_lava_diagnostics(temperature: wp.array(dtype=float),
                                 h: wp.array(dtype=float),
                                 solid: wp.array(dtype=wp.int32), dry: float,
                                 stats: wp.array(dtype=float)):
        idx = wp.tid()
        if solid[idx] == 0 and h[idx] > dry:
            wp.atomic_max(stats, 6, temperature[idx])


    @wp.kernel
    def _advect_flow_tracers(particles: wp.array(dtype=wp.vec3),
                             h: wp.array(dtype=float), u: wp.array(dtype=float),
                             v: wp.array(dtype=float), bed: wp.array(dtype=float),
                             solid: wp.array(dtype=wp.int32), width: int, height: int,
                             source_columns: int, dx: float, dt: float, dry: float,
                             lava_active: int, vent_x: float, vent_z: float):
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
                # Carried off the map: the tracer is gone, and the respawn
                # rules below pick it up next frame. This branch used to be
                # missing, so a tracer whose next step crossed the outer ring
                # was never moved again and never counted as dead -- every one
                # that reached the river outlet stopped in the last column, and
                # 36 000 of them drew a white dotted line across the river at
                # the map edge (v0.16.0, once the river visibly ran on past it).
                particles[n] = wp.vec3(float(width) * dx, -100.0, p.z)
        elif lava_active != 0:
            # VolcanoLab v0.14.0: a dead tracer respawns at the vent's own
            # disc, not the west edge -- the pre-existing respawn rule below
            # assumes water always enters from FLUID_SOURCE_COLUMNS, which a
            # volcano world never wets at all. Without this every one of the
            # 36 000 tracers sat forever at the dry west edge (y = -100,
            # literally never drawn) in ANY lava world, found only by reading
            # the actual tracer buffer back client-side -- the screen looked
            # fine because a SEPARATE system (the velocity-driven spray
            # points) happened to still show something moving near the vent.
            angle = float(n % 360) * 0.017453292519943295
            radius = 1.0 + float((n // 360) % 10) * 0.3
            sx = vent_x + radius * wp.cos(angle)
            sz = vent_z + radius * wp.sin(angle)
            si = int(wp.floor(sx / dx + float(width - 1) * 0.5 + 0.5))
            sj = int(wp.floor(sz / dx + float(height - 1) * 0.5 + 0.5))
            sy = -100.0
            if si > 0 and si < width - 1 and sj > 0 and sj < height - 1:
                spawn_idx = sj * width + si
                if solid[spawn_idx] == 0 and h[spawn_idx] > dry:
                    sy = bed[spawn_idx] + h[spawn_idx] + 0.08
            particles[n] = wp.vec3(sx, sy, sz)
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


    # One packed array instead of five: every entry is read back to the host
    # every frame, and each readback is a device sync. Packing them means the
    # whole diagnostic set costs one sync, which is what makes it affordable to
    # measure at both ends of a frame -- the CFL decision needs the state the
    # substeps will start from, and a client reading diagnostics needs the state
    # they ended at, and those are not the same state.
    STAT_DEPTH, STAT_SPEED, STAT_WAVE = 0, 1, 2
    STAT_VOLUME, STAT_WET, STAT_INLET = 3, 4, 5
    # v0.13.0: the hottest wet cell this frame, in Kelvin -- gated on lava being
    # enabled at all (see `_measure`), same pattern as STAT_INLET being gated on
    # `_inlet_enabled`.
    STAT_LAVA_TEMP_MAX = 6
    STAT_COUNT = 7

    @wp.kernel
    def _clear_diagnostics(stats: wp.array(dtype=float)):
        for k in range(7):
            stats[k] = 0.0


    @wp.kernel
    def _measure_inlet_flux(h: wp.array(dtype=float), u: wp.array(dtype=float),
                            bed: wp.array(dtype=float),
                            solid: wp.array(dtype=wp.int32),
                            inlet_q: wp.array(dtype=float),
                            width: int, dx: float,
                            stats: wp.array(dtype=float)):
        """Discharge crossing the first face inside the domain, in m3/s.

        Measured one cell downstream of the boundary, at the face between
        columns 1 and 2, and computed the way `_depth_step` transports it. The
        boundary's own face is prescribed, so measuring there would return the
        requested Q whatever the solver did with it -- an instrument that cannot
        disagree with the setting it is checking is not an instrument. This one
        can: it reads what the river is actually carrying just inside the map,
        which is the question the number is asked for.
        """
        idx = wp.tid()
        i = idx % width
        j = idx // width
        if i != 1 or solid[idx] != 0 or inlet_q[j] <= 0.0 or solid[idx + 1] != 0:
            return
        face = u[idx]
        if face >= 0.0:
            q = _face_flux(face, h[idx], bed[idx], bed[idx + 1])
        else:
            q = _face_flux(face, h[idx + 1], bed[idx + 1], bed[idx])
        wp.atomic_add(stats, 5, q * dx)


    @wp.kernel
    def _reduce_diagnostics(h: wp.array(dtype=float), u: wp.array(dtype=float),
                            v: wp.array(dtype=float), uc: wp.array(dtype=float),
                            vc: wp.array(dtype=float), solid: wp.array(dtype=wp.int32),
                            gravity: float, area: float, dry: float,
                            stats: wp.array(dtype=float)):
        idx = wp.tid()
        if solid[idx] == 0:
            depth = wp.max(0.0, h[idx])
            # speed as a gauge would read it (cell centre); the CFL wave speed
            # from this cell's own faces, which are what the time step must
            # keep from crossing more than a cell
            speed = wp.sqrt(uc[idx] * uc[idx] + vc[idx] * vc[idx])
            wave = wp.max(wp.abs(u[idx]), wp.abs(v[idx])) + wp.sqrt(gravity * depth)
            wp.atomic_max(stats, 0, depth)
            wp.atomic_max(stats, 1, speed)
            wp.atomic_max(stats, 2, wave)
            wp.atomic_add(stats, 3, depth * area)
            if depth > dry:
                wp.atomic_add(stats, 4, 1.0)


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
SOLID_OBSTACLE_TYPES = frozenset({"HOUSE", "BRIDGE", "BUILDING"})

# A BRIDGE is not a wall: water flows UNDER it. Only its piers obstruct, and
# only up to the deck. The deck itself is drawn and is what the water is
# compared against for the flooded-deck event, but it never enters the mask --
# a bridge rasterized as a solid block would dam the river it spans, which is
# the opposite of what a bridge does.
BRIDGE_PIER_TYPES = frozenset({"BRIDGE"})

# Bodies that are not walls but do change the bed they stand on. A ROAD is not
# solid -- water runs over a street, it does not stop at one -- so it never
# enters the obstacle mask; what it changes is roughness. Kept as a mapping
# rather than a type check so the next paved thing is one line, not a branch.
SURFACE_MANNING_TYPES = {"ROAD": config.PAVEMENT_MANNING_N}

# A ROCK of scale 1 raises the bed over this radius, in metres. Matches the
# radius of the ROCK mesh in frontend/src/world/ObjectFactory.ts so what the
# user sees is what the water feels.
ROCK_BASE_RADIUS_M = 1.5

# Length of a BRIDGE of scale 1, across the channel. Matches the deck mesh in
# frontend/src/world/ObjectFactory.ts so the piers the water feels line up with
# the piers the user sees.
BRIDGE_SPAN_M = 24.0


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
    def get_lava_temperature_field(self) -> np.ndarray: ...
    def sample_lava_contact(self, positions: np.ndarray) -> tuple: ...
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
    def get_lava_temperature_field(self) -> np.ndarray: return np.zeros(0, dtype=np.float32)
    def sample_lava_contact(self, positions: np.ndarray) -> tuple:
        n = len(positions)
        return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
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
        # v0.15.0: `_u`/`_v` are FACE velocities (see `_velocity_step`); these
        # are the cell-centred ones every "velocity at a cell" reader uses
        self._uc = self._vc = None
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
        # v0.12.0: outlet rows and the local discharge inlet. The defaults are
        # the 0.8.0 behaviour exactly -- outlet across the whole east edge, no
        # river inlet -- so every world that predates this is unaffected.
        self._outflow_rows = (0, 0)
        # v0.16.0: what lies beyond the open edge (config.OUTLET_KINDS) and, for
        # a "river", the bed slope per row it continues at
        self._outlet_kind = "overfall"
        self._outlet_slope = None
        self._outlet_slope_host = np.zeros(0, dtype=np.float32)
        self._inlet_enabled = False
        self._inlet_q = self._inlet_normal_depth = None
        self._inlet_q_host = np.zeros(0, dtype=np.float32)
        self._inlet_request = {"discharge_m3s": 0.0, "width_m": 0.0, "centre_z": 0.0}
        self._added_m3 = 0.0
        self._removed_m3 = 0.0
        self._sediment_out_m3 = 0.0
        self._sediment_in_m3 = 0.0
        self._volume_at_start = 0.0
        self._source_count = 0
        self._drain_count = 0
        self._src_centres = self._src_radii = self._src_levels = None
        self._drain_centres = self._drain_radii = None
        self._drain_strengths = self._drain_circulation = None
        self._drain_samples = None
        self._drain_removed_each = None
        # v0.17.0 storm sewer (set_sewer)
        self._inlet_count = self._outfall_count = 0
        self._inlet_centres = self._inlet_radii = self._inlet_strengths = None
        self._inlet_targets = self._inlet_circulation = self._inlet_samples = None
        self._inlet_step = self._inlet_frame = None
        self._outfall_centres = self._outfall_radii = None
        self._outfall_volume = self._outfall_cells = None
        self._inlet_flow_m3s: list = []
        self._inlet_capacity_m3s: list = []
        self._diag_sewer_in = self._diag_sewer_out = None
        self._section_faces = 0
        self._section_ids: list = []
        self._section_key = None
        self._section_readings: list = []
        self._inlet_demand_frame = None
        self._inlet_demand_m3s = []
        self._sewer_in_m3 = self._sewer_out_m3 = 0.0
        # VolcanoLab (v0.13.0). Lava mode is DERIVED from whether any VENT is
        # placed, the same way a placed SOURCE takes over from the edge inflow
        # entirely (v0.8.0) -- no separate toggle, so there is exactly one
        # answer to "is this world simulating lava". The consequence, named
        # here rather than left implicit: dropping a VENT onto a river world
        # turns all of its water into lava, because water/lava coexistence is
        # explicitly out of scope until v0.14.0 (docs/08_volcano_plan.md).
        self._lava_enabled = False
        self._temperature = self._next_temperature = None
        self._manning = None
        self._manning_host = np.zeros(0, dtype=np.float32)
        self._vent_count = 0
        self._vent_centres = self._vent_radii = None
        self._vent_discharges = self._vent_temps = None
        self._tracer_vent_x = self._tracer_vent_z = 0.0
        self._solidified_m3 = 0.0
        self._diag_solidified = None
        self._level = 0.5
        self._source_enabled = True
        # RainLab-1 (docs/14_rain_plan.md)
        self._edge_inflow_enabled = True
        self._rain_mm_h = 0.0
        self._rain_pending_m = 0.0
        self._rain_added_m3 = 0.0
        self._diag_rain = None
        self._tsunami_enabled = False
        self._tsunami_amplitude = 0.0
        self._tsunami_period = 60.0
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
        # RainLab-1: read here as well as live each tick, because the edge
        # prefill below has to know before the first substep whether the west
        # edge holds a level at all
        self._edge_inflow_enabled = bool(getattr(world.water, "edge_inflow_enabled", True))
        self._rain_mm_h = float(getattr(world.water, "rain_intensity_mm_h", 0.0))
        self._rain_pending_m = 0.0
        self._rain_added_m3 = 0.0
        # TsunamiLab wavemaker. Off unless the world asks for it, so every
        # non-tsunami world behaves exactly as it did before v0.14.1.
        self._tsunami_enabled = bool(getattr(world.water, "tsunami_enabled", False))
        self._tsunami_amplitude = float(getattr(world.water, "tsunami_amplitude_m", 0.0))
        self._tsunami_period = max(1.0e-3, float(
            getattr(world.water, "tsunami_period_s", 60.0)))
        # v0.14.3: a wave TRAIN, not one N-wave -- see WaterState.tsunami_wave_count.
        self._tsunami_wave_count = max(1, min(3, int(
            getattr(world.water, "tsunami_wave_count", 1))))
        spacing = float(getattr(world.water, "tsunami_wave_spacing_s", 0.0))
        # Auto default must clear 2*TSUNAMI_ACTIVE_WINDOW_PERIODS periods, or
        # consecutive pulses' active windows overlap and the gap the outlet
        # needs to drain through (see config.TSUNAMI_ACTIVE_WINDOW_PERIODS)
        # never opens. +1 period on top of that minimum leaves a real gap
        # rather than exactly touching it.
        auto_spacing = (2.0 * config.TSUNAMI_ACTIVE_WINDOW_PERIODS + 1.0) * self._tsunami_period
        self._tsunami_wave_spacing = spacing if spacing > 0.0 else auto_spacing
        self._tsunami_wave2_scale = float(getattr(world.water, "tsunami_wave2_scale", 1.3))
        self._tsunami_wave3_scale = float(getattr(world.water, "tsunami_wave3_scale", 0.6))
        self._width, self._height = world.terrain.width + 1, world.terrain.height + 1
        self._count = self._width * self._height
        bed_grid = np.ascontiguousarray(world.terrain.heights, dtype=np.float32)
        self._bed_host = bed_grid.ravel().copy()
        self._measure_outlet_slope(bed_grid, float(world.terrain.cell_size))
        depth_grid = np.zeros_like(bed_grid, dtype=np.float32)
        source_columns = min(config.FLUID_SOURCE_COLUMNS, self._width)
        water = getattr(world, "water", None)
        inlet_wanted = bool(getattr(water, "inlet_enabled", False))
        tsunami_wanted = bool(getattr(water, "tsunami_enabled", False))
        if tsunami_wanted:
            # A CALM SEA, everywhere the bed sits below `self._level` -- the
            # ocean existing before the wave reaches it is a precondition, not
            # the effect. No wave is seeded here at all: it arrives through the
            # east edge, driven per substep by `_apply_tsunami_edge`. v0.14.0
            # seeded an N-wave into the grid instead, and
            # docs/probe_tsunami_v2.py measured what that produced -- the sea
            # retreated 1.4 m and the land flooded 4.6 m, because a wave long
            # enough to draw the sea back cannot fit in the domain alongside
            # the shore and the land behind it. This branch still supersedes
            # the source-column prefill below, for the same reason as before: a
            # tsunami world's west edge is dry land (terrain_gen.coastline).
            depth_grid = np.maximum(
                self._level - bed_grid.astype(np.float64), 0.0).astype(np.float32)
        elif not inlet_wanted and self._edge_inflow_enabled:
            # A river inlet owns the west edge; pre-filling it from the level
            # control as well would put a wall of water across the floodplain at
            # t = 0 and then leave it to drain, which is not a river starting.
            depth_grid[:, :source_columns] = np.maximum(
                self._level - bed_grid[:, :source_columns], 0.0)
        depth = depth_grid.ravel()
        zeros = np.zeros(self._count, dtype=np.float32)
        face_u = face_v = zeros
        # v0.18.1: a settled world starts with its river already flowing --
        # depth AND face velocities, so it is at equilibrium from the first
        # substep instead of accelerating out of still water. Only when the
        # grid matches; a world whose terrain was resized since starts dry.
        stored = getattr(water, "initial_flow", None) if water is not None else None
        if stored and not tsunami_wanted and list(stored.get("shape", [])) == [self._height, self._width]:
            import base64
            decode = lambda key: np.frombuffer(base64.b64decode(stored[key]), dtype="<f4")
            fields = [decode(key) for key in ("h", "u", "v")]
            if all(f.size == self._count for f in fields):
                depth, face_u, face_v = (f.astype(np.float32) for f in fields)
        self._obstacle_host = np.zeros(self._count, dtype=np.int32)
        self._h = wp.array(depth, dtype=float, device=self.device)
        self._u = wp.array(face_u, dtype=float, device=self.device)
        self._v = wp.array(face_v, dtype=float, device=self.device)
        self._next_h = wp.empty(self._count, dtype=float, device=self.device)
        self._next_u = wp.empty(self._count, dtype=float, device=self.device)
        self._next_v = wp.empty(self._count, dtype=float, device=self.device)
        self._uc = wp.array(zeros, dtype=float, device=self.device)
        self._vc = wp.array(zeros, dtype=float, device=self.device)
        self._bed_terrain = wp.array(self._bed_host, dtype=float, device=self.device)
        self._bed_offset_host = np.zeros(self._count, dtype=np.float32)
        self._bed_offset = wp.array(self._bed_offset_host, dtype=float, device=self.device)
        self._bed = wp.array(self._bed_host.copy(), dtype=float, device=self.device)
        self._sediment = wp.zeros(self._count, dtype=float, device=self.device)
        self._next_sediment = wp.empty(self._count, dtype=float, device=self.device)
        self._obstacles = wp.array(self._obstacle_host, dtype=wp.int32, device=self.device)
        self._diag_stats = wp.zeros(STAT_COUNT, dtype=float, device=self.device)
        self._diag_added = wp.zeros(1, dtype=float, device=self.device)
        self._diag_removed = wp.zeros(1, dtype=float, device=self.device)
        self._diag_sediment_out = wp.zeros(1, dtype=float, device=self.device)
        self._diag_sediment_in = wp.zeros(1, dtype=float, device=self.device)
        self._diag_solidified = wp.zeros(1, dtype=float, device=self.device)
        self._diag_rain = wp.zeros(1, dtype=float, device=self.device)
        self._diag_sewer_in = wp.zeros(1, dtype=float, device=self.device)
        self._diag_sewer_out = wp.zeros(1, dtype=float, device=self.device)
        self._sewer_in_m3 = self._sewer_out_m3 = 0.0
        self._inlet_count = self._outfall_count = 0
        self._inlet_frame = None
        self._inlet_flow_m3s = []
        self._inlet_demand_frame = None
        self._inlet_demand_m3s = []
        self._inlet_capacity_m3s = []
        self._solidified_m3 = 0.0
        # VolcanoLab (v0.13.0). Manning friction defaults to the constant every
        # non-lava world already ran with, filled ONCE here from
        # `config.FLUID_MANNING_N` -- `advance()` never re-reads that module
        # constant itself any more, so a probe or test that patches it must do
        # so before `initialize()`, same as it already had to for every other
        # value this method bakes into a GPU array at start-up.
        # A tsunami world is a sea bed, not a river channel -- see config's
        # SEABED_MANNING_N for the measured difference this makes to how far the
        # sea can withdraw. Chosen by what the surface IS, not by which number
        # looked better.
        self._base_manning = (config.SEABED_MANNING_N if tsunami_wanted
                              else config.FLUID_MANNING_N)
        self._manning_host = np.full(self._count, self._base_manning, dtype=np.float32)
        self._manning = wp.array(self._manning_host.copy(),
                                 dtype=float, device=self.device)
        self._temperature = wp.array(
            np.full(self._count, config.LAVA_AMBIENT_TEMP_C + 273.15, dtype=np.float32),
            dtype=float, device=self.device)
        self._next_temperature = wp.empty(self._count, dtype=float, device=self.device)
        self._lava_enabled = False
        self._vent_count = 0
        self._tracer_vent_x = self._tracer_vent_z = 0.0
        self._inlet_q_host = np.zeros(self._height, dtype=np.float32)
        self._inlet_q = wp.zeros(self._height, dtype=float, device=self.device)
        self._inlet_normal_depth = wp.zeros(self._height, dtype=float,
                                            device=self.device)
        self._inlet_enabled = False
        self._inlet_request = {"discharge_m3s": 0.0, "width_m": 0.0, "centre_z": 0.0}
        self._outflow_rows = (0, self._height - 1)
        self._added_m3 = 0.0
        self._removed_m3 = 0.0
        self._sediment_out_m3 = 0.0
        self._sediment_in_m3 = 0.0
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
        if inlet_wanted:
            # The boundary is part of the world, so it is restored with the
            # world rather than waiting for the first tick's live sync.
            self.set_river_inlet(True, float(water.inlet_centre_z),
                                 float(water.inlet_width_m),
                                 float(water.inlet_discharge_m3s))
        if water is not None:
            # Erosion and the open edge are part of the world too, and for the
            # same reason as the inlet above they are restored here rather than
            # left to the first tick's live sync. They used to be left: the
            # solver's own defaults (erosion ON, outlet CLOSED) are the opposite
            # of a fresh WaterState's, and `_apply_water_settings` only runs
            # inside `step()`. Nothing moved at IDLE so nothing behaved wrongly,
            # but `diagnostics()` reported those defaults and the UI faithfully
            # drew both checkboxes inverted until the user pressed PLAY --
            # telling them a loaded scenario would erode its bed when it would
            # not, and that its river could not leave the map when it could.
            self.set_erosion(bool(getattr(water, "erosion_enabled", False)))
            # set_outflow forces this shut under the tsunami wavemaker; see
            # its docstring for why that belongs there and not here.
            self.set_outflow(
                config.FLUID_OUTFLOW_COLUMNS
                if bool(getattr(water, "outflow_enabled", True)) else 0,
                float(getattr(water, "outlet_centre_z", 0.0)),
                float(getattr(water, "outlet_width_m", 0.0)),
                str(getattr(water, "outlet_kind", "overfall")))
        self._recombine_bed()
        self._measure()
        self._volume_at_start = float(self._diag["volume_m3"])

    def _build_obstacle_mask(self, terrain, obstacles: dict) -> np.ndarray:
        mask = np.zeros(self._count, dtype=np.int32)
        positions = obstacles.get("positions", [])
        rotations = obstacles.get("rotations", [])
        scales = obstacles.get("scales", [])
        types = obstacles.get("types", [])
        bed_heights = obstacles.get("bed_heights", [])
        piers = obstacles.get("pier_counts", [])
        pier_radii = obstacles.get("pier_radii", [])
        half_extents = obstacles.get("half_extents", [])
        for n, position in enumerate(positions):
            if n >= len(types) or not self._is_solid(types[n], bed_heights, n):
                continue
            scale = scales[n] if n < len(scales) else [1.0, 1.0, 1.0]
            yaw = float(rotations[n][1]) if n < len(rotations) else 0.0
            cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
            if types[n] in BRIDGE_PIER_TYPES:
                self._rasterize_piers(mask, terrain, position, scale,
                                      cos_yaw, sin_yaw,
                                      piers[n] if n < len(piers) else 0.0,
                                      pier_radii[n] if n < len(pier_radii) else 0.0)
                continue
            # Read the real per-type footprint computed by
            # rigid_body.footprint_half_extents (already scale-multiplied)
            # rather than assuming HOUSE's own 2.0 m constant -- that
            # assumption was silently correct only because HOUSE used to be
            # the sole type on this path; BUILDING's footprint varies with
            # floors and would otherwise always rasterize at house size.
            if n < len(half_extents):
                half_x, half_z = float(half_extents[n][0]), float(half_extents[n][1])
            else:
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

    def _build_manning_map(self, terrain, obstacles: dict) -> np.ndarray:
        """Manning's n per cell, decided by what each cell's surface IS.

        Filled with this world's baseline, then overwritten under the footprint
        of every body in SURFACE_MANNING_TYPES. Deliberately separate from the
        obstacle mask: "does the water stop here" and "how rough is the bed
        here" are different questions, and for a street the answers are no and
        much smoother. The footprints are the same half extents the rigid
        system already reports to the solver, so a road rasterizes at the size
        it is drawn.
        """
        base = getattr(self, "_base_manning", config.FLUID_MANNING_N)
        field = np.full(self._count, base, dtype=np.float32)
        positions = obstacles.get("positions", []) if obstacles else []
        if len(positions) == 0:
            return field
        types = obstacles.get("types", [])
        rotations = obstacles.get("rotations", [])
        half_extents = obstacles.get("half_extents", [])
        grid = field.reshape(self._height, self._width)
        yy, xx = np.mgrid[0:self._height, 0:self._width]
        world_x = (xx - terrain.width / 2) * terrain.cell_size
        world_z = (yy - terrain.height / 2) * terrain.cell_size
        for index, position in enumerate(positions):
            surface_n = SURFACE_MANNING_TYPES.get(
                types[index] if index < len(types) else "")
            if surface_n is None or index >= len(half_extents):
                continue
            yaw = float(rotations[index][1]) if index < len(rotations) else 0.0
            cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
            dx = world_x - float(position[0])
            dz = world_z - float(position[2])
            local_x = cos_yaw * dx + sin_yaw * dz
            local_z = -sin_yaw * dx + cos_yaw * dz
            grid[(np.abs(local_x) <= float(half_extents[index][0]))
                 & (np.abs(local_z) <= float(half_extents[index][1]))] = surface_n
        return field

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

    def _rasterize_piers(self, mask: np.ndarray, terrain, position, scale,
                         cos_yaw: float, sin_yaw: float,
                         pier_count: float, pier_radius: float) -> None:
        """Solid discs for a bridge's piers, spaced evenly across its span.

        Only the piers obstruct. The deck is not rasterized at all, so water
        passes under the bridge, which is the entire point of building one --
        and it gives the real educational payoff: the piers constrict the
        channel, the flow speeds up between them, and debris piles against them.
        """
        count = int(round(pier_count))
        radius = float(pier_radius) * float(scale[0])
        if count <= 0 or radius <= 0.0:
            return
        span = BRIDGE_SPAN_M * float(scale[2])
        centre_x, centre_z = float(position[0]), float(position[2])
        cell = terrain.cell_size
        for index in range(count):
            # evenly spaced along the bridge's local z axis, ends included
            t = 0.5 if count == 1 else index / float(count - 1)
            offset = (t - 0.5) * span
            px = centre_x + sin_yaw * offset
            pz = centre_z + cos_yaw * offset
            gx = px / cell + terrain.width / 2
            gz = pz / cell + terrain.height / 2
            r_cells = max(1.0, radius / cell)
            lo_i = max(1, int(math.floor(gx - r_cells)))
            hi_i = min(self._width - 2, int(math.ceil(gx + r_cells)))
            lo_j = max(1, int(math.floor(gz - r_cells)))
            hi_j = min(self._height - 2, int(math.ceil(gz + r_cells)))
            for row in range(lo_j, hi_j + 1):
                for column in range(lo_i, hi_i + 1):
                    if (column - gx) ** 2 + (row - gz) ** 2 <= r_cells ** 2:
                        mask[row * self._width + column] = 1

    def _remap_obstacles(self, new_mask: np.ndarray) -> None:
        # face velocities, not `_host_fields()`'s centred ones: these are the
        # state the next substep starts from. A face left touching a new wall
        # is closed by `_velocity_step` on that substep regardless.
        h = np.asarray(self._h.numpy(), dtype=np.float32)
        u = np.asarray(self._u.numpy(), dtype=np.float32)
        v = np.asarray(self._v.numpy(), dtype=np.float32)
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
            self._measure_outlet_slope(terrain.heights, float(terrain.cell_size))
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
            new_manning = self._build_manning_map(terrain, obstacles)
            if not np.array_equal(new_manning, self._manning_host):
                self._manning_host = new_manning
                # While lava runs, `_lava_manning` owns this array and rewrites
                # every cell each substep from mu(T); the surface map is kept on
                # the host and handed back when lava stops. The two never apply
                # at once, which costs nothing real: a volcano has no asphalt.
                if not self._lava_enabled:
                    self._manning.assign(new_manning)
            self._seen_obstacle_revision = obstacle_revision
            self.obstacle_gpu_uploads += 1
            changed = True
        if changed:
            self._recombine_bed()
            self._measure()

    def _measure_outlet_slope(self, heights, cell: float) -> None:
        """Per row, the bed slope falling toward the east edge (positive = down).

        A least-squares fit over config.OUTLET_SLOPE_REACH_M of terrain next to
        the edge -- the terrain, not `_bed`, so a rock dome does not tilt the
        river beyond the map. Measured rather than carried from terrain_gen's
        `slope` parameter so a brushed or loaded valley gets its own slope too.
        Refreshed on every terrain revision; erosion during a run does not move
        it, which is deliberate: the river past the map is not being eroded.
        """
        grid = np.asarray(heights, dtype=np.float64).reshape(self._height, self._width)
        reach = max(1, min(self._width - 2,
                           int(round(config.OUTLET_SLOPE_REACH_M / max(cell, 1e-6)))))
        tail = grid[:, self._width - 1 - reach:]
        x = np.arange(reach + 1, dtype=np.float64) * cell
        x -= x.mean()
        slope = -((tail - tail.mean(axis=1, keepdims=True)) @ x) / float(x @ x)
        self._outlet_slope_host = slope.astype(np.float32)
        self._outlet_slope = wp.array(self._outlet_slope_host, dtype=float,
                                      device=self.device)

    def set_outflow(self, columns: int, centre_z: float = 0.0,
                    width_m: float = 0.0, kind: str = "overfall") -> None:
        """Width, in cells, of the transmissive outlet on the east edge.

        Zero restores the fully closed domain of 0.7.0 -- which is what the
        volume-conservation test wants, and it says so in its own name.

        v0.12.0: `width_m` narrows the outlet to a band of rows centred on
        `centre_z`, so a valley drains through its channel instead of through
        its floodplain. Zero (the default) keeps the whole edge open, which is
        what every world before this expects.

        v0.14.1: the tsunami wavemaker OWNS that edge, so the outlet is forced
        shut here rather than at the call sites. There are two of them --
        `set_boundaries` at load, and `SimulationManager._step_once` re-reading
        the toggle live every tick so it can be changed while RUNNING -- and
        the live one silently undid a fix applied only to the other. Measured
        cost of the outlet being open under the wavemaker: the wave floods 0 m
        inland instead of 258 m. Not because `_apply_outflow` drains it (that
        kernel is already skipped) but because `_velocity_step` reads the same
        column count and makes the edge transmissive for VELOCITY, so the
        arriving wave runs straight back out to sea.
        """
        if getattr(self, "_tsunami_enabled", False):
            columns = 0
            # between pulses the edge opens as an outlet onto the SEA, which a
            # "river" outlet on the sloping sea bed would drain
            kind = "overfall"
        self._outlet_kind = kind if kind in config.OUTLET_KINDS else "overfall"
        self._outflow_columns = max(0, int(columns))
        if width_m <= 0.0 or self._height <= 0:
            self._outflow_rows = (0, max(0, self._height - 1))
            return
        self._outflow_rows = self._edge_band(centre_z, width_m)

    def _edge_band(self, centre_z: float, width_m: float) -> tuple:
        """Rows covered by a band `width_m` wide centred on world z=`centre_z`."""
        cell = float(self._terrain.cell_size)
        centre_row = centre_z / cell + (self._height - 1) * 0.5
        half = max(0.5, width_m * 0.5 / cell)
        lo = int(max(0, math.ceil(centre_row - half)))
        hi = int(min(self._height - 1, math.floor(centre_row + half)))
        if hi < lo:
            lo = hi = int(min(self._height - 1, max(0, round(centre_row))))
        return lo, hi

    def set_river_inlet(self, enabled: bool, centre_z: float = 0.0,
                        width_m: float = 12.0, discharge_m3s: float = 0.0) -> None:
        """Prescribed-discharge inlet on a band of the west edge.

        Q is the control; the level the channel settles at is the answer, not a
        second knob -- prescribing both over-determines the boundary. The band
        carries q = Q/W per unit width, and the depth it arrives at is the
        normal depth for that q on the local bed slope, computed here on the
        host from Manning: h = (q*n/sqrt(S))^(3/5).

        That number is a floor, not a prescription: `_apply_river_inlet` takes
        the interior depth when the channel is deeper than normal, so backwater
        from downstream is respected. It exists because pure zero-gradient
        cannot start a dry channel -- h[0] = h[1] = 0 stays 0 for ever.
        """
        enabled = bool(enabled) and discharge_m3s > 0.0 and width_m > 0.0
        request = {"discharge_m3s": float(discharge_m3s), "width_m": float(width_m),
                   "centre_z": float(centre_z)}
        if enabled == self._inlet_enabled and request == self._inlet_request:
            return
        self._inlet_enabled = enabled
        self._inlet_request = request
        if self._inlet_q is None:
            return
        q_row = np.zeros(self._height, dtype=np.float32)
        depth_row = np.zeros(self._height, dtype=np.float32)
        if enabled:
            lo, hi = self._edge_band(centre_z, width_m)
            span_m = (hi - lo + 1) * float(self._terrain.cell_size)
            q = float(discharge_m3s) / max(span_m, 1.0e-6)     # m2/s per unit width
            q_row[lo:hi + 1] = q
            bed = self._bed_host.reshape(self._height, self._width)
            probe = min(self._width - 1, 8)
            slope = ((bed[lo:hi + 1, 0] - bed[lo:hi + 1, probe])
                     / (probe * float(self._terrain.cell_size)))
            # a flat or adverse bed has no normal depth; the floor keeps the
            # inlet finite there instead of demanding an infinite one
            slope = np.maximum(slope, 1.0e-4)
            normal = (q * config.FLUID_MANNING_N / np.sqrt(slope)) ** 0.6
            depth_row[lo:hi + 1] = np.minimum(normal, 10.0)
        self._inlet_q_host = q_row
        self._inlet_q.assign(q_row)
        self._inlet_normal_depth.assign(depth_row)

    def set_rain(self, intensity_mm_h: float) -> None:
        """RainLab-1: uniform rain in mm/h. Read live every tick, so a change
        acts from the next substep; 0 stops it (any undelivered portion below
        RAIN_APPLY_STEP_M stays pending until rain resumes)."""
        self._rain_mm_h = max(0.0, float(intensity_mm_h))

    def set_edge_inflow(self, enabled: bool) -> None:
        """Whether the west edge holds `level`. Off leaves those columns to the
        physics -- a rain-only world needs this, or the edge rewrites them every
        substep and drains whatever rain lands there."""
        self._edge_inflow_enabled = bool(enabled)

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
                self._drain_samples = wp.zeros(len(drains), dtype=float,
                                               device=self.device)
                self._drain_removed_each = wp.zeros(len(drains), dtype=float,
                                                    device=self.device)

    def _sink_weight(self, centre, radius: float, dx: float) -> float:
        """Sum of `_apply_drains`' falloff (1 - r^2/R^2) over a disc's open
        cells, times the cell area: the m2 a strength of 1 m/s drains."""
        width, height = self._width, self._height
        ci = centre[0] / dx + (width - 1) * 0.5
        cj = centre[2] / dx + (height - 1) * 0.5
        span = int(math.ceil(radius / dx)) + 1
        i0, i1 = max(0, int(ci) - span), min(width - 1, int(ci) + span)
        j0, j1 = max(0, int(cj) - span), min(height - 1, int(cj) + span)
        if i1 < i0 or j1 < j0:
            return 0.0
        ii, jj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1))
        x = (ii - (width - 1) * 0.5) * dx
        z = (jj - (height - 1) * 0.5) * dx
        r = np.hypot(x - centre[0], z - centre[2])
        inside = r <= radius
        if self._obstacle_host.size:
            inside &= self._obstacle_host.reshape(height, width)[jj, ii] == 0
        return float(np.sum((1.0 - (r / radius) ** 2)[inside])) * dx * dx

    def set_sections(self, sections: list, key=None) -> None:
        """v0.18.0 gauging lines: `sections` is (id, (cell_a, cell_b, axis,
        sign)) per line, from `sections.faces_for`. Re-uploaded only when `key`
        changes, so a line read every tick costs one kernel launch, not an
        upload. Readings restart from zero when the lines change."""
        if key is not None and key == self._section_key and self._h is not None:
            return
        self._section_key = key
        self._section_ids = [sid for sid, _faces in sections]
        count = len(sections)
        self._section_readings = []
        self._section_cumulative = [0.0] * count
        parts = [faces for _sid, faces in sections]
        total = sum(len(f[0]) for f in parts)
        self._section_faces = int(total)
        if not total:
            return
        owner = np.concatenate([np.full(len(f[0]), n, np.int32) for n, f in enumerate(parts)])
        cat = lambda k, dtype: np.concatenate([f[k] for f in parts]).astype(dtype)
        self._section_a = wp.array(cat(0, np.int32), dtype=wp.int32, device=self.device)
        self._section_b = wp.array(cat(1, np.int32), dtype=wp.int32, device=self.device)
        self._section_axis = wp.array(cat(2, np.int32), dtype=wp.int32, device=self.device)
        self._section_sign = wp.array(cat(3, np.float32), dtype=float, device=self.device)
        self._section_owner = wp.array(owner, dtype=wp.int32, device=self.device)
        self._section_volume = wp.zeros(count, dtype=float, device=self.device)
        self._section_area = wp.zeros(count, dtype=float, device=self.device)
        self._section_wet = wp.zeros(count, dtype=float, device=self.device)
        self._section_level = wp.zeros(count, dtype=float, device=self.device)

    def capture_flow(self) -> dict:
        """v0.18.1: the water on the map now, in WaterState.initial_flow's form."""
        import base64
        encode = lambda array: base64.b64encode(
            np.asarray(array.numpy(), dtype="<f4").tobytes()).decode("ascii")
        return {"shape": [self._height, self._width],
                "h": encode(self._h), "u": encode(self._u), "v": encode(self._v)}

    def section_readings(self) -> list:
        """Per line, over the last frame: (id, flow m3/s, area m2, wetted width
        m, mean surface level m or None, cumulative m3 since the lines were
        set)."""
        return list(self._section_readings)

    def set_sewer(self, inlets: list, outfalls: list, measure_demand: bool = False) -> None:
        """v0.17.0 storm sewer, from backend/app/sewer.py's `resolve`.

        `inlets` is (centre_xyz, radius_m, capacity_m3s, outfall_index[,
        path_capacity_m3s]) and `outfalls` is (centre_xyz, radius_m), read live
        every tick like SOURCE/DRAIN so any of them can be moved while RUNNING.
        `measure_demand` (v0.18.0, only when grates share a pipe) also books
        per frame what each grate would take at its path capacity, into
        `sewer_inlet_demand_m3s`.

        An inlet's sink strength is chosen so that, with water to spare, the
        disc takes exactly its pipe's capacity: strength = Q / (area * sum of
        the kernel's falloff over the disc's open cells). Not the continuous
        2 Q / (pi R^2): on 1 m cells a 1.5 m disc's falloff sums to 3.67 m2
        where pi R^2 / 2 is 3.53, and the inlet measurably took 0.0397 m3/s
        through a pipe rated 0.0383 (docs/16_sewer_plan.md). With less water
        about it takes less.

        Both discs are at least 0.75 of a cell in radius, which guarantees each
        covers the centre of the cell it sits in. An outfall whose own cell is
        solid (inside a house) or off the map could pour nowhere, and the water
        its inlets took would vanish, so those inlets get no capacity.
        """
        dx = float(self._terrain.cell_size) if self._terrain is not None else 1.0
        floor_radius = 0.75 * dx
        usable = []
        for centre, _radius in outfalls:
            i = int(round(centre[0] / dx + (self._width - 1) * 0.5))
            j = int(round(centre[2] / dx + (self._height - 1) * 0.5))
            inside = 0 <= i < self._width and 0 <= j < self._height
            usable.append(bool(inside and self._obstacle_host.size
                               and self._obstacle_host[j * self._width + i] == 0))
        rebuilt = len(inlets) != self._inlet_count
        self._inlet_count = len(inlets)
        self._outfall_count = len(outfalls)
        self._inlet_capacity_m3s = []
        if inlets:
            radii, strengths, targets, path_strengths = [], [], [], []
            for row in inlets:
                centre, radius, capacity, target = row[:4]
                path = row[4] if len(row) > 4 else capacity
                r = max(float(radius), floor_radius)
                if target < 0 or target >= len(outfalls) or not usable[target]:
                    capacity, path, target = 0.0, 0.0, -1
                weight = self._sink_weight(centre, r, dx)
                radii.append(r)
                strengths.append(float(capacity) / weight if weight > 0.0 else 0.0)
                path_strengths.append(float(path) / weight if weight > 0.0 else 0.0)
                targets.append(int(target))
                self._inlet_capacity_m3s.append(float(capacity))
            self._inlet_centres = wp.array(
                np.array([item[0] for item in inlets], dtype=np.float32),
                dtype=wp.vec3, device=self.device)
            self._inlet_radii = wp.array(np.array(radii, dtype=np.float32),
                                         dtype=float, device=self.device)
            self._inlet_strengths = wp.array(np.array(strengths, dtype=np.float32),
                                             dtype=float, device=self.device)
            self._inlet_targets = wp.array(np.array(targets, dtype=np.int32),
                                           dtype=wp.int32, device=self.device)
            self._inlet_path_strengths = wp.array(np.array(path_strengths, dtype=np.float32),
                                                  dtype=float, device=self.device)
            if not measure_demand:
                self._inlet_demand_frame = None
                self._inlet_demand_m3s = []
            elif (rebuilt or self._inlet_demand_frame is None
                  or len(self._inlet_demand_frame) != len(inlets)):
                self._inlet_demand_frame = wp.zeros(len(inlets), dtype=float, device=self.device)
                self._inlet_demand_scratch = wp.zeros(1, dtype=float, device=self.device)
                self._inlet_demand_m3s = []
            if rebuilt or self._inlet_frame is None:
                self._inlet_circulation = wp.zeros(len(inlets), dtype=float, device=self.device)
                self._inlet_samples = wp.zeros(len(inlets), dtype=float, device=self.device)
                self._inlet_step = wp.zeros(len(inlets), dtype=float, device=self.device)
                self._inlet_frame = wp.zeros(len(inlets), dtype=float, device=self.device)
                self._inlet_flow_m3s = [0.0] * len(inlets)
        elif rebuilt:
            self._inlet_frame = None
            self._inlet_flow_m3s = []
        if not inlets:
            self._inlet_demand_frame = None
            self._inlet_demand_m3s = []
        if outfalls:
            self._outfall_centres = wp.array(
                np.array([item[0] for item in outfalls], dtype=np.float32),
                dtype=wp.vec3, device=self.device)
            self._outfall_radii = wp.array(
                np.array([max(float(item[1]), floor_radius) if ok else 0.0
                          for item, ok in zip(outfalls, usable)], dtype=np.float32),
                dtype=float, device=self.device)
            if self._outfall_volume is None or len(self._outfall_volume) != len(outfalls):
                self._outfall_volume = wp.zeros(len(outfalls), dtype=float, device=self.device)
                self._outfall_cells = wp.zeros(len(outfalls), dtype=float, device=self.device)

    def set_lava_vents(self, vents: list) -> None:
        """Upload placeable VENT objects; their presence IS the lava toggle.

        `vents` is (centre_xyz, radius_m, discharge_m3s, temperature_c) tuples,
        read live every tick like SOURCE/DRAIN so a vent can be dragged while
        RUNNING. An empty list turns lava mode off -- and, because `_manning`
        would otherwise be left holding whatever spatially-varying mu(T) the
        last lava run computed, this explicitly refills it back to the plain
        constant so a world with lava removed behaves exactly like one that
        never had any, rather than keeping stale per-cell friction forever.
        """
        was_enabled = self._lava_enabled
        self._vent_count = len(vents)
        self._lava_enabled = bool(vents)
        if vents:
            # Representative spawn point for the flow tracers (see
            # _advect_flow_tracers) -- the average of every vent rather than
            # just the first, so two vents both get some tracer traffic
            # instead of one being visually silent.
            centres = np.array([item[0] for item in vents], dtype=np.float32)
            self._tracer_vent_x = float(centres[:, 0].mean())
            self._tracer_vent_z = float(centres[:, 2].mean())
            self._vent_centres = wp.array(
                np.array([item[0] for item in vents], dtype=np.float32),
                dtype=wp.vec3, device=self.device)
            self._vent_radii = wp.array(
                np.array([item[1] for item in vents], dtype=np.float32),
                dtype=float, device=self.device)
            self._vent_discharges = wp.array(
                np.array([item[2] for item in vents], dtype=np.float32),
                dtype=float, device=self.device)
            self._vent_temps = wp.array(
                np.array([float(item[3]) + 273.15 for item in vents], dtype=np.float32),
                dtype=float, device=self.device)
        elif was_enabled and self._manning is not None:
            # Back to this WORLD's own surfaces, not unconditionally to the
            # river channel's constant -- a tsunami world's bed is a sea bed
            # and stays one, and a street stays paved.
            self._manning.assign(self._manning_host if self._manning_host.size
                                 == self._count else
                                 np.full(self._count,
                                         getattr(self, "_base_manning",
                                                 config.FLUID_MANNING_N),
                                         dtype=np.float32))

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

    def _update_centre_velocity(self) -> None:
        wp.launch(_centre_velocity, dim=self._count,
                  inputs=[self._h, self._u, self._v, self._bed, self._obstacles,
                          self._inlet_q, self._uc, self._vc, self._width,
                          self._height, config.FLUID_DRY_DEPTH,
                          config.FLUID_MAX_VELOCITY], device=self.device)

    def _measure(self) -> None:
        if self._h is None:
            return
        # also refreshes the centred field for anything read between frames --
        # bodies, gauges, the velocity stream -- including after a probe or
        # test has written the face arrays directly
        self._update_centre_velocity()
        wp.launch(_clear_diagnostics, dim=1, inputs=[self._diag_stats],
                  device=self.device)
        wp.launch(_reduce_diagnostics, dim=self._count, inputs=[self._h, self._u,
                  self._v, self._uc, self._vc, self._obstacles,
                  float(self._world.environment.gravity),
                  float(self._terrain.cell_size ** 2), config.FLUID_DRY_DEPTH,
                  self._diag_stats], device=self.device)
        if self._inlet_enabled:
            wp.launch(_measure_inlet_flux, dim=self._count,
                      inputs=[self._h, self._u, self._bed, self._obstacles,
                              self._inlet_q, self._width,
                              float(self._terrain.cell_size),
                              self._diag_stats], device=self.device)
        if self._lava_enabled:
            wp.launch(_reduce_lava_diagnostics, dim=self._count,
                      inputs=[self._temperature, self._h, self._obstacles,
                              config.FLUID_DRY_DEPTH, self._diag_stats],
                      device=self.device)
        stats = self._diag_stats.numpy()          # the frame's one device sync
        max_wave = float(stats[2])
        cfl_dt = (config.FIXED_DT if max_wave <= 1.0e-8 else
                  config.FLUID_CFL * self._terrain.cell_size / max_wave)
        self._diag.update({
            "inlet_discharge_m3s": float(stats[5]),
            "cfl_dt": cfl_dt,
            "max_depth": float(stats[0]),
            "max_velocity": float(stats[1]),
            "wet_cells": int(stats[4]),
            "volume_m3": float(stats[3]),
            "lava_temp_max_c": (float(stats[6]) - 273.15) if self._lava_enabled else None,
        })

    def advance(self, global_dt: float, max_substeps: int, stability_dt: float) -> int:
        if self._h is None:
            return 0
        # Measured at both ends of the frame, for two different consumers. Here,
        # because the substep count must answer to the state these substeps
        # actually start from -- including any state written from outside since
        # the last frame. And again after the loop, because a client reading
        # diagnostics wants the frame that just ran: reporting the previous
        # frame's volume alongside this frame's inflow made the volume ledger
        # look out by exactly one frame of inflow. Two measurements cost two
        # device syncs, which is why the diagnostics were packed into one array.
        self._measure()
        required = max(1, math.ceil(global_dt / max(self._diag["cfl_dt"], 1.0e-8)))
        substeps = min(max_substeps, required)
        dt = global_dt / substeps
        self._diag["cfl_limited"] = required > max_substeps
        gravity = float(self._world.environment.gravity)
        area = float(self._terrain.cell_size ** 2)
        for _ in range(substeps):
            tsunami_active = True
            if self._tsunami_enabled:
                # v0.14.3: which regime this substep is in decides whether the
                # east edge is the wavemaker (velocity closed, hard Dirichlet
                # `_apply_tsunami_edge`) or a plain transmissive outlet
                # (velocity open, `_apply_outflow` -- reusing the river's own
                # mechanism). Velocity CANNOT be left open during an active
                # pulse: `_velocity_step`'s outlet branch clamps any inward
                # velocity to zero the instant outflow_columns > 0, so an
                # always-open edge blocks the crest from ever pushing water
                # in at all (measured: 0 m flood instead of 258 m, see
                # `_apply_tsunami_edge`'s docstring). So this has to be a
                # discrete gate, not a continuous blend.
                #
                # A pulse is "active" within this many periods of its own
                # centre (the N-wave shape is already small past that --
                # config.TSUNAMI_ACTIVE_WINDOW_PERIODS). `tsunami_wave_spacing`
                # is chosen (see `initialize`) so consecutive pulses' windows
                # do not overlap, which is what guarantees a real gap opens
                # up between them for the outlet to actually drain through.
                tsunami_active = False
                for wave_i in range(self._tsunami_wave_count):
                    centre = (2.0 * self._tsunami_period
                              + wave_i * self._tsunami_wave_spacing)
                    if (abs(self._time - centre)
                            < config.TSUNAMI_ACTIVE_WINDOW_PERIODS * self._tsunami_period):
                        tsunami_active = True
                        break
                # Overrides whatever `set_outflow` last wrote (it forces this
                # to 0 whenever tsunami_enabled, unconditionally -- see its
                # own docstring); recomputed fresh every substep because
                # "active" can change inside a single frame's worth of them.
                self._outflow_columns = 0 if tsunami_active else max(
                    1, config.FLUID_OUTFLOW_COLUMNS)
            if self._lava_enabled:
                # mu(T) from the PREVIOUS substep's temperature, written into the
                # same array `_velocity_step` already reads -- so the array swap
                # this replaced (a scalar `config.FLUID_MANNING_N`) costs nothing
                # extra on the non-lava path, which never launches this kernel.
                wp.launch(_lava_manning, dim=self._count,
                          inputs=[self._temperature, self._h, self._obstacles,
                                  self._manning, config.LAVA_MANNING_MIN,
                                  config.LAVA_MANNING_MAX,
                                  config.LAVA_SOLIDUS_TEMP_C + 273.15,
                                  config.LAVA_ERUPTION_TEMP_C + 273.15,
                                  config.FLUID_DRY_DEPTH], device=self.device)
            wp.launch(_velocity_step, dim=self._count, inputs=[self._h, self._u,
                      self._v, self._bed, self._obstacles, self._next_u,
                      self._next_v, self._inlet_q, self._width, self._height,
                      float(self._terrain.cell_size), dt, gravity,
                      config.FLUID_DRY_DEPTH, self._manning,
                      config.FLUID_FRICTION_MIN_DEPTH,
                      config.FLUID_MAX_VELOCITY, self._outflow_columns,
                      self._outflow_rows[0], self._outflow_rows[1],
                      1 if self._outlet_kind == "river" else 0,
                      self._outlet_slope],
                      device=self.device)
            self._u, self._next_u = self._next_u, self._u
            self._v, self._next_v = self._next_v, self._v
            if self._section_faces:
                wp.launch(_section_flux, dim=self._section_faces,
                          inputs=[self._h, self._u, self._v, self._bed, self._obstacles,
                                  self._inlet_q, self._section_a, self._section_b,
                                  self._section_axis, self._section_sign,
                                  self._section_owner, self._section_volume,
                                  self._section_area, self._section_wet,
                                  self._section_level, self._width,
                                  float(self._terrain.cell_size), dt,
                                  config.FLUID_DRY_DEPTH],
                          device=self.device)
            wp.launch(_depth_step, dim=self._count, inputs=[self._h, self._u,
                      self._v, self._bed, self._obstacles, self._next_h,
                      self._inlet_q,
                      self._width, self._height, float(self._terrain.cell_size), dt],
                      device=self.device)
            self._h, self._next_h = self._next_h, self._h
            if self._lava_enabled:
                # Right after the swap, `_next_h` is exactly the pre-depth-step
                # depth `_depth_step` computed its faces from, and `_h` is what
                # it produced -- the two ends of the same flux this substep
                # already moved. See `_advect_lava_energy` for why this replaces
                # a back-trace of the sediment kernel's kind at a wetting front.
                wp.launch(_advect_lava_energy, dim=self._count,
                          inputs=[self._next_h, self._h, self._temperature,
                                  self._next_temperature, self._u, self._v,
                                  self._bed, self._obstacles, self._width,
                                  self._height, float(self._terrain.cell_size), dt,
                                  config.FLUID_DRY_DEPTH,
                                  config.LAVA_AMBIENT_TEMP_C + 273.15],
                          device=self.device)
                self._temperature, self._next_temperature = (
                    self._next_temperature, self._temperature)
            self._update_centre_velocity()
            if self._inlet_enabled:
                # A local discharge inlet is the river's own boundary and takes
                # over from the edge-level source, for the same reason a placed
                # SOURCE does: one map, one answer to "where does the water come
                # from". The level mode stays available and unchanged.
                wp.launch(_apply_river_inlet, dim=self._count,
                          inputs=[self._h, self._uc, self._vc, self._sediment,
                                  self._obstacles,
                                  self._inlet_q, self._inlet_normal_depth,
                                  self._width, self._height,
                                  float(self._terrain.cell_size), dt, area,
                                  config.SEDIMENT_CAPACITY_SCALE,
                                  self._diag_added, self._diag_sediment_in],
                          device=self.device)
            elif (self._source_enabled and self._edge_inflow_enabled
                  and not self._source_count):
                # a placed SOURCE takes over from the edge inflow entirely --
                # otherwise the map has two sources and "where does the water
                # come from" stops having a single answer
                wp.launch(_apply_source, dim=self._count, inputs=[self._h,
                          self._bed, self._uc, self._vc, self._sediment,
                          self._obstacles, self._width,
                          self._height, config.FLUID_SOURCE_COLUMNS,
                          self._level, area, config.SEDIMENT_CAPACITY_SCALE,
                          self._diag_added, self._diag_sediment_in], device=self.device)
            if self._source_count:
                wp.launch(_apply_point_sources, dim=self._count,
                          inputs=[self._h, self._bed, self._obstacles,
                                  self._src_centres, self._src_radii,
                                  self._src_levels, self._source_count,
                                  self._width, self._height,
                                  float(self._terrain.cell_size), area,
                                  self._diag_added],
                          device=self.device)
            if self._vent_count:
                wp.launch(_apply_lava_vents, dim=self._count,
                          inputs=[self._h, self._u, self._v, self._temperature,
                                  self._obstacles, self._vent_centres,
                                  self._vent_radii, self._vent_discharges,
                                  self._vent_temps, self._vent_count,
                                  self._width, self._height,
                                  float(self._terrain.cell_size), dt,
                                  config.FLUID_DRY_DEPTH,
                                  config.FLUID_MAX_VELOCITY, area,
                                  self._diag_added], device=self.device)
            if self._rain_mm_h > 0.0 and not self._lava_enabled:
                # Poured in portions of RAIN_APPLY_STEP_M, not a float32 sliver
                # per substep -- see config. Skipped in a lava world: `h` there
                # IS lava, and rain would erupt from the sky.
                self._rain_pending_m += self._rain_mm_h / 3.6e6 * dt
                if self._rain_pending_m >= config.RAIN_APPLY_STEP_M:
                    wp.launch(_apply_rain, dim=self._count,
                              inputs=[self._h, self._obstacles,
                                      float(self._rain_pending_m), area,
                                      self._diag_rain], device=self.device)
                    self._rain_pending_m = 0.0
            if self._drain_count:
                self._drain_circulation.zero_()
                self._drain_samples.zero_()
                wp.launch(_measure_drain_circulation, dim=self._count,
                          inputs=[self._uc, self._vc, self._h, self._obstacles,
                                  self._drain_centres, self._drain_radii,
                                  self._drain_circulation, self._drain_samples,
                                  self._drain_count,
                                  self._width, self._height,
                                  float(self._terrain.cell_size),
                                  config.FLUID_DRY_DEPTH], device=self.device)
                wp.launch(_apply_drains, dim=self._count,
                          inputs=[self._h, self._u, self._v, self._obstacles,
                                  self._drain_centres, self._drain_radii,
                                  self._drain_strengths, self._drain_circulation,
                                  self._drain_samples,
                                  self._drain_count, self._width, self._height,
                                  float(self._terrain.cell_size), dt,
                                  config.FLUID_DRY_DEPTH, config.DRAIN_SWIRL_GAIN,
                                  config.FLUID_MAX_VELOCITY, area,
                                  self._diag_removed, self._drain_removed_each, 1, 1, 0],
                          device=self.device)
            if self._inlet_count and not self._lava_enabled:
                # v0.17.0 storm sewer (backend/app/sewer.py): inlets take water
                # through the drain kernel at a strength capped by their pipe,
                # and the same substep pours it out at the outfalls. Booked on
                # its own two counters, not added/removed: the water is moved,
                # not created or destroyed, and the HUD should not say otherwise.
                # Skipped in a lava world for the reason rain is.
                # Removal only (impose_flow = 0): the circulation arrays are
                # passed but never read, so they are not measured either.
                self._inlet_step.zero_()
                if self._inlet_demand_frame is not None:
                    # v0.18.0: what every grate would take at its whole path's
                    # capacity, before anything is taken -- the demand a shared
                    # pipe is split by on the next tick (backend/app/sewer.py)
                    wp.launch(_apply_drains, dim=self._count,
                              inputs=[self._h, self._u, self._v, self._obstacles,
                                      self._inlet_centres, self._inlet_radii,
                                      self._inlet_path_strengths, self._inlet_circulation,
                                      self._inlet_samples,
                                      self._inlet_count, self._width, self._height,
                                      float(self._terrain.cell_size), dt,
                                      config.FLUID_DRY_DEPTH, config.DRAIN_SWIRL_GAIN,
                                      config.FLUID_MAX_VELOCITY, area,
                                      self._inlet_demand_scratch, self._inlet_demand_frame,
                                      0, 0, 1],
                              device=self.device)
                wp.launch(_apply_drains, dim=self._count,
                          inputs=[self._h, self._u, self._v, self._obstacles,
                                  self._inlet_centres, self._inlet_radii,
                                  self._inlet_strengths, self._inlet_circulation,
                                  self._inlet_samples,
                                  self._inlet_count, self._width, self._height,
                                  float(self._terrain.cell_size), dt,
                                  config.FLUID_DRY_DEPTH, config.DRAIN_SWIRL_GAIN,
                                  config.FLUID_MAX_VELOCITY, area,
                                  self._diag_sewer_in, self._inlet_step, 0, 0, 0],
                          device=self.device)
                if self._outfall_count:
                    self._outfall_volume.zero_()
                    wp.launch(_sewer_route, dim=self._inlet_count,
                              inputs=[self._inlet_step, self._inlet_targets,
                                      self._outfall_volume, self._inlet_frame],
                              device=self.device)
                    self._outfall_cells.zero_()
                    wp.launch(_count_outfall_cells, dim=self._count,
                              inputs=[self._obstacles, self._outfall_centres,
                                      self._outfall_radii, self._outfall_count,
                                      self._width, self._height,
                                      float(self._terrain.cell_size),
                                      self._outfall_cells], device=self.device)
                    wp.launch(_apply_outfalls, dim=self._count,
                              inputs=[self._h, self._obstacles, self._outfall_centres,
                                      self._outfall_radii, self._outfall_volume,
                                      self._outfall_cells, self._outfall_count,
                                      self._width, self._height,
                                      float(self._terrain.cell_size), area,
                                      self._diag_sewer_out], device=self.device)
            if self._outflow_columns:
                # For a tsunami world this fires only in the GAP between
                # pulses (or after the last one) -- see the gate computed
                # above -- letting whatever the shore sends back out to sea
                # actually leave instead of reflecting off the wavemaker's
                # wall the rest of the time.
                wp.launch(_apply_outflow, dim=self._count,
                          inputs=[self._h, self._u, self._sediment,
                                  self._obstacles,
                                  self._width, self._height,
                                  self._outflow_columns,
                                  self._outflow_rows[0], self._outflow_rows[1],
                                  float(self._terrain.cell_size), dt, area,
                                  self._diag_removed, self._diag_sediment_out],
                          device=self.device)
            if self._tsunami_enabled and tsunami_active:
                # The sea level outside the map, now. An N-shape in TIME:
                # negative (the sea withdrawing) before the centre, positive
                # (the wave) after it, so the trough arrives first because it
                # is emitted first -- the same "ordering is the whole
                # mechanism" the in-domain seed used, moved to the boundary.
                # Peak magnitude is amplitude*exp(-0.5) = 0.607*amplitude, not
                # amplitude; v0.14.0's plan doc quoted the latter as if it were
                # the wave height, which it never was.
                # v0.14.3: a wave TRAIN, one N-wave per `_tsunami_wave_count`,
                # spaced `_tsunami_wave_spacing` apart and scaled per pulse --
                # a real tsunami's second or third crest is sometimes the
                # largest one (1960 Chile: 4.5 m at 15 min, 8 m an hour
                # later), so wave2/3 default ABOVE and below 1x respectively,
                # not as a decaying echo. wave_count=1 (the default) sums
                # exactly one term and reproduces the original single-pulse
                # formula bit for bit.
                level = 0.0
                for wave_i in range(self._tsunami_wave_count):
                    scale = 1.0
                    if wave_i == 1:
                        scale = self._tsunami_wave2_scale
                    elif wave_i == 2:
                        scale = self._tsunami_wave3_scale
                    centre = 2.0 * self._tsunami_period + wave_i * self._tsunami_wave_spacing
                    z = (self._time - centre) / self._tsunami_period
                    level += self._tsunami_amplitude * scale * z * math.exp(-0.5 * z * z)
                # z(0) = -2, not -infinity: at t=0 this curve is already
                # amplitude*(-2)*exp(-2) = -0.27*amplitude, e.g. -1.62 m at the
                # default amplitude=6 -- a real number, not a rounding error.
                # `initialize()` fills the whole sea to a flat, calm `level`
                # (deliberately, so the scene starts as a still sea and not
                # mid-event -- see test_tsunami_starts_as_a_calm_sea), so the
                # boundary demanding -1.62 m on substep one is a discontinuity
                # the interior never had: measured as a spurious ~5.6 m/s
                # front crossing the whole domain before the real drawback
                # even begins. Ramping the signal up from 0 over the first
                # tenth of a period removes that step (the curve itself is
                # still essentially flat on that timescale, so trough/crest
                # timing is unaffected) without touching the boundary's
                # physics once the ramp is done. Only the FIRST pulse needs
                # this -- by the time a second or third pulse's centre
                # arrives the ramp is long since 1.0.
                ramp = min(1.0, self._time / (0.1 * self._tsunami_period))
                level *= ramp
                wp.launch(_apply_tsunami_edge, dim=self._count,
                          inputs=[self._h, self._bed, self._obstacles,
                                  self._width, self._height,
                                  max(1, config.FLUID_OUTFLOW_COLUMNS),
                                  float(level), area, self._diag_added],
                          device=self.device)
            if self._drain_count or self._vent_count:
                # drains and vents write faces; what reads the cell after them
                # has to see the vortex or the eruption they just imposed
                self._update_centre_velocity()
            if self._erosion_enabled:
                # RiverLab (v0.6.0): pick material up where the flow is fast and
                # deep, drop it where it slows, then carry the suspended load
                # with the same velocity field. Runs AFTER the depth/source
                # update so it sees this substep's real h/u/v, and the bed is
                # recombined immediately so the next substep's velocity step
                # already feels the freshly cut channel -- that closed loop
                # (flow -> erosion -> terrain -> flow) is the whole point.
                wp.launch(_erode_deposit, dim=self._count,
                          inputs=[self._h, self._uc, self._vc, self._bed_terrain,
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
                          inputs=[self._sediment, self._next_sediment, self._uc,
                                  self._vc, self._h, self._obstacles, self._width,
                                  self._height, float(self._terrain.cell_size), dt,
                                  config.FLUID_DRY_DEPTH], device=self.device)
                self._sediment, self._next_sediment = self._next_sediment, self._sediment
                self._recombine_bed()
            if self._lava_enabled:
                # Transport already ran right after `_depth_step` (see
                # `_advect_lava_energy`); here: cool, then freeze -- checked
                # against the solidus only after this substep's heat loss.
                k0 = (config.LAVA_EMISSIVITY * config.LAVA_COOLING_ENHANCEMENT
                      * _LAVA_SIGMA / (_LAVA_RHO * _LAVA_CP))
                wp.launch(_cool_lava, dim=self._count,
                          inputs=[self._temperature, self._next_temperature,
                                  self._h, self._obstacles, dt, k0,
                                  config.LAVA_AMBIENT_TEMP_C + 273.15,
                                  config.LAVA_COOLING_MIN_DEPTH,
                                  config.FLUID_DRY_DEPTH], device=self.device)
                self._temperature, self._next_temperature = (
                    self._next_temperature, self._temperature)
                wp.launch(_solidify_lava, dim=self._count,
                          inputs=[self._h, self._temperature, self._bed_terrain,
                                  self._obstacles,
                                  config.LAVA_SOLIDUS_TEMP_C + 273.15,
                                  config.FLUID_DRY_DEPTH, area,
                                  self._diag_solidified], device=self.device)
                self._recombine_bed()
            wp.launch(_advect_flow_tracers, dim=config.FLOW_TRACER_COUNT,
                      inputs=[self._flow_particles, self._h, self._uc, self._vc,
                              self._bed, self._obstacles, self._width, self._height,
                              config.FLUID_SOURCE_COLUMNS,
                              float(self._terrain.cell_size), dt,
                              config.FLUID_DRY_DEPTH,
                              1 if self._lava_enabled else 0,
                              self._tracer_vent_x, self._tracer_vent_z],
                      device=self.device)
            self._time += dt
        self._frame_dt = global_dt
        self._fold_ledger()
        self._measure()
        self.last_substeps = substeps
        return substeps

    def _fold_ledger(self) -> None:
        """Move this frame's device-side volume counters into Python floats.

        Accumulated per frame and folded here rather than summed on the device
        for the whole run: a float32 accumulator that has reached thousands of
        m3 silently stops noticing the cubic metre being added to it. Folding
        also leaves the device counters at zero for the next frame, so every
        path that injects or removes water must call this afterwards.
        """
        if self._diag_added is None:
            return
        self._added_m3 += float(self._diag_added.numpy()[0])
        self._removed_m3 += float(self._diag_removed.numpy()[0])
        self._sediment_out_m3 += float(self._diag_sediment_out.numpy()[0])
        self._sediment_in_m3 += float(self._diag_sediment_in.numpy()[0])
        self._diag_sediment_in.zero_()
        self._diag_added.zero_()
        self._diag_removed.zero_()
        self._diag_sediment_out.zero_()
        if self._diag_rain is not None:
            # Rain enters `h` like any source, so it is booked on the same side
            # of the ledger as `added_m3`; kept separately as well so the UI and
            # the tests can read how much of the water on the map fell as rain.
            rain = float(self._diag_rain.numpy()[0])
            self._added_m3 += rain
            self._rain_added_m3 += rain
            self._diag_rain.zero_()
        if self._diag_solidified is not None:
            # Solidified lava leaves the fluid domain the same way outflow or a
            # drain does, so it belongs in the same ledger the volume-error
            # check reads -- otherwise a working solidification step would show
            # up as an unexplained conservation error, not as a feature.
            self._solidified_m3 += float(self._diag_solidified.numpy()[0])
            self._diag_solidified.zero_()
        if self._section_faces:
            # v0.18.0 gauging lines: time integrals over this frame -> means
            span = max(float(getattr(self, "_frame_dt", 0.0)), 1.0e-9)
            volume = np.asarray(self._section_volume.numpy(), dtype=np.float64)
            area = np.asarray(self._section_area.numpy(), dtype=np.float64)
            wet = np.asarray(self._section_wet.numpy(), dtype=np.float64)
            level = np.asarray(self._section_level.numpy(), dtype=np.float64)
            readings = []
            for n, sid in enumerate(self._section_ids):
                self._section_cumulative[n] += float(volume[n])
                readings.append((sid, float(volume[n]) / span, float(area[n]) / span,
                                 float(wet[n]) / span,
                                 float(level[n] / wet[n]) if wet[n] > 0.0 else None,
                                 self._section_cumulative[n]))
            self._section_readings = readings
            for array in (self._section_volume, self._section_area, self._section_wet,
                          self._section_level):
                array.zero_()
        if self._diag_sewer_in is not None:
            # v0.17.0: what the inlets took and what the outfalls poured out,
            # and per inlet the flow over this frame -- what a pipe is carrying
            self._sewer_in_m3 += float(self._diag_sewer_in.numpy()[0])
            self._sewer_out_m3 += float(self._diag_sewer_out.numpy()[0])
            self._diag_sewer_in.zero_()
            self._diag_sewer_out.zero_()
            if self._inlet_count and self._inlet_frame is not None:
                frame = np.asarray(self._inlet_frame.numpy(), dtype=np.float64)
                span = max(float(getattr(self, "_frame_dt", 0.0)), 1.0e-9)
                self._inlet_flow_m3s = (frame / span).tolist()
                self._inlet_frame.zero_()
                if self._inlet_demand_frame is not None:
                    demand = np.asarray(self._inlet_demand_frame.numpy(), dtype=np.float64)
                    self._inlet_demand_m3s = (demand / span).tolist()
                    self._inlet_demand_frame.zero_()

    def _host_fields(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._h is None:
            empty = np.zeros(0, dtype=np.float32)
            return empty, empty, empty
        return (np.asarray(self._h.numpy(), dtype=np.float32),
                np.asarray(self._uc.numpy(), dtype=np.float32),
                np.asarray(self._vc.numpy(), dtype=np.float32))

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
        # Refreshed here too, not only in `advance`: bodies are sampled from the
        # CELL velocity, and anything that wrote the face arrays since the last
        # substep (a probe, a test, a remap) would otherwise be read stale.
        self._update_centre_velocity()
        wp.launch(_sample_bodies, dim=count, inputs=[self._h, self._uc, self._vc,
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
        if (abs(level - self._level) > 1.0e-9 and self._h is not None
                and not self._inlet_enabled and self._edge_inflow_enabled):
            self._source_enabled = True
            wp.launch(_apply_source, dim=self._count, inputs=[self._h, self._bed,
                      self._uc, self._vc, self._sediment,
                      self._obstacles, self._width, self._height,
                      config.FLUID_SOURCE_COLUMNS, level,
                      float(self._terrain.cell_size ** 2),
                      config.SEDIMENT_CAPACITY_SCALE, self._diag_added,
                      self._diag_sediment_in],
                      device=self.device)
            self._fold_ledger()
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

    def get_lava_temperature_field(self) -> np.ndarray:
        """Per-cell lava temperature in Celsius, same grid and ordering as
        `get_water_height_field`. Empty whenever no VENT is in the world --
        see FrameKind.LAVA_TEMPERATURE in protocol.py -- so a river or dam
        scene never pays for a field that has nothing to show.
        """
        if not self._lava_enabled or self._temperature is None:
            return np.zeros(0, dtype=np.float32)
        return (np.asarray(self._temperature.numpy(), dtype=np.float32)
                - 273.15).astype(np.float32)

    def sample_lava_contact(self, positions: np.ndarray) -> tuple:
        """(depth_m, temperature_c) at each world (x, _, z) position, indexed
        into the same grid `get_water_height`/`get_lava_temperature_field`
        use. Deliberately separate from `sample_for_bodies`: that vectorized
        path runs for every rigid body every step to drive buoyancy/collision,
        while combustion (SimulationManager._check_lava_ignition) only needs
        this for the small, independent set of burnable objects -- folding it
        into sample_for_bodies would make every non-lava world pay for lava
        fields it never has.
        """
        n = len(positions)
        if not self._lava_enabled or self._h is None or not n:
            return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
        depth_field = np.asarray(self._h.numpy(), dtype=np.float32)
        temp_field = (np.asarray(self._temperature.numpy(), dtype=np.float32)
                      - 273.15).astype(np.float32)
        positions = np.asarray(positions, dtype=np.float64)
        i = np.clip(np.round(positions[:, 0] / self._terrain.cell_size
                             + self._terrain.width / 2).astype(np.int64),
                    0, self._width - 1)
        j = np.clip(np.round(positions[:, 2] / self._terrain.cell_size
                             + self._terrain.height / 2).astype(np.int64),
                    0, self._height - 1)
        idx = j * self._width + i
        return depth_field[idx], temp_field[idx]

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
                # v0.12.0 volume ledger. Without it a boundary condition cannot
                # be shown to work: "water appears at the inlet" and "the right
                # amount of water appears at the inlet" look identical on
                # screen. volume_m3 - (added - removed) is the solver's own
                # conservation error and should stay at numerical noise.
                "added_m3": self._added_m3,
                "removed_m3": self._removed_m3,
                # Solidified lava leaves `h` exactly like outflow or a drain
                # does, so it is added on the same side of the equation as
                # `removed_m3` -- otherwise a correctly working solidification
                # step reads as a conservation bug rather than as a feature.
                "volume_error_m3": (self._diag["volume_m3"] - self._volume_at_start
                                    - self._added_m3 + self._removed_m3
                                    + self._solidified_m3
                                    + self._sewer_in_m3 - self._sewer_out_m3),
                # v0.17.0 storm sewer: moved, not created or destroyed, so on
                # its own two counters. In and out differ only by float noise.
                "sewer_in_m3": self._sewer_in_m3,
                "sewer_out_m3": self._sewer_out_m3,
                "sewer_inlet_flow_m3s": list(self._inlet_flow_m3s),
                "sewer_inlet_capacity_m3s": list(self._inlet_capacity_m3s),
                "sewer_inlet_demand_m3s": list(self._inlet_demand_m3s),
                "sediment_out_m3": self._sediment_out_m3,
                # v0.18.0: suspended load the inflow boundaries assigned
                "sediment_in_m3": self._sediment_in_m3,
                "lava_enabled": self._lava_enabled,
                "vents": self._vent_count,
                "solidified_m3": self._solidified_m3,
                "inlet_enabled": self._inlet_enabled,
                "inlet_request_m3s": self._inlet_request["discharge_m3s"],
                "outflow_rows": list(self._outflow_rows),
                "outlet_kind": self._outlet_kind,
                # RainLab-1: what is actually being applied -- 0 in a lava world
                # whatever the slider says, so the UI never claims rain it is not
                # delivering
                "rain_mm_h": (self._rain_mm_h if not self._lava_enabled else 0.0),
                "rain_m3s": ((self._rain_mm_h / 3.6e6
                              * float(np.count_nonzero(self._obstacle_host == 0))
                              * float(self._terrain.cell_size ** 2))
                             if (not self._lava_enabled and self._terrain is not None)
                             else 0.0),
                "rain_added_m3": self._rain_added_m3,
                "rain_pending_m": self._rain_pending_m,
                "edge_inflow": self._edge_inflow_enabled,
                "sources": self._source_count,
                "drains": self._drain_count,
                "drain_swirl_mps": (
                    [float(c) / max(1.0, float(n)) for c, n
                     in zip(self._drain_circulation.numpy(),
                            self._drain_samples.numpy())]
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
