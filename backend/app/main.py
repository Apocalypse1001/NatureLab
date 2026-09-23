"""FastAPI backend: REST status, WebSocket control/sim channel, static frontend.

Run from the NatureLab directory:
    python -m uvicorn app.main:app --host 127.0.0.1 --port 8756
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config, persistence
from .simulation import SimulationManager

manager = SimulationManager()
clients: Set[WebSocket] = set()


async def broadcast_text(text: str) -> None:
    for client in list(clients):
        try:
            await client.send_text(text)
        except Exception:
            clients.discard(client)


async def broadcast_bytes(data: bytes) -> None:
    for client in list(clients):
        try:
            await client.send_bytes(data)
        except Exception:
            clients.discard(client)


@asynccontextmanager
async def lifespan(app: FastAPI):
    info = manager.engine_info()
    print(f"[naturelab] compute engine: {info['engine']} | device: {info['device']} | "
          f"CUDA: {info['cuda']} | GPU: {info['gpu_name']}")
    print(f"[naturelab] warp selftest: {info['selftest']}")
    yield
    manager.stop()


app = FastAPI(title="NatureLab backend", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/status")
async def status() -> JSONResponse:
    return JSONResponse({
        "app": "NatureLab",
        "backend": "online",
        "engine": manager.engine_info(),
        "sim_status": manager.status,
        "sim_time": manager.sim_time,
    })


@app.get("/api/worlds")
async def worlds() -> JSONResponse:
    return JSONResponse({"worlds": persistence.list_worlds()})


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    clients.add(ws)
    manager.attach(broadcast_text, broadcast_bytes)
    await ws.send_text(json.dumps({
        "type": "hello",
        "backend": "online",
        "engine": manager.engine_info(),
        "status": manager.status,
    }, separators=(",", ":")))
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_text(json.dumps({"type": "error", "error": "bad json"}))
                continue
            if not isinstance(msg, dict):
                await ws.send_text(json.dumps({"type": "error",
                                               "error": "message must be an object"}))
                continue
            reply = await _dispatch(ws, msg)
            if reply is not None:
                await ws.send_text(json.dumps(reply, separators=(",", ":")))
    except WebSocketDisconnect:
        clients.discard(ws)


def client_world() -> dict:
    """The world for the browser: the stored starting water (hundreds of kB of
    base64 the client never reads) is replaced by whether there is one."""
    world = manager.world.to_dict()
    world["water"]["initial_flow"] = world["water"].get("initial_flow") is not None
    return world


async def _dispatch(ws: WebSocket, msg: dict) -> dict | None:
    try:
        op = msg.get("op")
        if not isinstance(op, str):
            raise ValueError("op must be a string")
        if getattr(manager, "_settling", False) and op not in ("request_world", "set_speed"):
            # the world is put back when settling ends, so an edit made meanwhile
            # would silently vanish -- refuse it instead
            raise ValueError("the river is settling; wait for it to finish")
        if op == "request_world":
            return {"type": "world", "op": op, "world": client_world(),
                    "status": manager.status, "time": manager.sim_time}
        if op == "object_add":
            obj_dict = manager.apply_object_add(msg.get("object", {}))
            return {"type": "ack", "op": op, "id": obj_dict["id"], "object": obj_dict}
        if op in ("pipe_add", "pipe_update"):
            # v0.17.0 storm sewer: one reply carries every object the edit
            # created or changed (a pipe may create its inlet and outfall too)
            state = (manager.apply_pipe_add(msg.get("pipe", {})) if op == "pipe_add"
                     else manager.apply_pipe_update(msg.get("pipe", {})))
            return {"type": "pipe", "op": op, **state}
        if op == "object_update":
            manager.apply_object_update(str(msg["id"]), msg.get("fields", {}))
            return {"type": "ack", "op": op}
        if op == "object_remove":
            manager.apply_object_remove(str(msg["id"]))
            return {"type": "ack", "op": op}
        if op == "terrain_brush":
            patch = manager.apply_terrain_brush(msg["x"], msg["z"], msg["radius"],
                                                msg["strength"])
            return {"type": "terrain_patch", **patch}
        if op == "terrain_river":
            patch = manager.apply_terrain_river(msg.get("params") or {})
            return {"type": "terrain_patch", **patch}
        if op == "terrain_volcano":
            patch = manager.apply_terrain_volcano(msg.get("params") or {})
            return {"type": "terrain_patch", **patch}
        if op == "water_outflow":
            manager.apply_water_outflow(bool(msg.get("enabled", True)))
            return {"type": "ack", "op": op}
        if op == "river_inlet":
            state = manager.apply_river_inlet(msg.get("fields") or {})
            return {"type": "ack", "op": op, "inlet": state}
        if op == "river_outlet":
            state = manager.apply_river_outlet(msg.get("fields") or {})
            return {"type": "ack", "op": op, "outlet": state}
        if op == "rain":
            state = manager.apply_rain(msg.get("fields") or {})
            return {"type": "ack", "op": op, "rain": state}
        if op == "edge_inflow":
            manager.apply_edge_inflow(bool(msg.get("enabled", True)))
            return {"type": "ack", "op": op}
        if op == "water_erosion":
            manager.apply_water_erosion(bool(msg.get("enabled", False)))
            return {"type": "ack", "op": op}
        if op == "water_level":
            manager.apply_water_level(msg["level"])
            return {"type": "ack", "op": op}
        if op == "start":
            manager.start()
            return {"type": "ack", "op": op, "status": manager.status}
        if op == "pause":
            manager.pause()
            return {"type": "ack", "op": op, "status": manager.status}
        if op == "reset":
            manager.reset()
            return {"type": "world", "op": op, "world": client_world(),
                    "status": manager.status, "time": manager.sim_time}
        if op == "set_speed":
            manager.set_speed(msg["value"])
            return {"type": "ack", "op": op}
        if op == "save":
            path = manager.save(str(msg.get("name", "default")))
            return {"type": "saved", "name": msg.get("name", "default"),
                    "path": path}
        if op == "load":
            manager.load(str(msg.get("name", "default")))
            # op and name let the client frame its camera on a world that has
            # just arrived, and on a prepared scenario the way it is meant to
            return {"type": "world", "op": op, "name": str(msg.get("name", "default")),
                    "world": client_world(),
                    "status": manager.status, "time": manager.sim_time}
        if op == "water_settle":
            # v0.18.1: start the river flowing; the reply is the settled world
            result = await manager.apply_water_settle(msg.get("fields") or {})
            return {"type": "world", "op": op, "world": client_world(), "status": manager.status,
                    "time": manager.sim_time, "settle": result}
        return {"type": "error", "error": f"unknown op: {op}"}
    except Exception as exc:
        return {"type": "error", "error": str(exc)}


# Serve the built frontend (Vite `npm run build` output) if present.
if config.FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=config.FRONTEND_DIST / "assets"),
              name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str):
        target = config.FRONTEND_DIST / path
        if path and target.is_file():
            return FileResponse(target)
        return FileResponse(config.FRONTEND_DIST / "index.html")
