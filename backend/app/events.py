"""Event system: causal history of the simulation.

Events are structured so a future educational replay/analysis mode can show
"time / event / object / cause / parameters" without changing the core.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class EventType(str, enum.Enum):
    SIM_STARTED = "SIM_STARTED"
    SIM_PAUSED = "SIM_PAUSED"
    SIM_RESET = "SIM_RESET"
    OBJECT_STARTED_MOVING = "OBJECT_STARTED_MOVING"
    OBJECT_COLLISION = "OBJECT_COLLISION"
    OBJECT_DAMAGED = "OBJECT_DAMAGED"
    OBJECT_BROKEN = "OBJECT_BROKEN"
    OBJECT_FLOATING = "OBJECT_FLOATING"
    OBJECT_SETTLED = "OBJECT_SETTLED"
    WATER_ENTERED_AREA = "WATER_ENTERED_AREA"
    WORLD_LOADED = "WORLD_LOADED"
    WORLD_SAVED = "WORLD_SAVED"


@dataclass
class SimEvent:
    time: float
    type: str
    object_id: Optional[str] = None
    cause: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        params = {k: (v.item() if hasattr(v, "item") else v)
                  for k, v in self.parameters.items()}
        return {"time": round(self.time, 3), "type": self.type,
                "object_id": self.object_id, "cause": self.cause,
                "parameters": params}


class EventLog:
    """Append-only ring of events; streamed to the frontend as they occur."""

    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self._all: List[SimEvent] = []
        self._pending: List[SimEvent] = []

    def record(self, sim_time: float, etype: EventType, object_id: Optional[str] = None,
               cause: str = "", **parameters: Any) -> SimEvent:
        event = SimEvent(time=sim_time, type=etype.value, object_id=object_id,
                         cause=cause, parameters=parameters)
        self._all.append(event)
        if len(self._all) > self.capacity:
            self._all = self._all[-self.capacity:]
        self._pending.append(event)
        return event

    def take_pending(self) -> List[Dict[str, Any]]:
        out = [e.to_dict() for e in self._pending]
        self._pending.clear()
        return out

    def all(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self._all]

    def clear(self) -> None:
        self._all.clear()
        self._pending.clear()
