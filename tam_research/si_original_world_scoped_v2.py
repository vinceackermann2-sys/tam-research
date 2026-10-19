"""World-scoped inference-time fast adaptation for user's integrated SI architecture.

ENGINEERING CORRECTION ONLY: V1's factual long-term memory was world scoped,
but rank-two online fast weights were shared across all environments. That
allowed learning from world A's reward to change world B's forward path.

This V2 subclass preserves the exact original SI hybrid and V1 modules and
isolates ONLY the online fast-weight state per explicitly named world.
It does not make caller-claimed feedback cryptographically authentic, prevent
shared-backbone learning across tasks, or establish positive scientific results.
"""
from __future__ import annotations

from typing import Any, Sequence

import torch
from torch.nn import functional as F

from tam_research.si_original_integrated_v1 import (
    PROVENANCE, SIOriginalIntegratedV1,
)

SCHEMA = "si-original-world-scoped-fast-v2/1"


class SIOriginalWorldScopedV2(SIOriginalIntegratedV1):
    """Sequential use only; mutable active-world dispatch is not thread-safe.

    Memory is already world scoped in V1. This subclass changes neither the
    unmodified original SIHybridBlock nor its action/encoder/reasoning paths.
    Max 32 distinct named worlds in a checkpoint to bound external state.
    """

    def __init__(self, *, max_worlds: int = 32, **kwargs: Any) -> None:
        if type(max_worlds) is not int or not 1 <= max_worlds <= 32:
            raise ValueError("bounded max_worlds required")
        super().__init__(**kwargs)
        self.max_worlds = max_worlds
        self._active_world: str | None = None
        self._world_fast: dict[str, tuple[torch.Tensor, torch.Tensor, int]] = {}

    def _fast_correction(self, state: torch.Tensor) -> torch.Tensor:
        world = self._active_world
        if world is None:
            raise RuntimeError("world-scoped fast weights require forward(world=...)")
        if world in self._world_fast:
            u, v, _ = self._world_fast[world]
            u = u.to(device=state.device, dtype=state.dtype)
            v = v.to(device=state.device, dtype=state.dtype)
        else:
            # Neutral initial state, NOT another world's most recent adapter.
            u, v = self.fast_u, self.fast_v
        features = F.normalize(state, dim=-1)
        residual = (features @ u) @ v
        return state + residual[:, None] * self.fast_direction[None]

    def forward(self, steps: Sequence[tuple[str, torch.Tensor]], *,
                world: str = "default", **kwargs: Any) -> dict[str, Any]:
        if self._active_world is not None:
            raise RuntimeError("nested or concurrent world forwards are unsupported")
        if type(world) is not str or not world:
            raise ValueError("nonempty world key required")
        self._active_world = world
        try:
            return super().forward(steps, world=world, **kwargs)
        finally:
            self._active_world = None

    @torch.no_grad()
    def ingest_executed(self, before: Sequence[tuple[str, torch.Tensor]],
                        action: int, reward: float,
                        after: Sequence[tuple[str, torch.Tensor]], *,
                        world: str, skill: str | None = None,
                        provenance: str = PROVENANCE) -> None:
        if type(world) is not str or not world:
            raise ValueError("nonempty world key required")
        if world not in self._world_fast and len(self._world_fast) >= self.max_worlds:
            raise ValueError("world-scoped fast-state capacity exceeded")
        # Parent V1 updates fast_u and fast_v in place. Temporarily load ONLY
        # this world's state and restore neutral seed buffers unconditionally.
        # A failed public-feedback validation must not create a world record.
        neutral_u, neutral_v = self.fast_u.clone(), self.fast_v.clone()
        u, v, previous_updates = self._world_fast.get(
            world, (neutral_u, neutral_v, 0))
        try:
            self.fast_u.copy_(u.to(self.fast_u))
            self.fast_v.copy_(v.to(self.fast_v))
            super().ingest_executed(
                before, action, reward, after, world=world,
                skill=skill, provenance=provenance)
            self._world_fast[world] = (
                self.fast_u.detach().cpu().clone(),
                self.fast_v.detach().cpu().clone(),
                previous_updates + 1,
            )
        finally:
            self.fast_u.copy_(neutral_u)
            self.fast_v.copy_(neutral_v)

    def export_persistent_state(self) -> dict[str, Any]:
        record = super().export_persistent_state()
        record["schema"] = SCHEMA
        record["fast_worlds"] = {
            name: {"u": u.tolist(), "v": v.tolist(), "updates": count}
            for name, (u, v, count) in sorted(self._world_fast.items())
        }
        return record

    @torch.no_grad()
    def import_persistent_state(self, record: dict[str, Any]) -> None:
        expected = {
            "schema", "width", "memory", "fast_u", "fast_v",
            "fast_updates", "fast_worlds",
        }
        if (type(record) is not dict or set(record) != expected
                or record["schema"] != SCHEMA
                or type(record["fast_worlds"]) is not dict
                or len(record["fast_worlds"]) > self.max_worlds):
            raise ValueError("invalid world-scoped SI checkpoint")
        checked: dict[str, tuple[torch.Tensor, torch.Tensor, int]] = {}
        total = 0
        for name, fields in record["fast_worlds"].items():
            if (type(name) is not str or not name
                    or type(fields) is not dict
                    or set(fields) != {"u", "v", "updates"}
                    or type(fields["updates"]) is not int
                    or fields["updates"] < 1):
                raise ValueError("invalid world fast-state entry")
            try:
                u = torch.tensor(fields["u"], dtype=self.fast_u.dtype)
                v = torch.tensor(fields["v"], dtype=self.fast_v.dtype)
            except (TypeError, ValueError, RuntimeError) as error:
                raise ValueError("invalid world fast tensor") from error
            if (u.shape != self.fast_u.shape
                    or v.shape != self.fast_v.shape
                    or not bool(torch.isfinite(u).all())
                    or not bool(torch.isfinite(v).all())):
                raise ValueError("invalid world fast-state contents")
            checked[name] = (u, v, fields["updates"])
            total += fields["updates"]
        if (type(record["fast_updates"]) is not int
                or total != record["fast_updates"]):
            raise ValueError("world update ledger disagrees with total")
        parent = {key: value for key, value in record.items()
                  if key != "fast_worlds"}
        parent["schema"] = "si-original-integrated-v1/1"
        super().import_persistent_state(parent)
        self._world_fast = checked

    def world_fast_update_counts(self) -> dict[str, int]:
        return {w: n for w, (_, _, n) in sorted(self._world_fast.items())}
