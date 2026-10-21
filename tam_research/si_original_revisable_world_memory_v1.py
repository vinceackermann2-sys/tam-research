"""Revisable world memory (RWM1) for the user's world-scoped SI V2.

ENGINEERING REPAIR ONLY. The 21 October component diagnostic of the selected
V2 found that its factual memory returns stale values for revised facts
(duplicate keys append; cosine argmax returns the first), shares one FIFO
bound across worlds, ignores the executed action, always returns a nearest
record, and its native export omits replay.

This subclass replaces ONLY the storage objects that the unmodified V1/V2
ingest path writes into, and adds explicit lifetime persistence. Validation,
the fast-weight rule, the forward path and every neural parameter are
inherited unchanged. It is not a continual-learning, decision-utility,
planning or breakthrough claim. Sequential use only; not thread-safe.
Feedback provenance remains a caller-attested string, not a signed receipt.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from torch.nn import functional as F

from tam_research.si_original_integrated_v1 import PROVENANCE, PublicExperience
from tam_research.si_original_world_scoped_v2 import SIOriginalWorldScopedV2

SCHEMA = "si-original-revisable-world-memory-v1/1"
# Two keys are the same context iff their max absolute difference is <= this.
IDENTITY_TOL = 2e-6
# Every parameter/buffer that determines a memory key (the backbone state).
KEY_PATH = ("text_encoder.", "image_encoder.", "audio_encoder.",
            "video_encoder.", "motor_encoder.", "modality.", "position.",
            "blocks.", "norm.")
_MEMORY_FIELDS = {"action", "key", "value", "reward", "writes", "changes",
                  "first_write", "last_write"}
_AUDIT_FIELDS = {"identity", "old_value_sha256", "old_last_write",
                 "superseded_by"}
_REPLAY_FIELDS = {"skill", "before", "action", "reward", "after",
                  "provenance", "write_index"}
_CONFIG_FIELDS = {"width", "layers", "experts", "max_steps", "max_worlds",
                  "memory_world_capacity", "replay_world_capacity",
                  "audit_capacity", "match_threshold", "identity_tolerance"}


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    flat = tensor.detach().cpu().contiguous().reshape(-1)
    return flat.view(torch.uint8).numpy().tobytes()


def _max_difference(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max())


def _identity_digest(world: str, action: int, key: torch.Tensor) -> str:
    digest = hashlib.sha256(world.encode())
    digest.update(bytes([action]))
    digest.update(_tensor_bytes(key))
    return digest.hexdigest()


def _legal_action(action: Any) -> bool:
    return type(action) is int and action in range(4)


def _legal_reward(reward: Any) -> bool:
    return (type(reward) in (int, float) and math.isfinite(reward)
            and -1 <= reward <= 1)


def state_digest(value: Any) -> str:
    """Canonical SHA256 of nested lifetime records (dict keys sorted)."""
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            digest.update(f"T{item.dtype}{tuple(item.shape)}".encode())
            digest.update(_tensor_bytes(item))
        elif isinstance(item, dict):
            digest.update(b"D%d" % len(item))
            for key in sorted(item):
                digest.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"L%d" % len(item))
            for element in item:
                visit(element)
        else:
            digest.update(f"S{item!r}".encode())

    visit(value)
    return digest.hexdigest()


@dataclass
class MemoryRecord:
    action: int
    key: torch.Tensor
    value: torch.Tensor
    reward: float
    writes: int
    changes: int
    first_write: int
    last_write: int


@dataclass(frozen=True)
class IndexedExperience(PublicExperience):
    write_index: int


class RevisableWorldMemory:
    """World-partitioned, action-keyed, latest-wins verified transition memory.

    Identity is (world, action, key). Writing an existing identity replaces
    its value (the original key is kept) and logs the superseded value's
    digest in a bounded per-world audit list that retrieval never reads.
    Each world has its own quota and evicts only its own oldest-written
    record. Reads never mutate stored state.
    """

    def __init__(self, width: int, *, world_capacity: int = 32,
                 audit_capacity: int = 64,
                 match_threshold: float | None = None) -> None:
        if (type(width) is not int or width < 1
                or type(world_capacity) is not int or world_capacity < 1
                or type(audit_capacity) is not int or audit_capacity < 1):
            raise ValueError("positive bounded memory dimensions required")
        if match_threshold is not None and (
                type(match_threshold) is not float
                or not -1. <= match_threshold <= 1.):
            raise ValueError("cosine match threshold must be None or in [-1, 1]")
        self.width = width
        self.world_capacity = world_capacity
        self.audit_capacity = audit_capacity
        self.match_threshold = match_threshold
        self.rows: dict[str, list[MemoryRecord]] = {}
        self.audit: dict[str, list[dict[str, Any]]] = {}
        self.audit_dropped: dict[str, int] = {}
        self.write_count = 0
        self._staged: tuple[str, torch.Tensor, torch.Tensor] | None = None
        # Transient per-forward hit collector set by the model; not state.
        self._observer: list[bool] | None = None

    def _check(self, world: Any, key: Any, value: Any, provenance: Any) -> None:
        if type(world) is not str or not world or provenance != PROVENANCE:
            raise ValueError("verified executed feedback required")
        if (not isinstance(key, torch.Tensor)
                or not isinstance(value, torch.Tensor)
                or key.shape != (self.width,) or value.shape != (self.width,)
                or not key.is_floating_point()
                or not value.is_floating_point()
                or not bool(torch.isfinite(key).all())
                or not bool(torch.isfinite(value).all())):
            raise ValueError("invalid fact representation")

    @staticmethod
    def _find(rows: Sequence[MemoryRecord], action: int,
              key: torch.Tensor) -> int | None:
        best, best_distance = None, math.inf
        for i, record in enumerate(rows):
            if record.action != action:
                continue
            distance = _max_difference(record.key, key.to(record.key.dtype))
            if distance <= IDENTITY_TOL and distance < best_distance:
                best, best_distance = i, distance
        return best

    def find(self, world: str, key: torch.Tensor,
             action: int) -> MemoryRecord | None:
        rows = self.rows.get(world, [])
        index = self._find(rows, action, key.detach().cpu())
        return None if index is None else rows[index]

    def put(self, world: str, action: int, key: torch.Tensor,
            value: torch.Tensor, *, reward: float, provenance: str) -> int:
        """Write one executed (world, action, key) -> value fact; return its index."""
        self._check(world, key, value, provenance)
        if not _legal_action(action):
            raise ValueError("legal executed action 0..3 required")
        if not _legal_reward(reward):
            raise ValueError("finite executed reward in [-1, 1] required")
        key = key.detach().cpu().clone()
        value = value.detach().cpu().clone()
        index = self.write_count
        rows = self.rows.setdefault(world, [])
        self.audit.setdefault(world, [])
        self.audit_dropped.setdefault(world, 0)
        found = self._find(rows, action, key)
        if found is not None:
            record = rows[found]
            if _max_difference(record.value, value) > IDENTITY_TOL:
                self._log_superseded(world, record, index)
                record.changes += 1
            record.value = value
            record.reward = float(reward)
            record.writes += 1
            record.last_write = index
        else:
            if len(rows) >= self.world_capacity:
                # Own-world eviction only: other worlds' facts are never touched.
                del rows[min(range(len(rows)), key=lambda i: rows[i].last_write)]
            rows.append(MemoryRecord(action, key, value, float(reward),
                                     1, 0, index, index))
        self.write_count += 1
        return index

    def _log_superseded(self, world: str, record: MemoryRecord,
                        index: int) -> None:
        entries = self.audit[world]
        if len(entries) >= self.audit_capacity:
            del entries[0]
            self.audit_dropped[world] += 1
        entries.append({
            "identity": _identity_digest(world, record.action, record.key),
            "old_value_sha256": hashlib.sha256(
                _tensor_bytes(record.value)).hexdigest(),
            "old_last_write": record.last_write,
            "superseded_by": index,
        })

    # V1's ingest calls write(world, prior, delta, provenance=...) and then
    # appends a replay record that carries the executed action and reward.
    # write() therefore validates and stages; WorldReplay.append commits.
    def write(self, world: str, key: torch.Tensor, value: torch.Tensor,
              *, provenance: str) -> None:
        self._check(world, key, value, provenance)
        self._staged = (world, key.detach().cpu().clone(),
                        value.detach().cpu().clone())

    def clear_staged(self) -> None:
        self._staged = None

    def commit_staged(self, experience: PublicExperience) -> int:
        staged, self._staged = self._staged, None
        if (staged is None or staged[0] != experience.world
                or not torch.equal(staged[1], experience.before)):
            raise RuntimeError("replay record without its staged memory write")
        world, key, value = staged
        return self.put(world, experience.action, key, value,
                        reward=experience.reward,
                        provenance=experience.provenance)

    def lookup(self, world: str, query: torch.Tensor,
               action: int | None = None) -> tuple[torch.Tensor, bool]:
        """Return (value, hit). A miss returns zeros, never a nearest record."""
        if (not isinstance(query, torch.Tensor) or query.ndim != 1
                or query.numel() != self.width):
            raise ValueError("invalid memory query")
        if action is not None and not _legal_action(action):
            raise ValueError("legal action filter 0..3 required")
        rows = [r for r in self.rows.get(world, ())
                if action is None or r.action == action]
        if not rows:
            return torch.zeros_like(query), False
        probe = query.detach().cpu().to(rows[0].key.dtype)
        chosen = [i for i, r in enumerate(rows)
                  if _max_difference(r.key, probe) <= IDENTITY_TOL]
        if not chosen and self.match_threshold is not None:
            similarity = F.cosine_similarity(
                torch.stack([r.key for r in rows]), probe[None], dim=-1)
            top = float(similarity.max())
            if top >= self.match_threshold:
                chosen = [i for i in range(len(rows))
                          if float(similarity[i]) == top]
        if not chosen:
            return torch.zeros_like(query), False
        # Several actions (or exact cosine ties) share the best context: the
        # latest executed observation wins, never the first.
        best = max(chosen, key=lambda i: rows[i].last_write)
        return rows[best].value.to(device=query.device, dtype=query.dtype), True

    def read(self, world: str, query: torch.Tensor,
             action: int | None = None) -> torch.Tensor:
        value, hit = self.lookup(world, query, action)
        if self._observer is not None:
            self._observer.append(hit)
        return value

    def dump(self) -> dict[str, Any]:
        return {
            "write_count": self.write_count,
            "rows": {world: [{
                "action": r.action, "key": r.key.clone(),
                "value": r.value.clone(), "reward": r.reward,
                "writes": r.writes, "changes": r.changes,
                "first_write": r.first_write, "last_write": r.last_write,
            } for r in rows] for world, rows in sorted(self.rows.items())},
            "audit": {world: {
                "entries": [dict(entry) for entry in self.audit[world]],
                "dropped": self.audit_dropped[world],
            } for world in sorted(self.rows)},
        }

    def restored(self, record: Any) -> RevisableWorldMemory:
        """Validated copy built from dump(); never mutates self."""
        if (type(record) is not dict
                or set(record) != {"write_count", "rows", "audit"}
                or type(record["write_count"]) is not int
                or record["write_count"] < 0
                or type(record["rows"]) is not dict
                or type(record["audit"]) is not dict
                or set(record["rows"]) != set(record["audit"])):
            raise ValueError("invalid revisable memory export")
        count = record["write_count"]
        fresh = RevisableWorldMemory(
            self.width, world_capacity=self.world_capacity,
            audit_capacity=self.audit_capacity,
            match_threshold=self.match_threshold)
        fresh.write_count = count
        last_writes: set[int] = set()
        for world, entries in record["rows"].items():
            if (type(entries) is not list
                    or not 1 <= len(entries) <= self.world_capacity):
                raise ValueError("invalid world memory rows or quota exceeded")
            checked: list[MemoryRecord] = []
            for e in entries:
                if type(e) is not dict or set(e) != _MEMORY_FIELDS:
                    raise ValueError("invalid memory record schema")
                self._check(world, e["key"], e["value"], PROVENANCE)
                ints = (e["writes"], e["changes"], e["first_write"],
                        e["last_write"])
                if (not _legal_action(e["action"])
                        or type(e["reward"]) is not float
                        or not _legal_reward(e["reward"])
                        or any(type(n) is not int for n in ints)
                        or e["key"].dtype != e["value"].dtype
                        or not 0 <= e["changes"] < e["writes"]
                        or not 0 <= e["first_write"] <= e["last_write"] < count
                        or e["last_write"] in last_writes):
                    raise ValueError("invalid memory record ledger")
                if self._find(checked, e["action"], e["key"]) is not None:
                    raise ValueError("duplicate memory identity")
                last_writes.add(e["last_write"])
                checked.append(MemoryRecord(
                    e["action"], e["key"].detach().cpu().clone(),
                    e["value"].detach().cpu().clone(), e["reward"],
                    e["writes"], e["changes"], e["first_write"],
                    e["last_write"]))
            audit = record["audit"][world]
            if (type(audit) is not dict or set(audit) != {"entries", "dropped"}
                    or type(audit["entries"]) is not list
                    or len(audit["entries"]) > self.audit_capacity
                    or type(audit["dropped"]) is not int
                    or audit["dropped"] < 0):
                raise ValueError("invalid memory audit log")
            for entry in audit["entries"]:
                if (type(entry) is not dict or set(entry) != _AUDIT_FIELDS
                        or any(type(entry[k]) is not str or len(entry[k]) != 64
                               for k in ("identity", "old_value_sha256"))
                        or type(entry["old_last_write"]) is not int
                        or type(entry["superseded_by"]) is not int
                        or not 0 <= entry["old_last_write"]
                        < entry["superseded_by"] < count):
                    raise ValueError("invalid memory audit entry")
            fresh.rows[world] = checked
            fresh.audit[world] = [dict(entry) for entry in audit["entries"]]
            fresh.audit_dropped[world] = audit["dropped"]
        return fresh


class WorldReplay:
    """Append-only executed experience, bounded per world (own-world FIFO).

    Old outcomes stay: they actually happened. Iteration is in global write
    order, so V1's consolidate_verified_replay (``list(self.replay)[-count:]``)
    selects exactly as before whenever no world has overflowed its quota.
    """

    def __init__(self, memory: RevisableWorldMemory, *,
                 world_capacity: int = 32) -> None:
        if type(world_capacity) is not int or world_capacity < 1:
            raise ValueError("positive bounded replay quota required")
        self._memory = memory
        self.world_capacity = world_capacity
        self.rows: dict[str, list[IndexedExperience]] = {}

    def append(self, experience: PublicExperience) -> None:
        """Called by the inherited V1 ingest right after its staged write."""
        index = self._memory.commit_staged(experience)
        self._add(IndexedExperience(
            experience.world, experience.skill, experience.before,
            experience.action, experience.reward, experience.after,
            experience.provenance, index))

    def _add(self, record: IndexedExperience) -> None:
        rows = self.rows.setdefault(record.world, [])
        if len(rows) >= self.world_capacity:
            del rows[0]
        rows.append(record)

    def __iter__(self) -> Iterator[IndexedExperience]:
        return iter(sorted((r for rows in self.rows.values() for r in rows),
                           key=lambda r: r.write_index))

    def __len__(self) -> int:
        return sum(len(rows) for rows in self.rows.values())

    def dump(self) -> dict[str, list[dict[str, Any]]]:
        return {world: [{
            "skill": r.skill, "before": r.before.clone(), "action": r.action,
            "reward": r.reward, "after": r.after.clone(),
            "provenance": r.provenance, "write_index": r.write_index,
        } for r in rows] for world, rows in sorted(self.rows.items())}

    def restored(self, record: Any, memory: RevisableWorldMemory,
                 skills: Sequence[str]) -> WorldReplay:
        """Validated copy bound to ``memory``; never mutates self."""
        if type(record) is not dict:
            raise ValueError("invalid world replay export")
        fresh = WorldReplay(memory, world_capacity=self.world_capacity)
        indices: set[int] = set()
        for world, entries in record.items():
            if (type(world) is not str or not world
                    or type(entries) is not list
                    or not 1 <= len(entries) <= self.world_capacity):
                raise ValueError("invalid world replay rows or quota exceeded")
            previous = -1
            for e in entries:
                if type(e) is not dict or set(e) != _REPLAY_FIELDS:
                    raise ValueError("invalid replay record schema")
                memory._check(world, e["before"], e["after"], e["provenance"])
                if (not _legal_action(e["action"])
                        or type(e["reward"]) is not float
                        or not _legal_reward(e["reward"])
                        or (e["skill"] is not None and e["skill"] not in skills)
                        or type(e["write_index"]) is not int
                        or not previous < e["write_index"] < memory.write_count
                        or e["write_index"] in indices):
                    raise ValueError("invalid replay record ledger")
                previous = e["write_index"]
                indices.add(previous)
                fresh._add(IndexedExperience(
                    world, e["skill"], e["before"].detach().cpu().clone(),
                    e["action"], e["reward"], e["after"].detach().cpu().clone(),
                    e["provenance"], e["write_index"]))
        return fresh


class SIOriginalRevisableWorldMemoryV1(SIOriginalWorldScopedV2):
    """V2 with revisable, action-keyed, world-partitioned memory and replay.

    The inherited V1/V2 ingest, validation, fast-weight rule and forward path
    run unmodified; only the storage objects they write into differ. V2's
    max_worlds bounds the shared world registry (fast state, memory, replay).
    No neural parameter or buffer is added.
    """

    def __init__(self, *, memory_world_capacity: int = 32,
                 replay_world_capacity: int = 32, audit_capacity: int = 64,
                 match_threshold: float | None = None, **kwargs: Any) -> None:
        if "memory_capacity" in kwargs or "replay_capacity" in kwargs:
            raise ValueError("RWM1 bounds memory and replay per world")
        super().__init__(**kwargs)
        self.memory = RevisableWorldMemory(
            self.width, world_capacity=memory_world_capacity,
            audit_capacity=audit_capacity, match_threshold=match_threshold)
        self.replay = WorldReplay(self.memory,
                                  world_capacity=replay_world_capacity)

    def lifetime_config(self) -> dict[str, Any]:
        return {
            "width": self.width, "layers": self.cfg.layers,
            "experts": self.cfg.experts, "max_steps": self.max_steps,
            "max_worlds": self.max_worlds,
            "memory_world_capacity": self.memory.world_capacity,
            "replay_world_capacity": self.replay.world_capacity,
            "audit_capacity": self.memory.audit_capacity,
            "match_threshold": self.memory.match_threshold,
            "identity_tolerance": IDENTITY_TOL,
        }

    def representation_sha256(self) -> str:
        """Fingerprint of everything that produces a memory key."""
        digest = hashlib.sha256()
        for name, tensor in sorted(self.state_dict().items()):
            if name.startswith(KEY_PATH):
                digest.update(
                    f"{name}|{tensor.dtype}|{tuple(tensor.shape)}|".encode())
                digest.update(_tensor_bytes(tensor))
        return digest.hexdigest()

    def forward(self, steps: Sequence[tuple[str, torch.Tensor]], *,
                world: str = "default", **kwargs: Any) -> dict[str, Any]:
        hits: list[bool] = []
        self.memory._observer = hits
        try:
            output = super().forward(steps, world=world, **kwargs)
        finally:
            self.memory._observer = None
        logits = output["logits"]
        output["memory_hit"] = (
            torch.tensor(hits, dtype=torch.bool, device=logits.device) if hits
            else torch.zeros(logits.size(0), dtype=torch.bool,
                             device=logits.device))
        return output

    @torch.no_grad()
    def ingest_executed(self, before: Sequence[tuple[str, torch.Tensor]],
                        action: int, reward: float,
                        after: Sequence[tuple[str, torch.Tensor]], *,
                        world: str, skill: str | None = None,
                        provenance: str = PROVENANCE) -> None:
        # Inherited path: validate, stage the memory write, append the replay
        # record that commits it with the executed action, then update only
        # this world's fast state. Never leave a stale staged write behind.
        self.memory.clear_staged()
        try:
            super().ingest_executed(before, action, reward, after,
                                    world=world, skill=skill,
                                    provenance=provenance)
        finally:
            self.memory.clear_staged()

    def export_lifetime_state(self) -> dict[str, Any]:
        """Complete external state; loads with torch.load(weights_only=True)."""
        return {
            "schema": SCHEMA,
            "config": self.lifetime_config(),
            "representation_sha256": self.representation_sha256(),
            "skills": sorted(self.skills),
            "fast_updates": self.fast_updates,
            "latest_replay_loss": self.latest_replay_loss,
            "fast_worlds": {name: {"u": u.clone(), "v": v.clone(),
                                   "updates": count}
                            for name, (u, v, count)
                            in sorted(self._world_fast.items())},
            "memory": self.memory.dump(),
            "replay": self.replay.dump(),
        }

    def _checked_fast(self, record: Any, total: Any
                      ) -> dict[str, tuple[torch.Tensor, torch.Tensor, int]]:
        if (type(record) is not dict or len(record) > self.max_worlds
                or type(total) is not int or total < 0):
            raise ValueError("invalid world fast-state export")
        checked = {}
        for name, fields in record.items():
            if (type(name) is not str or not name or type(fields) is not dict
                    or set(fields) != {"u", "v", "updates"}
                    or type(fields["updates"]) is not int
                    or fields["updates"] < 1):
                raise ValueError("invalid world fast-state entry")
            u, v = fields["u"], fields["v"]
            if (not isinstance(u, torch.Tensor) or not isinstance(v, torch.Tensor)
                    or u.shape != self.fast_u.shape
                    or v.shape != self.fast_v.shape
                    or u.dtype != self.fast_u.dtype
                    or v.dtype != self.fast_v.dtype
                    or not bool(torch.isfinite(u).all())
                    or not bool(torch.isfinite(v).all())):
                raise ValueError("invalid world fast-state contents")
            checked[name] = (u.detach().cpu().clone(), v.detach().cpu().clone(),
                             fields["updates"])
        if sum(n for _, _, n in checked.values()) != total:
            raise ValueError("world update ledger disagrees with total")
        return checked

    @torch.no_grad()
    def import_lifetime_state(self, record: Any) -> None:
        """Validate everything, then swap state in; refuses V1/V2 schemas."""
        expected = {"schema", "config", "representation_sha256", "skills",
                    "fast_updates", "latest_replay_loss", "fast_worlds",
                    "memory", "replay"}
        if (type(record) is not dict or set(record) != expected
                or record["schema"] != SCHEMA):
            raise ValueError("invalid revisable-memory lifetime record")
        if record["config"] != self.lifetime_config():
            raise ValueError("lifetime configuration differs from this model")
        if record["skills"] != sorted(self.skills):
            raise ValueError("lifetime skills differ from this model")
        if record["representation_sha256"] != self.representation_sha256():
            raise ValueError(
                "representation fingerprint mismatch: stored keys are stale")
        loss = record["latest_replay_loss"]
        if loss is not None and (type(loss) is not float
                                 or not math.isfinite(loss)):
            raise ValueError("invalid replay loss record")
        memory = self.memory.restored(record["memory"])
        replay = self.replay.restored(record["replay"], memory,
                                      sorted(self.skills))
        fast = self._checked_fast(record["fast_worlds"],
                                  record["fast_updates"])
        if not set(memory.rows) == set(replay.rows) == set(fast):
            raise ValueError("world registry disagrees across stores")
        self.memory, self.replay, self._world_fast = memory, replay, fast
        self.fast_updates = record["fast_updates"]
        self.latest_replay_loss = loss

    # The V2 export omitted replay; RWM1's persistent state is the lifetime.
    def export_persistent_state(self) -> dict[str, Any]:
        return self.export_lifetime_state()

    def import_persistent_state(self, record: Any) -> None:
        self.import_lifetime_state(record)

    def save_lifetime(self, path: str | Path) -> None:
        """Weights plus lifetime state; never overwrites an existing file."""
        payload = {
            "weights": {name: tensor.detach().cpu().clone()
                        for name, tensor in self.state_dict().items()},
            "lifetime": self.export_lifetime_state(),
        }
        with Path(path).open("xb") as stream:
            torch.save(payload, stream)

    @classmethod
    def load_lifetime(cls, path: str | Path
                      ) -> SIOriginalRevisableWorldMemoryV1:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if (type(payload) is not dict or set(payload) != {"weights", "lifetime"}
                or type(payload["lifetime"]) is not dict
                or type(payload["lifetime"].get("config")) is not dict
                or set(payload["lifetime"]["config"]) != _CONFIG_FIELDS
                or type(payload["lifetime"].get("skills")) is not list):
            raise ValueError("invalid revisable-memory lifetime file")
        config = payload["lifetime"]["config"]
        if config["identity_tolerance"] != IDENTITY_TOL:
            raise ValueError("lifetime identity tolerance differs")
        model = cls(
            width=config["width"], layers=config["layers"],
            experts=config["experts"], max_steps=config["max_steps"],
            max_worlds=config["max_worlds"],
            memory_world_capacity=config["memory_world_capacity"],
            replay_world_capacity=config["replay_world_capacity"],
            audit_capacity=config["audit_capacity"],
            match_threshold=config["match_threshold"])
        for name in payload["lifetime"]["skills"]:
            model.add_skill(name)
        try:
            model.load_state_dict(payload["weights"], strict=True)
        except RuntimeError as error:
            raise ValueError("lifetime weights do not match its configuration"
                             ) from error
        model.import_lifetime_state(payload["lifetime"])
        return model
