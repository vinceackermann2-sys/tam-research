"""PCMA-v1 durable factual memory checkpoint (engineering, nonconfirmatory).

Serialize ONLY the completed public factual two-step memory and its bounded
online residual. Never serialize the hidden world/rule, unrevealed actions,
current pending action, neural training data, or the frozen model weights.

The checkpoint is a canonical JSON + SHA256 accidental-corruption check, NOT
an authenticated or confidential record. Do not accept checkpoints from
untrusted parties; a separate signed event ledger/process boundary is needed.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import torch

from tam_research.si_pcma_v1 import PublicObservation
from tam_research.si_pcma_v1_neural_probe import (
    ActualTwoStepRecord, FrozenNeuralTwoStepPolicy, TwoStepPCMAState,
)


CHECKPOINT_FORMAT = "pcma-v1-public-factual-session-checkpoint/1"
MAX_CHECKPOINT_BYTES = 2 * 1024 * 1024


def _encode(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _int(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"invalid {label}")
    return value


def _float(value: Any, label: str, limit: float = 1e6) -> float:
    if type(value) not in (float, int) or not math.isfinite(value) or abs(value) > limit:
        raise ValueError(f"invalid {label}")
    return float(value)


def model_weight_fingerprint(policy: FrozenNeuralTwoStepPolicy) -> str:
    """Checks the exact frozen neural parameters without storing tensors.

    Model loading/training belongs to a separate trusted workflow. A fresh
    policy must have IDENTICAL externally supplied trained weights.
    """
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.predictor.state_dict().items()):
        cpu = tensor.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(cpu.dtype).encode("ascii"))
        digest.update(_encode(list(cpu.shape)))
        digest.update(cpu.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _require_closed(policy: FrozenNeuralTwoStepPolicy) -> None:
    if type(policy) is not FrozenNeuralTwoStepPolicy:
        raise TypeError("only the frozen factual two-step policy is supported")
    if (policy._pending is not None or policy._first_event is not None
            or policy.history or policy.state.session_updates != 0):
        raise RuntimeError("checkpoint requires a clean restarted session boundary")
    if policy._last_time != policy.state._last_time:
        raise RuntimeError("policy/state factual clocks disagree")


def checkpoint_bytes(policy: FrozenNeuralTwoStepPolicy) -> bytes:
    """Construct a checkpoint only after completed session + restart_session()."""
    _require_closed(policy)
    s = policy.state
    payload = {
        "format": CHECKPOINT_FORMAT,
        "predictor_kind": policy.predictor.kind,
        "predictor_sha256": model_weight_fingerprint(policy),
        "continuation": list(policy.continuation.action_by_symbol),
        "allow_probe": policy.allow_probe,
        "pcma_weight": policy.pcma_weight,
        "memory_enabled": s.memory_enabled,
        "adaptation_enabled": s.adaptation_enabled,
        "capacity": s.capacity,
        "rank": s.rank,
        "update_limit": s.update_limit,
        "lr": s.lr,
        "memory": [asdict(record) for record in s.memory],
        "recent_provenance": list(s._seen_queue),
        "last_time": s._last_time,
        "u": s.u,
        "v": s.v,
        "session_updates": s.session_updates,
        "total_updates": s.total_updates,
    }
    raw = _encode(payload)
    envelope = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "payload": payload,
    }
    output = _encode(envelope)
    if len(output) > MAX_CHECKPOINT_BYTES:
        raise ValueError("checkpoint exceeds bounded storage")
    return output


def restore_checkpoint_bytes(policy: FrozenNeuralTwoStepPolicy, data: bytes) -> None:
    """Validate a complete checkpoint BEFORE changing the active policy.

    Refuses mismatched frozen neural weights, continuation, ablations or
    resource limits, non-finite adapter values, broken clock/provenance and
    malformed/oversized records. The unkeyed hash is NOT tamper protection.
    """
    _require_closed(policy)
    if type(data) is not bytes or len(data) > MAX_CHECKPOINT_BYTES:
        raise ValueError("checkpoint must be bounded JSON bytes")
    try:
        envelope = json.loads(data.decode("utf-8"))
        if type(envelope) is not dict or set(envelope) != {"sha256", "payload"}:
            raise ValueError("invalid checkpoint envelope")
        payload = envelope["payload"]
        if type(payload) is not dict:
            raise ValueError("invalid checkpoint payload")
        raw = _encode(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, OverflowError) as exc:
        raise ValueError("invalid checkpoint JSON") from exc
    if hashlib.sha256(raw).hexdigest() != envelope["sha256"]:
        raise ValueError("checksum mismatch")
    s = policy.state
    expected = {
        "format": CHECKPOINT_FORMAT,
        "predictor_kind": policy.predictor.kind,
        "predictor_sha256": model_weight_fingerprint(policy),
        "continuation": list(policy.continuation.action_by_symbol),
        "allow_probe": policy.allow_probe,
        "pcma_weight": policy.pcma_weight,
        "memory_enabled": s.memory_enabled,
        "adaptation_enabled": s.adaptation_enabled,
        "capacity": s.capacity,
        "rank": s.rank,
        "update_limit": s.update_limit,
        "lr": s.lr,
    }
    for key, value in expected.items():
        if type(payload.get(key)) is not type(value) or payload[key] != value:
            raise ValueError(f"checkpoint incompatible: {key}")
    if set(payload) != set(expected) | {
        "memory", "recent_provenance", "last_time", "u", "v",
        "session_updates", "total_updates",
    }:
        raise ValueError("unsupported checkpoint fields")

    clock = _int(payload["last_time"], "last factual time")
    session_updates = _int(payload["session_updates"], "session updates")
    total_updates = _int(payload["total_updates"], "total updates")
    if session_updates != 0 or total_updates > clock // 2:
        raise ValueError("not at a completed session boundary")
    memory_input = payload["memory"]
    if type(memory_input) is not list or len(memory_input) > s.capacity:
        raise ValueError("invalid factual memory capacity")
    parsed: list[ActualTwoStepRecord] = []
    previous = -1
    all_provenance: set[str] = set()
    for row in memory_input:
        if type(row) is not dict or set(row) != {
            "before", "first_action", "first_time", "observed_net",
            "first_provenance", "second_provenance",
        }:
            raise ValueError("invalid factual memory record")
        before = row["before"]
        if type(before) is not dict or set(before) != {"symbol", "steps_remaining"}:
            raise ValueError("invalid public observation record")
        symbol = _int(before["symbol"], "symbol")
        steps = _int(before["steps_remaining"], "steps")
        action = _int(row["first_action"], "action")
        first_time = _int(row["first_time"], "first factual time", 1)
        net = _float(row["observed_net"], "factual two-step reward", 2.0)
        if (symbol not in range(3) or steps > 16 or action not in range(4)
                or first_time <= previous or first_time + 1 > clock
                or net < -0.1):
            raise ValueError("invalid or unordered factual record")
        for name in ("first_provenance", "second_provenance"):
            provenance = row[name]
            if (type(provenance) is not str or not provenance
                    or len(provenance) > 256 or provenance in all_provenance):
                raise ValueError("invalid or duplicate factual provenance")
            all_provenance.add(provenance)
        parsed.append(ActualTwoStepRecord(
            PublicObservation(symbol, steps), action, first_time, net,
            row["first_provenance"], row["second_provenance"],
        ))
        previous = first_time
    if not s.memory_enabled and parsed:
        raise ValueError("disabled memory cannot contain factual rows")

    recent = payload["recent_provenance"]
    if (type(recent) is not list or len(recent) > s.provenance_capacity
            or len(recent) != len(set(map(str, recent)))):
        raise ValueError("invalid bounded provenance history")
    if any(type(x) is not str or not x or len(x) > 256 for x in recent):
        raise ValueError("invalid provenance ID")
    u, v = payload["u"], payload["v"]
    if type(u) is not list or len(u) != s.rank or type(v) is not list or len(v) != s.rank:
        raise ValueError("invalid low-rank adapter")
    safe_u = []
    for row in u:
        if type(row) is not list or len(row) != 8:
            raise ValueError("invalid adapter width")
        safe_u.append([_float(x, "adapter row") for x in row])
    safe_v = [_float(x, "adapter coefficient") for x in v]
    if not s.adaptation_enabled and total_updates:
        raise ValueError("disabled adapter cannot have updates")
    if clock == 0 and (parsed or recent or total_updates):
        raise ValueError("nonempty memory cannot precede first factual event")
    # Two-phase commit: only validated values mutate the policy.
    s.memory.clear()
    s.memory.extend(parsed)
    s._seen_queue.clear()
    s._seen_queue.extend(recent)
    s._seen_provenance = set(recent)
    s._last_time = clock
    s.u, s.v = safe_u, safe_v
    s.session_updates = session_updates
    s.total_updates = total_updates
    policy._last_time = clock


def save_checkpoint_atomic(policy: FrozenNeuralTwoStepPolicy, path: str | Path) -> None:
    """Atomic same-directory JSON replacement; no pickle or network actions."""
    payload = checkpoint_bytes(policy)
    destination = Path(path)
    if not destination.parent.is_dir():
        raise FileNotFoundError(destination.parent)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=".pcma-", delete=False
        ) as file:
            temp_name = file.name
            os.chmod(temp_name, 0o600)
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_name, destination)
        # Directory sync makes the rename more durable on POSIX filesystems.
        if os.name == "posix":
            fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        if temp_name is not None and os.path.exists(temp_name):
            os.unlink(temp_name)


def load_checkpoint(policy: FrozenNeuralTwoStepPolicy, path: str | Path) -> None:
    with Path(path).open("rb") as file:
        payload = file.read(MAX_CHECKPOINT_BYTES + 1)
    restore_checkpoint_bytes(policy, payload)
