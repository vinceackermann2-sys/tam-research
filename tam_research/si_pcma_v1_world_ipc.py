"""Public-feedback-only IPC boundary for the SI PCMA-v1 development world.

Only a separate child interpreter owns the world instance; the controller
receives PublicObservation and its OWN executed FactualTransition. No worker
handles, seeds, rules or private context are passed to the policy callbacks.
This improves accidental state isolation, NOT malicious-code sandboxing.

The IPC runner DENIES heldout worlds unconditionally; not a scientific
evaluator and cannot be used to approve an experiment or reserve seeds.
"""
from __future__ import annotations

import json
import os
import selectors
import subprocess
import sys
from typing import Any

from tam_research.si_pcma_v1 import FactualTransition, PublicObservation
from tam_research.si_pcma_v1_ood_world import OODWorldConfig, PublicPolicy

MAX_WIRE_BYTES = 4096


def _public_observation(value: Any) -> PublicObservation:
    if type(value) is not dict or set(value) != {"symbol", "steps_remaining"}:
        raise ValueError("invalid public observation schema")
    symbol, remaining = value["symbol"], value["steps_remaining"]
    if (type(symbol) is not int or symbol not in range(3)
            or type(remaining) is not int or not 0 <= remaining <= 16):
        raise ValueError("invalid public observation values")
    return PublicObservation(symbol, remaining)


def _factual_transition(value: Any) -> FactualTransition:
    if type(value) is not dict or set(value) != {
        "before", "action", "after", "reward", "fee", "time",
        "provenance", "done",
    }:
        raise ValueError("invalid factual feedback schema")
    before, after = (_public_observation(value["before"]),
                     _public_observation(value["after"]))
    action, time = value["action"], value["time"]
    reward, fee, provenance, done = (
        value["reward"], value["fee"], value["provenance"], value["done"])
    if (type(action) is not int or action not in range(4)
            or type(time) is not int or time < 1
            or type(reward) not in (int, float) or reward not in (0, 1)
            or type(fee) not in (int, float)
            or fee != (0.1 if action == 3 else 0.0)
            or (action == 3 and reward != 0)
            or type(provenance) is not str or len(provenance) > 128
            or not provenance or type(done) is not bool
            or after.steps_remaining != before.steps_remaining - 1):
        raise ValueError("invalid factual feedback fields")
    return FactualTransition(
        before, action, after, float(reward), float(fee),
        time, provenance, done,
    )


class DevelopmentWorldIPC:
    """One bounded child world, one request/reply at a time.

    The worker path and runtime are trusted. No untrusted external code runs
    inside the worker. Only the chosen action passes into the child.
    """

    def __init__(self, config: OODWorldConfig, *, timeout: float = 10.0,
                 world_kind: str = "ood"):
        if type(config) is not OODWorldConfig or config.split != "development":
            raise ValueError("heldout worlds prohibited by IPC development runner")
        if type(world_kind) is not str or world_kind not in ("ood", "history"):
            raise ValueError("unsupported development world kind")
        if type(timeout) not in (int, float) or not 0 < timeout <= 60:
            raise ValueError("invalid request timeout")
        self.config = config
        self.world_kind = world_kind
        self.timeout = float(timeout)
        self._proc: subprocess.Popen[bytes] | None = None
        self.observation: PublicObservation | None = None
        self._last_time = 0
        self._seen: set[str] = set()
        self._closed = False

    def __enter__(self) -> DevelopmentWorldIPC:
        if self._proc is not None or self._closed:
            raise RuntimeError("world client may only start once")
        # Do not inherit unrelated tokens or account credentials into the
        # trusted development child. This is NOT a security sandbox.
        allowed = ("PATH", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_PREFIX",
                   "LD_LIBRARY_PATH", "SystemRoot", "WINDIR",
                   "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        env = {key: os.environ[key] for key in allowed if key in os.environ}
        cmd = [sys.executable, "-u", "-m",
               "tam_research.si_pcma_v1_world_worker",
               "--split", "development",
               "--seed", str(self.config.seed),
               "--scenario", self.config.scenario,
               "--world-kind", self.world_kind]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, bufsize=0, env=env,
        )
        try:
            result = self._exchange({"op": "start"})
            if result.get("type") != "observation":
                raise RuntimeError("worker failed development initialization")
            self.observation = _public_observation(result.get("observation"))
            if self.observation.steps_remaining != self.config.session_horizon:
                raise RuntimeError("wrong development session horizon")
            return self
        except BaseException:
            self.close()
            raise

    def _exchange(self, message: dict[str, Any]) -> dict[str, Any]:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise RuntimeError("worker not running")
        payload = json.dumps(message, separators=(",", ":"),
                             allow_nan=False).encode("utf-8") + b"\n"
        if len(payload) > MAX_WIRE_BYTES:
            raise ValueError("request too large")
        try:
            proc.stdin.write(payload)
            proc.stdin.flush()
            with selectors.DefaultSelector() as selector:
                selector.register(proc.stdout, selectors.EVENT_READ)
                ready = selector.select(self.timeout)
            if not ready:
                raise TimeoutError("development world worker did not respond")
            wire = proc.stdout.readline(MAX_WIRE_BYTES + 1)
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError("development world worker unavailable") from exc
        if (not wire or len(wire) > MAX_WIRE_BYTES or
                not wire.endswith(b"\n")):
            raise RuntimeError("bad or truncated world worker response")
        try:
            result = json.loads(wire.decode("utf-8"))
        except (UnicodeError, ValueError) as exc:
            raise RuntimeError("malformed world worker response") from exc
        if type(result) is not dict:
            raise RuntimeError("invalid response envelope")
        if result.get("type") == "error":
            raise ValueError("development world worker rejected request")
        return result

    def step(self, action: int) -> FactualTransition:
        before = self.observation
        if before is None or self._closed or self._proc is None:
            raise RuntimeError("worker has not started")
        if before.steps_remaining == 0:
            raise RuntimeError("completed session requires explicit restart")
        if type(action) is not int or action not in range(4):
            raise ValueError("invalid public action")
        result = self._exchange({"op": "step", "action": action})
        if result.get("type") != "transition":
            raise RuntimeError("expected only chosen factual transition")
        event = _factual_transition(result.get("transition"))
        if (event.action != action or event.before != before
                or event.time != self._last_time + 1
                or event.provenance in self._seen
                or event.done != (event.after.steps_remaining == 0)):
            raise ValueError("worker factual causality/clock mismatch")
        self._last_time = event.time
        self._seen.add(event.provenance)
        self.observation = event.after
        return event

    def restart_session(self) -> PublicObservation:
        if self.observation is None or self.observation.steps_remaining != 0:
            raise RuntimeError("cannot restart incomplete development session")
        result = self._exchange({"op": "restart"})
        if result.get("type") != "observation":
            raise RuntimeError("world restart did not return public observation")
        obs = _public_observation(result.get("observation"))
        if obs.steps_remaining != self.config.session_horizon:
            raise ValueError("incorrect development restart horizon")
        self.observation = obs
        return obs

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.poll() is None:
                try:
                    self._exchange({"op": "close"})
                except (ValueError, RuntimeError, OSError, TimeoutError):
                    pass
            if proc.stdin is not None:
                proc.stdin.close()
            if proc.poll() is None:
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
        finally:
            if proc.stdout is not None:
                proc.stdout.close()
            self._proc = None

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def run_ipc_development_sessions(
    policy: PublicPolicy, config: OODWorldConfig, *,
    sessions: int = 2, erase_persistent: bool = False,
    world_kind: str = "ood",
) -> tuple[tuple[FactualTransition, ...], ...]:
    """Development only; policy callbacks see nothing except public facts.

    A single owned worker persists hidden world across *session* restarts,
    but its identity is NEVER passed to the policy. No offline replay and no
    withheld-action outcomes are generated.
    """
    if type(config) is not OODWorldConfig or config.split != "development":
        raise ValueError("heldout worlds are not available for engineering IPC")
    if type(sessions) is not int or not 1 <= sessions <= 32:
        raise ValueError("invalid bounded development session count")
    traces: list[tuple[FactualTransition, ...]] = []
    with DevelopmentWorldIPC(config, world_kind=world_kind) as world:
        for session in range(sessions):
            if session:
                world.restart_session()
                policy.restart_session(erase_persistent=erase_persistent)
            facts: list[FactualTransition] = []
            for _ in range(config.session_horizon):
                before = world.observation
                if before is None:
                    raise RuntimeError("no public observation")
                action = policy.choose(before)
                factual = world.step(action)
                policy.observe(factual)
                facts.append(factual)
            traces.append(tuple(facts))
    return tuple(traces)
