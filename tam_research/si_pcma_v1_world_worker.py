"""Development-only private-world JSON-lines worker for SI PCMA-v1.

Run as `python -u -m tam_research.si_pcma_v1_world_worker` with explicit
development fixture arguments. The *worker* alone constructs OODRuleWorld.
Messages contain ONLY public observations and executed factual transitions.
This does NOT sandbox a malicious local policy: source, OS processes and
command-line seed values may remain inspectable by same-privilege code.

NO heldout rules, scientific attempts, paid compute or seed reservations.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from typing import Any

from tam_research.si_pcma_v1_ood_world import OODRuleWorld, OODWorldConfig

MAX_REQUEST_BYTES = 4096
MAX_RESPONSE_BYTES = 4096


def _response(message: dict[str, Any]) -> None:
    raw = json.dumps(message, separators=(",", ":"), allow_nan=False)
    if len(raw.encode("utf-8")) > MAX_RESPONSE_BYTES:
        raise RuntimeError("response budget exceeded")
    sys.stdout.write(raw + "\n")
    sys.stdout.flush()


def worker_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SI PCMA-v1 development-only world RPC")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--scenario", choices=("stable", "shift", "noisy", "long"),
                        default="stable")
    parser.add_argument("--split", choices=("development",), default="development")
    args = parser.parse_args(argv)
    # No `heldout` option exists, even when a caller bypasses the harness.
    world = OODRuleWorld(OODWorldConfig(args.seed, "development", args.scenario))
    started = False
    done = False
    closed = False
    for wire in sys.stdin.buffer:
        if len(wire) > MAX_REQUEST_BYTES or not wire.endswith(b"\n"):
            _response({"type": "error", "code": "INVALID_MESSAGE"})
            return 2
        try:
            request = json.loads(wire.decode("utf-8"))
        except (UnicodeError, ValueError):
            _response({"type": "error", "code": "INVALID_MESSAGE"})
            continue
        if type(request) is not dict or type(request.get("op")) is not str:
            _response({"type": "error", "code": "INVALID_MESSAGE"})
            continue
        op = request["op"]
        if op == "start" and set(request) == {"op"} and not started:
            started = True
            _response({"type": "observation", "observation": asdict(world.observe())})
        elif op == "step" and set(request) == {"op", "action"} and started and not closed:
            action = request["action"]
            if type(action) is not int or action not in range(4) or done:
                _response({"type": "error", "code": "INVALID_ACTION_OR_SESSION"})
                continue
            transition = world.step(action)
            done = transition.done
            _response({"type": "transition", "transition": asdict(transition)})
        elif op == "restart" and set(request) == {"op"} and started and done:
            done = False
            _response({"type": "observation",
                       "observation": asdict(world.restart_session())})
        elif op == "close" and set(request) == {"op"}:
            closed = True
            _response({"type": "closed"})
            return 0
        else:
            # Never expose error tracebacks, latent context, hidden rule or
            # private-world configuration in an RPC response.
            _response({"type": "error", "code": "INVALID_REQUEST_OR_PHASE"})
    return 0 if closed else 2


if __name__ == "__main__":
    raise SystemExit(worker_main())
