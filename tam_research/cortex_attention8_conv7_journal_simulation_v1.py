"""Local synthetic-only journal. NO Modal, GPU, seeds, or real reservation."""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any
from tam_research.cortex_attention8_conv7_dispatch_dryrun_v1 import MODELS, preflight_dry_run

MARK = "SYNTHETIC_CONV7_JOURNAL_V1_NO_REAL_RESERVATION"


class SyntheticJournal:
    """Exclusive immutable local markers; only for CPU sandbox experiments."""
    def __init__(self, directory: Path):
        self.root = Path(directory)
        if not self.root.is_dir() or self.root.is_symlink():
            raise ValueError("sandbox must already be an ordinary directory")

    def _path(self, model: str | None, action: str) -> Path:
        if model is None and action == "reserve":
            return self.root / "reservation.json"
        if model not in MODELS or action not in ("start", "terminal"):
            raise ValueError("unknown simulated model/action")
        return self.root / f"{MODELS.index(model):02d}-{action}.json"

    def _read(self, model: str | None, action: str) -> dict[str, Any] | None:
        path = self._path(model, action)
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError) as exc:
            raise RuntimeError("corrupt or partial synthetic marker") from exc
        keys = {"mode", "model", "action", "paid_authority"}
        if action == "terminal":
            keys.add("outcome")
        if not isinstance(record, dict) or set(record) != keys:
            raise RuntimeError("invalid synthetic marker schema")
        if record["mode"] != MARK or record["model"] != model or record["action"] != action:
            raise RuntimeError("invalid synthetic marker identity")
        if record["paid_authority"] is not False:
            raise RuntimeError("synthetic marker cannot authorize GPU execution")
        if action == "terminal" and record["outcome"] not in ("COMPLETE", "ERROR"):
            raise RuntimeError("unknown synthetic terminal outcome")
        return record

    def _create_exclusive(self, model: str | None, action: str, outcome: str | None = None) -> None:
        record = {"mode": MARK, "model": model, "action": action, "paid_authority": False}
        if action == "terminal":
            if outcome not in ("COMPLETE", "ERROR"):
                raise ValueError("bad simulated terminal outcome")
            record["outcome"] = outcome
        payload = (json.dumps(record, sort_keys=True) + "\n").encode()
        path = self._path(model, action)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise RuntimeError("synthetic attempt marker already consumed") from exc
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            dfd = os.open(self.root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except BaseException:
            # Do not remove partial markers: interrupted writes must block replay.
            raise

    def snapshot(self) -> dict[str, Any]:
        allowed = {self._path(None, "reserve").name}
        for name in MODELS:
            allowed.add(self._path(name, "start").name)
            allowed.add(self._path(name, "terminal").name)
        if any(p.name not in allowed for p in self.root.iterdir()):
            raise RuntimeError("unexpected file in journal")
        statuses = []
        for model in MODELS:
            start = self._read(model, "start")
            terminal = self._read(model, "terminal")
            if terminal and not start:
                raise RuntimeError("completion before attempt start")
            statuses.append(terminal["outcome"] if terminal else "STARTED" if start else "NONE")
        reserved = self._read(None, "reserve")
        if reserved is None:
            if any(s != "NONE" for s in statuses):
                raise RuntimeError("attempt without reservation")
            state = "UNRESERVED"
        else:
            state = "SYNTHETIC_ALL_COMPLETE_NOT_EVIDENCE"
            for i, status in enumerate(statuses):
                if i and statuses[i-1] != "COMPLETE" and status != "NONE":
                    raise RuntimeError("out of order, resumed, or post-failure attempt")
                if status == "ERROR":
                    state = "TERMINAL_FAILURE_NO_RETRY"
                    break
                if status == "STARTED":
                    state = "INCOMPLETE_START_NO_REPLAY"
                    break
                if status == "NONE":
                    state = "READY_FOR_NEXT_SIMULATED_MODEL"
                    break
        return {"state": state, "statuses": dict(zip(MODELS, statuses)),
                "synthetic_only": True, "real_reservation": False, "paid_authority": False,
                "real_seed_consumed": False, "gpu_allocated": False, "scientific_evidence": False}

    def reserve(self) -> None:
        verified = preflight_dry_run()
        if verified["classification"] != "CONV7_DISPATCH_DRY_RUN_READY_REAL_DISPATCH_BLOCKED":
            raise RuntimeError("existing preflight no longer blocks paid dispatch")
        if self.snapshot()["state"] != "UNRESERVED":
            raise RuntimeError("reservation already issued")
        self._create_exclusive(None, "reserve")

    def start(self, model: str) -> None:
        if model not in MODELS:
            raise ValueError("unknown model")
        report = self.snapshot()
        i = MODELS.index(model)
        if report["state"] != "READY_FOR_NEXT_SIMULATED_MODEL":
            raise RuntimeError("simulated journal not ready")
        if report["statuses"][model] != "NONE":
            raise RuntimeError("model already attempted")
        if any(report["statuses"][name] != "COMPLETE" for name in MODELS[:i]):
            raise RuntimeError("earlier model has not completed")
        if any(report["statuses"][name] != "NONE" for name in MODELS[i+1:]):
            raise RuntimeError("later model already attempted")
        self._create_exclusive(model, "start")

    def finish(self, model: str, outcome: str) -> None:
        if model not in MODELS or outcome not in ("COMPLETE", "ERROR"):
            raise ValueError("unknown model or outcome")
        report = self.snapshot()
        if report["state"] != "INCOMPLETE_START_NO_REPLAY" or report["statuses"][model] != "STARTED":
            raise RuntimeError("cannot finish a nonactive attempt")
        self._create_exclusive(model, "terminal", outcome)
