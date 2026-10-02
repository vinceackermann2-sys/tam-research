from __future__ import annotations

"""Read-only public health snapshot for both configured Modal accounts."""

import importlib.util
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tam_research" / "modal_dual_account_v3.py"
MODULE_NAME = "_tam_research_modal_dual_account_v3_health_snapshot"

_spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load Modal selector module from {MODULE_PATH}")
_module = importlib.util.module_from_spec(_spec)
sys.modules[MODULE_NAME] = _module
_spec.loader.exec_module(_module)

probe_account = _module.probe_account
resolve_secondary_credentials = _module.resolve_secondary_credentials

REQUIRED_VOLUME = "tam-research-data"
RUNTIME_PROBE_PATH = "modal_runtime_admission_probe_1067_v1.py"


def _github_output(name: str, value: str) -> None:
    output = os.environ.get("GITHUB_OUTPUT")
    if not output:
        return
    if "\n" in value or "\r" in value:
        raise ValueError("GitHub output must be single-line")
    with Path(output).open("a", encoding="utf-8") as handle:
        handle.write(f"{name}={value}\n")


def main() -> None:
    env = os.environ
    secondary_id, secondary_secret, aliases = resolve_secondary_credentials(env)

    primary = probe_account(
        label="primary",
        token_id=env.get("MODAL_TOKEN_ID"),
        token_secret=env.get("MODAL_TOKEN_SECRET"),
        required_volumes=(REQUIRED_VOLUME,),
        runtime_probe_path=RUNTIME_PROBE_PATH,
    )
    secondary = probe_account(
        label="secondary",
        token_id=secondary_id,
        token_secret=secondary_secret,
        required_volumes=(REQUIRED_VOLUME,),
        runtime_probe_path=RUNTIME_PROBE_PATH,
    )

    payload = {
        "classification": "ZERO_GPU_MODAL_ACCOUNT_HEALTH_SNAPSHOT_V1",
        "required_volume": REQUIRED_VOLUME,
        "runtime_probe_path": RUNTIME_PROBE_PATH,
        "primary": primary.public_evidence(),
        "secondary": {
            **secondary.public_evidence(),
            "credential_aliases": list(aliases),
        },
        "gpu_allocated": False,
        "writes_performed": False,
        "seed_consumed": False,
    }
    compact = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    print(compact)
    _github_output("snapshot_json", compact)


if __name__ == "__main__":
    main()
