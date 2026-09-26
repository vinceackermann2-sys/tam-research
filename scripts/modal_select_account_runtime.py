from __future__ import annotations

"""Standalone runtime-admission-aware Modal account selector (#1067)."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SELECTOR_PATH = ROOT / "tam_research" / "modal_dual_account_runtime.py"
PROBE_PATH = ROOT / "scripts" / "modal_runtime_admission_probe.py"

SELECTOR_BLOB = "e3489db84794471bb3ef00f1938939e92d0183a9"
PROBE_BLOB = "f758544147d365e1e16019a37edad2d558697b2b"


def _git_blob_sha(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _verify(path: Path, expected: str, name: str) -> None:
    actual = _git_blob_sha(path.read_bytes())
    if actual != expected:
        raise RuntimeError(f"#1067 {name} blob drift: {actual} != {expected}")


_verify(SELECTOR_PATH, SELECTOR_BLOB, "runtime selector")
_verify(PROBE_PATH, PROBE_BLOB, "runtime admission probe")

_MODULE_NAME = "_tam_modal_dual_account_runtime_standalone"
_spec = importlib.util.spec_from_file_location(_MODULE_NAME, SELECTOR_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load runtime selector from {SELECTOR_PATH}")
_module = importlib.util.module_from_spec(_spec)
sys.modules[_MODULE_NAME] = _module
_spec.loader.exec_module(_module)
select_from_environment = _module.select_from_environment


def _write_github_output(values: dict[str, str]) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    path = Path(output_path)
    with path.open("a", encoding="utf-8") as handle:
        for key, value in values.items():
            if "\n" in value or "\r" in value:
                raise ValueError(f"GitHub output {key} must be single-line")
            handle.write(f"{key}={value}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--required-volume", action="append", default=[])
    parser.add_argument("--force-account", choices=("primary", "secondary"))
    parser.add_argument("--low-credit-account", choices=("primary", "secondary", "none"))
    args = parser.parse_args()

    selection = select_from_environment(
        required_volumes=args.required_volume,
        force_account=args.force_account,
        low_credit_account=args.low_credit_account,
    )
    evidence = selection.public_evidence()
    compact = json.dumps(evidence, sort_keys=True, separators=(",", ":"))
    print(compact)
    _write_github_output(
        {
            "selected_account": selection.selected_account,
            "selected_workspace_name": selection.selected_workspace_name or "",
            "selection_reason": selection.reason,
            "selection_evidence_json": compact,
        }
    )


if __name__ == "__main__":
    main()
