from __future__ import annotations

"""CLI for the runtime-admission-aware Modal selector (#1067)."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tam_research"
    / "modal_dual_account_v2.py"
)
_MODULE_NAME = "_tam_research_modal_dual_account_v2_standalone"
_spec = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load Modal selector module from {_MODULE_PATH}")
_module = importlib.util.module_from_spec(_spec)
sys.modules[_MODULE_NAME] = _module
_spec.loader.exec_module(_module)
select_from_environment = _module.select_from_environment


def _write_github_output(values: dict[str, str]) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    with Path(output_path).open("a", encoding="utf-8") as handle:
        for key, value in values.items():
            if "\n" in value or "\r" in value:
                raise ValueError(f"GitHub output {key} must be single-line")
            handle.write(f"{key}={value}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--required-volume", action="append", default=[])
    parser.add_argument("--force-account", choices=("primary", "secondary"))
    parser.add_argument(
        "--low-credit-account", choices=("primary", "secondary", "none")
    )
    parser.add_argument(
        "--runtime-probe-path",
        default="modal_runtime_admission_probe_1067_v1.py",
    )
    args = parser.parse_args()

    selection = select_from_environment(
        required_volumes=args.required_volume,
        force_account=args.force_account,
        low_credit_account=args.low_credit_account,
        runtime_probe_path=args.runtime_probe_path,
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
