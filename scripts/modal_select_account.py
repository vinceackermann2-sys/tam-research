from __future__ import annotations

"""CLI wrapper for tam_research.modal_dual_account (#1031)."""

import argparse
import json
import os
from pathlib import Path

from tam_research.modal_dual_account import select_from_environment


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
