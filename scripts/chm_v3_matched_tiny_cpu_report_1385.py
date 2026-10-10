from __future__ import annotations

"""Issue #1385 scored CPU-only report. No model tuning, replays or GPU."""
import json
import hashlib
from pathlib import Path

from tam_research.chm_v3_matched_tiny_cpu_report_1385 import (
    SUCCESS_LABEL, run_cpu_one_shot_report,
)


def main() -> None:
    result = run_cpu_one_shot_report()
    if result["classification"] != SUCCESS_LABEL:
        raise RuntimeError("#1385 wrong synthetic engineering report classification")
    if result["gpu_used"] or result["new_scientific_attempt"]:
        raise RuntimeError("#1385 GPU/science authorization violation")
    if len(result["case_rows"]) != 192 or len(result["per_family"]) != 24:
        raise RuntimeError("#1385 incomplete independent test panel")
    path = Path("chm-v3-1385-cpu-only-scored-report.json")
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    path.write_text(encoded, encoding="utf-8")
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    compact = {
        "classification": result["classification"],
        "control_issue": result["control_issue"],
        "historical_scientific_classification": result["historical_scientific_classification"],
        "historical_scientific_seed_consumed": True,
        "report_sha256": digest,
        "training_steps_per_arm": result["optimizer_updates_per_arm"],
        "heldout_cases": len(result["case_rows"]),
        "panels": result["per_family"],
        "training_wall_seconds_total": result["training_wall_seconds_total"],
        "evaluation_wall_seconds_by_arm": result["evaluation_wall_seconds_by_arm"],
        "parameters": result["training"]["parameters"],
        "parameter_parity_within_1_percent": result["parameter_parity_within_1_percent"],
        "compute_parity_verified": result["compute_parity_verified"],
        "redacted_controls": result["redacted_controls"],
        "gpu_used": False,
        "scientific_quality_pass_authorized": False,
    }
    print("CHM_V3_1385_TINY_CPU_ONE_SHOT_SUMMARY=" + json.dumps(compact,sort_keys=True))


if __name__ == "__main__":
    main()
