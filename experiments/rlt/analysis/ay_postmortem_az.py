"""AZ: read-only forensic audit of frozen AW/AY CPU evidence.

This tool NEVER trains, loads a checkpoint, or calls a GPU. Results are derived
only from immutable committed JSON and tracked-file inventory. AY .pt checkpoint
files were created on the runner but absent from the final commit due to the
repository's *.pt ignore rule. Do not rerun consumed AY to recover them.
"""
from __future__ import annotations

import fnmatch
import json
from pathlib import Path
import subprocess


BASE = Path("experiments/rlt")
AW_RESULT = BASE / "cpu/results/rlt-bounded-streaming-vs-reset-20261010-aw.json"
AY_RESULT = BASE / "cpu/results/rlt-bounded-threeway-20261010-ay.json"
AY_JOB = BASE / "cpu/jobs/rlt-bounded-threeway-20261010-ay.json"
AY_CLAIM = BASE / "cpu/claims/rlt-bounded-threeway-20261010-ay.json"
CHECKPOINTS = BASE / "cpu/checkpoints/rlt-bounded-threeway-20261010-ay"
GAPS = ("1", "4", "16", "64", "127")
LONG_GAPS = ("16", "64", "127")


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def git_tracked(path: Path) -> bool:
    r = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", str(path)],
        check=False, capture_output=True, text=True,
    )
    return r.returncode == 0


def audit() -> dict:
    aw, ay = read(AW_RESULT), read(AY_RESULT)
    job, claim = read(AY_JOB), read(AY_CLAIM)
    assert aw["status"] == ay["status"] == "complete"
    assert aw["job_id"] == "rlt-cpu-bounded-streaming-vs-reset-20261010-aw"
    assert ay["job_id"] == job["job_id"] == claim["job_id"] == "rlt-cpu-bounded-threeway-20261010-ay"
    assert ay["scientific_execution"] is False and ay["gpu_requested"] is False
    assert ay["breakthrough_claim_supported"] is False
    assert ay["repository_provenance"]["trigger_sha"] == "633ad44b3e94b9fed2e75ffdf8fa806c7ed4798f"
    assert job["runner_blob_sha"] == ay["repository_provenance"]["runner_blob_sha"]
    assert [x["name"] for x in ay["results"]] == ["carry", "reset", "sliding_kv"]
    assert job["expected_parameters"] == {"carry": 27712, "reset": 27712, "sliding_kv": 27680}
    assert claim["automatic_retry"] is False
    ignore_rules = Path(".gitignore").read_text().splitlines()
    ignore_pt = any(line.strip() == "*.pt" for line in ignore_rules)

    aw_carry = next(x for x in aw["results"] if x["name"] == "streaming_carry")
    candidates = {x["name"]: x for x in ay["results"]}
    summaries = {}
    for name, row in candidates.items():
        final = row["final"]
        assert set(final) == set(GAPS)
        long_gaps = {}
        for d in LONG_GAPS:
            f = final[d]
            assert f["examples"] == 256
            long_gaps[d] = {
                "accuracy": f["accuracy"],
                "binary_nll": f["binary_nll"],
                "class0_accuracy": f["class0_accuracy"],
                "class1_accuracy": f["class1_accuracy"],
                "majority_class_collapse": (
                    (f["class0_accuracy"] == 1 and f["class1_accuracy"] == 0)
                    or (f["class0_accuracy"] == 0 and f["class1_accuracy"] == 1)
                ),
            }
        summaries[name] = {
            "parameters": row["parameters"],
            "accuracy_by_distance": {k: final[k]["accuracy"] for k in GAPS},
            "long_distance": long_gaps,
            "steps": row["long_training"]["steps"],
            "tokens_seen": row["long_training"]["tokens_seen"],
            "tokens_per_second": row["long_training"]["tokens_per_second"],
            "long_seconds": row["long_training"]["seconds"],
        }

    missing = []
    for row in candidates.values():
        p = Path(row["checkpoint"]["path"])
        if not git_tracked(p) or not p.is_file():
            missing.append(str(p))
    ratio = summaries["sliding_kv"]["tokens_per_second"] / summaries["carry"]["tokens_per_second"]
    aw_to_ay = {
        d: {
            "AW_carry_accuracy": aw_carry["final"][d]["accuracy"],
            "AY_carry_accuracy": candidates["carry"]["final"][d]["accuracy"],
            "percentage_point_change": 100.0 * (
                candidates["carry"]["final"][d]["accuracy"]
                - aw_carry["final"][d]["accuracy"]
            ),
        } for d in LONG_GAPS
    }
    direct_retention = {
        d: {
            cls: ay["carry_retention_diagnostics_after_ay_training"][f"{d}-class{cls}"]["mean_direct_retention_over_gap"]
            for cls in (0, 1)
        } for d in LONG_GAPS
    }
    return {
        "audit": "AZ-readonly-AW-AY-20261010",
        "gpu_used": False,
        "new_training": False,
        "frozen_AY_run_id": 38073108161,
        "ay_metrics_persisted": True,
        "checkpoints_in_repo": len(missing) == 0,
        "missing_checkpoints": missing,
        "pt_gitignore_rule_active": ignore_pt,
        "checkpoint_recovery_by_rerun_prohibited": True,
        "transformer_to_carry_cpu_throughput_ratio": ratio,
        "models": summaries,
        "AW_vs_AY_carry": aw_to_ay,
        "AY_carry_direct_retention": direct_retention,
        "limits": [
            "AW/AY use different model seeds/data and are engineering screens, not a preregistered scientific replication",
            "Two-layer eight-token window has 14-token full receptive distance; distant-set tests are provably inaccessible to AX",
            "AY .pt files were omitted from git by repository ignore rule; no downloadable artifact was uploaded",
            "Recorded gate summaries are valid but trained-state causal interventions cannot be conducted without checkpoints",
            "Equal wall time is not equal tokens or FLOPs; no language-model superiority claim",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
