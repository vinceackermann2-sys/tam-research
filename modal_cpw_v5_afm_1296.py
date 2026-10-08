from __future__ import annotations

import json
import math
from pathlib import Path
import re
import time

import modal

from tam_research.cpw_v5_afm.protocol import (
    AFM_DISTANCE_256_ACCURACY,
    AFM_LONG_MEAN_ACCURACY,
    AFM_PARAMETERS,
    AFM_SEQUENCE_MARGIN,
    ARMS,
    INVALID_CLASSIFICATION,
    ISSUE,
    REPLICATION_EVAL_EXAMPLES_PER_DISTANCE,
    REPLICATION_SEEDS,
    REPLICATION_STEPS,
    RESULT_ROOT,
    SEQUENCE_LEAKAGE_ALARM_LONG_ACCURACY,
    SEQUENCE_PARAMETERS,
    SMOKE_EVAL_EXAMPLES_PER_DISTANCE,
    SMOKE_SEED,
    SMOKE_STEPS,
    STOP_CLASSIFICATION,
    STRONG_CLASSIFICATION,
    SUPPORTED_CLASSIFICATION,
    TRANSFORMER_PARAMETERS,
    TRANSFORMER_VALID_DISTANCE_256_ACCURACY,
    TRANSFORMER_VALID_LONG_MEAN_ACCURACY,
)

APP_NAME = "tam-research-cpw-v5-afm-1296"
VOLUME_NAME = "tam-research-data"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.7,<2.11", "numpy>=2.0,<3")
    .add_local_python_source("tam_research")
    .add_local_python_source("architectures")
)


def _finite_result(result: dict) -> bool:
    numbers = [
        float(result["training_seconds"]),
        float(result["total_compute_seconds"]),
        float(result["examples_per_second"]),
        float(result["peak_vram_gb"]),
        float(result["last_train_query_loss"]),
        float(result["evaluation"]["long_mean_accuracy"]),
        float(result["evaluation"]["long_mean_nll"]),
    ]
    for row in result["evaluation"]["by_distance"].values():
        numbers += [float(row["accuracy"]), float(row["nll"])]
    return all(math.isfinite(v) and v >= 0 for v in numbers)


def _summary(result: dict) -> dict:
    return {
        "arm": result["arm"],
        "seed": int(result["seed"]),
        "parameters": int(result["parameters"]),
        "steps": int(result["steps"]),
        "examples_seen": int(result["examples_seen"]),
        "training_seconds": float(result["training_seconds"]),
        "total_compute_seconds": float(result["total_compute_seconds"]),
        "examples_per_second": float(result["examples_per_second"]),
        "peak_vram_gb": float(result["peak_vram_gb"]),
        "last_train_query_loss": float(result["last_train_query_loss"]),
        "evaluation": result["evaluation"],
    }


def _seed_gate(row: dict) -> dict:
    afm = row["afm_first1"]["evaluation"]
    sequence = row["sequence_only"]["evaluation"]
    transformer = row["transformer"]["evaluation"]
    long_acc = float(afm["long_mean_accuracy"])
    sequence_acc = float(sequence["long_mean_accuracy"])
    distance_256_acc = float(afm["by_distance"]["256"]["accuracy"])
    transformer_valid = (
        float(transformer["long_mean_accuracy"]) >= TRANSFORMER_VALID_LONG_MEAN_ACCURACY
        and float(transformer["by_distance"]["256"]["accuracy"])
        >= TRANSFORMER_VALID_DISTANCE_256_ACCURACY
    )
    leakage_negative = sequence_acc < SEQUENCE_LEAKAGE_ALARM_LONG_ACCURACY
    supported = (
        transformer_valid
        and leakage_negative
        and long_acc >= AFM_LONG_MEAN_ACCURACY
        and distance_256_acc >= AFM_DISTANCE_256_ACCURACY
        and long_acc - sequence_acc >= AFM_SEQUENCE_MARGIN
    )
    return {
        "seed": int(row["afm_first1"]["seed"]),
        "supported": supported,
        "transformer_valid": transformer_valid,
        "sequence_leakage_negative": leakage_negative,
        "transformer_long_mean_accuracy": float(transformer["long_mean_accuracy"]),
        "sequence_long_mean_accuracy": sequence_acc,
        "afm_long_mean_accuracy": long_acc,
        "afm_distance_256_accuracy": distance_256_acc,
        "afm_minus_sequence_accuracy": long_acc - sequence_acc,
    }


@app.function(
    image=image,
    gpu="H100!",
    cpu=8,
    memory=32768,
    timeout=3 * 60 * 60,
    retries=0,
    volumes={"/vol": volume},
)
def run_benchmark(source_sha: str) -> dict:
    from tam_research.cpw_v5_afm.train import train_arm

    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise RuntimeError("source SHA must be exact 40-character lowercase hex")

    root = Path(RESULT_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    attempt_path = root / "ATTEMPT.json"
    result_path = root / "RESULT.json"
    failure_path = root / "FAILURE.json"

    if attempt_path.exists() or result_path.exists() or failure_path.exists():
        raise RuntimeError("CPW-v5 AFM namespace already consumed")
    if any(root.iterdir()):
        raise RuntimeError("CPW-v5 AFM namespace contains unexpected prior artifacts")

    attempt = {
        "issue": ISSUE,
        "source_sha": source_sha,
        "arms": list(ARMS),
        "smoke_seed": SMOKE_SEED,
        "replication_seeds": list(REPLICATION_SEEDS),
        "smoke_steps": SMOKE_STEPS,
        "replication_steps": REPLICATION_STEPS,
        "gpu": "H100!",
        "retry_authorized": False,
        "resume_authorized": False,
        "checkpoint_reuse_authorized": False,
        "breakthrough_claim_allowed": False,
        "scale_up_authorized": False,
        "started_unix": time.time(),
    }
    attempt_path.write_text(json.dumps(attempt, indent=2), encoding="utf-8")
    volume.commit()
    print("CPW_V5_AFM_ATTEMPT=" + json.dumps(attempt, sort_keys=True), flush=True)

    try:
        smoke_raw: dict[str, dict] = {}
        for arm in ARMS:
            smoke_raw[arm] = train_arm(
                arm=arm,
                seed=SMOKE_SEED,
                run_root=str(root / "smoke"),
                steps=SMOKE_STEPS,
                eval_examples_per_distance=SMOKE_EVAL_EXAMPLES_PER_DISTANCE,
            )
            volume.commit()

        smoke_ok = (
            all(_finite_result(x) for x in smoke_raw.values())
            and int(smoke_raw["transformer"]["parameters"]) == TRANSFORMER_PARAMETERS
            and int(smoke_raw["sequence_only"]["parameters"]) == SEQUENCE_PARAMETERS
            and int(smoke_raw["afm_first1"]["parameters"]) == AFM_PARAMETERS
        )
        print(
            "CPW_V5_AFM_SMOKE="
            + json.dumps({
                "pass": smoke_ok,
                "arms": {arm: _summary(x) for arm, x in smoke_raw.items()},
            }, sort_keys=True),
            flush=True,
        )

        replication: list[dict] = []
        if smoke_ok:
            for seed in REPLICATION_SEEDS:
                print(
                    "CPW_V5_AFM_SEED_START=" + json.dumps({"seed": seed}),
                    flush=True,
                )
                row: dict[str, dict] = {}
                for arm in ARMS:
                    raw = train_arm(
                        arm=arm,
                        seed=seed,
                        run_root=str(root / "replication" / f"seed-{seed}"),
                        steps=REPLICATION_STEPS,
                        eval_examples_per_distance=(
                            REPLICATION_EVAL_EXAMPLES_PER_DISTANCE
                        ),
                    )
                    volume.commit()
                    row[arm] = _summary(raw)
                    print(
                        "CPW_V5_AFM_ARM=" + json.dumps({
                            "seed": seed, "arm": arm, "summary": row[arm]
                        }, sort_keys=True),
                        flush=True,
                    )
                replication.append(row)

        integrity = (
            smoke_ok
            and len(replication) == len(REPLICATION_SEEDS)
            and all(
                row["transformer"]["parameters"] == TRANSFORMER_PARAMETERS
                and row["sequence_only"]["parameters"] == SEQUENCE_PARAMETERS
                and row["afm_first1"]["parameters"] == AFM_PARAMETERS
                and all(_finite_result(row[arm]) for arm in ARMS)
                for row in replication
            )
        )

        if integrity:
            gates = [_seed_gate(row) for row in replication]
            supported = sum(g["supported"] for g in gates)
            transformer_valid = all(g["transformer_valid"] for g in gates)
            leakage_negative = all(g["sequence_leakage_negative"] for g in gates)
            memory_supported = transformer_valid and leakage_negative and supported >= 2
            strong_supported = memory_supported and supported == len(REPLICATION_SEEDS)
            if not transformer_valid or not leakage_negative:
                classification = INVALID_CLASSIFICATION
            elif strong_supported:
                classification = STRONG_CLASSIFICATION
            elif memory_supported:
                classification = SUPPORTED_CLASSIFICATION
            else:
                classification = STOP_CLASSIFICATION
            means = {
                arm: {
                    str(distance): {
                        metric: sum(
                            float(row[arm]["evaluation"]["by_distance"][str(distance)][metric])
                            for row in replication
                        ) / len(replication)
                        for metric in ("accuracy", "nll")
                    }
                    for distance in (32, 64, 128, 256)
                }
                for arm in ARMS
            }
        else:
            gates = []
            supported = 0
            transformer_valid = False
            leakage_negative = False
            memory_supported = False
            strong_supported = False
            means = {}
            classification = STOP_CLASSIFICATION

        final = {
            "issue": ISSUE,
            "source_sha": source_sha,
            "classification": classification,
            "smoke_pass": smoke_ok,
            "replication_integrity": integrity,
            "supported_seed_count": supported,
            "transformer_valid": transformer_valid,
            "sequence_leakage_negative": leakage_negative,
            "memory_supported": memory_supported,
            "strong_memory_supported": strong_supported,
            "seed_gates": gates,
            "distance_means": means,
            "replication": replication,
            "breakthrough_claim_allowed": False,
            "scale_up_authorized": False,
            "completed_unix": time.time(),
        }
        result_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
        volume.commit()
        print("CPW_V5_AFM_RESULT=" + json.dumps(final, sort_keys=True), flush=True)
        return final

    except BaseException as exc:
        failure = {
            "issue": ISSUE,
            "source_sha": source_sha,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "scientific_interpretation": False,
            "retry_authorized": False,
            "resume_authorized": False,
            "breakthrough_claim_allowed": False,
            "failed_unix": time.time(),
        }
        failure_path.write_text(json.dumps(failure, indent=2), encoding="utf-8")
        volume.commit()
        print("CPW_V5_AFM_FAILURE=" + json.dumps(failure, sort_keys=True), flush=True)
        raise


@app.local_entrypoint()
def main(source_sha: str) -> None:
    result = run_benchmark.remote(source_sha)
    print("CPW_V5_AFM_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
