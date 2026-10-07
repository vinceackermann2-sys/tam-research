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
    AFM_SEQUENCE_MARGIN,
    ARMS,
    EXPECTED_AFM_PARAMETERS,
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

APP_NAME = "tam-research-cpw-v5-afm-1290"
VOLUME_NAME = "tam-research-data"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.7,<2.11",
        "numpy>=2.0,<3",
    )
    .add_local_python_source("tam_research")
    .add_local_python_source("architectures")
)


def _finite(result: dict) -> bool:
    values = [
        float(result["training_seconds"]),
        float(result["eval_seconds"]),
        float(result["total_compute_seconds"]),
        float(result["examples_per_second"]),
        float(result["peak_vram_gb"]),
        float(result["last_train_query_loss"]),
        float(result["evaluation"]["long_mean_accuracy"]),
        float(result["evaluation"]["long_mean_nll"]),
    ]
    for row in result["evaluation"]["by_distance"].values():
        values.extend((float(row["accuracy"]), float(row["nll"])))
    return all(math.isfinite(v) and v >= 0.0 for v in values)


def _summary(result: dict) -> dict:
    out = {
        "arm": result["arm"],
        "seed": int(result["seed"]),
        "parameters": int(result["parameters"]),
        "steps": int(result["steps"]),
        "examples_seen": int(result["examples_seen"]),
        "training_seconds": float(result["training_seconds"]),
        "total_compute_seconds": float(result["total_compute_seconds"]),
        "examples_per_second": float(result["examples_per_second"]),
        "peak_vram_gb": float(result["peak_vram_gb"]),
        "evaluation": result["evaluation"],
    }
    if "memory_stats" in result:
        out["memory_stats"] = result["memory_stats"]
    return out


def _mean(rows: list[dict], arm: str, key: str) -> float:
    return sum(float(row[arm]["evaluation"][key]) for row in rows) / len(rows)


def _distance_mean(rows: list[dict], arm: str, distance: int, metric: str) -> float:
    d = str(distance)
    return sum(
        float(row[arm]["evaluation"]["by_distance"][d][metric])
        for row in rows
    ) / len(rows)


def _seed_gate(row: dict) -> dict:
    afm = row["afm_last1"]["evaluation"]
    sequence = row["sequence_only"]["evaluation"]
    distances = ("64", "128", "256")
    accuracy_margin_ok = all(
        float(afm["by_distance"][d]["accuracy"])
        - float(sequence["by_distance"][d]["accuracy"])
        >= AFM_SEQUENCE_MARGIN
        for d in distances
    )
    nll_ok = all(
        float(afm["by_distance"][d]["nll"])
        < float(sequence["by_distance"][d]["nll"])
        for d in distances
    )
    afm_accuracy_ok = (
        float(afm["long_mean_accuracy"]) >= AFM_LONG_MEAN_ACCURACY
        and float(afm["by_distance"]["256"]["accuracy"])
        >= AFM_DISTANCE_256_ACCURACY
    )
    supported = afm_accuracy_ok and accuracy_margin_ok and nll_ok
    return {
        "seed": int(row["afm_last1"]["seed"]),
        "supported": supported,
        "afm_accuracy_ok": afm_accuracy_ok,
        "accuracy_margin_ok": accuracy_margin_ok,
        "nll_ok": nll_ok,
        "sequence_long_mean_accuracy": float(sequence["long_mean_accuracy"]),
        "afm_long_mean_accuracy": float(afm["long_mean_accuracy"]),
        "afm_distance_256_accuracy": float(
            afm["by_distance"]["256"]["accuracy"]
        ),
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
        raise RuntimeError(f"invalid source SHA: {source_sha!r}")

    root = Path(RESULT_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    attempt_path = root / "ATTEMPT.json"
    result_path = root / "RESULT.json"
    failure_path = root / "FAILURE.json"
    if attempt_path.exists() or result_path.exists() or failure_path.exists():
        raise RuntimeError("CPW-v5 AFM namespace already consumed")

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
        "language_panel_authorized": False,
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

        smoke_integrity = (
            all(_finite(x) for x in smoke_raw.values())
            and int(smoke_raw["transformer"]["parameters"]) == TRANSFORMER_PARAMETERS
            and int(smoke_raw["sequence_only"]["parameters"]) == SEQUENCE_PARAMETERS
            and int(smoke_raw["afm_last1"]["parameters"]) == EXPECTED_AFM_PARAMETERS
        )
        smoke = {arm: _summary(x) for arm, x in smoke_raw.items()}
        print(
            "CPW_V5_AFM_SMOKE="
            + json.dumps({"pass": smoke_integrity, "arms": smoke}, sort_keys=True),
            flush=True,
        )

        replication: list[dict] = []
        if smoke_integrity:
            for seed in REPLICATION_SEEDS:
                print(
                    "CPW_V5_AFM_SEED_START="
                    + json.dumps({"seed": seed}, sort_keys=True),
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
                        "CPW_V5_AFM_ARM="
                        + json.dumps(
                            {"seed": seed, "arm": arm, "summary": row[arm]},
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                replication.append(row)

        integrity = (
            smoke_integrity
            and len(replication) == len(REPLICATION_SEEDS)
            and all(
                row["transformer"]["parameters"] == TRANSFORMER_PARAMETERS
                and row["sequence_only"]["parameters"] == SEQUENCE_PARAMETERS
                and row["afm_last1"]["parameters"] == EXPECTED_AFM_PARAMETERS
                for row in replication
            )
        )

        if integrity:
            transformer_long = _mean(
                replication, "transformer", "long_mean_accuracy"
            )
            transformer_256 = _distance_mean(
                replication, "transformer", 256, "accuracy"
            )
            sequence_long = _mean(
                replication, "sequence_only", "long_mean_accuracy"
            )
            afm_long = _mean(replication, "afm_last1", "long_mean_accuracy")
            afm_256 = _distance_mean(
                replication, "afm_last1", 256, "accuracy"
            )
            transformer_learns = (
                transformer_long >= TRANSFORMER_VALID_LONG_MEAN_ACCURACY
                and transformer_256 >= TRANSFORMER_VALID_DISTANCE_256_ACCURACY
            )
            leakage_alarm = (
                sequence_long > SEQUENCE_LEAKAGE_ALARM_LONG_ACCURACY
            )
            seed_gates = [_seed_gate(row) for row in replication]
            supported_seeds = sum(bool(g["supported"]) for g in seed_gates)
            memory_supported = (
                transformer_learns
                and not leakage_alarm
                and afm_long >= AFM_LONG_MEAN_ACCURACY
                and afm_256 >= AFM_DISTANCE_256_ACCURACY
                and supported_seeds >= 2
            )
            strong_supported = (
                memory_supported
                and supported_seeds == len(REPLICATION_SEEDS)
                and abs(afm_long - transformer_long) <= 0.05
            )
            if not transformer_learns or leakage_alarm:
                classification = INVALID_CLASSIFICATION
            elif strong_supported:
                classification = STRONG_CLASSIFICATION
            elif memory_supported:
                classification = SUPPORTED_CLASSIFICATION
            else:
                classification = STOP_CLASSIFICATION

            afm_tps_ratio = sum(
                row["afm_last1"]["examples_per_second"]
                / max(row["transformer"]["examples_per_second"], 1e-12)
                for row in replication
            ) / len(replication)
            afm_train_ratio = sum(
                row["afm_last1"]["training_seconds"]
                / max(row["transformer"]["training_seconds"], 1e-12)
                for row in replication
            ) / len(replication)
            afm_compute_ratio = sum(
                row["afm_last1"]["total_compute_seconds"]
                / max(row["transformer"]["total_compute_seconds"], 1e-12)
                for row in replication
            ) / len(replication)
        else:
            transformer_long = transformer_256 = None
            sequence_long = afm_long = afm_256 = None
            transformer_learns = leakage_alarm = False
            seed_gates = []
            supported_seeds = 0
            memory_supported = strong_supported = False
            afm_tps_ratio = afm_train_ratio = afm_compute_ratio = None
            classification = STOP_CLASSIFICATION

        aggregate = {
            "transformer_long_mean_accuracy": transformer_long,
            "transformer_distance_256_accuracy": transformer_256,
            "sequence_long_mean_accuracy": sequence_long,
            "afm_long_mean_accuracy": afm_long,
            "afm_distance_256_accuracy": afm_256,
            "transformer_learns": transformer_learns,
            "sequence_leakage_alarm": leakage_alarm,
            "supported_seed_count": supported_seeds,
            "memory_supported": memory_supported,
            "strong_memory_supported": strong_supported,
            "seed_gates": seed_gates,
            "afm_examples_per_second_ratio": afm_tps_ratio,
            "afm_training_seconds_ratio": afm_train_ratio,
            "afm_total_compute_ratio": afm_compute_ratio,
            "afm_parameter_fraction": (
                EXPECTED_AFM_PARAMETERS / TRANSFORMER_PARAMETERS
            ),
        }
        if integrity:
            aggregate["distance_means"] = {
                arm: {
                    str(distance): {
                        "accuracy": _distance_mean(
                            replication, arm, distance, "accuracy"
                        ),
                        "nll": _distance_mean(
                            replication, arm, distance, "nll"
                        ),
                    }
                    for distance in (32, 64, 128, 256)
                }
                for arm in ARMS
            }

        final = {
            "issue": ISSUE,
            "source_sha": source_sha,
            "classification": classification,
            "smoke_pass": smoke_integrity,
            "replication_integrity": integrity,
            "aggregate": aggregate,
            "replication": replication,
            "breakthrough_claim_allowed": False,
            "scale_up_authorized": False,
            "language_panel_authorized": False,
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
