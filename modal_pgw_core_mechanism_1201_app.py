from __future__ import annotations

import json
import math
from pathlib import Path
import time

import modal

from tam_research.pgw_core_mechanism.protocol import (
    ARMS,
    EFFECT_THRESHOLD_NLL,
    EXPECTED_PARAMETERS,
    ISSUE,
    REPLICATION_SEEDS,
    REPLICATION_TOKENS,
    RESULT_ROOT,
    SEQ_LEN,
    MICRO_BATCH_SIZE,
    GRAD_ACCUM_STEPS,
    SMOKE_SEED,
    SMOKE_TOKENS,
)

APP_NAME = "tam-research-pgw-core-mechanism-1201"
VOLUME_NAME = "tam-research-data"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.7,<2.11",
        "datasets>=4.0,<5",
        "transformers>=4.55,<5",
        "tokenizers>=0.21,<1",
        "numpy>=2.0,<3",
        "huggingface-hub>=0.34,<1",
    )
    .add_local_python_source("tam_research")
)


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=60 * 60,
    volumes={"/vol": volume},
)
def ensure_data() -> dict:
    from tam_research.data import prepare_fineweb

    result = prepare_fineweb(
        "/vol/data/fineweb-edu-gpt2",
        train_tokens=25_000_000,
        val_tokens=2_000_000,
    )
    volume.commit()
    print(
        "PGW_CORE_MECHANISM_DATA_GUARD=" + json.dumps(result, sort_keys=True),
        flush=True,
    )
    return result


def _finite(result: dict) -> bool:
    ev = result["final_eval"]
    values = [
        float(ev["nll"]),
        float(ev["perplexity"]),
        float(result["training_tokens_per_second"]),
        float(result["peak_vram_gb"]),
    ]
    return all(math.isfinite(value) for value in values)


def _selected_fraction(result: dict) -> float | None:
    router = result["final_eval"].get("router")
    if not router:
        return None
    return float(router["mean"]["selected_fraction"])


def _summary(result: dict) -> dict:
    return {
        "seed": int(result["seed"]),
        "nll": float(result["final_eval"]["nll"]),
        "perplexity": float(result["final_eval"]["perplexity"]),
        "training_tps": float(result["training_tokens_per_second"]),
        "peak_vram_gib": float(result["peak_vram_gb"]),
        "parameters": int(result["parameters"]),
        "selected_fraction": _selected_fraction(result),
    }


def _telemetry_ok(arm: str, summary: dict) -> bool:
    value = summary["selected_fraction"]
    if arm in {"transformer", "chunk_local_256"}:
        return value is None
    return value is not None and abs(float(value)) <= 1e-12


def _support(
    *,
    paired: list[dict[str, dict]],
    comparator: str,
) -> dict:
    candidate = "local224_predictor_carry"
    candidate_mean = sum(row[candidate]["nll"] for row in paired) / len(paired)
    comparator_mean = sum(row[comparator]["nll"] for row in paired) / len(paired)
    effect = comparator_mean - candidate_mean
    wins = sum(
        row[candidate]["nll"] < row[comparator]["nll"]
        for row in paired
    )
    return {
        "comparator": comparator,
        "effect_nll": effect,
        "candidate_wins": wins,
        "supported": effect >= EFFECT_THRESHOLD_NLL and wins >= 2,
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
def run_panel(source_sha: str) -> dict:
    import re

    from tam_research.pgw_core_mechanism.train import train_core_mechanism_arm

    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise RuntimeError(f"invalid source SHA: {source_sha!r}")

    root = Path(RESULT_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    attempt_path = root / "ATTEMPT.json"
    result_path = root / "RESULT.json"
    failure_path = root / "FAILURE.json"
    if attempt_path.exists() or result_path.exists() or failure_path.exists():
        raise RuntimeError("PGW core mechanism panel #1201 namespace is already consumed")

    attempt = {
        "issue": ISSUE,
        "source_sha": source_sha,
        "arms": list(ARMS),
        "smoke_seed": SMOKE_SEED,
        "replication_seeds": list(REPLICATION_SEEDS),
        "started_unix": time.time(),
        "gpu": "H100!",
        "retry_authorized": False,
        "resume_authorized": False,
        "breakthrough_claim_allowed": False,
        "scale_up_authorized": False,
    }
    attempt_path.write_text(json.dumps(attempt, indent=2), encoding="utf-8")
    volume.commit()
    print(
        "PGW_CORE_MECHANISM_ATTEMPT=" + json.dumps(attempt, sort_keys=True),
        flush=True,
    )

    try:
        smoke_raw: dict[str, dict] = {}
        smoke: dict[str, dict] = {}
        for arm in ARMS:
            print(json.dumps({
                "stage": "smoke_start",
                "arm": arm,
                "seed": SMOKE_SEED,
                "tokens": SMOKE_TOKENS,
            }, sort_keys=True), flush=True)
            result = train_core_mechanism_arm(
                arm=arm,
                seed=SMOKE_SEED,
                data_dir="/vol/data/fineweb-edu-gpt2",
                run_root=str(root / "smoke"),
                token_budget=SMOKE_TOKENS,
                seq_len=SEQ_LEN,
                micro_batch_size=MICRO_BATCH_SIZE,
                grad_accum_steps=GRAD_ACCUM_STEPS,
            )
            smoke_raw[arm] = result
            smoke[arm] = _summary(result)
            volume.commit()

        smoke_pass = all(
            _finite(smoke_raw[arm])
            and smoke[arm]["parameters"] == EXPECTED_PARAMETERS
            and _telemetry_ok(arm, smoke[arm])
            for arm in ARMS
        )
        print(
            "PGW_CORE_MECHANISM_SMOKE=" + json.dumps({
                "arms": smoke,
                "stage2_authorized": smoke_pass,
            }, sort_keys=True),
            flush=True,
        )

        paired: list[dict[str, dict]] = []
        raw_replication: list[dict[str, dict]] = []
        if smoke_pass:
            for seed in REPLICATION_SEEDS:
                row_raw: dict[str, dict] = {}
                row: dict[str, dict] = {}
                for arm in ARMS:
                    print(json.dumps({
                        "stage": "replication_start",
                        "arm": arm,
                        "seed": seed,
                        "tokens": REPLICATION_TOKENS,
                    }, sort_keys=True), flush=True)
                    result = train_core_mechanism_arm(
                        arm=arm,
                        seed=seed,
                        data_dir="/vol/data/fineweb-edu-gpt2",
                        run_root=str(root / "replication" / f"seed-{seed}"),
                        token_budget=REPLICATION_TOKENS,
                        seq_len=SEQ_LEN,
                        micro_batch_size=MICRO_BATCH_SIZE,
                        grad_accum_steps=GRAD_ACCUM_STEPS,
                    )
                    row_raw[arm] = result
                    row[arm] = _summary(result)
                    volume.commit()
                raw_replication.append(row_raw)
                paired.append(row)
                print(
                    "PGW_CORE_MECHANISM_PAIR=" + json.dumps(
                        {"seed": seed, "arms": row},
                        sort_keys=True,
                    ),
                    flush=True,
                )

        replication_integrity = (
            len(paired) == len(REPLICATION_SEEDS)
            and all(
                _finite(row_raw[arm])
                and row[arm]["parameters"] == EXPECTED_PARAMETERS
                and _telemetry_ok(arm, row[arm])
                for row_raw, row in zip(raw_replication, paired)
                for arm in ARMS
            )
        )

        if replication_integrity:
            mean_nll = {
                arm: sum(row[arm]["nll"] for row in paired) / len(paired)
                for arm in ARMS
            }
            mean_tps = {
                arm: sum(row[arm]["training_tps"] for row in paired) / len(paired)
                for arm in ARMS
            }
            transformer_tps = mean_tps["transformer"]
            throughput_vs_transformer = {
                arm: mean_tps[arm] / max(transformer_tps, 1e-12)
                for arm in ARMS
            }

            predictor = _support(
                paired=paired,
                comparator="local224_only",
            )
            carry = _support(
                paired=paired,
                comparator="local224_predictor_reset",
            )
            predictor_supported = bool(predictor["supported"])
            carry_supported = bool(carry["supported"])
            core_supported = predictor_supported

            candidate = "local224_predictor_carry"
            descriptive = {
                "chunk_local_256_minus_transformer_nll": (
                    mean_nll["chunk_local_256"] - mean_nll["transformer"]
                ),
                "carry_minus_chunk_local_256_nll": (
                    mean_nll[candidate] - mean_nll["chunk_local_256"]
                ),
                "carry_minus_transformer_nll": (
                    mean_nll[candidate] - mean_nll["transformer"]
                ),
                "carry_wins_vs_chunk_local_256": sum(
                    row[candidate]["nll"] < row["chunk_local_256"]["nll"]
                    for row in paired
                ),
                "carry_wins_vs_transformer": sum(
                    row[candidate]["nll"] < row["transformer"]["nll"]
                    for row in paired
                ),
            }
            attribution = {
                "predictor": predictor,
                "cross_chunk_carry": carry,
                "PREDICTOR_SUPPORTED": predictor_supported,
                "CROSS_CHUNK_CARRY_SUPPORTED": carry_supported,
                "LOCAL_PREDICTOR_CORE_SUPPORTED": core_supported,
            }
            classification = (
                "PGW_LOCAL_PREDICTOR_CORE_SUPPORTED"
                if core_supported
                else "PGW_LOCAL_PREDICTOR_CORE_UNSUPPORTED"
            )
        else:
            mean_nll = {}
            mean_tps = {}
            throughput_vs_transformer = {}
            descriptive = {}
            attribution = {
                "PREDICTOR_SUPPORTED": False,
                "CROSS_CHUNK_CARRY_SUPPORTED": False,
                "LOCAL_PREDICTOR_CORE_SUPPORTED": False,
            }
            classification = "PGW_LOCAL_PREDICTOR_CORE_INTEGRITY_STOP"

        final = {
            "issue": ISSUE,
            "source_sha": source_sha,
            "classification": classification,
            "smoke_pass": smoke_pass,
            "smoke": smoke,
            "replication": paired,
            "replication_integrity": replication_integrity,
            "mean_nll": mean_nll,
            "mean_training_tps": mean_tps,
            "throughput_vs_transformer": throughput_vs_transformer,
            "attribution": attribution,
            "descriptive": descriptive,
            "breakthrough_claim_allowed": False,
            "scale_up_authorized": False,
            "completed_unix": time.time(),
        }
        result_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
        volume.commit()
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
            "scale_up_authorized": False,
            "failed_unix": time.time(),
        }
        failure_path.write_text(json.dumps(failure, indent=2), encoding="utf-8")
        volume.commit()
        print(
            "PGW_CORE_MECHANISM_FAILURE=" + json.dumps(failure, sort_keys=True),
            flush=True,
        )
        raise


@app.local_entrypoint()
def main(source_sha: str) -> None:
    ensure_data.remote()
    result = run_panel.remote(source_sha)
    print(
        "PGW_CORE_MECHANISM_RESULT=" + json.dumps(result, sort_keys=True),
        flush=True,
    )
