from __future__ import annotations

import json
import math
from pathlib import Path
import re
import time

import modal

from tam_research.cpw_v1.protocol import (
    EXPECTED_CPW_V1_PARAMETERS,
    GRAD_ACCUM_STEPS,
    MICRO_BATCH_SIZE,
    SEQ_LEN,
    TRANSFORMER_PARAMETERS,
)

ISSUE = 1250
SMOKE_SEED = 1_250_001
REPLICATION_SEEDS = (1_250_101, 1_250_102, 1_250_103)
SMOKE_TOKENS = 1_000_000
REPLICATION_TOKENS = 5_000_000
RESULT_ROOT = "/vol/cpw-v1/fast-scientific-replication-v1"
APP_NAME = "tam-research-cpw-v1-fast-scientific-1250"
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
    .add_local_python_source("architectures")
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
    return result


def _finite(result: dict) -> bool:
    ev = result["final_eval"]
    values = [
        float(ev["nll"]),
        float(ev["perplexity"]),
        float(result["training_tokens_per_second"]),
        float(result["training_seconds"]),
        float(result["total_compute_seconds"]),
        float(result["peak_vram_gb"]),
    ]
    return all(math.isfinite(v) and v >= 0.0 for v in values)


def _fast_telemetry_ok(result: dict) -> bool:
    stats = result["final_eval"].get("router")
    if not stats:
        return False
    mean = stats.get("mean") or {}
    try:
        return (
            float(mean["active_predictors"]) == 3.0
            and float(mean["attention_present"]) == 0.0
            and float(mean["router_present"]) == 0.0
            and math.isfinite(float(mean["world_state_norm"]))
            and float(mean["world_state_norm"]) >= 0.0
        )
    except (KeyError, TypeError, ValueError):
        return False


def _summary(r: dict) -> dict:
    return {
        "architecture": r["architecture"],
        "seed": int(r["seed"]),
        "parameters": int(r["parameters"]),
        "tokens_seen": int(r["tokens_seen"]),
        "nll": float(r["final_eval"]["nll"]),
        "perplexity": float(r["final_eval"]["perplexity"]),
        "training_tps": float(r["training_tokens_per_second"]),
        "training_seconds": float(r["training_seconds"]),
        "total_compute_seconds": float(r["total_compute_seconds"]),
        "peak_vram_gib": float(r["peak_vram_gb"]),
        "telemetry": r["final_eval"].get("router"),
    }


def _pair(t: dict, c: dict) -> dict:
    ts = _summary(t)
    cs = _summary(c)
    return {
        "seed": int(t["seed"]),
        "transformer": ts,
        "cpwv1fast": cs,
        "transformer_minus_cpw_nll": ts["nll"] - cs["nll"],
        "cpw_training_tps_ratio": cs["training_tps"] / max(ts["training_tps"], 1e-12),
        "cpw_training_seconds_ratio": cs["training_seconds"] / max(ts["training_seconds"], 1e-12),
        "cpw_total_compute_ratio": cs["total_compute_seconds"] / max(ts["total_compute_seconds"], 1e-12),
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
def run_replication(source_sha: str) -> dict:
    from tam_research.cpw_v1_fast.model import fast_parameter_count, triton_available
    from tam_research.cpw_v1_fast.train import train_cpw_v1_fast
    from tam_research.train import train_language_model

    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise RuntimeError(f"invalid source SHA: {source_sha!r}")
    if not triton_available():
        raise RuntimeError("Triton is unavailable on H100 image")
    if fast_parameter_count() != EXPECTED_CPW_V1_PARAMETERS:
        raise RuntimeError("optimized CPW parameter count drift")

    root = Path(RESULT_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    attempt_path = root / "ATTEMPT.json"
    result_path = root / "RESULT.json"
    failure_path = root / "FAILURE.json"
    if attempt_path.exists() or result_path.exists() or failure_path.exists():
        raise RuntimeError("CPW-v1-fast scientific namespace already consumed")

    attempt = {
        "issue": ISSUE,
        "source_sha": source_sha,
        "smoke_seed": SMOKE_SEED,
        "replication_seeds": list(REPLICATION_SEEDS),
        "smoke_tokens": SMOKE_TOKENS,
        "replication_tokens": REPLICATION_TOKENS,
        "transformer_parameters": TRANSFORMER_PARAMETERS,
        "cpw_parameters": EXPECTED_CPW_V1_PARAMETERS,
        "gpu": "H100!",
        "retry_authorized": False,
        "resume_authorized": False,
        "breakthrough_claim_allowed": False,
        "scale_up_authorized": False,
        "started_unix": time.time(),
    }
    attempt_path.write_text(json.dumps(attempt, indent=2), encoding="utf-8")
    volume.commit()
    print("CPW_V1_FAST_ATTEMPT=" + json.dumps(attempt, sort_keys=True), flush=True)

    def train_pair(seed: int, token_budget: int, run_root: Path) -> tuple[dict, dict]:
        transformer = train_language_model(
            architecture="transformer",
            seed=seed,
            data_dir="/vol/data/fineweb-edu-gpt2",
            run_root=str(run_root / "transformer"),
            token_budget=token_budget,
            seq_len=SEQ_LEN,
            micro_batch_size=MICRO_BATCH_SIZE,
            grad_accum_steps=GRAD_ACCUM_STEPS,
            eval_every_tokens=token_budget,
            checkpoint_every_tokens=token_budget,
            resume=False,
            compile_model=False,
        )
        volume.commit()
        cpw = train_cpw_v1_fast(
            seed=seed,
            data_dir="/vol/data/fineweb-edu-gpt2",
            run_root=str(run_root / "cpwv1fast"),
            token_budget=token_budget,
            seq_len=SEQ_LEN,
            micro_batch_size=MICRO_BATCH_SIZE,
            grad_accum_steps=GRAD_ACCUM_STEPS,
        )
        volume.commit()
        return transformer, cpw

    try:
        smoke_t, smoke_c = train_pair(
            SMOKE_SEED,
            SMOKE_TOKENS,
            root / "smoke",
        )
        smoke = _pair(smoke_t, smoke_c)
        smoke_pass = (
            _finite(smoke_t)
            and _finite(smoke_c)
            and int(smoke_t["parameters"]) == TRANSFORMER_PARAMETERS
            and int(smoke_c["parameters"]) == EXPECTED_CPW_V1_PARAMETERS
            and _fast_telemetry_ok(smoke_c)
        )
        print(
            "CPW_V1_FAST_SMOKE="
            + json.dumps(
                {"pair": smoke, "replication_authorized": smoke_pass},
                sort_keys=True,
            ),
            flush=True,
        )

        pairs: list[dict] = []
        raw_pairs: list[tuple[dict, dict]] = []
        if smoke_pass:
            for seed in REPLICATION_SEEDS:
                print(
                    "CPW_V1_FAST_PAIR_START="
                    + json.dumps(
                        {"seed": seed, "tokens_per_model": REPLICATION_TOKENS},
                        sort_keys=True,
                    ),
                    flush=True,
                )
                t, c = train_pair(
                    seed,
                    REPLICATION_TOKENS,
                    root / "replication" / f"seed-{seed}",
                )
                raw_pairs.append((t, c))
                pair = _pair(t, c)
                pairs.append(pair)
                print("CPW_V1_FAST_PAIR=" + json.dumps(pair, sort_keys=True), flush=True)

        integrity = (
            smoke_pass
            and len(pairs) == len(REPLICATION_SEEDS)
            and all(
                _finite(t)
                and _finite(c)
                and int(t["parameters"]) == TRANSFORMER_PARAMETERS
                and int(c["parameters"]) == EXPECTED_CPW_V1_PARAMETERS
                and _fast_telemetry_ok(c)
                for t, c in raw_pairs
            )
        )

        if integrity:
            t_mean = sum(p["transformer"]["nll"] for p in pairs) / len(pairs)
            c_mean = sum(p["cpwv1fast"]["nll"] for p in pairs) / len(pairs)
            wins = sum(
                p["cpwv1fast"]["nll"] < p["transformer"]["nll"]
                for p in pairs
            )
            mean_tps_ratio = sum(p["cpw_training_tps_ratio"] for p in pairs) / len(pairs)
            mean_train_seconds_ratio = sum(
                p["cpw_training_seconds_ratio"] for p in pairs
            ) / len(pairs)
            mean_total_compute_ratio = sum(
                p["cpw_total_compute_ratio"] for p in pairs
            ) / len(pairs)
            quality_supported = c_mean < t_mean and wins >= 2
            if (
                quality_supported
                and mean_tps_ratio >= 1.0
                and mean_train_seconds_ratio <= 1.0
            ):
                classification = "CPW_V1_FAST_PARETO_SIGNAL"
            elif quality_supported:
                classification = "CPW_V1_FAST_QUALITY_SIGNAL"
            else:
                classification = "CPW_V1_FAST_STOP_OR_REDESIGN"
        else:
            t_mean = None
            c_mean = None
            wins = 0
            mean_tps_ratio = None
            mean_train_seconds_ratio = None
            mean_total_compute_ratio = None
            quality_supported = False
            classification = "CPW_V1_FAST_STOP_OR_REDESIGN"

        final = {
            "issue": ISSUE,
            "source_sha": source_sha,
            "classification": classification,
            "smoke_pass": smoke_pass,
            "smoke": smoke,
            "replication_integrity": integrity,
            "replication": pairs,
            "transformer_mean_nll": t_mean,
            "cpwv1fast_mean_nll": c_mean,
            "cpwv1fast_wins": wins,
            "quality_supported": quality_supported,
            "mean_cpw_training_tps_ratio": mean_tps_ratio,
            "mean_cpw_training_seconds_ratio": mean_train_seconds_ratio,
            "mean_cpw_total_compute_ratio": mean_total_compute_ratio,
            "cpw_parameter_fraction_of_transformer": (
                EXPECTED_CPW_V1_PARAMETERS / TRANSFORMER_PARAMETERS
            ),
            "breakthrough_claim_allowed": False,
            "scale_up_authorized": False,
            "completed_unix": time.time(),
        }
        result_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
        volume.commit()
        print("CPW_V1_FAST_RESULT=" + json.dumps(final, sort_keys=True), flush=True)
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
        print("CPW_V1_FAST_FAILURE=" + json.dumps(failure, sort_keys=True), flush=True)
        raise


@app.local_entrypoint()
def main(source_sha: str) -> None:
    ensure_data.remote()
    result = run_replication.remote(source_sha)
    print("CPW_V1_FAST_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
