from __future__ import annotations

import json
import math
from pathlib import Path
import re
import time

import modal

from tam_research.cpw_v0.protocol import (
    MAX_PARAMETER_MISMATCH_FRACTION,
    MIN_THROUGHPUT_RATIO,
    POSITIVE_CLASSIFICATION,
    REPLICATION_SEEDS,
    REPLICATION_TOKENS,
    RESULT_ROOT,
    ROUTER_TOP_K,
    SEQ_LEN,
    MICRO_BATCH_SIZE,
    GRAD_ACCUM_STEPS,
    SMOKE_SEED,
    SMOKE_TOKENS,
    STOP_CLASSIFICATION,
    TRANSFORMER_PARAMETERS,
)

APP_NAME = "tam-research-cpw-v0-1245"
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
    print("CPW_V0_DATA_GUARD=" + json.dumps(result, sort_keys=True), flush=True)
    return result


def _finite(result: dict) -> bool:
    ev = result["final_eval"]
    values = [
        float(ev["nll"]),
        float(ev["perplexity"]),
        float(result["training_tokens_per_second"]),
        float(result["peak_vram_gb"]),
        float(result["total_compute_seconds"]),
    ]
    return all(math.isfinite(v) and v >= 0.0 for v in values)


def _router_ok(result: dict) -> bool:
    router = result["final_eval"].get("router")
    if not router:
        return False
    mean = router.get("mean") or {}
    keys = (
        "active_fraction",
        "router_entropy",
        "attention_usage",
        "world_usage",
        "sequence_usage",
        "memory_usage",
        "world_state_norm",
    )
    try:
        values = [float(mean[k]) for k in keys]
    except (KeyError, TypeError, ValueError):
        return False
    if not all(math.isfinite(v) for v in values):
        return False
    expected_active = ROUTER_TOP_K / 4.0
    usage_sum = sum(values[2:6])
    return (
        abs(values[0] - expected_active) <= 1e-6
        and abs(usage_sum - ROUTER_TOP_K) <= 1e-4
        and values[1] > 0.0
        and values[6] >= 0.0
    )


def _parameter_ok(result: dict) -> bool:
    actual = int(result["parameters"])
    mismatch = abs(actual - TRANSFORMER_PARAMETERS) / TRANSFORMER_PARAMETERS
    return mismatch <= MAX_PARAMETER_MISMATCH_FRACTION


def _summarize(result: dict) -> dict:
    router = result["final_eval"].get("router")
    return {
        "architecture": result["architecture"],
        "seed": int(result["seed"]),
        "parameters": int(result["parameters"]),
        "tokens_seen": int(result["tokens_seen"]),
        "nll": float(result["final_eval"]["nll"]),
        "perplexity": float(result["final_eval"]["perplexity"]),
        "training_tps": float(result["training_tokens_per_second"]),
        "total_compute_seconds": float(result["total_compute_seconds"]),
        "peak_vram_gib": float(result["peak_vram_gb"]),
        "router": router,
        "last_auxiliary_loss": (
            float(result["last_auxiliary_loss"])
            if "last_auxiliary_loss" in result
            else None
        ),
    }


def _pair_summary(transformer: dict, cpw: dict) -> dict:
    t = _summarize(transformer)
    c = _summarize(cpw)
    return {
        "seed": int(transformer["seed"]),
        "transformer": t,
        "cpw": c,
        "transformer_minus_cpw_nll": t["nll"] - c["nll"],
        "cpw_training_throughput_ratio": c["training_tps"] / max(t["training_tps"], 1e-12),
        "cpw_compute_ratio": c["total_compute_seconds"] / max(t["total_compute_seconds"], 1e-12),
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
def run_screen(source_sha: str) -> dict:
    from tam_research.cpw_v0.model import cpwv0_parameter_count
    from tam_research.cpw_v0.train import train_cpw_v0_candidate

    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise RuntimeError(f"invalid source SHA: {source_sha!r}")

    root = Path(RESULT_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    attempt_path = root / "ATTEMPT.json"
    result_path = root / "RESULT.json"
    failure_path = root / "FAILURE.json"
    if attempt_path.exists() or result_path.exists() or failure_path.exists():
        raise RuntimeError("CPW-v0 scientific namespace is already consumed")

    cpw_parameters = cpwv0_parameter_count()
    mismatch = abs(cpw_parameters - TRANSFORMER_PARAMETERS) / TRANSFORMER_PARAMETERS
    if mismatch > MAX_PARAMETER_MISMATCH_FRACTION:
        raise RuntimeError(
            f"CPW parameter mismatch {mismatch:.8f} exceeds frozen tolerance"
        )

    attempt = {
        "issue": 1245,
        "source_sha": source_sha,
        "smoke_seed": SMOKE_SEED,
        "replication_seeds": list(REPLICATION_SEEDS),
        "smoke_tokens": SMOKE_TOKENS,
        "replication_tokens": REPLICATION_TOKENS,
        "transformer_parameters": TRANSFORMER_PARAMETERS,
        "cpw_parameters": cpw_parameters,
        "parameter_mismatch_fraction": mismatch,
        "gpu": "H100!",
        "retry_authorized": False,
        "resume_authorized": False,
        "breakthrough_claim_allowed": False,
        "scale_up_authorized": False,
        "started_unix": time.time(),
    }
    attempt_path.write_text(json.dumps(attempt, indent=2), encoding="utf-8")
    volume.commit()
    print("CPW_V0_ATTEMPT=" + json.dumps(attempt, sort_keys=True), flush=True)

    try:
        smoke_raw: dict[str, dict] = {}
        for architecture in ("transformer", "cpwv0"):
            print(json.dumps({
                "stage": "smoke_start",
                "architecture": architecture,
                "seed": SMOKE_SEED,
                "tokens": SMOKE_TOKENS,
            }, sort_keys=True), flush=True)
            smoke_raw[architecture] = train_cpw_v0_candidate(
                architecture=architecture,
                seed=SMOKE_SEED,
                data_dir="/vol/data/fineweb-edu-gpt2",
                run_root=str(root / "smoke"),
                token_budget=SMOKE_TOKENS,
                seq_len=SEQ_LEN,
                micro_batch_size=MICRO_BATCH_SIZE,
                grad_accum_steps=GRAD_ACCUM_STEPS,
            )
            volume.commit()

        smoke = _pair_summary(
            smoke_raw["transformer"],
            smoke_raw["cpwv0"],
        )
        smoke_pass = (
            _finite(smoke_raw["transformer"])
            and _finite(smoke_raw["cpwv0"])
            and int(smoke_raw["transformer"]["parameters"]) == TRANSFORMER_PARAMETERS
            and _parameter_ok(smoke_raw["cpwv0"])
            and _router_ok(smoke_raw["cpwv0"])
        )
        print("CPW_V0_SMOKE=" + json.dumps({
            "pair": smoke,
            "replication_authorized": smoke_pass,
        }, sort_keys=True), flush=True)

        pairs: list[dict] = []
        raw_pairs: list[tuple[dict, dict]] = []
        if smoke_pass:
            for seed in REPLICATION_SEEDS:
                print(json.dumps({
                    "stage": "replication_pair_start",
                    "seed": seed,
                    "tokens_per_model": REPLICATION_TOKENS,
                }, sort_keys=True), flush=True)
                transformer = train_cpw_v0_candidate(
                    architecture="transformer",
                    seed=seed,
                    data_dir="/vol/data/fineweb-edu-gpt2",
                    run_root=str(root / "replication" / f"seed-{seed}"),
                    token_budget=REPLICATION_TOKENS,
                    seq_len=SEQ_LEN,
                    micro_batch_size=MICRO_BATCH_SIZE,
                    grad_accum_steps=GRAD_ACCUM_STEPS,
                )
                volume.commit()
                cpw = train_cpw_v0_candidate(
                    architecture="cpwv0",
                    seed=seed,
                    data_dir="/vol/data/fineweb-edu-gpt2",
                    run_root=str(root / "replication" / f"seed-{seed}"),
                    token_budget=REPLICATION_TOKENS,
                    seq_len=SEQ_LEN,
                    micro_batch_size=MICRO_BATCH_SIZE,
                    grad_accum_steps=GRAD_ACCUM_STEPS,
                )
                volume.commit()
                raw_pairs.append((transformer, cpw))
                pair = _pair_summary(transformer, cpw)
                pairs.append(pair)
                print("CPW_V0_PAIR=" + json.dumps(pair, sort_keys=True), flush=True)

        integrity = (
            smoke_pass
            and len(pairs) == len(REPLICATION_SEEDS)
            and all(
                _finite(t)
                and _finite(c)
                and int(t["parameters"]) == TRANSFORMER_PARAMETERS
                and _parameter_ok(c)
                and _router_ok(c)
                for t, c in raw_pairs
            )
        )

        if integrity:
            transformer_mean_nll = sum(
                p["transformer"]["nll"] for p in pairs
            ) / len(pairs)
            cpw_mean_nll = sum(p["cpw"]["nll"] for p in pairs) / len(pairs)
            cpw_wins = sum(
                p["cpw"]["nll"] < p["transformer"]["nll"]
                for p in pairs
            )
            throughput_ratio = sum(
                p["cpw_training_throughput_ratio"] for p in pairs
            ) / len(pairs)
            positive = (
                cpw_mean_nll < transformer_mean_nll
                and cpw_wins >= 2
                and throughput_ratio >= MIN_THROUGHPUT_RATIO
            )
            classification = (
                POSITIVE_CLASSIFICATION if positive else STOP_CLASSIFICATION
            )
        else:
            transformer_mean_nll = None
            cpw_mean_nll = None
            cpw_wins = 0
            throughput_ratio = None
            classification = STOP_CLASSIFICATION

        final = {
            "issue": 1245,
            "source_sha": source_sha,
            "classification": classification,
            "smoke_pass": smoke_pass,
            "smoke": smoke,
            "replication_integrity": integrity,
            "replication": pairs,
            "transformer_mean_nll": transformer_mean_nll,
            "cpw_mean_nll": cpw_mean_nll,
            "cpw_wins": cpw_wins,
            "mean_cpw_training_throughput_ratio": throughput_ratio,
            "breakthrough_claim_allowed": False,
            "scale_up_authorized": False,
            "completed_unix": time.time(),
        }
        result_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
        volume.commit()
        print("CPW_V0_RESULT=" + json.dumps(final, sort_keys=True), flush=True)
        return final
    except BaseException as exc:
        failure = {
            "issue": 1245,
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
        print("CPW_V0_FAILURE=" + json.dumps(failure, sort_keys=True), flush=True)
        raise


@app.local_entrypoint()
def main(source_sha: str) -> None:
    ensure_data.remote()
    result = run_screen.remote(source_sha)
    print("CPW_V0_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
