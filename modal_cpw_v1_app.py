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
    ISSUE,
    MICRO_BATCH_SIZE,
    MIN_THROUGHPUT_RATIO,
    POSITIVE_CLASSIFICATION,
    QUALITY_ONLY_CLASSIFICATION,
    REPLICATION_SEEDS,
    REPLICATION_TOKENS,
    RESULT_ROOT,
    SEQ_LEN,
    SMOKE_SEED,
    SMOKE_TOKENS,
    STOP_CLASSIFICATION,
    TRANSFORMER_PARAMETERS,
)

APP_NAME = "tam-research-cpw-v1-1248"
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
    print("CPW_V1_DATA_GUARD=" + json.dumps(result, sort_keys=True), flush=True)
    return result


def _finite(r: dict) -> bool:
    ev = r["final_eval"]
    values = [
        float(ev["nll"]),
        float(ev["perplexity"]),
        float(r["training_tokens_per_second"]),
        float(r["peak_vram_gb"]),
        float(r["total_compute_seconds"]),
    ]
    return all(math.isfinite(v) and v >= 0.0 for v in values)


def _telemetry_ok(r: dict) -> bool:
    stats = r["final_eval"].get("router")
    if not stats:
        return False
    mean = stats.get("mean") or {}
    try:
        active = float(mean["active_predictors"])
        attention = float(mean["attention_present"])
        router = float(mean["router_present"])
        state_norm = float(mean["world_state_norm"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        active == 3.0
        and attention == 0.0
        and router == 0.0
        and math.isfinite(state_norm)
        and state_norm >= 0.0
    )


def _summary(r: dict) -> dict:
    return {
        "architecture": r["architecture"],
        "seed": int(r["seed"]),
        "parameters": int(r["parameters"]),
        "tokens_seen": int(r["tokens_seen"]),
        "nll": float(r["final_eval"]["nll"]),
        "perplexity": float(r["final_eval"]["perplexity"]),
        "training_tps": float(r["training_tokens_per_second"]),
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
        "cpwv1": cs,
        "transformer_minus_cpwv1_nll": ts["nll"] - cs["nll"],
        "cpwv1_training_throughput_ratio": cs["training_tps"] / max(ts["training_tps"], 1e-12),
        "cpwv1_compute_ratio": cs["total_compute_seconds"] / max(ts["total_compute_seconds"], 1e-12),
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
    from tam_research.cpw_v1.model import cpwv1_parameter_count
    from tam_research.cpw_v1.train import train_cpw_v1_candidate

    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise RuntimeError(f"invalid source SHA: {source_sha!r}")

    root = Path(RESULT_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    attempt_path = root / "ATTEMPT.json"
    result_path = root / "RESULT.json"
    failure_path = root / "FAILURE.json"
    if attempt_path.exists() or result_path.exists() or failure_path.exists():
        raise RuntimeError("CPW-v1 namespace already consumed")

    params = cpwv1_parameter_count()
    if params != EXPECTED_CPW_V1_PARAMETERS:
        raise RuntimeError(f"CPW-v1 parameter drift: {params}")
    if params >= TRANSFORMER_PARAMETERS:
        raise RuntimeError("CPW-v1 is not smaller than Transformer")

    attempt = {
        "issue": ISSUE,
        "source_sha": source_sha,
        "smoke_seed": SMOKE_SEED,
        "replication_seeds": list(REPLICATION_SEEDS),
        "smoke_tokens": SMOKE_TOKENS,
        "replication_tokens": REPLICATION_TOKENS,
        "transformer_parameters": TRANSFORMER_PARAMETERS,
        "cpwv1_parameters": params,
        "gpu": "H100!",
        "retry_authorized": False,
        "resume_authorized": False,
        "breakthrough_claim_allowed": False,
        "scale_up_authorized": False,
        "started_unix": time.time(),
    }
    attempt_path.write_text(json.dumps(attempt, indent=2), encoding="utf-8")
    volume.commit()
    print("CPW_V1_ATTEMPT=" + json.dumps(attempt, sort_keys=True), flush=True)

    try:
        smoke_raw = {}
        for architecture in ("transformer", "cpwv1"):
            smoke_raw[architecture] = train_cpw_v1_candidate(
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

        smoke = _pair(smoke_raw["transformer"], smoke_raw["cpwv1"])
        smoke_pass = (
            _finite(smoke_raw["transformer"])
            and _finite(smoke_raw["cpwv1"])
            and int(smoke_raw["transformer"]["parameters"]) == TRANSFORMER_PARAMETERS
            and int(smoke_raw["cpwv1"]["parameters"]) == EXPECTED_CPW_V1_PARAMETERS
            and _telemetry_ok(smoke_raw["cpwv1"])
        )
        print("CPW_V1_SMOKE=" + json.dumps({
            "pair": smoke,
            "replication_authorized": smoke_pass,
        }, sort_keys=True), flush=True)

        pairs = []
        raw_pairs = []
        if smoke_pass:
            for seed in REPLICATION_SEEDS:
                transformer = train_cpw_v1_candidate(
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
                cpw = train_cpw_v1_candidate(
                    architecture="cpwv1",
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
                pair = _pair(transformer, cpw)
                pairs.append(pair)
                print("CPW_V1_PAIR=" + json.dumps(pair, sort_keys=True), flush=True)

        integrity = (
            smoke_pass
            and len(pairs) == len(REPLICATION_SEEDS)
            and all(
                _finite(t)
                and _finite(c)
                and int(t["parameters"]) == TRANSFORMER_PARAMETERS
                and int(c["parameters"]) == EXPECTED_CPW_V1_PARAMETERS
                and _telemetry_ok(c)
                for t, c in raw_pairs
            )
        )

        if integrity:
            t_mean = sum(p["transformer"]["nll"] for p in pairs) / len(pairs)
            c_mean = sum(p["cpwv1"]["nll"] for p in pairs) / len(pairs)
            wins = sum(p["cpwv1"]["nll"] < p["transformer"]["nll"] for p in pairs)
            tps_ratio = sum(p["cpwv1_training_throughput_ratio"] for p in pairs) / len(pairs)
            quality = c_mean < t_mean and wins >= 2
            if quality and tps_ratio >= MIN_THROUGHPUT_RATIO:
                classification = POSITIVE_CLASSIFICATION
            elif quality:
                classification = QUALITY_ONLY_CLASSIFICATION
            else:
                classification = STOP_CLASSIFICATION
        else:
            t_mean = None
            c_mean = None
            wins = 0
            tps_ratio = None
            classification = STOP_CLASSIFICATION

        final = {
            "issue": ISSUE,
            "source_sha": source_sha,
            "classification": classification,
            "smoke_pass": smoke_pass,
            "smoke": smoke,
            "replication_integrity": integrity,
            "replication": pairs,
            "transformer_mean_nll": t_mean,
            "cpwv1_mean_nll": c_mean,
            "cpwv1_wins": wins,
            "mean_cpwv1_training_throughput_ratio": tps_ratio,
            "cpwv1_parameter_fraction_of_transformer": (
                EXPECTED_CPW_V1_PARAMETERS / TRANSFORMER_PARAMETERS
            ),
            "breakthrough_claim_allowed": False,
            "scale_up_authorized": False,
            "completed_unix": time.time(),
        }
        result_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
        volume.commit()
        print("CPW_V1_RESULT=" + json.dumps(final, sort_keys=True), flush=True)
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
        print("CPW_V1_FAILURE=" + json.dumps(failure, sort_keys=True), flush=True)
        raise


@app.local_entrypoint()
def main(source_sha: str) -> None:
    ensure_data.remote()
    result = run_screen.remote(source_sha)
    print("CPW_V1_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
