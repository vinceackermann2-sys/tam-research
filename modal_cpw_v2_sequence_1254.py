from __future__ import annotations

import json
import math
from pathlib import Path
import re
import time

import modal

from tam_research.cpw_v2_sequence.protocol import (
    GRAD_ACCUM_STEPS,
    ISSUE,
    MICRO_BATCH_SIZE,
    POSITIVE_CLASSIFICATION,
    REPLICATION_SEEDS,
    REPLICATION_TOKENS,
    RESULT_ROOT,
    SEQ_LEN,
    SEQUENCE_PARAMETERS,
    SMOKE_SEED,
    SMOKE_TOKENS,
    STOP_CLASSIFICATION,
    TRANSFORMER_PARAMETERS,
)

APP_NAME = "tam-research-cpw-v2-sequence-1254"
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
    if int(result.get("train_tokens", -1)) != 25_000_000 or int(result.get("val_tokens", -1)) != 2_000_000:\n        raise RuntimeError(f"unexpected data boundary: {result}")\n    volume.commit()\n    print("CPW_V2_SEQUENCE_DATA_GUARD=" + json.dumps(result, sort_keys=True), flush=True)
    return result


def _finite(r: dict) -> bool:
    vals = [
        float(r["final_eval"]["nll"]),
        float(r["final_eval"]["perplexity"]),
        float(r["training_tokens_per_second"]),
        float(r["training_seconds"]),
        float(r["total_compute_seconds"]),
        float(r["peak_vram_gb"]),
    ]
    return all(math.isfinite(v) and v >= 0.0 for v in vals)


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
        "learning_curve": r.get("learning_curve", []),
    }


def _pair(t: dict, c: dict) -> dict:
    ts = _summary(t)
    cs = _summary(c)
    return {
        "seed": ts["seed"],
        "transformer": ts,
        "sequence": cs,
        "transformer_minus_sequence_nll": ts["nll"] - cs["nll"],
        "sequence_training_tps_ratio": cs["training_tps"] / max(ts["training_tps"], 1e-12),
        "sequence_training_seconds_ratio": cs["training_seconds"] / max(ts["training_seconds"], 1e-12),
        "sequence_total_compute_ratio": cs["total_compute_seconds"] / max(ts["total_compute_seconds"], 1e-12),
    }


def _curve_effect(pair: dict) -> dict[str, float]:
    t = {int(x["tokens_seen"]): float(x["nll"]) for x in pair["transformer"]["learning_curve"]}
    c = {int(x["tokens_seen"]): float(x["nll"]) for x in pair["sequence"]["learning_curve"]}
    return {
        str(h): t[h] - c[h]
        for h in sorted(set(t) & set(c))
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
    from tam_research.cpw_v2_sequence.train import train_sequence_candidate

    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise RuntimeError(f"invalid source SHA: {source_sha!r}")

    root = Path(RESULT_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    attempt_path = root / "ATTEMPT.json"
    result_path = root / "RESULT.json"
    failure_path = root / "FAILURE.json"
    if attempt_path.exists() or result_path.exists() or failure_path.exists():
        raise RuntimeError("CPW-v2 sequence namespace already consumed")

    attempt = {
        "issue": ISSUE,
        "source_sha": source_sha,
        "smoke_seed": SMOKE_SEED,
        "replication_seeds": list(REPLICATION_SEEDS),
        "smoke_tokens": SMOKE_TOKENS,
        "replication_tokens": REPLICATION_TOKENS,
        "gpu": "H100!",
        "retry_authorized": False,
        "resume_authorized": False,
        "breakthrough_claim_allowed": False,
        "scale_up_authorized": False,
        "started_unix": time.time(),
    }
    attempt_path.write_text(json.dumps(attempt, indent=2), encoding="utf-8")
    volume.commit()
    print("CPW_V2_SEQUENCE_ATTEMPT=" + json.dumps(attempt, sort_keys=True), flush=True)

    try:
        smoke_raw = {}
        for arch in ("transformer", "cpwv2seq"):
            smoke_raw[arch] = train_sequence_candidate(
                architecture=arch,
                seed=SMOKE_SEED,
                data_dir="/vol/data/fineweb-edu-gpt2",
                run_root=str(root / "smoke"),
                token_budget=SMOKE_TOKENS,
                seq_len=SEQ_LEN,
                micro_batch_size=MICRO_BATCH_SIZE,
                grad_accum_steps=GRAD_ACCUM_STEPS,
            )
            volume.commit()

        smoke = _pair(smoke_raw["transformer"], smoke_raw["cpwv2seq"])
        smoke_pass = (
            _finite(smoke_raw["transformer"])
            and _finite(smoke_raw["cpwv2seq"])
            and int(smoke_raw["transformer"]["parameters"]) == TRANSFORMER_PARAMETERS
            and int(smoke_raw["cpwv2seq"]["parameters"]) == SEQUENCE_PARAMETERS
        )
        print("CPW_V2_SEQUENCE_SMOKE=" + json.dumps({
            "pass": smoke_pass,
            "pair": smoke,
        }, sort_keys=True), flush=True)

        replication = []
        if smoke_pass:
            for seed in REPLICATION_SEEDS:
                print("CPW_V2_SEQUENCE_PAIR_START=" + json.dumps({
                    "seed": seed,
                    "tokens_per_model": REPLICATION_TOKENS,
                }, sort_keys=True), flush=True)
                t = train_sequence_candidate(
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
                c = train_sequence_candidate(
                    architecture="cpwv2seq",
                    seed=seed,
                    data_dir="/vol/data/fineweb-edu-gpt2",
                    run_root=str(root / "replication" / f"seed-{seed}"),
                    token_budget=REPLICATION_TOKENS,
                    seq_len=SEQ_LEN,
                    micro_batch_size=MICRO_BATCH_SIZE,
                    grad_accum_steps=GRAD_ACCUM_STEPS,
                )
                volume.commit()
                p = _pair(t, c)
                p["curve_effect_transformer_minus_sequence"] = _curve_effect(p)
                replication.append(p)
                print("CPW_V2_SEQUENCE_PAIR=" + json.dumps(p, sort_keys=True), flush=True)

        integrity = (
            smoke_pass
            and len(replication) == len(REPLICATION_SEEDS)
            and all(
                p["transformer"]["parameters"] == TRANSFORMER_PARAMETERS
                and p["sequence"]["parameters"] == SEQUENCE_PARAMETERS
                for p in replication
            )
        )

        if integrity:
            t_mean = sum(p["transformer"]["nll"] for p in replication) / len(replication)
            c_mean = sum(p["sequence"]["nll"] for p in replication) / len(replication)
            wins = sum(p["sequence"]["nll"] < p["transformer"]["nll"] for p in replication)
            train_ratio = sum(p["sequence_training_seconds_ratio"] for p in replication) / len(replication)
            compute_ratio = sum(p["sequence_total_compute_ratio"] for p in replication) / len(replication)
            tps_ratio = sum(p["sequence_training_tps_ratio"] for p in replication) / len(replication)
            strong_pareto = all(
                p["sequence"]["nll"] < p["transformer"]["nll"]
                and p["sequence_total_compute_ratio"] < 1.0
                for p in replication
            )
            horizons = sorted({
                int(h)
                for p in replication
                for h in p["curve_effect_transformer_minus_sequence"]
            })
            curve_mean = {
                str(h): sum(
                    p["curve_effect_transformer_minus_sequence"][str(h)]
                    for p in replication
                    if str(h) in p["curve_effect_transformer_minus_sequence"]
                ) / sum(
                    str(h) in p["curve_effect_transformer_minus_sequence"]
                    for p in replication
                )
                for h in horizons
            }
            positive = (
                c_mean < t_mean
                and wins >= 2
                and train_ratio <= 1.0
                and compute_ratio <= 1.0
            )
            classification = POSITIVE_CLASSIFICATION if positive else STOP_CLASSIFICATION
        else:
            t_mean = c_mean = float("nan")
            wins = 0
            train_ratio = compute_ratio = tps_ratio = float("nan")
            strong_pareto = False
            curve_mean = {}
            classification = STOP_CLASSIFICATION

        final = {
            "issue": ISSUE,
            "source_sha": source_sha,
            "classification": classification,
            "smoke_pass": smoke_pass,
            "replication_integrity": integrity,
            "transformer_mean_nll": t_mean,
            "sequence_mean_nll": c_mean,
            "sequence_wins": wins,
            "mean_sequence_training_seconds_ratio": train_ratio,
            "mean_sequence_total_compute_ratio": compute_ratio,
            "mean_sequence_training_tps_ratio": tps_ratio,
            "strong_pareto_3_of_3": strong_pareto,
            "mean_curve_transformer_minus_sequence_nll": curve_mean,
            "replication": replication,
            "breakthrough_claim_allowed": False,
            "scale_up_authorized": False,
            "completed_unix": time.time(),
        }
        result_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
        volume.commit()
        print("CPW_V2_SEQUENCE_RESULT=" + json.dumps(final, sort_keys=True), flush=True)
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
        print("CPW_V2_SEQUENCE_FAILURE=" + json.dumps(failure, sort_keys=True), flush=True)
        raise


@app.local_entrypoint()
def main(source_sha: str) -> None:
    ensure_data.remote()
    result = run_screen.remote(source_sha)
    print("CPW_V2_SEQUENCE_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
