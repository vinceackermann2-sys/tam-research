from __future__ import annotations

import json
import math
from pathlib import Path
import re
import time

import modal

from tam_research.cpw_v1_pairwise.protocol import (
    ARM_ORDER,
    EXPECTED_PARAMETERS,
    GRAD_ACCUM_STEPS,
    ISSUE,
    MICRO_BATCH_SIZE,
    REPLICATION_SEEDS,
    REPLICATION_TOKENS,
    RESULT_ROOT,
    SEQ_LEN,
    SMOKE_SEED,
    SMOKE_TOKENS,
)

APP_NAME = "tam-research-cpw-v1-pairwise-1252"
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


def _finite(r: dict) -> bool:
    ev = r["final_eval"]
    values = [
        float(ev["nll"]),
        float(ev["perplexity"]),
        float(r["training_tokens_per_second"]),
        float(r["training_seconds"]),
        float(r["total_compute_seconds"]),
        float(r["peak_vram_gb"]),
    ]
    return all(math.isfinite(v) and v >= 0.0 for v in values)


def _summary(arm: str, r: dict) -> dict:
    return {
        "arm": arm,
        "seed": int(r["seed"]),
        "parameters": int(r["parameters"]),
        "nll": float(r["final_eval"]["nll"]),
        "perplexity": float(r["final_eval"]["perplexity"]),
        "training_tps": float(r["training_tokens_per_second"]),
        "training_seconds": float(r["training_seconds"]),
        "total_compute_seconds": float(r["total_compute_seconds"]),
        "peak_vram_gib": float(r["peak_vram_gb"]),
        "telemetry": r["final_eval"].get("router"),
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
    from tam_research.cpw_v1_pairwise.train import train_pairwise_candidate

    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise RuntimeError(f"invalid source SHA: {source_sha!r}")

    root = Path(RESULT_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    attempt_path = root / "ATTEMPT.json"
    result_path = root / "RESULT.json"
    failure_path = root / "FAILURE.json"
    if attempt_path.exists() or result_path.exists() or failure_path.exists():
        raise RuntimeError("CPW-v1 pairwise panel namespace already consumed")

    attempt = {
        "issue": ISSUE,
        "source_sha": source_sha,
        "arms": list(ARM_ORDER),
        "smoke_seed": SMOKE_SEED,
        "replication_seeds": list(REPLICATION_SEEDS),
        "smoke_tokens": SMOKE_TOKENS,
        "replication_tokens": REPLICATION_TOKENS,
        "retry_authorized": False,
        "resume_authorized": False,
        "breakthrough_claim_allowed": False,
        "scale_up_authorized": False,
        "started_unix": time.time(),
    }
    attempt_path.write_text(json.dumps(attempt, indent=2), encoding="utf-8")
    volume.commit()
    print("CPW_V1_PAIRWISE_ATTEMPT=" + json.dumps(attempt, sort_keys=True), flush=True)

    try:
        smoke_raw: dict[str, dict] = {}
        smoke: dict[str, dict] = {}
        for arm in ARM_ORDER:
            raw = train_pairwise_candidate(
                arm=arm,
                seed=SMOKE_SEED,
                data_dir="/vol/data/fineweb-edu-gpt2",
                run_root=str(root / "smoke"),
                token_budget=SMOKE_TOKENS,
                seq_len=SEQ_LEN,
                micro_batch_size=MICRO_BATCH_SIZE,
                grad_accum_steps=GRAD_ACCUM_STEPS,
            )
            smoke_raw[arm] = raw
            smoke[arm] = _summary(arm, raw)
            volume.commit()
            print("CPW_V1_PAIRWISE_SMOKE_ARM=" + json.dumps(smoke[arm], sort_keys=True), flush=True)

        smoke_pass = all(
            _finite(smoke_raw[arm])
            and int(smoke_raw[arm]["parameters"]) == EXPECTED_PARAMETERS[arm]
            for arm in ARM_ORDER
        )
        print("CPW_V1_PAIRWISE_SMOKE_GATE=" + json.dumps({"pass": smoke_pass}, sort_keys=True), flush=True)

        by_seed: dict[int, dict[str, dict]] = {}
        if smoke_pass:
            for seed in REPLICATION_SEEDS:
                row: dict[str, dict] = {}
                for arm in ARM_ORDER:
                    raw = train_pairwise_candidate(
                        arm=arm,
                        seed=seed,
                        data_dir="/vol/data/fineweb-edu-gpt2",
                        run_root=str(root / "replication" / f"seed-{seed}"),
                        token_budget=REPLICATION_TOKENS,
                        seq_len=SEQ_LEN,
                        micro_batch_size=MICRO_BATCH_SIZE,
                        grad_accum_steps=GRAD_ACCUM_STEPS,
                    )
                    if not _finite(raw):
                        raise RuntimeError(f"non-finite arm={arm} seed={seed}")
                    if int(raw["parameters"]) != EXPECTED_PARAMETERS[arm]:
                        raise RuntimeError(f"parameter drift arm={arm}")
                    row[arm] = _summary(arm, raw)
                    volume.commit()
                    print("CPW_V1_PAIRWISE_RUN=" + json.dumps(row[arm], sort_keys=True), flush=True)
                by_seed[int(seed)] = row

        integrity = smoke_pass and len(by_seed) == len(REPLICATION_SEEDS)

        arm_summary: dict[str, dict] = {}
        candidate_results: dict[str, dict] = {}
        successor: str | None = None
        if integrity:
            for arm in ARM_ORDER:
                vals = [by_seed[s][arm] for s in REPLICATION_SEEDS]
                arm_summary[arm] = {
                    "mean_nll": sum(v["nll"] for v in vals) / len(vals),
                    "mean_training_tps": sum(v["training_tps"] for v in vals) / len(vals),
                    "mean_training_seconds": sum(v["training_seconds"] for v in vals) / len(vals),
                    "mean_total_compute_seconds": sum(v["total_compute_seconds"] for v in vals) / len(vals),
                    "mean_peak_vram_gib": sum(v["peak_vram_gib"] for v in vals) / len(vals),
                    "parameters": EXPECTED_PARAMETERS[arm],
                    "per_seed_nll": {str(s): by_seed[s][arm]["nll"] for s in REPLICATION_SEEDS},
                }

            t = arm_summary["transformer"]
            for arm in ("sequence_memory", "world_memory", "world_sequence"):
                a = arm_summary[arm]
                wins = sum(
                    by_seed[s][arm]["nll"] < by_seed[s]["transformer"]["nll"]
                    for s in REPLICATION_SEEDS
                )
                tps_ratio = a["mean_training_tps"] / max(t["mean_training_tps"], 1e-12)
                training_seconds_ratio = a["mean_training_seconds"] / max(t["mean_training_seconds"], 1e-12)
                total_compute_ratio = a["mean_total_compute_seconds"] / max(t["mean_total_compute_seconds"], 1e-12)
                quality = a["mean_nll"] < t["mean_nll"] and wins >= 2
                efficiency = quality and tps_ratio >= 0.90
                pareto = quality and tps_ratio >= 1.0 and training_seconds_ratio <= 1.0
                candidate_results[arm] = {
                    "transformer_minus_candidate_mean_nll": t["mean_nll"] - a["mean_nll"],
                    "wins_vs_transformer": wins,
                    "training_tps_ratio": tps_ratio,
                    "training_seconds_ratio": training_seconds_ratio,
                    "total_compute_ratio": total_compute_ratio,
                    "parameter_fraction_of_transformer": EXPECTED_PARAMETERS[arm] / EXPECTED_PARAMETERS["transformer"],
                    "quality_supported": quality,
                    "efficiency_supported": efficiency,
                    "pareto_supported": pareto,
                }

            pareto_candidates = [
                arm for arm, r in candidate_results.items()
                if r["pareto_supported"]
            ]
            if pareto_candidates:
                successor = min(
                    pareto_candidates,
                    key=lambda arm: arm_summary[arm]["mean_nll"],
                )
            else:
                efficient = [
                    arm for arm, r in candidate_results.items()
                    if r["efficiency_supported"]
                ]
                if efficient:
                    successor = max(
                        efficient,
                        key=lambda arm: candidate_results[arm]["training_tps_ratio"],
                    )

        final = {
            "issue": ISSUE,
            "source_sha": source_sha,
            "classification": (
                "CPW_V1_PAIRWISE_PANEL_COMPLETE"
                if integrity
                else "CPW_V1_PAIRWISE_PANEL_INVALID"
            ),
            "smoke_pass": smoke_pass,
            "replication_integrity": integrity,
            "arm_summary": arm_summary,
            "candidate_results": candidate_results,
            "selected_successor": successor,
            "breakthrough_claim_allowed": False,
            "scale_up_authorized": False,
            "completed_unix": time.time(),
        }
        result_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
        volume.commit()
        print("CPW_V1_PAIRWISE_RESULT=" + json.dumps(final, sort_keys=True), flush=True)
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
        print("CPW_V1_PAIRWISE_FAILURE=" + json.dumps(failure, sort_keys=True), flush=True)
        raise


@app.local_entrypoint()
def main(source_sha: str) -> None:
    ensure_data.remote()
    result = run_panel.remote(source_sha)
    print("CPW_V1_PAIRWISE_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
