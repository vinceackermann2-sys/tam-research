from __future__ import annotations

import json
import math
from pathlib import Path
import re
import time

import modal

from tam_research.cpw_v0_mechanism.protocol import (
    ABLATION_MAP,
    ARM_ORDER,
    CPW_PARAMETERS,
    EFFECT_THRESHOLD,
    GRAD_ACCUM_STEPS,
    ISSUE,
    MICRO_BATCH_SIZE,
    REPLICATION_SEEDS,
    REPLICATION_TOKENS,
    RESULT_ROOT,
    SEQ_LEN,
    SMOKE_SEED,
    SMOKE_TOKENS,
    TRANSFORMER_PARAMETERS,
)

APP_NAME = "tam-research-cpw-v0-mechanism-1247"
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
    return result


def _finite(r: dict) -> bool:
    values = [
        float(r["final_eval"]["nll"]),
        float(r["final_eval"]["perplexity"]),
        float(r["training_tokens_per_second"]),
        float(r["peak_vram_gb"]),
        float(r["total_compute_seconds"]),
    ]
    return all(math.isfinite(v) and v >= 0.0 for v in values)


def _cpw_integrity(arm: str, r: dict) -> bool:
    if int(r["parameters"]) != CPW_PARAMETERS or not _finite(r):
        return False
    router = r["final_eval"].get("router")
    if not router:
        return False
    mean = router.get("mean") or {}
    try:
        active = float(mean["active_fraction"])
        entropy = float(mean["router_entropy"])
        usages = [
            float(mean["attention_usage"]),
            float(mean["world_usage"]),
            float(mean["sequence_usage"]),
            float(mean["memory_usage"]),
        ]
    except (KeyError, TypeError, ValueError):
        return False
    expected_active = 1.0 if arm == "uniform_all" else 0.5
    if not (
        math.isfinite(active)
        and abs(active - expected_active) <= 2e-3
        and math.isfinite(entropy)
        and entropy > 0.0
        and all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in usages)
    ):
        return False
    if arm == "no_aux" and abs(float(r["last_auxiliary_loss"])) > 1e-8:
        return False
    disabled = {
        "no_attention": "attention_usage",
        "no_world": "world_usage",
        "no_sequence": "sequence_usage",
        "no_memory": "memory_usage",
    }.get(arm)
    if disabled is not None and abs(float(mean[disabled])) > 2e-3:
        return False
    return True


def _summary(r: dict) -> dict:
    return {
        "arm": r["arm"],
        "seed": int(r["seed"]),
        "parameters": int(r["parameters"]),
        "nll": float(r["final_eval"]["nll"]),
        "perplexity": float(r["final_eval"]["perplexity"]),
        "training_tps": float(r["training_tokens_per_second"]),
        "total_compute_seconds": float(r["total_compute_seconds"]),
        "peak_vram_gib": float(r["peak_vram_gb"]),
        "last_auxiliary_loss": (
            float(r["last_auxiliary_loss"])
            if "last_auxiliary_loss" in r
            else None
        ),
        "router": r["final_eval"].get("router"),
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
    from tam_research.cpw_v0_mechanism.train import train_panel_arm

    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise RuntimeError(f"invalid source SHA: {source_sha!r}")

    root = Path(RESULT_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    attempt_path = root / "ATTEMPT.json"
    result_path = root / "RESULT.json"
    failure_path = root / "FAILURE.json"
    if attempt_path.exists() or result_path.exists() or failure_path.exists():
        raise RuntimeError("CPW-v0 mechanism namespace is already consumed")

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
    print("CPW_V0_MECHANISM_ATTEMPT=" + json.dumps(attempt, sort_keys=True), flush=True)

    try:
        smoke: dict[str, dict] = {}
        for arm in ARM_ORDER:
            raw = train_panel_arm(
                arm=arm,
                seed=SMOKE_SEED,
                data_dir="/vol/data/fineweb-edu-gpt2",
                run_root=str(root / "smoke"),
                token_budget=SMOKE_TOKENS,
                seq_len=SEQ_LEN,
                micro_batch_size=MICRO_BATCH_SIZE,
                grad_accum_steps=GRAD_ACCUM_STEPS,
            )
            smoke[arm] = raw
            volume.commit()
            print("CPW_V0_MECHANISM_SMOKE_ARM=" + json.dumps(_summary(raw), sort_keys=True), flush=True)

        smoke_pass = (
            _finite(smoke["transformer"])
            and int(smoke["transformer"]["parameters"]) == TRANSFORMER_PARAMETERS
            and all(_cpw_integrity(arm, smoke[arm]) for arm in ARM_ORDER if arm != "transformer")
        )
        print("CPW_V0_MECHANISM_SMOKE_GATE=" + json.dumps({"pass": smoke_pass}, sort_keys=True), flush=True)

        by_seed: dict[int, dict[str, dict]] = {}
        if smoke_pass:
            for seed in REPLICATION_SEEDS:
                row: dict[str, dict] = {}
                for arm in ARM_ORDER:
                    raw = train_panel_arm(
                        arm=arm,
                        seed=seed,
                        data_dir="/vol/data/fineweb-edu-gpt2",
                        run_root=str(root / "replication" / f"seed-{seed}"),
                        token_budget=REPLICATION_TOKENS,
                        seq_len=SEQ_LEN,
                        micro_batch_size=MICRO_BATCH_SIZE,
                        grad_accum_steps=GRAD_ACCUM_STEPS,
                    )
                    row[arm] = raw
                    volume.commit()
                    print("CPW_V0_MECHANISM_RUN=" + json.dumps(_summary(raw), sort_keys=True), flush=True)
                by_seed[int(seed)] = row

        integrity = (
            smoke_pass
            and len(by_seed) == len(REPLICATION_SEEDS)
            and all(
                _finite(row["transformer"])
                and int(row["transformer"]["parameters"]) == TRANSFORMER_PARAMETERS
                and all(_cpw_integrity(arm, row[arm]) for arm in ARM_ORDER if arm != "transformer")
                for row in by_seed.values()
            )
        )

        arm_summary: dict[str, dict] = {}
        effects: dict[str, dict] = {}
        if integrity:
            for arm in ARM_ORDER:
                vals = [by_seed[s][arm] for s in REPLICATION_SEEDS]
                arm_summary[arm] = {
                    "mean_nll": sum(float(v["final_eval"]["nll"]) for v in vals) / len(vals),
                    "mean_training_tps": sum(float(v["training_tokens_per_second"]) for v in vals) / len(vals),
                    "mean_peak_vram_gib": sum(float(v["peak_vram_gb"]) for v in vals) / len(vals),
                    "per_seed_nll": {str(s): float(by_seed[s][arm]["final_eval"]["nll"]) for s in REPLICATION_SEEDS},
                }

            full = arm_summary["full"]
            transformer = arm_summary["transformer"]
            full_vs_transformer = {
                "transformer_minus_full_mean_nll": transformer["mean_nll"] - full["mean_nll"],
                "full_wins": sum(
                    float(by_seed[s]["full"]["final_eval"]["nll"])
                    < float(by_seed[s]["transformer"]["final_eval"]["nll"])
                    for s in REPLICATION_SEEDS
                ),
                "full_training_tps_ratio": full["mean_training_tps"] / max(transformer["mean_training_tps"], 1e-12),
            }

            for mechanism, ablation in ABLATION_MAP.items():
                deltas = {
                    str(s): (
                        float(by_seed[s][ablation]["final_eval"]["nll"])
                        - float(by_seed[s]["full"]["final_eval"]["nll"])
                    )
                    for s in REPLICATION_SEEDS
                }
                mean_effect = sum(deltas.values()) / len(deltas)
                full_wins = sum(v > 0.0 for v in deltas.values())
                effects[mechanism] = {
                    "ablation": ablation,
                    "ablation_minus_full_nll_by_seed": deltas,
                    "mean_effect": mean_effect,
                    "full_wins": full_wins,
                    "supported": mean_effect >= EFFECT_THRESHOLD and full_wins >= 2,
                }
        else:
            full_vs_transformer = None

        final = {
            "issue": ISSUE,
            "source_sha": source_sha,
            "classification": (
                "CPW_V0_MECHANISM_PANEL_COMPLETE"
                if integrity
                else "CPW_V0_MECHANISM_PANEL_INVALID"
            ),
            "smoke_pass": smoke_pass,
            "replication_integrity": integrity,
            "arm_summary": arm_summary,
            "full_vs_transformer": full_vs_transformer,
            "effects": effects,
            "breakthrough_claim_allowed": False,
            "scale_up_authorized": False,
            "completed_unix": time.time(),
        }
        result_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
        volume.commit()
        print("CPW_V0_MECHANISM_RESULT=" + json.dumps(final, sort_keys=True), flush=True)
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
        print("CPW_V0_MECHANISM_FAILURE=" + json.dumps(failure, sort_keys=True), flush=True)
        raise


@app.local_entrypoint()
def main(source_sha: str) -> None:
    ensure_data.remote()
    result = run_panel.remote(source_sha)
    print("CPW_V0_MECHANISM_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
