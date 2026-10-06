from __future__ import annotations

import json
import math
from pathlib import Path
import re
import time

import modal

from tam_research.cpw_v3_sparse_world.protocol import (
    ARMS,
    CPW_V3_PARAMETERS,
    ISSUE,
    MICRO_BATCH_SIZE,
    POSITIVE_CLASSIFICATION,
    QUALITY_CLASSIFICATION,
    REPLICATION_SEEDS,
    REPLICATION_TOKENS,
    RESULT_ROOT,
    SEQ_LEN,
    SMOKE_SEED,
    SMOKE_TOKENS,
    SPARSE_WORLD_ARMS,
    STOP_CLASSIFICATION,
    TRANSFORMER_PARAMETERS,
    GRAD_ACCUM_STEPS,
)

APP_NAME = "tam-research-cpw-v3-sparse-world-1261"
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


def _verify_frozen_data() -> dict:
    root = Path("/vol/data/fineweb-edu-gpt2")
    meta_path = root / "meta.json"
    train_path = root / "train.bin"
    val_path = root / "val.bin"
    if not (meta_path.exists() and train_path.exists() and val_path.exists()):
        raise RuntimeError("frozen FineWeb-Edu token shard is missing")
    meta = json.loads(meta_path.read_text())
    expected = {
        "dataset": "HuggingFaceFW/fineweb-edu",
        "dataset_config": "sample-10BT",
        "tokenizer": "gpt2",
        "seed": 1234,
        "train_tokens": 25_000_000,
        "val_tokens": 2_000_000,
        "dtype": "uint16",
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise RuntimeError(
                f"frozen data mismatch for {key}: {meta.get(key)!r} != {value!r}"
            )
    if train_path.stat().st_size != 25_000_000 * 2:
        raise RuntimeError("unexpected train.bin byte size")
    if val_path.stat().st_size != 2_000_000 * 2:
        raise RuntimeError("unexpected val.bin byte size")
    return {
        "verified": True,
        "writes_performed": False,
        "meta": meta,
        "train_bytes": train_path.stat().st_size,
        "val_bytes": val_path.stat().st_size,
    }


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
        "telemetry": r["final_eval"].get("router"),
    }


def _candidate_stats(
    arm: str,
    rows: list[dict],
) -> dict:
    transformer = [row["transformer"] for row in rows]
    candidate = [row[arm] for row in rows]
    t_mean = sum(x["nll"] for x in transformer) / len(transformer)
    c_mean = sum(x["nll"] for x in candidate) / len(candidate)
    wins = sum(c["nll"] < t["nll"] for t, c in zip(transformer, candidate))
    tps_ratios = [
        c["training_tps"] / max(t["training_tps"], 1e-12)
        for t, c in zip(transformer, candidate)
    ]
    compute_ratios = [
        c["total_compute_seconds"] / max(t["total_compute_seconds"], 1e-12)
        for t, c in zip(transformer, candidate)
    ]
    per_seed = [
        {
            "seed": int(t["seed"]),
            "transformer_nll": t["nll"],
            "candidate_nll": c["nll"],
            "transformer_minus_candidate_nll": t["nll"] - c["nll"],
            "training_tps_ratio": tr,
            "total_compute_ratio": cr,
            "candidate_peak_vram_gib": c["peak_vram_gib"],
        }
        for t, c, tr, cr in zip(transformer, candidate, tps_ratios, compute_ratios)
    ]
    quality = c_mean < t_mean and wins >= 2
    pareto = (
        quality
        and sum(tps_ratios) / len(tps_ratios) >= 1.0
        and sum(compute_ratios) / len(compute_ratios) <= 1.0
    )
    strong = all(
        row["candidate_nll"] < row["transformer_nll"]
        and row["training_tps_ratio"] >= 1.0
        and row["total_compute_ratio"] <= 1.0
        for row in per_seed
    )
    return {
        "arm": arm,
        "transformer_mean_nll": t_mean,
        "candidate_mean_nll": c_mean,
        "mean_transformer_minus_candidate_nll": t_mean - c_mean,
        "wins": wins,
        "mean_training_tps_ratio": sum(tps_ratios) / len(tps_ratios),
        "mean_total_compute_ratio": sum(compute_ratios) / len(compute_ratios),
        "parameters": candidate[0]["parameters"],
        "parameter_fraction_of_transformer": (
            candidate[0]["parameters"] / TRANSFORMER_PARAMETERS
        ),
        "quality_supported": quality,
        "pareto_supported": pareto,
        "strong_pareto_3_of_3": strong,
        "per_seed": per_seed,
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
    from tam_research.cpw_v3_sparse_world.model import (
        ARM_WORLD_LAYERS,
        sparse_world_parameter_count,
    )
    from tam_research.cpw_v3_sparse_world.train import train_sparse_world_candidate

    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise RuntimeError(f"invalid source SHA: {source_sha!r}")

    data_guard = _verify_frozen_data()
    for arm in ARM_WORLD_LAYERS:
        if sparse_world_parameter_count(arm) != CPW_V3_PARAMETERS:
            raise RuntimeError(f"parameter drift for {arm}")

    root = Path(RESULT_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    attempt_path = root / "ATTEMPT.json"
    result_path = root / "RESULT.json"
    failure_path = root / "FAILURE.json"
    if attempt_path.exists() or result_path.exists() or failure_path.exists():
        raise RuntimeError("CPW-v3 namespace already consumed")

    attempt = {
        "issue": ISSUE,
        "source_sha": source_sha,
        "arms": list(ARMS),
        "smoke_seed": SMOKE_SEED,
        "replication_seeds": list(REPLICATION_SEEDS),
        "smoke_tokens": SMOKE_TOKENS,
        "replication_tokens": REPLICATION_TOKENS,
        "gpu": "H100!",
        "data_guard": data_guard,
        "retry_authorized": False,
        "resume_authorized": False,
        "breakthrough_claim_allowed": False,
        "scale_up_authorized": False,
        "started_unix": time.time(),
    }
    attempt_path.write_text(json.dumps(attempt, indent=2), encoding="utf-8")
    volume.commit()
    print("CPW_V3_ATTEMPT=" + json.dumps(attempt, sort_keys=True), flush=True)

    try:
        smoke_raw: dict[str, dict] = {}
        for arm in ARMS:
            smoke_raw[arm] = train_sparse_world_candidate(
                arm=arm,
                seed=SMOKE_SEED,
                data_dir="/vol/data/fineweb-edu-gpt2",
                run_root=str(root / "smoke"),
                token_budget=SMOKE_TOKENS,
                seq_len=SEQ_LEN,
                micro_batch_size=MICRO_BATCH_SIZE,
                grad_accum_steps=GRAD_ACCUM_STEPS,
            )
            volume.commit()

        smoke_integrity = (
            all(_finite(r) for r in smoke_raw.values())
            and int(smoke_raw["transformer"]["parameters"]) == TRANSFORMER_PARAMETERS
            and all(
                int(smoke_raw[a]["parameters"]) == CPW_V3_PARAMETERS
                for a in ARMS
                if a != "transformer"
            )
        )
        smoke = {arm: _summary(r) for arm, r in smoke_raw.items()}
        print(
            "CPW_V3_SMOKE="
            + json.dumps(
                {"pass": smoke_integrity, "arms": smoke},
                sort_keys=True,
            ),
            flush=True,
        )

        panels: list[dict] = []
        if smoke_integrity:
            for seed in REPLICATION_SEEDS:
                print(
                    "CPW_V3_SEED_START="
                    + json.dumps({"seed": seed}, sort_keys=True),
                    flush=True,
                )
                row: dict[str, dict] = {}
                for arm in ARMS:
                    result = train_sparse_world_candidate(
                        arm=arm,
                        seed=seed,
                        data_dir="/vol/data/fineweb-edu-gpt2",
                        run_root=str(root / "replication" / f"seed-{seed}"),
                        token_budget=REPLICATION_TOKENS,
                        seq_len=SEQ_LEN,
                        micro_batch_size=MICRO_BATCH_SIZE,
                        grad_accum_steps=GRAD_ACCUM_STEPS,
                    )
                    volume.commit()
                    row[arm] = _summary(result)
                    print(
                        "CPW_V3_ARM="
                        + json.dumps(
                            {"seed": seed, "arm": arm, "summary": row[arm]},
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                panels.append(row)

        integrity = (
            smoke_integrity
            and len(panels) == len(REPLICATION_SEEDS)
            and all(
                row["transformer"]["parameters"] == TRANSFORMER_PARAMETERS
                and all(
                    row[a]["parameters"] == CPW_V3_PARAMETERS
                    for a in ARMS
                    if a != "transformer"
                )
                for row in panels
            )
        )

        stats: dict[str, dict] = {}
        selected: str | None = None
        classification = STOP_CLASSIFICATION
        if integrity:
            for arm in SPARSE_WORLD_ARMS:
                stats[arm] = _candidate_stats(arm, panels)

            strong = [s for s in stats.values() if s["strong_pareto_3_of_3"]]
            pareto = [s for s in stats.values() if s["pareto_supported"]]
            quality = [s for s in stats.values() if s["quality_supported"]]

            if strong:
                selected = min(strong, key=lambda s: s["candidate_mean_nll"])["arm"]
                classification = POSITIVE_CLASSIFICATION
            elif pareto:
                selected = min(pareto, key=lambda s: s["candidate_mean_nll"])["arm"]
                classification = POSITIVE_CLASSIFICATION
            elif quality:
                selected = max(quality, key=lambda s: s["mean_training_tps_ratio"])["arm"]
                classification = QUALITY_CLASSIFICATION

        final = {
            "issue": ISSUE,
            "source_sha": source_sha,
            "classification": classification,
            "smoke_pass": smoke_integrity,
            "replication_integrity": integrity,
            "selected_successor": selected,
            "candidate_stats": stats,
            "replication": panels,
            "sequence_only_reference": (
                _candidate_stats("sequence_only", panels)
                if integrity
                else None
            ),
            "breakthrough_claim_allowed": False,
            "scale_up_authorized": False,
            "completed_unix": time.time(),
        }
        result_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
        volume.commit()
        print("CPW_V3_RESULT=" + json.dumps(final, sort_keys=True), flush=True)
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
        print("CPW_V3_FAILURE=" + json.dumps(failure, sort_keys=True), flush=True)
        raise


@app.local_entrypoint()
def main(source_sha: str) -> None:
    result = run_panel.remote(source_sha)
    print("CPW_V3_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
