from __future__ import annotations

import json
import math
from pathlib import Path
import time

import modal

from tam_research.attention_width_100m.protocol import (
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
    MIN_THROUGHPUT_RATIO,
    SMOKE_SEED,
    SMOKE_TOKENS,
)

APP_NAME = "tam-research-attention-width-100m-1238"
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
        "ATTENTION_WIDTH_100M_DATA_GUARD="
        + json.dumps(result, sort_keys=True),
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


def _summary(result: dict) -> dict:
    parameters = int(result["parameters"])
    return {
        "seed": int(result["seed"]),
        "nll": float(result["final_eval"]["nll"]),
        "perplexity": float(result["final_eval"]["perplexity"]),
        "training_tps": float(result["training_tokens_per_second"]),
        "peak_vram_gib": float(result["peak_vram_gb"]),
        "parameters": parameters,
        "active_parameters": parameters,
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
    import re

    from tam_research.attention_width_100m.train import (
        train_attention_width_100m_arm,
    )

    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise RuntimeError(f"invalid source SHA: {source_sha!r}")

    root = Path(RESULT_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    attempt_path = root / "ATTEMPT.json"
    result_path = root / "RESULT.json"
    failure_path = root / "FAILURE.json"
    if attempt_path.exists() or result_path.exists() or failure_path.exists():
        raise RuntimeError(
            "attention-width 100M #1238 namespace is already consumed"
        )

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
        "combination_authorized": False,
        "production_claim_allowed": False,
    }
    attempt_path.write_text(json.dumps(attempt, indent=2), encoding="utf-8")
    volume.commit()
    print(
        "ATTENTION_WIDTH_100M_ATTEMPT="
        + json.dumps(attempt, sort_keys=True),
        flush=True,
    )

    try:
        smoke_raw: dict[str, dict] = {}
        smoke: dict[str, dict] = {}
        for arm in ARMS:
            print(
                json.dumps(
                    {
                        "stage": "smoke_start",
                        "arm": arm,
                        "seed": SMOKE_SEED,
                        "tokens": SMOKE_TOKENS,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            result = train_attention_width_100m_arm(
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
            and smoke[arm]["active_parameters"] == EXPECTED_PARAMETERS
            for arm in ARMS
        )
        print(
            "ATTENTION_WIDTH_100M_SMOKE="
            + json.dumps(
                {"arms": smoke, "stage2_authorized": smoke_pass},
                sort_keys=True,
            ),
            flush=True,
        )

        paired: list[dict[str, dict]] = []
        raw_replication: list[dict[str, dict]] = []
        if smoke_pass:
            for seed in REPLICATION_SEEDS:
                row_raw: dict[str, dict] = {}
                row: dict[str, dict] = {}
                for arm in ARMS:
                    print(
                        json.dumps(
                            {
                                "stage": "replication_start",
                                "arm": arm,
                                "seed": seed,
                                "tokens": REPLICATION_TOKENS,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    result = train_attention_width_100m_arm(
                        arm=arm,
                        seed=seed,
                        data_dir="/vol/data/fineweb-edu-gpt2",
                        run_root=str(
                            root / "replication" / f"seed-{seed}"
                        ),
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
                    "ATTENTION_WIDTH_100M_PAIR="
                    + json.dumps(
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
                and row[arm]["active_parameters"] == EXPECTED_PARAMETERS
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
                arm: sum(row[arm]["training_tps"] for row in paired)
                / len(paired)
                for arm in ARMS
            }
            baseline = "global512_ff2048"
            reduced = "global448_ff2176"
            effect = mean_nll[baseline] - mean_nll[reduced]
            reduced_wins = sum(
                row[reduced]["nll"] < row[baseline]["nll"]
                for row in paired
            )
            throughput_ratio = (
                mean_tps[reduced] / max(mean_tps[baseline], 1e-12)
            )
            replicated = (
                effect >= EFFECT_THRESHOLD_NLL
                and reduced_wins >= 2
                and throughput_ratio >= MIN_THROUGHPUT_RATIO
            )
            attribution = {
                "GLOBAL_WIDTH_EFFECT_100M": effect,
                "reduced_wins_vs_baseline": reduced_wins,
                "throughput_ratio": throughput_ratio,
                "GLOBAL_REDUCED_WIDTH_100M_REPLICATED": replicated,
            }
            classification = (
                "ATTENTION_WIDTH_GLOBAL_100M_REPLICATED"
                if replicated
                else "ATTENTION_WIDTH_GLOBAL_100M_NOT_REPLICATED"
            )
        else:
            mean_nll = {}
            mean_tps = {}
            attribution = {
                "GLOBAL_REDUCED_WIDTH_100M_REPLICATED": False,
            }
            classification = "ATTENTION_WIDTH_GLOBAL_100M_INTEGRITY_STOP"

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
            "attribution": attribution,
            "breakthrough_claim_allowed": False,
            "combination_authorized": False,
            "production_claim_allowed": False,
            "completed_unix": time.time(),
        }
        result_path.write_text(
            json.dumps(final, indent=2),
            encoding="utf-8",
        )
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
            "combination_authorized": False,
            "production_claim_allowed": False,
            "failed_unix": time.time(),
        }
        failure_path.write_text(
            json.dumps(failure, indent=2),
            encoding="utf-8",
        )
        volume.commit()
        print(
            "ATTENTION_WIDTH_100M_FAILURE="
            + json.dumps(failure, sort_keys=True),
            flush=True,
        )
        raise


@app.local_entrypoint()
def main(source_sha: str) -> None:
    ensure_data.remote()
    result = run_replication.remote(source_sha)
    print(
        "ATTENTION_WIDTH_100M_RESULT="
        + json.dumps(result, sort_keys=True),
        flush=True,
    )
