from __future__ import annotations

import json
import math
from pathlib import Path
import time

import modal

from tam_research.attention_width_factorial.protocol import (
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

APP_NAME = "tam-research-attention-width-factorial-1218"
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
        "ATTENTION_WIDTH_FACTORIAL_DATA_GUARD="
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
def run_panel(source_sha: str) -> dict:
    import re

    from tam_research.attention_width_factorial.train import (
        train_attention_width_arm,
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
            "attention-width factorial #1218 namespace is already consumed"
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
        "scale_up_authorized": False,
    }
    attempt_path.write_text(json.dumps(attempt, indent=2), encoding="utf-8")
    volume.commit()
    print(
        "ATTENTION_WIDTH_FACTORIAL_ATTEMPT="
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
            result = train_attention_width_arm(
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
            "ATTENTION_WIDTH_FACTORIAL_SMOKE="
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
                    result = train_attention_width_arm(
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
                    "ATTENTION_WIDTH_FACTORIAL_PAIR="
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
            a = "global256_ff1024"
            b = "local256_ff1024"
            c = "global224_ff1088"
            d = "local224_ff1088"

            global_effect = mean_nll[a] - mean_nll[c]
            local_effect = mean_nll[b] - mean_nll[d]
            locality_cost_256 = mean_nll[b] - mean_nll[a]
            locality_cost_224 = mean_nll[d] - mean_nll[c]
            interaction = local_effect - global_effect

            global_wins = sum(
                row[c]["nll"] < row[a]["nll"] for row in paired
            )
            local_wins = sum(
                row[d]["nll"] < row[b]["nll"] for row in paired
            )
            global_supported = (
                global_effect >= EFFECT_THRESHOLD_NLL
                and global_wins >= 2
            )
            local_supported = (
                local_effect >= EFFECT_THRESHOLD_NLL
                and local_wins >= 2
            )
            interaction_supported = (
                local_supported
                and interaction >= EFFECT_THRESHOLD_NLL
            )

            attribution = {
                "GLOBAL_WIDTH_EFFECT": global_effect,
                "LOCAL_WIDTH_EFFECT": local_effect,
                "LOCALITY_COST_256": locality_cost_256,
                "LOCALITY_COST_224": locality_cost_224,
                "WIDTH_LOCALITY_INTERACTION": interaction,
                "global224_wins_vs_global256": global_wins,
                "local224_wins_vs_local256": local_wins,
                "GLOBAL_REDUCED_WIDTH_SUPPORTED": global_supported,
                "LOCAL_REDUCED_WIDTH_SUPPORTED": local_supported,
                "WIDTH_LOCALITY_INTERACTION_SUPPORTED": interaction_supported,
            }

            descriptive = {
                "local224_minus_global256_nll": mean_nll[d] - mean_nll[a],
                "local224_wins_vs_global256": sum(
                    row[d]["nll"] < row[a]["nll"] for row in paired
                ),
                "global224_minus_global256_nll": mean_nll[c] - mean_nll[a],
                "local224_minus_global224_nll": mean_nll[d] - mean_nll[c],
            }
            baseline_tps = mean_tps[a]
            throughput_vs_global256 = {
                arm: mean_tps[arm] / max(baseline_tps, 1e-12)
                for arm in ARMS
            }

            if interaction_supported:
                classification = (
                    "ATTENTION_WIDTH_LOCALITY_INTERACTION_SUPPORTED"
                )
            elif global_supported or local_supported:
                classification = "ATTENTION_WIDTH_REDUCTION_SUPPORTED"
            else:
                classification = "ATTENTION_WIDTH_FACTORIAL_UNSUPPORTED"
        else:
            mean_nll = {}
            mean_tps = {}
            throughput_vs_global256 = {}
            descriptive = {}
            attribution = {
                "GLOBAL_REDUCED_WIDTH_SUPPORTED": False,
                "LOCAL_REDUCED_WIDTH_SUPPORTED": False,
                "WIDTH_LOCALITY_INTERACTION_SUPPORTED": False,
            }
            classification = "ATTENTION_WIDTH_FACTORIAL_INTEGRITY_STOP"

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
            "throughput_vs_global256": throughput_vs_global256,
            "attribution": attribution,
            "descriptive": descriptive,
            "breakthrough_claim_allowed": False,
            "scale_up_authorized": False,
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
            "scale_up_authorized": False,
            "failed_unix": time.time(),
        }
        failure_path.write_text(
            json.dumps(failure, indent=2),
            encoding="utf-8",
        )
        volume.commit()
        print(
            "ATTENTION_WIDTH_FACTORIAL_FAILURE="
            + json.dumps(failure, sort_keys=True),
            flush=True,
        )
        raise


@app.local_entrypoint()
def main(source_sha: str) -> None:
    ensure_data.remote()
    result = run_panel.remote(source_sha)
    print(
        "ATTENTION_WIDTH_FACTORIAL_RESULT="
        + json.dumps(result, sort_keys=True),
        flush=True,
    )
