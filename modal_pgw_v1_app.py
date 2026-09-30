from __future__ import annotations

import json
import math
from pathlib import Path
import time

import modal

from tam_research.pgw_v1.protocol import (
    EXPECTED_PARAMETERS,
    EXPECTED_SELECTED_FRACTION,
    ISSUE,
    MIN_PGW_THROUGHPUT_RATIO,
    POSITIVE_CLASSIFICATION,
    REPLICATION_SEEDS,
    REPLICATION_TOKENS,
    RESULT_ROOT,
    SEQ_LEN,
    MICRO_BATCH_SIZE,
    GRAD_ACCUM_STEPS,
    SMOKE_SEED,
    SMOKE_TOKENS,
    STOP_CLASSIFICATION,
)

APP_NAME = "tam-research-pgw-v1-1159"
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
    print("PGW_V1_DATA_GUARD=" + json.dumps(result, sort_keys=True), flush=True)
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


def _selected_fraction(result: dict) -> float:
    router = result["final_eval"].get("router")
    if not router:
        return float("nan")
    return float(router["mean"]["selected_fraction"])


def _pair_summary(transformer: dict, pgw: dict) -> dict:
    t_nll = float(transformer["final_eval"]["nll"])
    p_nll = float(pgw["final_eval"]["nll"])
    t_tps = float(transformer["training_tokens_per_second"])
    p_tps = float(pgw["training_tokens_per_second"])
    return {
        "seed": int(transformer["seed"]),
        "transformer_nll": t_nll,
        "pgw_nll": p_nll,
        "transformer_minus_pgw_nll": t_nll - p_nll,
        "transformer_training_tps": t_tps,
        "pgw_training_tps": p_tps,
        "throughput_ratio": p_tps / max(t_tps, 1e-12),
        "transformer_peak_vram_gib": float(transformer["peak_vram_gb"]),
        "pgw_peak_vram_gib": float(pgw["peak_vram_gb"]),
        "pgw_selected_fraction": _selected_fraction(pgw),
        "transformer_parameters": int(transformer["parameters"]),
        "pgw_parameters": int(pgw["parameters"]),
    }


def _pair_parameters_exact(pair: dict) -> bool:
    return (
        pair["transformer_parameters"] == EXPECTED_PARAMETERS
        and pair["pgw_parameters"] == EXPECTED_PARAMETERS
    )


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
    import re

    from tam_research.pgw_v1.train import train_pgw_v1_candidate

    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise RuntimeError(f"invalid source SHA: {source_sha!r}")

    root = Path(RESULT_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    attempt_path = root / "ATTEMPT.json"
    result_path = root / "RESULT.json"
    failure_path = root / "FAILURE.json"
    if attempt_path.exists() or result_path.exists() or failure_path.exists():
        raise RuntimeError("PGW-v1 #1159 result namespace is already consumed")

    attempt = {
        "issue": ISSUE,
        "source_sha": source_sha,
        "smoke_seed": SMOKE_SEED,
        "replication_seeds": list(REPLICATION_SEEDS),
        "started_unix": time.time(),
        "gpu": "H100!",
        "retry_authorized": False,
        "resume_authorized": False,
    }
    attempt_path.write_text(json.dumps(attempt, indent=2), encoding="utf-8")
    volume.commit()
    print("PGW_V1_ATTEMPT=" + json.dumps(attempt, sort_keys=True), flush=True)

    try:
        smoke: dict[str, dict] = {}
        for architecture in ("transformer", "pgwv1"):
            print(
                json.dumps(
                    {
                        "stage": "smoke_start",
                        "architecture": architecture,
                        "seed": SMOKE_SEED,
                        "tokens": SMOKE_TOKENS,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            smoke[architecture] = train_pgw_v1_candidate(
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

        smoke_pair = _pair_summary(smoke["transformer"], smoke["pgwv1"])
        smoke_pass = (
            _finite(smoke["transformer"])
            and _finite(smoke["pgwv1"])
            and _pair_parameters_exact(smoke_pair)
            and abs(
                smoke_pair["pgw_selected_fraction"]
                - EXPECTED_SELECTED_FRACTION
            )
            <= 1e-12
        )
        print(
            "PGW_V1_SMOKE="
            + json.dumps(
                {
                    **smoke_pair,
                    "stage2_authorized": smoke_pass,
                },
                sort_keys=True,
            ),
            flush=True,
        )

        replication_pairs: list[dict] = []
        raw_replication: list[dict[str, dict]] = []
        if smoke_pass:
            for seed in REPLICATION_SEEDS:
                pair_raw: dict[str, dict] = {}
                for architecture in ("transformer", "pgwv1"):
                    print(
                        json.dumps(
                            {
                                "stage": "replication_start",
                                "architecture": architecture,
                                "seed": seed,
                                "tokens": REPLICATION_TOKENS,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    pair_raw[architecture] = train_pgw_v1_candidate(
                        architecture=architecture,
                        seed=seed,
                        data_dir="/vol/data/fineweb-edu-gpt2",
                        run_root=str(root / "replication"),
                        token_budget=REPLICATION_TOKENS,
                        seq_len=SEQ_LEN,
                        micro_batch_size=MICRO_BATCH_SIZE,
                        grad_accum_steps=GRAD_ACCUM_STEPS,
                    )
                    volume.commit()
                raw_replication.append(pair_raw)
                pair = _pair_summary(
                    pair_raw["transformer"],
                    pair_raw["pgwv1"],
                )
                replication_pairs.append(pair)
                print(
                    "PGW_V1_PAIR=" + json.dumps(pair, sort_keys=True),
                    flush=True,
                )

        if replication_pairs:
            t_mean = sum(
                pair["transformer_nll"] for pair in replication_pairs
            ) / len(replication_pairs)
            p_mean = sum(
                pair["pgw_nll"] for pair in replication_pairs
            ) / len(replication_pairs)
            wins = sum(
                pair["pgw_nll"] < pair["transformer_nll"]
                for pair in replication_pairs
            )
            throughput_ratio = sum(
                pair["throughput_ratio"] for pair in replication_pairs
            ) / len(replication_pairs)
            finite_ok = all(
                _finite(pair_raw["transformer"])
                and _finite(pair_raw["pgwv1"])
                for pair_raw in raw_replication
            )
            params_exact = all(
                _pair_parameters_exact(pair)
                for pair in replication_pairs
            )
            selected_ok = all(
                abs(
                    pair["pgw_selected_fraction"]
                    - EXPECTED_SELECTED_FRACTION
                )
                <= 1e-12
                for pair in replication_pairs
            )
            positive = (
                finite_ok
                and params_exact
                and selected_ok
                and p_mean < t_mean
                and wins >= 2
                and throughput_ratio >= MIN_PGW_THROUGHPUT_RATIO
            )
        else:
            t_mean = float("nan")
            p_mean = float("nan")
            wins = 0
            throughput_ratio = float("nan")
            finite_ok = False
            params_exact = False
            selected_ok = False
            positive = False

        classification = (
            POSITIVE_CLASSIFICATION if positive else STOP_CLASSIFICATION
        )
        final = {
            "issue": ISSUE,
            "source_sha": source_sha,
            "classification": classification,
            "smoke_pass": smoke_pass,
            "smoke": smoke_pair,
            "replication": replication_pairs,
            "replication_summary": {
                "transformer_mean_nll": t_mean,
                "pgw_mean_nll": p_mean,
                "pgw_wins": wins,
                "mean_throughput_ratio": throughput_ratio,
                "finite": finite_ok,
                "parameters_exact": params_exact,
                "selected_fraction_exact": selected_ok,
            },
            "breakthrough_claim_allowed": False,
            "next_step_if_positive": (
                "preregister mechanism controls and equal-compute/compiled evaluation"
            ),
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
            "failed_unix": time.time(),
        }
        failure_path.write_text(
            json.dumps(failure, indent=2),
            encoding="utf-8",
        )
        volume.commit()
        print("PGW_V1_FAILURE=" + json.dumps(failure, sort_keys=True), flush=True)
        raise


@app.local_entrypoint()
def main(source_sha: str) -> None:
    ensure_data.remote()
    result = run_screen.remote(source_sha)
    print("PGW_V1_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
