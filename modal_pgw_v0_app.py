from __future__ import annotations

import json
import math
from pathlib import Path
import time

import modal

from tam_research.pgw_v0.protocol import (
    EXPECTED_SELECTED_FRACTION,
    GRAD_ACCUM_STEPS,
    ISSUE,
    MICRO_BATCH_SIZE,
    MIN_PGW_THROUGHPUT_RATIO,
    PARAMETER_MISMATCH_LIMIT,
    POSITIVE_CLASSIFICATION,
    REPLICATION_SEEDS,
    REPLICATION_TOKENS,
    RESULT_ROOT,
    SEQ_LEN,
    SMOKE_SEED,
    SMOKE_TOKENS,
    STOP_CLASSIFICATION,
)

APP_NAME = "tam-research-pgw-v0-1145"
VOLUME_NAME = "tam-research-data"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
github_secret = modal.Secret.from_name("github-secret")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.7,<2.11",
        "datasets>=4.0,<5",
        "transformers>=4.55,<5",
        "tokenizers>=0.21,<1",
        "numpy>=2.0,<3",
        "huggingface-hub>=0.34,<1",
        "PyGithub>=2.3,<3",
    )
    .add_local_python_source("tam_research")
)


def _comment(repo_full_name: str, issue_number: int, body: str) -> None:
    if not repo_full_name or not issue_number:
        print(body, flush=True)
        return
    import os
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print(body, flush=True)
        return
    try:
        import github
        client = github.Github(auth=github.Auth.Token(token))
        client.get_repo(repo_full_name).get_issue(number=issue_number).create_comment(body)
    except Exception as exc:
        print(f"[nonfatal-comment-error] {type(exc).__name__}: {exc}", flush=True)


def _both_comments(repo_full_name: str, trigger_issue: int, body: str) -> None:
    _comment(repo_full_name, trigger_issue, body)
    if trigger_issue != ISSUE:
        _comment(repo_full_name, ISSUE, body)


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=60 * 60,
    volumes={"/vol": volume},
    secrets=[github_secret],
)
def ensure_data(repo_full_name: str = "", issue_number: int = 0) -> dict:
    from tam_research.data import prepare_fineweb

    result = prepare_fineweb(
        "/vol/data/fineweb-edu-gpt2",
        train_tokens=25_000_000,
        val_tokens=2_000_000,
    )
    volume.commit()
    _both_comments(
        repo_full_name,
        issue_number,
        "PGW-v0 data guard passed: shared FineWeb-Edu/GPT-2 token shards are available.",
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
    return all(math.isfinite(x) for x in values)


def _selected_fraction(result: dict) -> float:
    router = result["final_eval"].get("router")
    if not router:
        return float("nan")
    return float(router["mean"]["selected_fraction"])


def _pair_summary(transformer: dict, pgw: dict) -> dict:
    t_nll = float(transformer["final_eval"]["nll"])
    p_nll = float(pgw["final_eval"]["nll"])
    return {
        "seed": int(transformer["seed"]),
        "transformer_nll": t_nll,
        "pgw_nll": p_nll,
        "transformer_minus_pgw_nll": t_nll - p_nll,
        "transformer_training_tps": float(transformer["training_tokens_per_second"]),
        "pgw_training_tps": float(pgw["training_tokens_per_second"]),
        "throughput_ratio": float(pgw["training_tokens_per_second"])
        / max(float(transformer["training_tokens_per_second"]), 1e-12),
        "transformer_peak_vram_gib": float(transformer["peak_vram_gb"]),
        "pgw_peak_vram_gib": float(pgw["peak_vram_gb"]),
        "pgw_selected_fraction": _selected_fraction(pgw),
        "transformer_parameters": int(transformer["parameters"]),
        "pgw_parameters": int(pgw["parameters"]),
    }


@app.function(
    image=image,
    gpu="H100!",
    cpu=8,
    memory=32768,
    timeout=3 * 60 * 60,
    volumes={"/vol": volume},
    secrets=[github_secret],
)
def run_screen(
    repo_full_name: str,
    issue_number: int,
    source_sha: str,
) -> dict:
    import re
    from tam_research.pgw_v0.train import train_pgw_candidate

    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise RuntimeError(f"invalid source SHA: {source_sha!r}")

    root = Path(RESULT_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    attempt_path = root / "ATTEMPT.json"
    result_path = root / "RESULT.json"
    failure_path = root / "FAILURE.json"
    if attempt_path.exists() or result_path.exists() or failure_path.exists():
        raise RuntimeError("PGW-v0 #1145 result namespace is already consumed")

    attempt = {
        "issue": ISSUE,
        "source_sha": source_sha,
        "smoke_seed": SMOKE_SEED,
        "replication_seeds": list(REPLICATION_SEEDS),
        "started_unix": time.time(),
        "gpu": "H100!",
        "retry_authorized": False,
    }
    attempt_path.write_text(json.dumps(attempt, indent=2), encoding="utf-8")
    volume.commit()

    _both_comments(
        repo_full_name,
        issue_number,
        f"PGW-v0 scientific attempt started on H100 from `{source_sha}`. "
        f"Smoke seed={SMOKE_SEED}; no retry/resume is authorized for this namespace.",
    )

    try:
        smoke: dict[str, dict] = {}
        for architecture in ("transformer", "pgw"):
            _both_comments(
                repo_full_name,
                issue_number,
                f"PGW-v0 smoke: training **{architecture}** seed {SMOKE_SEED} for {SMOKE_TOKENS:,} tokens.",
            )
            smoke[architecture] = train_pgw_candidate(
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

        smoke_pair = _pair_summary(smoke["transformer"], smoke["pgw"])
        parameter_gap = abs(
            smoke_pair["pgw_parameters"] - smoke_pair["transformer_parameters"]
        ) / smoke_pair["transformer_parameters"]
        smoke_pass = (
            _finite(smoke["transformer"])
            and _finite(smoke["pgw"])
            and parameter_gap <= PARAMETER_MISMATCH_LIMIT
            and abs(
                smoke_pair["pgw_selected_fraction"] - EXPECTED_SELECTED_FRACTION
            ) <= 1e-8
        )

        _both_comments(
            repo_full_name,
            issue_number,
            "PGW-v0 smoke result: "
            f"Transformer NLL={smoke_pair['transformer_nll']:.4f}; "
            f"PGW NLL={smoke_pair['pgw_nll']:.4f}; "
            f"PGW/T throughput={smoke_pair['throughput_ratio']:.3f}; "
            f"parameter gap={100*parameter_gap:.4f}%; "
            f"workspace selected fraction={smoke_pair['pgw_selected_fraction']:.4f}; "
            f"stage2_authorized={str(smoke_pass).lower()}.",
        )

        replication_pairs: list[dict] = []
        raw_replication: list[dict] = []
        if smoke_pass:
            for seed in REPLICATION_SEEDS:
                pair_raw: dict[str, dict] = {}
                for architecture in ("transformer", "pgw"):
                    _both_comments(
                        repo_full_name,
                        issue_number,
                        f"PGW-v0 replication: training **{architecture}** seed {seed} "
                        f"for {REPLICATION_TOKENS:,} tokens.",
                    )
                    pair_raw[architecture] = train_pgw_candidate(
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
                pair = _pair_summary(pair_raw["transformer"], pair_raw["pgw"])
                replication_pairs.append(pair)
                _both_comments(
                    repo_full_name,
                    issue_number,
                    f"PGW-v0 seed {seed}: Transformer NLL={pair['transformer_nll']:.4f}; "
                    f"PGW NLL={pair['pgw_nll']:.4f}; delta(T-PGW)={pair['transformer_minus_pgw_nll']:+.4f}; "
                    f"throughput ratio={pair['throughput_ratio']:.3f}.",
                )

        if replication_pairs:
            t_mean = sum(x["transformer_nll"] for x in replication_pairs) / len(replication_pairs)
            p_mean = sum(x["pgw_nll"] for x in replication_pairs) / len(replication_pairs)
            wins = sum(x["pgw_nll"] < x["transformer_nll"] for x in replication_pairs)
            throughput_ratio = sum(x["throughput_ratio"] for x in replication_pairs) / len(replication_pairs)
            selected_ok = all(
                abs(x["pgw_selected_fraction"] - EXPECTED_SELECTED_FRACTION) <= 1e-8
                for x in replication_pairs
            )
            finite_ok = all(
                _finite(pair_raw["transformer"]) and _finite(pair_raw["pgw"])
                for pair_raw in raw_replication
            )
            positive = (
                finite_ok
                and p_mean < t_mean
                and wins >= 2
                and throughput_ratio >= MIN_PGW_THROUGHPUT_RATIO
                and selected_ok
            )
        else:
            t_mean = p_mean = throughput_ratio = float("nan")
            wins = 0
            finite_ok = False
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
            "parameter_gap_fraction": parameter_gap,
            "smoke": smoke_pair,
            "replication": replication_pairs,
            "replication_summary": {
                "transformer_mean_nll": t_mean,
                "pgw_mean_nll": p_mean,
                "pgw_wins": wins,
                "mean_throughput_ratio": throughput_ratio,
                "finite": finite_ok,
                "selected_fraction_exact": selected_ok,
            },
            "breakthrough_claim_allowed": False,
            "next_step_if_positive": "mechanism ablations and equal-compute replication",
            "completed_unix": time.time(),
        }
        result_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
        volume.commit()

        _both_comments(
            repo_full_name,
            issue_number,
            f"**PGW-v0 terminal classification: `{classification}`**\n\n"
            f"Replication Transformer mean NLL={t_mean:.4f}; PGW mean NLL={p_mean:.4f}; "
            f"PGW wins={wins}/{len(replication_pairs)}; mean PGW/T throughput={throughput_ratio:.3f}.\n\n"
            "This is a development screen only; `breakthrough_claim_allowed=false`.",
        )
        return final
    except Exception as exc:
        failure = {
            "issue": ISSUE,
            "source_sha": source_sha,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "scientific_interpretation": False,
            "retry_authorized": False,
            "failed_unix": time.time(),
        }
        failure_path.write_text(json.dumps(failure, indent=2), encoding="utf-8")
        volume.commit()
        _both_comments(
            repo_full_name,
            issue_number,
            f"PGW-v0 terminal infrastructure/runtime failure: `{type(exc).__name__}: {exc}`. "
            "No scientific interpretation and no automatic retry.",
        )
        raise


@app.local_entrypoint()
def main(
    repo_full_name: str,
    issue_number: int,
    source_sha: str,
):
    ensure_data.remote(repo_full_name, issue_number)
    result = run_screen.remote(repo_full_name, issue_number, source_sha)
    print("PGW_V0_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
