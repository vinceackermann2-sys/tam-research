from __future__ import annotations

import json
import modal

APP_NAME = "tam-research-full-train"
VOLUME_NAME = "tam-research-data"
MAX_GPU_SECONDS = 1_300
PRETRAIN_TOKENS = 300_000_000
SEED = 7400
SEQ_LEN = 512
MICRO_BATCH = 64
GRAD_ACCUM = 2

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
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
        return
    import os

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print(f"[status] {body}", flush=True)
        return
    try:
        import github

        client = github.Github(auth=github.Auth.Token(token))
        client.get_repo(repo_full_name).get_issue(number=issue_number).create_comment(body)
    except Exception as exc:
        print(f"[status-report-nonfatal] {type(exc).__name__}: {exc}; body={body}", flush=True)


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=2 * 60 * 60,
    volumes={"/vol": volume},
    secrets=[github_secret],
)
def prepare_full_data(repo_full_name: str = "", issue_number: int = 0) -> dict:
    from tam_research.data import prepare_fineweb
    from tam_research.posttrain_data import prepare_posttrain_data

    _comment(
        repo_full_name,
        issue_number,
        "🟦 **Full-model data preparation started** — isolated 310M-token FineWeb shard + UltraFeedback SFT/DPO arrays.",
    )
    # Use an isolated FineWeb directory so data preparation cannot rewrite the shard
    # currently being read by the active 100M scaling gate.
    fineweb = prepare_fineweb(
        "/vol/data/fineweb-edu-gpt2-full25m",
        train_tokens=310_000_000,
        val_tokens=2_000_000,
        seed=1234,
    )
    posttrain = prepare_posttrain_data(
        "/vol/data/ultrafeedback-gpt2-full25m",
        seq_len=SEQ_LEN,
        sft_train_rows=20_000,
        sft_eval_rows=1_000,
        preference_train_rows=5_000,
        preference_eval_rows=500,
        seed=SEED,
    )
    volume.commit()
    result = {"fineweb": fineweb, "posttrain": posttrain}
    _comment(repo_full_name, issue_number, "🟩 **Full-model data preparation finished.**")
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.function(
    image=image,
    gpu="H100!",
    cpu=8,
    memory=32768,
    timeout=MAX_GPU_SECONDS,
    volumes={"/vol": volume},
    secrets=[github_secret],
)
def train_full_tam25m(
    repo_full_name: str = "",
    issue_number: int = 0,
    max_gpu_seconds: int = MAX_GPU_SECONDS,
) -> dict:
    if max_gpu_seconds > MAX_GPU_SECONDS or max_gpu_seconds < 300:
        raise ValueError(f"max_gpu_seconds must be between 300 and {MAX_GPU_SECONDS}")

    import os
    import time
    from importlib.metadata import version as package_version
    from pathlib import Path

    from tam_research.compile_cache import (
        DEFAULT_COMPILE_MODE,
        compiler_cache_dir,
        compiler_cache_env,
    )

    volume.reload()
    cache_dir = compiler_cache_dir(
        "/vol/compile-cache",
        architecture="tamv3",
        model_scale="25m",
        seq_len=SEQ_LEN,
        micro_batch_size=MICRO_BATCH,
        grad_accum_steps=GRAD_ACCUM,
        torch_build=package_version("torch"),
        compile_mode=DEFAULT_COMPILE_MODE,
    )
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_preexisting = any(Path(cache_dir).iterdir())
    os.environ.update(compiler_cache_env(cache_dir))

    # Import torch/training only after the persistent compiler cache is configured.
    import torch
    from tam_research.posttrain import run_posttraining
    from tam_research.train_scaled import train_scaled_language_model

    started = time.perf_counter()
    _comment(
        repo_full_name,
        issue_number,
        f"🔥 **TAM-v3-25M-Full H100 started** — pretrain={PRETRAIN_TOKENS:,} tokens, seed={SEED}, "
        f"hard GPU ceiling={max_gpu_seconds}s, compiler-cache={'warm' if cache_preexisting else 'cold'}.",
    )

    try:
        pretrain = train_scaled_language_model(
            architecture="tamv3",
            model_scale="25m",
            seed=SEED,
            data_dir="/vol/data/fineweb-edu-gpt2-full25m",
            run_root="/vol/full-model-runs/pretrain",
            token_budget=PRETRAIN_TOKENS,
            seq_len=SEQ_LEN,
            micro_batch_size=MICRO_BATCH,
            grad_accum_steps=GRAD_ACCUM,
            compile_model=True,
            eval_every_tokens=50_000_000,
            checkpoint_every_tokens=50_000_000,
        )
        volume.commit()
        elapsed_after_pretrain = time.perf_counter() - started
        _comment(
            repo_full_name,
            issue_number,
            f"🧠 **Pretraining finished** — NLL={pretrain['final_eval']['nll']:.4f}, "
            f"PPL={pretrain['final_eval']['perplexity']:.2f}, steady={pretrain['training_tokens_per_second']:.0f} tok/s, "
            f"GPU-function elapsed={elapsed_after_pretrain:.1f}s.",
        )

        # Leave an explicit reserve for both post-training stages and final evaluation.
        if elapsed_after_pretrain > max_gpu_seconds - 180:
            raise RuntimeError(
                f"budget guard: only {max_gpu_seconds - elapsed_after_pretrain:.1f}s remain after pretraining; "
                "refusing to start post-training and risk an uncheckpointed timeout"
            )

        pretrained_checkpoint = (
            "/vol/full-model-runs/pretrain/25m/ctx512-mb64-ga2/"
            f"tamv3-25m-compiled-seed{SEED}/latest.pt"
        )
        if not Path(pretrained_checkpoint).exists():
            raise FileNotFoundError(pretrained_checkpoint)

        post = run_posttraining(
            pretrained_checkpoint=pretrained_checkpoint,
            posttrain_data_dir="/vol/data/ultrafeedback-gpt2-full25m",
            fineweb_data_dir="/vol/data/fineweb-edu-gpt2-full25m",
            run_dir=f"/vol/full-model-runs/TAM-v3-25M-Full-seed{SEED}",
            seed=SEED,
        )
        total_elapsed = time.perf_counter() - started
        if total_elapsed > max_gpu_seconds:
            raise RuntimeError("budget guard exceeded after post-training")
        volume.commit()

        result = {
            "model": "TAM-v3-25M-Full",
            "seed": SEED,
            "pretrain_tokens": PRETRAIN_TOKENS,
            "compiler_cache_preexisting": cache_preexisting,
            "compiler_cache_dir": cache_dir,
            "max_gpu_seconds": max_gpu_seconds,
            "gpu_function_elapsed_seconds": total_elapsed,
            "pretrain": pretrain,
            "posttrain": post,
        }
        final_path = Path(f"/vol/full-model-runs/TAM-v3-25M-Full-seed{SEED}/pipeline_summary.json")
        final_path.write_text(json.dumps(result, indent=2))
        volume.commit()
        dpo = post["dpo"]
        _comment(
            repo_full_name,
            issue_number,
            f"✅ **TAM-v3-25M-Full completed** — pretrain NLL={pretrain['final_eval']['nll']:.4f}; "
            f"SFT held-out assistant NLL={post['sft']['heldout']['assistant_nll']:.4f}; "
            f"DPO held-out implicit reward accuracy={dpo['heldout']['implicit_reward_accuracy']:.3f}; "
            f"final FineWeb NLL={post['final_language_eval']['nll']:.4f}; total GPU-function elapsed={total_elapsed:.1f}s.\n"
            f"Final checkpoint: `{post['final_checkpoint']}`",
        )
        print(json.dumps(result, indent=2), flush=True)
        return result
    except Exception as exc:
        # Pretraining checkpoints are committed before post-training begins, so even a
        # budget abort preserves all completed work.
        try:
            volume.commit()
        except Exception:
            pass
        _comment(
            repo_full_name,
            issue_number,
            f"❌ **TAM-v3-25M-Full failed/aborted:** `{type(exc).__name__}: {exc}`",
        )
        raise


@app.local_entrypoint()
def main(
    repo_full_name: str = "",
    issue_number: int = 0,
    max_gpu_seconds: int = MAX_GPU_SECONDS,
):
    if max_gpu_seconds > MAX_GPU_SECONDS:
        raise ValueError(f"hard budget ceiling is {MAX_GPU_SECONDS} GPU seconds")
    prepare_full_data.remote(repo_full_name, issue_number)
    call = train_full_tam25m.spawn(repo_full_name, issue_number, max_gpu_seconds)
    print(json.dumps({"call_id": call.object_id, "max_gpu_seconds": max_gpu_seconds}, indent=2))
