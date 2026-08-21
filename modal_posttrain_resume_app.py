from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "tam-research-posttrain-resume"
VOLUME_NAME = "tam-research-data"
MAX_GPU_SECONDS = 900
SEED = 7400

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
github_secret = modal.Secret.from_name("github-secret")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.7,<2.11",
        "transformers>=4.55,<5",
        "tokenizers>=0.21,<1",
        "numpy>=2.0,<3",
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
    gpu="H100!",
    cpu=8,
    memory=32768,
    timeout=MAX_GPU_SECONDS,
    volumes={"/vol": volume},
    secrets=[github_secret],
)
def resume_posttraining(
    repo_full_name: str = "",
    issue_number: int = 0,
    max_gpu_seconds: int = MAX_GPU_SECONDS,
) -> dict:
    if not 300 <= max_gpu_seconds <= MAX_GPU_SECONDS:
        raise ValueError(f"max_gpu_seconds must be between 300 and {MAX_GPU_SECONDS}")

    import time
    from tam_research.posttrain import run_posttraining

    volume.reload()
    pretrained_checkpoint = Path(
        "/vol/full-model-runs/pretrain/25m/ctx512-mb64-ga2/"
        f"tamv3-25m-compiled-seed{SEED}/latest.pt"
    )
    if not pretrained_checkpoint.exists():
        raise FileNotFoundError(f"required pretrained checkpoint missing: {pretrained_checkpoint}")

    started = time.perf_counter()
    _comment(
        repo_full_name,
        issue_number,
        f"🔥 **Post-training resume H100 started** — using existing 300M-token checkpoint, "
        f"seed={SEED}, hard GPU ceiling={max_gpu_seconds}s. No pretraining will run.",
    )

    try:
        post = run_posttraining(
            pretrained_checkpoint=str(pretrained_checkpoint),
            posttrain_data_dir="/vol/data/ultrafeedback-gpt2-full25m",
            fineweb_data_dir="/vol/data/fineweb-edu-gpt2-full25m",
            run_dir=f"/vol/full-model-runs/TAM-v3-25M-Full-seed{SEED}",
            seed=SEED,
        )
        elapsed = time.perf_counter() - started
        if elapsed > max_gpu_seconds:
            raise RuntimeError(f"budget guard exceeded: {elapsed:.1f}s > {max_gpu_seconds}s")

        result = {
            "model": "TAM-v3-25M-Full",
            "seed": SEED,
            "resumed_from_pretrained_checkpoint": str(pretrained_checkpoint),
            "max_gpu_seconds": max_gpu_seconds,
            "gpu_function_elapsed_seconds": elapsed,
            "posttrain": post,
        }
        final_path = Path(f"/vol/full-model-runs/TAM-v3-25M-Full-seed{SEED}/resume_pipeline_summary.json")
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.write_text(json.dumps(result, indent=2))
        volume.commit()

        dpo = post["dpo"]
        _comment(
            repo_full_name,
            issue_number,
            f"✅ **TAM-v3-25M post-training completed** — "
            f"SFT held-out assistant NLL={post['sft']['heldout']['assistant_nll']:.4f}; "
            f"DPO held-out implicit reward accuracy={dpo['heldout']['implicit_reward_accuracy']:.3f}; "
            f"final FineWeb NLL={post['final_language_eval']['nll']:.4f}; "
            f"GPU-function elapsed={elapsed:.1f}s.\n"
            f"Final checkpoint: `{post['final_checkpoint']}`",
        )
        print(json.dumps(result, indent=2), flush=True)
        return result
    except Exception as exc:
        try:
            volume.commit()
        except Exception:
            pass
        _comment(
            repo_full_name,
            issue_number,
            f"❌ **Post-training resume failed/aborted:** `{type(exc).__name__}: {exc}`",
        )
        raise


@app.local_entrypoint()
def main(
    repo_full_name: str = "",
    issue_number: int = 0,
    max_gpu_seconds: int = MAX_GPU_SECONDS,
):
    call = resume_posttraining.spawn(repo_full_name, issue_number, max_gpu_seconds)
    print(json.dumps({"call_id": call.object_id, "max_gpu_seconds": max_gpu_seconds}, indent=2))
