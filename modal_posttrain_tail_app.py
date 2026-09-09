from __future__ import annotations

import json
from pathlib import Path
import time

import modal

app = modal.App("tam-research-posttrain-tail")
volume = modal.Volume.from_name("tam-research-data", create_if_missing=True)
github_secret = modal.Secret.from_name("github-secret")

SEED = 7400
MAX_GPU_SECONDS = 55
SOURCE_DIR = Path("/vol/data/ultrafeedback-gpt2-full25m")
TAIL_DIR = Path("/vol/data/ultrafeedback-gpt2-tail")
RUN_DIR = Path("/vol/full-model-runs/TAM-v3-25M-BudgetTail-seed7400")
PRETRAINED = Path(
    "/vol/full-model-runs/pretrain/25m/ctx512-mb64-ga2/"
    "tamv3-25m-compiled-seed7400/latest.pt"
)

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
        print(body, flush=True)
        return
    try:
        import github
        client = github.Github(auth=github.Auth.Token(token))
        client.get_repo(repo_full_name).get_issue(number=issue_number).create_comment(body)
    except Exception as exc:
        print(f"[status-report-nonfatal] {type(exc).__name__}: {exc}", flush=True)


@app.function(
    image=image,
    cpu=1,
    memory=2048,
    timeout=120,
    volumes={"/vol": volume},
    secrets=[github_secret],
)
def prepare_tail_data(repo_full_name: str = "", issue_number: int = 0) -> dict:
    import numpy as np

    volume.reload()
    TAIL_DIR.mkdir(parents=True, exist_ok=True)
    specs = {
        "sft_train_inputs.npy": 2048,
        "sft_train_labels.npy": 2048,
        "sft_eval_inputs.npy": 256,
        "sft_eval_labels.npy": 256,
        "pref_train_chosen_inputs.npy": 512,
        "pref_train_chosen_labels.npy": 512,
        "pref_train_rejected_inputs.npy": 512,
        "pref_train_rejected_labels.npy": 512,
        "pref_eval_chosen_inputs.npy": 128,
        "pref_eval_chosen_labels.npy": 128,
        "pref_eval_rejected_inputs.npy": 128,
        "pref_eval_rejected_labels.npy": 128,
    }
    for name, rows in specs.items():
        src = SOURCE_DIR / name
        if not src.exists():
            raise FileNotFoundError(src)
        dst = TAIL_DIR / name
        arr = np.load(src, mmap_mode="r")
        np.save(dst, np.asarray(arr[:rows]).copy())
    volume.commit()
    result = {"sft_train_rows": 2048, "sft_eval_rows": 256, "dpo_train_pairs": 512, "dpo_eval_pairs": 128}
    _comment(repo_full_name, issue_number, f"🧩 **Budget-tail posttrain data ready** — {result}.")
    return result


@app.function(
    image=image,
    gpu="H100!",
    cpu=4,
    memory=8192,
    timeout=MAX_GPU_SECONDS,
    volumes={"/vol": volume},
    secrets=[github_secret],
)
def train_tail(repo_full_name: str = "", issue_number: int = 0) -> dict:
    volume.reload()
    if not PRETRAINED.exists():
        raise FileNotFoundError(PRETRAINED)
    if not (TAIL_DIR / "sft_train_inputs.npy").exists():
        raise FileNotFoundError(TAIL_DIR / "sft_train_inputs.npy")

    import torch
    from tam_research.data import TokenBin
    from tam_research.posttrain import generate_samples, load_checkpoint_model, train_dpo, train_sft
    from tam_research.train import evaluate

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    _comment(
        repo_full_name,
        issue_number,
        "🔥 **Budget-tail H100 started** — eager SFT + DPO from saved 300M-token base; hard ceiling=55s; no compile/pretraining.",
    )

    try:
        sft_path, sft = train_sft(
            str(PRETRAINED),
            str(TAIL_DIR),
            str(RUN_DIR),
            seed=SEED,
            micro_batch_size=64,
            grad_accum_steps=2,
            compile_model=False,
        )
        volume.commit()
        elapsed = time.perf_counter() - started
        _comment(
            repo_full_name,
            issue_number,
            f"🟦 **Budget-tail SFT committed** — assistant NLL={sft['heldout']['assistant_nll']:.4f}; elapsed={elapsed:.1f}s.",
        )
        if MAX_GPU_SECONDS - elapsed < 18:
            raise RuntimeError(
                f"budget guard: only {MAX_GPU_SECONDS - elapsed:.1f}s remain after SFT; preserving SFT and skipping DPO"
            )

        dpo_path, dpo = train_dpo(
            sft_path,
            str(TAIL_DIR),
            str(RUN_DIR),
            seed=SEED,
            beta=0.1,
            micro_batch_size=32,
            grad_accum_steps=2,
            learning_rate=1e-5,
        )
        volume.commit()
        elapsed = time.perf_counter() - started
        _comment(
            repo_full_name,
            issue_number,
            f"🟪 **Budget-tail DPO committed** — implicit reward accuracy={dpo['heldout']['implicit_reward_accuracy']:.3f}; elapsed={elapsed:.1f}s.",
        )

        result = {
            "model": "TAM-v3-25M-BudgetTail",
            "seed": SEED,
            "pretrained_checkpoint": str(PRETRAINED),
            "sft_checkpoint": sft_path,
            "final_checkpoint": dpo_path,
            "sft": sft,
            "dpo": dpo,
            "status": "posttraining_complete",
            "elapsed_seconds": elapsed,
        }
        # Only spend leftover seconds on optional quality inspection.
        if MAX_GPU_SECONDS - elapsed >= 6:
            final_model = load_checkpoint_model(dpo_path, torch.device("cuda"))
            result["final_language_eval"] = evaluate(
                final_model,
                TokenBin("/vol/data/fineweb-edu-gpt2-full25m/val.bin"),
                seq_len=512,
                batches=4,
                batch_size=16,
                seed=SEED + 400,
            )
            result["samples"] = generate_samples(
                dpo_path,
                prompts=[
                    "Explain why the sky looks blue in simple language.",
                    "Write a short Python function that adds two numbers.",
                    "A user says 17 * 6 = 112. Correct the mistake.",
                ],
                max_new_tokens=20,
                seed=SEED,
            )
        result["elapsed_seconds"] = time.perf_counter() - started
        (RUN_DIR / "budget_tail_summary.json").write_text(json.dumps(result, indent=2))
        volume.commit()
        _comment(
            repo_full_name,
            issue_number,
            f"✅ **TAM-v3-25M budget-tail posttraining completed** — final checkpoint `{dpo_path}`; total elapsed={result['elapsed_seconds']:.1f}s.",
        )
        print(json.dumps(result, indent=2), flush=True)
        return result
    except Exception as exc:
        try:
            volume.commit()
        except Exception:
            pass
        _comment(repo_full_name, issue_number, f"❌ **Budget-tail posttraining stopped:** `{type(exc).__name__}: {exc}`")
        raise


@app.local_entrypoint()
def main(repo_full_name: str = "", issue_number: int = 0):
    prepare_tail_data.remote(repo_full_name, issue_number)
    call = train_tail.spawn(repo_full_name, issue_number)
    print(json.dumps({"call_id": call.object_id, "hard_gpu_seconds": MAX_GPU_SECONDS}, indent=2))
