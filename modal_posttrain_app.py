from __future__ import annotations

import json
import modal

APP_NAME = "tam-research-posttrain"
VOLUME_NAME = "tam-research-data"
MAX_GPU_SECONDS = 150
SEED = 7400
SEQ_LEN = 512
SFT_TRAIN_EXAMPLES = 10_000

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


def _prepare_sft_subset(source_dir: str, destination_dir: str, rows: int) -> str:
    """Create a deterministic small SFT train shard in local ephemeral storage.

    The original 20k UltraFeedback arrays on the persistent Volume remain untouched.
    Evaluation arrays are copied in full so held-out metrics stay comparable.
    """
    import shutil
    from pathlib import Path

    import numpy as np

    src = Path(source_dir)
    dst = Path(destination_dir)
    dst.mkdir(parents=True, exist_ok=True)

    train_inputs = np.load(src / "sft_train_inputs.npy", mmap_mode="r")
    train_labels = np.load(src / "sft_train_labels.npy", mmap_mode="r")
    n = min(rows, len(train_inputs), len(train_labels))
    if n < 128:
        raise RuntimeError(f"SFT subset too small: {n}")
    np.save(dst / "sft_train_inputs.npy", np.asarray(train_inputs[:n]).copy())
    np.save(dst / "sft_train_labels.npy", np.asarray(train_labels[:n]).copy())
    shutil.copyfile(src / "sft_eval_inputs.npy", dst / "sft_eval_inputs.npy")
    shutil.copyfile(src / "sft_eval_labels.npy", dst / "sft_eval_labels.npy")
    return str(dst)


@app.function(
    image=image,
    gpu="H100!",
    cpu=8,
    memory=32768,
    timeout=MAX_GPU_SECONDS,
    volumes={"/vol": volume},
    secrets=[github_secret],
)
def posttrain_tam25m(
    repo_full_name: str = "",
    issue_number: int = 0,
    max_gpu_seconds: int = MAX_GPU_SECONDS,
) -> dict:
    if max_gpu_seconds != MAX_GPU_SECONDS:
        raise ValueError(f"this budget-tight continuation is fixed at {MAX_GPU_SECONDS} GPU seconds")

    import time
    from pathlib import Path

    volume.reload()

    pretrained_checkpoint = (
        "/vol/full-model-runs/pretrain/25m/ctx512-mb64-ga2/"
        f"tamv3-25m-compiled-seed{SEED}/latest.pt"
    )
    posttrain_data_dir = "/vol/data/ultrafeedback-gpt2-full25m"
    fineweb_data_dir = "/vol/data/fineweb-edu-gpt2-full25m"
    run_dir = Path(f"/vol/full-model-runs/TAM-v3-25M-Full-seed{SEED}")
    run_dir.mkdir(parents=True, exist_ok=True)

    required = [
        Path(pretrained_checkpoint),
        Path(posttrain_data_dir) / "sft_train_inputs.npy",
        Path(posttrain_data_dir) / "sft_train_labels.npy",
        Path(posttrain_data_dir) / "sft_eval_inputs.npy",
        Path(posttrain_data_dir) / "sft_eval_labels.npy",
        Path(posttrain_data_dir) / "pref_train_chosen_inputs.npy",
        Path(posttrain_data_dir) / "pref_train_rejected_inputs.npy",
        Path(posttrain_data_dir) / "pref_eval_chosen_inputs.npy",
        Path(posttrain_data_dir) / "pref_eval_rejected_inputs.npy",
        Path(fineweb_data_dir) / "val.bin",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"required continuation artifacts missing: {missing}")

    import torch
    from tam_research.data import TokenBin
    from tam_research.posttrain import generate_samples, load_checkpoint_model, train_dpo, train_sft
    from tam_research.train import evaluate

    started = time.perf_counter()
    _comment(
        repo_full_name,
        issue_number,
        f"🔥 **TAM-v3-25M budget-v2 post-training started** — saved 300M-token base; "
        f"SFT={SFT_TRAIN_EXAMPLES:,} examples eager/no-compile; DPO=full 5k pairs; hard ceiling={MAX_GPU_SECONDS}s.",
    )

    try:
        sft_path = run_dir / "sft.pt"
        sft_summary_path = run_dir / "sft_summary.json"
        if sft_path.exists() and sft_summary_path.exists():
            sft_summary = json.loads(sft_summary_path.read_text())
            _comment(repo_full_name, issue_number, "♻️ **Existing SFT checkpoint found** — reusing it.")
        else:
            subset_dir = _prepare_sft_subset(posttrain_data_dir, "/tmp/tam-sft-10k", SFT_TRAIN_EXAMPLES)
            sft_checkpoint, sft_summary = train_sft(
                pretrained_checkpoint,
                subset_dir,
                str(run_dir),
                seed=SEED,
                micro_batch_size=64,
                grad_accum_steps=2,
                compile_model=False,
            )
            sft_path = Path(sft_checkpoint)
            volume.commit()
            elapsed = time.perf_counter() - started
            _comment(
                repo_full_name,
                issue_number,
                f"🟦 **SFT finished and committed** — train_examples={sft_summary['train_examples']}; "
                f"held-out assistant NLL={sft_summary['heldout']['assistant_nll']:.4f}; "
                f"training={sft_summary['training_seconds']:.1f}s; elapsed={elapsed:.1f}s; compile=0.",
            )

        elapsed = time.perf_counter() - started
        remaining = MAX_GPU_SECONDS - elapsed
        if remaining < 62:
            partial = {
                "model": "TAM-v3-25M-Full",
                "seed": SEED,
                "status": "sft_complete_dpo_deferred",
                "pretrained_checkpoint": pretrained_checkpoint,
                "sft_checkpoint": str(sft_path),
                "sft": sft_summary,
                "gpu_function_elapsed_seconds": elapsed,
            }
            (run_dir / "posttrain_summary.json").write_text(json.dumps(partial, indent=2))
            volume.commit()
            _comment(
                repo_full_name,
                issue_number,
                f"✅ **SFT is durable; DPO deferred by budget guard** — {remaining:.1f}s remained. "
                f"SFT checkpoint: `{sft_path}`",
            )
            return partial

        dpo_path = run_dir / "final_dpo.pt"
        dpo_summary_path = run_dir / "dpo_summary.json"
        if dpo_path.exists() and dpo_summary_path.exists():
            dpo_summary = json.loads(dpo_summary_path.read_text())
            _comment(repo_full_name, issue_number, "♻️ **Existing DPO checkpoint found** — reusing it.")
        else:
            dpo_checkpoint, dpo_summary = train_dpo(
                str(sft_path),
                posttrain_data_dir,
                str(run_dir),
                seed=SEED,
                beta=0.1,
                micro_batch_size=32,
                grad_accum_steps=2,
                learning_rate=1e-5,
            )
            dpo_path = Path(dpo_checkpoint)
            volume.commit()
            elapsed = time.perf_counter() - started
            _comment(
                repo_full_name,
                issue_number,
                f"🟪 **DPO finished and committed** — pairs={dpo_summary['pairs_seen']}; "
                f"held-out implicit reward accuracy={dpo_summary['heldout']['implicit_reward_accuracy']:.3f}; "
                f"training={dpo_summary['training_seconds']:.1f}s; elapsed={elapsed:.1f}s.",
            )

        elapsed = time.perf_counter() - started
        remaining = MAX_GPU_SECONDS - elapsed
        if remaining < 8:
            partial = {
                "model": "TAM-v3-25M-Full",
                "seed": SEED,
                "status": "posttraining_complete_eval_deferred",
                "pretrained_checkpoint": pretrained_checkpoint,
                "sft_checkpoint": str(sft_path),
                "final_checkpoint": str(dpo_path),
                "sft": sft_summary,
                "dpo": dpo_summary,
                "gpu_function_elapsed_seconds": elapsed,
            }
            (run_dir / "posttrain_summary.json").write_text(json.dumps(partial, indent=2))
            volume.commit()
            _comment(
                repo_full_name,
                issue_number,
                f"✅ **SFT + DPO complete; optional eval deferred.** Final checkpoint: `{dpo_path}`",
            )
            return partial

        final_model = load_checkpoint_model(str(dpo_path), torch.device("cuda"))
        final_language = evaluate(
            final_model,
            TokenBin(str(Path(fineweb_data_dir) / "val.bin")),
            seq_len=512,
            batches=6,
            batch_size=32,
            seed=SEED + 400,
        )
        samples = generate_samples(str(dpo_path), max_new_tokens=16, seed=SEED)
        total_elapsed = time.perf_counter() - started
        result = {
            "model": "TAM-v3-25M-Full",
            "seed": SEED,
            "status": "complete",
            "pretrained_checkpoint": pretrained_checkpoint,
            "sft_checkpoint": str(sft_path),
            "final_checkpoint": str(dpo_path),
            "sft": sft_summary,
            "dpo": dpo_summary,
            "final_language_eval": final_language,
            "samples": samples,
            "sft_train_examples_budget_v2": SFT_TRAIN_EXAMPLES,
            "max_gpu_seconds": MAX_GPU_SECONDS,
            "gpu_function_elapsed_seconds": total_elapsed,
        }
        (run_dir / "final_summary.json").write_text(json.dumps(result, indent=2))
        volume.commit()
        _comment(
            repo_full_name,
            issue_number,
            f"✅ **TAM-v3-25M-Full post-training completed** — SFT assistant NLL="
            f"{sft_summary['heldout']['assistant_nll']:.4f}; DPO implicit reward accuracy="
            f"{dpo_summary['heldout']['implicit_reward_accuracy']:.3f}; final FineWeb NLL="
            f"{final_language['nll']:.4f}; elapsed={total_elapsed:.1f}s.\n"
            f"Final checkpoint: `{dpo_path}`",
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
            f"❌ **TAM-v3-25M budget-v2 post-training failed/aborted:** `{type(exc).__name__}: {exc}`",
        )
        raise


@app.local_entrypoint()
def main(
    repo_full_name: str = "",
    issue_number: int = 0,
    max_gpu_seconds: int = MAX_GPU_SECONDS,
):
    if max_gpu_seconds != MAX_GPU_SECONDS:
        raise ValueError(f"hard post-training budget is fixed at {MAX_GPU_SECONDS} GPU seconds")
    call = posttrain_tam25m.spawn(repo_full_name, issue_number, max_gpu_seconds)
    print(json.dumps({"call_id": call.object_id, "max_gpu_seconds": max_gpu_seconds}, indent=2))
