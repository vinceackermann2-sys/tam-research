from __future__ import annotations

import json
import modal

APP_NAME = "tam-research-full100m"
VOLUME_NAME = "tam-research-data"
SEED = 8100
PRETRAIN_TOKENS = 2_000_000_000
SEQ_LEN = 512
MICRO_BATCH = 64
GRAD_ACCUM = 2

# Hard cost controls. The H100 function contains pretraining + SFT + DPO, so this is
# a true aggregate GPU ceiling rather than a per-stage estimate. At the 2026-08-22
# Modal H100 list rate ($0.001097/s), 11,700s is $12.83 maximum GPU spend.
MAX_GPU_SECONDS = 11_700
POSTTRAIN_RESERVE_SECONDS = 1_500
DATA_PREP_TIMEOUT_SECONDS = 4 * 60 * 60

PRETRAIN_DATA_DIR = "/vol/data/tam100m-2b-curated-v1"
POSTTRAIN_DATA_DIR = "/vol/data/tam100m-posttrain-v1"
PRETRAIN_RUN_ROOT = "/vol/full100m-runs/pretrain"
FINAL_RUN_DIR = f"/vol/full100m-runs/TAM-v3-100M-2B-seed{SEED}"

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
        print(
            f"[status-report-nonfatal] {type(exc).__name__}: {exc}; body={body}",
            flush=True,
        )


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=DATA_PREP_TIMEOUT_SECONDS,
    nonpreemptible=True,
    volumes={"/vol": volume},
    secrets=[github_secret],
)
def prepare_full100m_data(repo_full_name: str = "", issue_number: int = 0) -> dict:
    from tam_research.posttrain100_data import prepare_100m_posttrain_data
    from tam_research.pretrain_mixture import prepare_pretrain_mixture

    _comment(
        repo_full_name,
        issue_number,
        "🟦 **TAM-100M data preparation started** — 2B curated GPT-2 tokens + "
        "Smol-SmolTalk SFT + UltraFeedback DPO. CPU only; no GPU allocated.",
    )
    pretrain = prepare_pretrain_mixture(PRETRAIN_DATA_DIR, seed=SEED)
    posttrain = prepare_100m_posttrain_data(
        POSTTRAIN_DATA_DIR,
        seq_len=SEQ_LEN,
        sft_train_rows=100_000,
        sft_eval_rows=2_000,
        preference_train_rows=10_000,
        preference_eval_rows=1_000,
        seed=SEED,
    )
    volume.commit()
    result = {"pretrain": pretrain, "posttrain": posttrain}
    _comment(
        repo_full_name,
        issue_number,
        "🟩 **TAM-100M data preparation finished and committed** — "
        f"train={pretrain['train_tokens']:,} tokens; val={pretrain['val_tokens']:,} tokens; "
        f"SFT={posttrain['sft_train_rows']:,}; DPO={posttrain['preference_train_rows']:,}.",
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.function(
    image=image,
    gpu="H100!",
    cpu=8,
    memory=65536,
    timeout=MAX_GPU_SECONDS,
    volumes={"/vol": volume},
    secrets=[github_secret],
)
def train_full_tam100m(
    repo_full_name: str = "",
    issue_number: int = 0,
    max_gpu_seconds: int = MAX_GPU_SECONDS,
) -> dict:
    if max_gpu_seconds != MAX_GPU_SECONDS:
        raise ValueError(
            f"production 100M run is fixed to hard ceiling={MAX_GPU_SECONDS}s; "
            "refusing a caller-provided override"
        )

    import os
    import time
    from importlib.metadata import version as package_version
    from pathlib import Path

    from tam_research.compile_cache import (
        DEFAULT_COMPILE_MODE,
        compiler_cache_dir,
        compiler_cache_env,
    )

    # Pull all CPU-prepared data before Torch opens compiler-cache files.
    volume.reload()
    required = [
        Path(PRETRAIN_DATA_DIR) / "train.bin",
        Path(PRETRAIN_DATA_DIR) / "val.bin",
        Path(POSTTRAIN_DATA_DIR) / "sft_train_inputs.npy",
        Path(POSTTRAIN_DATA_DIR) / "pref_train_chosen_inputs.npy",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"100M full-run artifacts missing: {missing}")

    cache_dir = compiler_cache_dir(
        "/vol/compile-cache",
        architecture="tamv3",
        model_scale="100m",
        seq_len=SEQ_LEN,
        micro_batch_size=MICRO_BATCH,
        grad_accum_steps=GRAD_ACCUM,
        torch_build=package_version("torch"),
        compile_mode=DEFAULT_COMPILE_MODE,
    )
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_preexisting = any(Path(cache_dir).iterdir())
    os.environ.update(compiler_cache_env(cache_dir))

    import torch
    from tam_research.data import TokenBin
    from tam_research.posttrain import (
        generate_samples,
        load_checkpoint_model,
        train_dpo,
        train_sft,
    )
    from tam_research.train import evaluate
    from tam_research.train_scaled import train_scaled_language_model

    started = time.perf_counter()
    _comment(
        repo_full_name,
        issue_number,
        "🔥 **TAM-v3-100M-Full H100 started** — "
        f"params≈101.8M, pretrain={PRETRAIN_TOKENS:,} tokens, context={SEQ_LEN}, "
        f"micro={MICRO_BATCH}, accum={GRAD_ACCUM}, seed={SEED}; "
        f"compiler-cache={'warm' if cache_preexisting else 'cold'}; "
        f"hard aggregate GPU ceiling={MAX_GPU_SECONDS}s (cannot be overridden).",
    )

    try:
        pretrain = train_scaled_language_model(
            architecture="tamv3",
            model_scale="100m",
            seed=SEED,
            data_dir=PRETRAIN_DATA_DIR,
            run_root=PRETRAIN_RUN_ROOT,
            token_budget=PRETRAIN_TOKENS,
            seq_len=SEQ_LEN,
            micro_batch_size=MICRO_BATCH,
            grad_accum_steps=GRAD_ACCUM,
            compile_model=True,
            eval_every_tokens=200_000_000,
            checkpoint_every_tokens=200_000_000,
        )
        volume.commit()
        elapsed = time.perf_counter() - started
        _comment(
            repo_full_name,
            issue_number,
            f"🧠 **2B-token pretraining finished and committed** — "
            f"NLL={pretrain['final_eval']['nll']:.4f}; "
            f"PPL={pretrain['final_eval']['perplexity']:.2f}; "
            f"steady={pretrain['training_tokens_per_second']:.0f} tok/s; "
            f"elapsed={elapsed:.1f}s.",
        )

        remaining = MAX_GPU_SECONDS - elapsed
        if remaining < POSTTRAIN_RESERVE_SECONDS:
            result = {
                "model": "TAM-v3-100M-2B",
                "status": "pretraining_complete_posttraining_deferred_by_budget_guard",
                "seed": SEED,
                "pretrain": pretrain,
                "remaining_gpu_seconds": remaining,
            }
            Path(FINAL_RUN_DIR).mkdir(parents=True, exist_ok=True)
            Path(FINAL_RUN_DIR, "pipeline_summary.json").write_text(
                json.dumps(result, indent=2)
            )
            volume.commit()
            _comment(
                repo_full_name,
                issue_number,
                f"✅ **Pretraining is durable; post-training deferred by budget guard** — "
                f"only {remaining:.1f}s remained.",
            )
            return result

        pretrain_dir = (
            Path(PRETRAIN_RUN_ROOT)
            / "100m"
            / f"ctx{SEQ_LEN}-mb{MICRO_BATCH}-ga{GRAD_ACCUM}"
            / str(pretrain["run_id"])
        )
        pretrained_checkpoint = pretrain_dir / "latest.pt"
        if not pretrained_checkpoint.exists():
            raise FileNotFoundError(str(pretrained_checkpoint))

        final_dir = Path(FINAL_RUN_DIR)
        final_dir.mkdir(parents=True, exist_ok=True)
        sft_path = final_dir / "sft.pt"
        sft_summary_path = final_dir / "sft_summary.json"
        if sft_path.exists() and sft_summary_path.exists():
            sft_summary = json.loads(sft_summary_path.read_text())
            _comment(repo_full_name, issue_number, "♻️ **Existing 100M SFT checkpoint found** — reusing it.")
        else:
            sft_checkpoint, sft_summary = train_sft(
                str(pretrained_checkpoint),
                POSTTRAIN_DATA_DIR,
                FINAL_RUN_DIR,
                seed=SEED,
                micro_batch_size=MICRO_BATCH,
                grad_accum_steps=GRAD_ACCUM,
                learning_rate=5e-5,
                compile_model=True,
            )
            sft_path = Path(sft_checkpoint)
            volume.commit()
            _comment(
                repo_full_name,
                issue_number,
                f"🟦 **100M SFT finished and committed** — "
                f"examples={sft_summary['train_examples']:,}; "
                f"held-out assistant NLL={sft_summary['heldout']['assistant_nll']:.4f}; "
                f"train={sft_summary['training_seconds']:.1f}s; "
                f"compile={sft_summary['compile_seconds']:.1f}s.",
            )

        elapsed = time.perf_counter() - started
        if MAX_GPU_SECONDS - elapsed < 450:
            result = {
                "model": "TAM-v3-100M-2B",
                "status": "pretrain_sft_complete_dpo_deferred_by_budget_guard",
                "seed": SEED,
                "pretrain": pretrain,
                "sft": sft_summary,
                "sft_checkpoint": str(sft_path),
                "gpu_function_elapsed_seconds": elapsed,
            }
            Path(FINAL_RUN_DIR, "pipeline_summary.json").write_text(
                json.dumps(result, indent=2)
            )
            volume.commit()
            _comment(
                repo_full_name,
                issue_number,
                "✅ **Pretrain + SFT durable; DPO deferred by budget guard.**",
            )
            return result

        dpo_path = final_dir / "final_dpo.pt"
        dpo_summary_path = final_dir / "dpo_summary.json"
        if dpo_path.exists() and dpo_summary_path.exists():
            dpo_summary = json.loads(dpo_summary_path.read_text())
            _comment(repo_full_name, issue_number, "♻️ **Existing 100M DPO checkpoint found** — reusing it.")
        else:
            dpo_checkpoint, dpo_summary = train_dpo(
                str(sft_path),
                POSTTRAIN_DATA_DIR,
                FINAL_RUN_DIR,
                seed=SEED,
                beta=0.1,
                micro_batch_size=16,
                grad_accum_steps=4,
                learning_rate=1e-5,
            )
            dpo_path = Path(dpo_checkpoint)
            volume.commit()
            _comment(
                repo_full_name,
                issue_number,
                f"🟪 **100M DPO finished and committed** — "
                f"pairs={dpo_summary['pairs_seen']:,}; "
                f"held-out implicit reward acc={dpo_summary['heldout']['implicit_reward_accuracy']:.3f}; "
                f"train={dpo_summary['training_seconds']:.1f}s.",
            )

        # Final evaluation is deliberately small; durable model checkpoints take priority.
        final_model = load_checkpoint_model(str(dpo_path), torch.device("cuda"))
        final_language = evaluate(
            final_model,
            TokenBin(str(Path(PRETRAIN_DATA_DIR) / "val.bin")),
            seq_len=SEQ_LEN,
            batches=12,
            batch_size=16,
            seed=SEED + 500,
        )
        samples = generate_samples(
            str(dpo_path),
            max_new_tokens=48,
            seed=SEED,
        )
        total_elapsed = time.perf_counter() - started
        result = {
            "model": "TAM-v3-100M-2B-Full",
            "status": "complete",
            "seed": SEED,
            "parameters": pretrain["parameters"],
            "pretrain_tokens": PRETRAIN_TOKENS,
            "tokens_per_parameter": PRETRAIN_TOKENS / float(pretrain["parameters"]),
            "pretrain_checkpoint": str(pretrained_checkpoint),
            "sft_checkpoint": str(sft_path),
            "final_checkpoint": str(dpo_path),
            "pretrain": pretrain,
            "sft": sft_summary,
            "dpo": dpo_summary,
            "final_language_eval": final_language,
            "samples": samples,
            "compiler_cache_preexisting": cache_preexisting,
            "max_gpu_seconds": MAX_GPU_SECONDS,
            "gpu_function_elapsed_seconds": total_elapsed,
        }
        Path(FINAL_RUN_DIR, "pipeline_summary.json").write_text(
            json.dumps(result, indent=2)
        )
        volume.commit()
        _comment(
            repo_full_name,
            issue_number,
            f"✅ **TAM-v3-100M-2B-Full complete** — "
            f"pretrain NLL={pretrain['final_eval']['nll']:.4f}; "
            f"SFT assistant NLL={sft_summary['heldout']['assistant_nll']:.4f}; "
            f"DPO reward acc={dpo_summary['heldout']['implicit_reward_accuracy']:.3f}; "
            f"final mixture NLL={final_language['nll']:.4f}; "
            f"elapsed={total_elapsed:.1f}s.\nFinal checkpoint: `{dpo_path}`",
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
            f"❌ **TAM-v3-100M-2B-Full failed/aborted:** `{type(exc).__name__}: {exc}`",
        )
        raise


@app.local_entrypoint()
def main(repo_full_name: str = "", issue_number: int = 0):
    # Deliberately no caller-supplied timeout/budget argument: the code-level cap is
    # fixed and cannot be raised by an issue payload or workflow-dispatch input.
    prepare_full100m_data.remote(repo_full_name, issue_number)
    call = train_full_tam100m.spawn(repo_full_name, issue_number, MAX_GPU_SECONDS)
    print(
        json.dumps(
            {
                "call_id": call.object_id,
                "hard_gpu_seconds": MAX_GPU_SECONDS,
                "pretrain_tokens": PRETRAIN_TOKENS,
            },
            indent=2,
        )
    )