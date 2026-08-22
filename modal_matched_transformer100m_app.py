from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "tam-research-matched-transformer100m"
VOLUME_NAME = "tam-research-data"
SEED = 8100
PRETRAIN_TOKENS = 2_000_000_000
VAL_TOKENS = 5_000_000
SEQ_LEN = 512
MICRO_BATCH = 64
GRAD_ACCUM = 2
EXPECTED_TRANSFORMER_PARAMS = 101_803_520
TAM_REFERENCE_PARAMS = 101_806_616

# Identical aggregate H100 ceiling to the TAM-v3 production run.
# At the rate recorded in modal_full100m_app.py on 2026-08-22 this is $12.83 max.
MAX_GPU_SECONDS = 11_700
POSTTRAIN_RESERVE_SECONDS = 1_500

# Reuse the exact immutable prepared bytes consumed by TAM.
PRETRAIN_DATA_DIR = "/vol/data/tam100m-2b-curated-v1"
POSTTRAIN_DATA_DIR = "/vol/data/tam100m-posttrain-v1"
PRETRAIN_RUN_ROOT = "/vol/full100m-runs/transformer-pretrain"
FINAL_RUN_DIR = f"/vol/full100m-runs/Transformer-100M-2B-seed{SEED}"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
github_secret = modal.Secret.from_name("github-secret")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.7,<2.11",
        "numpy>=2.0,<3",
        "transformers>=4.55,<5",
        "tokenizers>=0.21,<1",
        "PyGithub>=2.3,<3",
    )
    .add_local_python_source("tam_research")
)


def _comment(repo_full_name: str, issue_number: int, body: str) -> None:
    if not repo_full_name or not issue_number:
        print(f"[status] {body}", flush=True)
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


def _validate_protocol_bytes() -> dict:
    pre_root = Path(PRETRAIN_DATA_DIR)
    post_root = Path(POSTTRAIN_DATA_DIR)
    required = [
        pre_root / "train.bin",
        pre_root / "val.bin",
        pre_root / "meta.json",
        post_root / "meta.json",
        post_root / "sft_train_inputs.npy",
        post_root / "sft_train_labels.npy",
        post_root / "sft_eval_inputs.npy",
        post_root / "sft_eval_labels.npy",
        post_root / "pref_train_chosen_inputs.npy",
        post_root / "pref_train_chosen_labels.npy",
        post_root / "pref_train_rejected_inputs.npy",
        post_root / "pref_train_rejected_labels.npy",
        post_root / "pref_eval_chosen_inputs.npy",
        post_root / "pref_eval_chosen_labels.npy",
        post_root / "pref_eval_rejected_inputs.npy",
        post_root / "pref_eval_rejected_labels.npy",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"matched-control source artifacts missing: {missing}")

    pre = json.loads((pre_root / "meta.json").read_text())
    post = json.loads((post_root / "meta.json").read_text())
    expected_pre = {
        "assembly_version": 3,
        "train_tokens": PRETRAIN_TOKENS,
        "val_tokens": VAL_TOKENS,
        "seed": SEED,
        "tokenizer": "gpt2",
        "dtype": "uint16",
    }
    for key, value in expected_pre.items():
        if pre.get(key) != value:
            raise RuntimeError(f"pretrain metadata mismatch for {key}: {pre.get(key)!r} != {value!r}")
    if (pre_root / "train.bin").stat().st_size != PRETRAIN_TOKENS * 2:
        raise RuntimeError("train.bin byte size does not match exact 2B uint16 tokens")
    if (pre_root / "val.bin").stat().st_size != VAL_TOKENS * 2:
        raise RuntimeError("val.bin byte size does not match exact 5M uint16 tokens")

    expected_post = {
        "seq_len": SEQ_LEN,
        "sft_train_rows": 100_000,
        "sft_eval_rows": 2_000,
        "preference_train_rows": 10_000,
        "preference_eval_rows": 1_000,
        "seed": SEED,
        "input_dtype": "int32",
        "tokenizer": "gpt2",
    }
    for key, value in expected_post.items():
        if post.get(key) != value:
            raise RuntimeError(f"posttrain metadata mismatch for {key}: {post.get(key)!r} != {value!r}")

    return {"pretrain": pre, "posttrain": post}


@app.function(
    image=image,
    cpu=2,
    memory=2048,
    timeout=300,
    volumes={"/vol": volume},
    secrets=[github_secret],
)
def preflight_matched_control(repo_full_name: str = "", issue_number: int = 0) -> dict:
    from tam_research.models import ResearchLM, parameter_count
    from tam_research.scales import model_config_for_scale

    volume.reload()
    meta = _validate_protocol_bytes()
    # train_language_model deliberately uses max(1024, seq_len) for positional capacity.
    cfg = model_config_for_scale("transformer", "100m", max_seq_len=max(1024, SEQ_LEN))
    model = ResearchLM(cfg)
    params = parameter_count(model)
    if params != EXPECTED_TRANSFORMER_PARAMS:
        raise RuntimeError(f"Transformer parameter count changed: {params:,} != {EXPECTED_TRANSFORMER_PARAMS:,}")
    delta = abs(params - TAM_REFERENCE_PARAMS)
    _comment(
        repo_full_name,
        issue_number,
        "🟩 **Matched Transformer preflight passed** — exact TAM train/val/SFT/DPO bytes verified; "
        f"Transformer params={params:,}, TAM params={TAM_REFERENCE_PARAMS:,}, delta={delta:,} "
        f"({100*delta/TAM_REFERENCE_PARAMS:.4f}%); no GPU allocated yet.",
    )
    return {"parameters": params, "parameter_delta": delta, "data": meta}


@app.function(
    image=image,
    gpu="H100!",
    cpu=8,
    memory=65536,
    timeout=MAX_GPU_SECONDS,
    volumes={"/vol": volume},
    secrets=[github_secret],
)
def train_matched_transformer100m(
    repo_full_name: str = "",
    issue_number: int = 0,
    max_gpu_seconds: int = MAX_GPU_SECONDS,
) -> dict:
    if max_gpu_seconds != MAX_GPU_SECONDS:
        raise ValueError(f"matched control is fixed to {MAX_GPU_SECONDS}s; override refused")

    import os
    import time
    from importlib.metadata import version as package_version

    from tam_research.compile_cache import DEFAULT_COMPILE_MODE, compiler_cache_dir, compiler_cache_env

    volume.reload()
    _validate_protocol_bytes()
    cache_dir = compiler_cache_dir(
        "/vol/compile-cache",
        architecture="transformer",
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
    from tam_research.posttrain import generate_samples, load_checkpoint_model, train_dpo, train_sft
    from tam_research.train import evaluate
    from tam_research.train_scaled import train_scaled_language_model

    started = time.perf_counter()
    _comment(
        repo_full_name,
        issue_number,
        "🔥 **Matched Transformer-100M H100 started** — "
        f"params={EXPECTED_TRANSFORMER_PARAMS:,}; exact same 2B corpus + 5M val + SFT/DPO bytes as TAM; "
        f"context={SEQ_LEN}, micro={MICRO_BATCH}, accum={GRAD_ACCUM}, seed={SEED}; "
        f"compiler-cache={'warm' if cache_preexisting else 'cold'}; "
        f"hard aggregate H100 ceiling={MAX_GPU_SECONDS}s.",
    )

    try:
        pretrain = train_scaled_language_model(
            architecture="transformer",
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
        if int(pretrain["parameters"]) != EXPECTED_TRANSFORMER_PARAMS:
            raise RuntimeError(f"trained parameter count mismatch: {pretrain['parameters']}")
        volume.commit()
        elapsed = time.perf_counter() - started
        _comment(
            repo_full_name,
            issue_number,
            "🧠 **Matched Transformer 2B pretraining finished and committed** — "
            f"NLL={pretrain['final_eval']['nll']:.4f}; PPL={pretrain['final_eval']['perplexity']:.2f}; "
            f"steady={pretrain['training_tokens_per_second']:.0f} tok/s; elapsed={elapsed:.1f}s.",
        )

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
        remaining = MAX_GPU_SECONDS - elapsed
        if remaining < POSTTRAIN_RESERVE_SECONDS:
            result = {
                "model": "Transformer-100M-2B",
                "status": "pretraining_complete_posttraining_deferred_by_budget_guard",
                "seed": SEED,
                "pretrain": pretrain,
                "remaining_gpu_seconds": remaining,
            }
            (final_dir / "pipeline_summary.json").write_text(json.dumps(result, indent=2))
            volume.commit()
            _comment(repo_full_name, issue_number, f"✅ **Transformer pretraining durable; post-training deferred** — {remaining:.1f}s remained.")
            return result

        sft_path = final_dir / "sft.pt"
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
            "🟦 **Matched Transformer SFT finished and committed** — "
            f"examples={sft_summary['train_examples']:,}; held-out assistant NLL={sft_summary['heldout']['assistant_nll']:.4f}; "
            f"train={sft_summary['training_seconds']:.1f}s; compile={sft_summary['compile_seconds']:.1f}s.",
        )

        elapsed = time.perf_counter() - started
        if MAX_GPU_SECONDS - elapsed < 450:
            result = {
                "model": "Transformer-100M-2B",
                "status": "pretrain_sft_complete_dpo_deferred_by_budget_guard",
                "seed": SEED,
                "pretrain": pretrain,
                "sft": sft_summary,
                "sft_checkpoint": str(sft_path),
                "gpu_function_elapsed_seconds": elapsed,
            }
            (final_dir / "pipeline_summary.json").write_text(json.dumps(result, indent=2))
            volume.commit()
            _comment(repo_full_name, issue_number, "✅ **Transformer pretrain + SFT durable; DPO deferred by budget guard.**")
            return result

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
            "🟪 **Matched Transformer DPO finished and committed** — "
            f"pairs={dpo_summary['pairs_seen']:,}; held-out implicit reward acc={dpo_summary['heldout']['implicit_reward_accuracy']:.3f}; "
            f"train={dpo_summary['training_seconds']:.1f}s.",
        )

        final_model = load_checkpoint_model(str(dpo_path), torch.device("cuda"))
        final_language = evaluate(
            final_model,
            TokenBin(str(Path(PRETRAIN_DATA_DIR) / "val.bin")),
            seq_len=SEQ_LEN,
            batches=12,
            batch_size=16,
            seed=SEED + 500,
        )
        samples = generate_samples(str(dpo_path), max_new_tokens=48, seed=SEED)
        total_elapsed = time.perf_counter() - started
        result = {
            "model": "Transformer-100M-2B-Full",
            "status": "complete",
            "seed": SEED,
            "parameters": pretrain["parameters"],
            "tam_reference_parameters": TAM_REFERENCE_PARAMS,
            "parameter_delta": abs(int(pretrain["parameters"]) - TAM_REFERENCE_PARAMS),
            "pretrain_tokens": PRETRAIN_TOKENS,
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
            "fairness": {
                "same_pretrain_bytes": True,
                "same_validation_bytes": True,
                "same_sft_bytes": True,
                "same_dpo_bytes": True,
                "same_seed": True,
                "same_context": True,
                "same_batching": True,
                "same_optimizer_schedule": True,
                "same_h100_ceiling_seconds": True,
            },
        }
        (final_dir / "pipeline_summary.json").write_text(json.dumps(result, indent=2))
        volume.commit()
        _comment(
            repo_full_name,
            issue_number,
            "✅ **Matched Transformer-100M-2B-Full complete** — "
            f"pretrain NLL={pretrain['final_eval']['nll']:.4f}; SFT assistant NLL={sft_summary['heldout']['assistant_nll']:.4f}; "
            f"DPO reward acc={dpo_summary['heldout']['implicit_reward_accuracy']:.3f}; final mixture NLL={final_language['nll']:.4f}; "
            f"elapsed={total_elapsed:.1f}s.\nFinal checkpoint: `{dpo_path}`",
        )
        print(json.dumps(result, indent=2), flush=True)
        return result
    except Exception as exc:
        try:
            volume.commit()
        except Exception:
            pass
        _comment(repo_full_name, issue_number, f"❌ **Matched Transformer-100M-2B failed/aborted:** `{type(exc).__name__}: {exc}`")
        raise


@app.local_entrypoint()
def main(repo_full_name: str = "", issue_number: int = 0):
    preflight = preflight_matched_control.remote(repo_full_name, issue_number)
    call = train_matched_transformer100m.spawn(repo_full_name, issue_number, MAX_GPU_SECONDS)
    print(json.dumps({"preflight": preflight, "call_id": call.object_id, "hard_gpu_seconds": MAX_GPU_SECONDS}, indent=2))
