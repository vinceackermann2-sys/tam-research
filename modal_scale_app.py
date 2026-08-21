from __future__ import annotations

import json
import modal

APP_NAME = "tam-research-scaling"
VOLUME_NAME = "tam-research-data"

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
    import github

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return
    client = github.Github(auth=github.Auth.Token(token))
    client.get_repo(repo_full_name).get_issue(number=issue_number).create_comment(body)


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=24 * 60 * 60,
    volumes={"/vol": volume},
    secrets=[github_secret],
)
def prepare_scale_data(
    train_tokens: int,
    repo_full_name: str = "",
    issue_number: int = 0,
) -> dict:
    from tam_research.data import prepare_fineweb

    _comment(
        repo_full_name,
        issue_number,
        f"🟦 **Scaling data preparation started** — target train tokens={train_tokens:,}, validation tokens=2,000,000.",
    )
    result = prepare_fineweb(
        "/vol/data/fineweb-edu-gpt2",
        train_tokens=train_tokens,
        val_tokens=2_000_000,
    )
    volume.commit()
    _comment(repo_full_name, issue_number, "🟩 **Scaling data preparation finished.**")
    return result


@app.function(
    image=image,
    gpu="H100!",
    cpu=8,
    memory=32768,
    timeout=24 * 60 * 60,
    volumes={"/vol": volume},
    secrets=[github_secret],
)
def train_scaled_one(
    architecture: str,
    model_scale: str,
    seed: int,
    token_budget: int,
    seq_len: int,
    micro_batch_size: int,
    grad_accum_steps: int,
    repo_full_name: str = "",
    issue_number: int = 0,
) -> dict:
    from tam_research.train_scaled import train_scaled_language_model

    _comment(
        repo_full_name,
        issue_number,
        f"🔥 **Scaled H100 training started** — architecture={architecture}, scale={model_scale}, seed={seed}, "
        f"token budget={token_budget:,}, context={seq_len}, micro={micro_batch_size}, accum={grad_accum_steps}.",
    )
    try:
        volume.reload()
        result = train_scaled_language_model(
            architecture=architecture,
            model_scale=model_scale,
            seed=seed,
            data_dir="/vol/data/fineweb-edu-gpt2",
            run_root="/vol/scaling-runs",
            token_budget=token_budget,
            seq_len=seq_len,
            micro_batch_size=micro_batch_size,
            grad_accum_steps=grad_accum_steps,
            compile_model=True,
        )
        volume.commit()
        ev = result["final_eval"]
        _comment(
            repo_full_name,
            issue_number,
            f"✅ **{architecture} {model_scale} seed {seed} finished** — tokens={result['tokens_seen']:,}, "
            f"params={result['parameters']:,}, NLL={ev['nll']:.4f}, PPL={ev['perplexity']:.2f}.\n"
            f"Steady training: {result['training_tokens_per_second']:.0f} tok/s in {result['training_seconds']:.2f}s; "
            f"peak VRAM={result['peak_vram_gb']:.2f} GiB.\n"
            f"Compile={result['compile_seconds']:.2f}s; end-to-end measured={result['elapsed_seconds']:.2f}s.",
        )
        print(json.dumps(result, indent=2), flush=True)
        return result
    except Exception as exc:
        _comment(
            repo_full_name,
            issue_number,
            f"❌ **{architecture} {model_scale} seed {seed} failed:** `{type(exc).__name__}: {exc}`",
        )
        raise


@app.local_entrypoint()
def main(
    model_scale: str = "50m",
    token_budget: int = 5_000_000,
    seq_len: int = 512,
    architectures: str = "transformer,tamv3",
    seeds: str = "7100",
    micro_batch_size: int = 32,
    grad_accum_steps: int = 4,
    repo_full_name: str = "",
    issue_number: int = 0,
):
    scale = model_scale.strip().lower()
    allowed_scales = {"25m", "50m", "100m", "300m", "1b"}
    if scale not in allowed_scales:
        raise ValueError("model_scale must be one of 25m, 50m, 100m, 300m, or 1b")
    if micro_batch_size * grad_accum_steps != 128:
        raise ValueError("micro_batch_size * grad_accum_steps must equal 128")
    if scale in {"300m", "1b"} and micro_batch_size > 32:
        raise ValueError("300m/1b single-H100 runs require micro_batch_size <= 32")

    archs = [a.strip() for a in architectures.split(",") if a.strip()]
    seed_values = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    if not archs or not set(archs) <= {"transformer", "tamv3"}:
        raise ValueError("architectures must be transformer and/or tamv3")

    prepare_scale_data.remote(
        train_tokens=token_budget + 10_000_000,
        repo_full_name=repo_full_name,
        issue_number=issue_number,
    )

    spawned = []
    for architecture in archs:
        for seed in seed_values:
            call = train_scaled_one.spawn(
                architecture,
                scale,
                seed,
                token_budget,
                seq_len,
                micro_batch_size,
                grad_accum_steps,
                repo_full_name,
                issue_number,
            )
            spawned.append(
                {
                    "call_id": call.object_id,
                    "architecture": architecture,
                    "model_scale": scale,
                    "seed": seed,
                }
            )
    print(json.dumps({"spawned": spawned}, indent=2), flush=True)
