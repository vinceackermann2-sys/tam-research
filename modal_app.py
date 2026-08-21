from __future__ import annotations

import json
import modal

APP_NAME = "tam-research"
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


def _comment_on_issue(repo_full_name: str, issue_number: int, body: str) -> None:
    if not repo_full_name or not issue_number:
        return
    import os
    import github
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return
    g = github.Github(auth=github.Auth.Token(token))
    g.get_repo(repo_full_name).get_issue(number=issue_number).create_comment(body)


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=24 * 60 * 60,
    volumes={"/vol": volume},
    secrets=[github_secret],
)
def prepare_data(
    train_tokens: int = 110_000_000,
    val_tokens: int = 2_000_000,
    repo_full_name: str = "",
    issue_number: int = 0,
) -> dict:
    from tam_research.data import prepare_fineweb
    _comment_on_issue(
        repo_full_name,
        issue_number,
        f"🟦 **Modal connected. Data preparation started** — target train tokens={train_tokens:,}, validation tokens={val_tokens:,}.",
    )
    try:
        result = prepare_fineweb("/vol/data/fineweb-edu-gpt2", train_tokens=train_tokens, val_tokens=val_tokens)
        volume.commit()
        _comment_on_issue(
            repo_full_name,
            issue_number,
            f"🟩 **Data preparation finished** — training stream is committed to the Modal Volume.",
        )
        print(json.dumps(result, indent=2), flush=True)
        return result
    except Exception as exc:
        _comment_on_issue(
            repo_full_name,
            issue_number,
            f"❌ **Data preparation failed:** `{type(exc).__name__}: {exc}`",
        )
        raise


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=24 * 60 * 60,
    volumes={"/vol": volume},
    secrets=[github_secret],
)
def train_one(
    architecture: str,
    seed: int,
    token_budget: int = 100_000_000,
    seq_len: int = 512,
    repo_full_name: str = "",
    issue_number: int = 0,
) -> dict:
    from tam_research.train import train_language_model
    _comment_on_issue(
        repo_full_name,
        issue_number,
        f"🔥 **H100 training started** — architecture={architecture}, seed={seed}, token budget={token_budget:,}, context={seq_len}.",
    )
    try:
        volume.reload()
        result = train_language_model(
            architecture=architecture,
            seed=seed,
            data_dir="/vol/data/fineweb-edu-gpt2",
            run_root="/vol/runs",
            token_budget=token_budget,
            seq_len=seq_len,
        )
        volume.commit()
        ev = result["final_eval"]
        router = ev.get("router")
        route_text = ""
        if router and router.get("mean"):
            m = router["mean"]
            route_text = f"\nRouter: attention={m['attention']:.3f}, memory={m['memory']:.3f}, world={m['world']:.3f}"
        _comment_on_issue(
            repo_full_name,
            issue_number,
            f"✅ **{architecture} seed {seed} finished** — {result['tokens_seen']:,} tokens, "
            f"NLL={ev['nll']:.4f}, PPL={ev['perplexity']:.2f}, params={result['parameters']:,}."
            f"{route_text}",
        )
        print(json.dumps(result, indent=2), flush=True)
        return result
    except Exception as exc:
        _comment_on_issue(repo_full_name, issue_number, f"❌ **{architecture} seed {seed} failed:** `{type(exc).__name__}: {exc}`")
        raise


@app.local_entrypoint()
def main(
    action: str = "suite",
    token_budget: int = 100_000_000,
    seq_len: int = 512,
    architectures: str = "transformer,tamv2",
    seeds: str = "7025,7026,7027",
    repo_full_name: str = "",
    issue_number: int = 0,
):
    if action == "prepare":
        print(
            prepare_data.remote(
                train_tokens=token_budget + 10_000_000,
                val_tokens=2_000_000,
                repo_full_name=repo_full_name,
                issue_number=issue_number,
            )
        )
        return
    if action == "train-one":
        arch = architectures.split(",")[0].strip()
        seed = int(seeds.split(",")[0])
        print(train_one.remote(arch, seed, token_budget, seq_len, repo_full_name, issue_number))
        return
    if action != "suite":
        raise ValueError("action must be prepare, train-one, or suite")

    # Data preparation is blocking so every GPU run sees the exact same committed token stream.
    prepare_data.remote(
        train_tokens=token_budget + 10_000_000,
        val_tokens=2_000_000,
        repo_full_name=repo_full_name,
        issue_number=issue_number,
    )
    spawned = []
    for arch in [a.strip() for a in architectures.split(",") if a.strip()]:
        for seed_text in [s.strip() for s in seeds.split(",") if s.strip()]:
            seed = int(seed_text)
            call = train_one.spawn(arch, seed, token_budget, seq_len, repo_full_name, issue_number)
            spawned.append({"call_id": call.object_id, "architecture": arch, "seed": seed})
    _comment_on_issue(
        repo_full_name,
        issue_number,
        "🚀 **GPU jobs spawned on Modal** — " + ", ".join(f"{x['architecture']} seed {x['seed']}" for x in spawned),
    )
    print(json.dumps({"spawned": spawned}, indent=2), flush=True)
    # Intentionally return immediately. `modal run --detach` leaves spawned GPU jobs alive.
