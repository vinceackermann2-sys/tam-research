from __future__ import annotations

import json
import modal

APP_NAME = "tam-research"
VOLUME_NAME = "tam-research-data"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

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


@app.function(image=image, cpu=8, memory=32768, timeout=24 * 60 * 60, volumes={"/vol": volume})
def prepare_data(train_tokens: int = 110_000_000, val_tokens: int = 2_000_000) -> dict:
    from tam_research.data import prepare_fineweb
    result = prepare_fineweb("/vol/data/fineweb-edu-gpt2", train_tokens=train_tokens, val_tokens=val_tokens)
    volume.commit()
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.function(image=image, gpu="H100", cpu=8, memory=32768, timeout=24 * 60 * 60, volumes={"/vol": volume})
def train_one(architecture: str, seed: int, token_budget: int = 100_000_000, seq_len: int = 512) -> dict:
    from tam_research.train import train_language_model
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
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main(
    action: str = "suite",
    token_budget: int = 100_000_000,
    seq_len: int = 512,
    architectures: str = "transformer,tamv2",
    seeds: str = "7025,7026,7027",
):
    if action == "prepare":
        print(prepare_data.remote(train_tokens=token_budget + 10_000_000, val_tokens=2_000_000))
        return
    if action == "train-one":
        arch = architectures.split(",")[0].strip()
        seed = int(seeds.split(",")[0])
        print(train_one.remote(arch, seed, token_budget, seq_len))
        return
    if action != "suite":
        raise ValueError("action must be prepare, train-one, or suite")

    prepare_data.remote(train_tokens=token_budget + 10_000_000, val_tokens=2_000_000)
    calls = []
    for arch in [a.strip() for a in architectures.split(",") if a.strip()]:
        for seed_text in [s.strip() for s in seeds.split(",") if s.strip()]:
            seed = int(seed_text)
            call = train_one.spawn(arch, seed, token_budget, seq_len)
            calls.append((arch, seed, call))
            print(json.dumps({"spawned": call.object_id, "architecture": arch, "seed": seed}), flush=True)

    results = []
    for arch, seed, call in calls:
        results.append(call.get())
    print(json.dumps({"suite": results}, indent=2), flush=True)
