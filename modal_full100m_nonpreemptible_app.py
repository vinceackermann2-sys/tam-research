from __future__ import annotations

import json

from modal_full100m_app import (
    app,
    github_secret,
    image,
    volume,
    _comment,
    DATA_PREP_TIMEOUT_SECONDS,
    FINAL_RUN_DIR,
    GRAD_ACCUM,
    MAX_GPU_SECONDS,
    MICRO_BATCH,
    POSTTRAIN_DATA_DIR,
    PRETRAIN_DATA_DIR,
    PRETRAIN_TOKENS,
    SEED,
    SEQ_LEN,
    train_full_tam100m,
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
def prepare_full100m_data_nonpreemptible(
    repo_full_name: str = "", issue_number: int = 0
) -> dict:
    """Prepare the production 2B-token corpus on non-preemptible CPU capacity.

    This is intentionally a separate production launcher so the scientifically frozen
    model/training implementation remains unchanged. Modal CPU non-preemptibility is
    used only for data preparation after repeated preemptions were observed on #106.
    """
    from tam_research.posttrain100_data import prepare_100m_posttrain_data
    from tam_research.pretrain_mixture import prepare_pretrain_mixture

    _comment(
        repo_full_name,
        issue_number,
        "🟦 **TAM-100M non-preemptible data preparation started** — 2B curated GPT-2 "
        "tokens + Smol-SmolTalk SFT + UltraFeedback DPO. CPU only; no GPU allocated; "
        "Modal nonpreemptible=True.",
    )

    pretrain = prepare_pretrain_mixture(PRETRAIN_DATA_DIR, seed=SEED)
    # Make completed pretraining shards/assembly durable before post-training prep.
    volume.commit()

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


@app.local_entrypoint()
def main(repo_full_name: str = "", issue_number: int = 0):
    # The CPU phase is guaranteed non-preemptible. The H100 phase remains exactly the
    # frozen production function with the immutable 11,700-second aggregate ceiling.
    prepare_full100m_data_nonpreemptible.remote(repo_full_name, issue_number)
    call = train_full_tam100m.spawn(repo_full_name, issue_number, MAX_GPU_SECONDS)
    print(
        json.dumps(
            {
                "call_id": call.object_id,
                "hard_gpu_seconds": MAX_GPU_SECONDS,
                "pretrain_tokens": PRETRAIN_TOKENS,
                "seed": SEED,
                "context": SEQ_LEN,
                "micro_batch": MICRO_BATCH,
                "grad_accum": GRAD_ACCUM,
                "final_run_dir": FINAL_RUN_DIR,
                "cpu_prep_nonpreemptible": True,
            },
            indent=2,
        ),
        flush=True,
    )
