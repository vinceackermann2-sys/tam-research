"""Offline, zero-GPU sampler parity audit for the unapproved Conv7 3-way pilot.

Read-only: all sample files are provided by the caller, and the only supported
seed is a dedicated synthetic CPU test value. No Modal, CUDA, GPU allocation,
training, checkpoints, real corpus reads, or paid run authority.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from scripts.cortex_attention8_conv7_readiness_v1 import check_readiness
from tam_research.data import TokenBin

REPO = Path(__file__).resolve().parents[1]
SAMPLER_PATH = REPO / "tam_research/data.py"
FROZEN_RUNNER_PATH = REPO / "modal_cortex_reduced_attention_v2_attention8_pair1_successor_100m_2b_v1.py"
PROPOSAL_PATH = REPO / "research/reduced_attention/attention8_conv7_engineering_proposal_v1.json"
LOCKED_BLOBS = {
    SAMPLER_PATH.relative_to(REPO).as_posix(): "f73e85ebfa861927827b59716a1f7152dd3bcfc6",
    FROZEN_RUNNER_PATH.relative_to(REPO).as_posix(): "5442ed44568206797e80365c5f6d0d65a1b9a964",
}
MODELS = (
    "fresh_transformer", "reduced_attention_dense_v2_attention8",
    "attention8_causal_depthwise_conv7",
)
SYNTHETIC_SEED = 9124013
BATCHES = 3
MOCK_BATCH = 4
MOCK_LENGTH = 12


def _blob_sha(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\x00" + raw).hexdigest()


def audit_frozen_data_stream_source() -> dict[str, Any]:
    if check_readiness()["gpu_launch_permitted"] is not False:
        raise RuntimeError("source-bound readiness no longer blocks paid dispatch")
    for name, expected in LOCKED_BLOBS.items():
        if _blob_sha((REPO / name).read_bytes()) != expected:
            raise RuntimeError(f"frozen sampler/runner blob drift: {name}")

    frozen = FROZEN_RUNNER_PATH.read_text(encoding="utf-8")
    for anchor in (
        "manual_seed(PAIR_SEED + 99_999)",  # throwaway compile warmup
        "manual_seed(PAIR_SEED + 10_000)",  # actual training offsets
        "seed=PAIR_SEED + 30_000",        # independently reset final heldout
        "seed_all(PAIR_SEED)",            # reset model RNG after warmup
        "x, y = train_data.batch(",
        "x, y = val_data.batch(EVAL_BATCH_SIZE, SEQ_LEN, generator, device)",
        "TOTAL_OPTIMIZER_STEPS = 30_518",
        "GRAD_ACCUM_STEPS = 2",
        "SEQ_LEN = 512",
        "MICRO_BATCH_SIZE = 64",
        "FINAL_EVAL_BATCHES = 50",
        "EVAL_BATCH_SIZE = 32",
    ):
        if anchor not in frozen:
            raise RuntimeError("frozen stream/seed contract drift")
    proposal = json.loads(PROPOSAL_PATH.read_text(encoding="utf-8"))
    if proposal["stage"] != "ZERO_GPU_PROPOSAL_NOT_PREREGISTERED":
        raise RuntimeError("unexpected proposal stage")
    if any(value is not False for value in proposal["authority"].values()):
        raise RuntimeError("paid authority unexpectedly available")
    p = proposal["draft_engineering_pilot"]
    if tuple(p["comparators"]) != MODELS:
        raise RuntimeError("three-way model order changed")
    if p["engineering_seed"] is not None or p["strict_max_gpu_spend_usd"] is not None:
        raise RuntimeError("engineering seed or spending cap unexpectedly populated")
    if not p["shared_heldout_batch_stream"] or not p["same_training_seed_per_model"]:
        raise RuntimeError("matched RNG streams disabled")
    if (p["per_model_optimizer_steps"], p["tokens_per_step"],
        p["full_horizon_optimizer_steps"]) != (3052, 65536, 30518):
        raise RuntimeError("training geometry drift")
    if (p["final_evaluation_batches"], p["evaluation_batch_size"],
        p["sequence_length"]) != (50, 32, 512):
        raise RuntimeError("evaluation geometry drift")
    return {
        "classification": "CONV7_SOURCE_BOUND_SAMPLER_PARITY_CPU_ONLY",
        "frozen_blobs_verified": dict(LOCKED_BLOBS),
        "training_tokens_per_optimizer_step": 65536,
        "final_heldout_tokens_per_model": 819200,
        "train_rng_offset": 10000,
        "compile_warmup_rng_offset": 99999,
        "final_eval_rng_offset": 30000,
        "engineering_seed_assigned": False,
        "gpu_allocated": False,
        "paid_run_authorized": False,
        "scientific_evidence": False,
    }


def synthetic_cpu_stream_digest(path: Path, offset: int) -> str:
    """Fingerprint three tiny TokenBin batches from a test fixture, never from /vol."""
    path = Path(path)
    if path.resolve().as_posix().startswith("/vol/"):
        raise ValueError("real Modal dataset paths forbidden in synthetic audit")
    if path.suffix != ".bin" or not path.is_file():
        raise ValueError("synthetic existing uint16 shard required")
    if path.stat().st_size < 2 * (MOCK_LENGTH + 2):
        raise ValueError("synthetic shard too short")
    if path.stat().st_size > 2 * 100000:
        raise ValueError("only tiny synthetic shards are supported")
    if offset not in (10000, 30000, 99999):
        raise ValueError("unexpected RNG offset")
    sample = TokenBin(str(path))
    gen = torch.Generator(device="cpu").manual_seed(SYNTHETIC_SEED + offset)
    digest = hashlib.sha256()
    for _ in range(BATCHES):
        x, y = sample.batch(MOCK_BATCH, MOCK_LENGTH, gen, torch.device("cpu"))
        assert x.shape == (MOCK_BATCH, MOCK_LENGTH)
        assert y.shape == x.shape
        digest.update(x.numpy().tobytes())
        digest.update(y.numpy().tobytes())
    return digest.hexdigest()


def compare_synthetic_three_way_streams(train_path: Path, val_path: Path) -> dict[str, Any]:
    lock = audit_frozen_data_stream_source()
    if Path(train_path).resolve() == Path(val_path).resolve():
        raise ValueError("training and heldout synthetic files must be distinct")
    # Each model must get a separately seeded generator, not a shared mutable
    # generator passed serially from one model to the next.
    train = {m: synthetic_cpu_stream_digest(train_path, 10000) for m in MODELS}
    heldout = {m: synthetic_cpu_stream_digest(val_path, 30000) for m in MODELS}
    if len(set(train.values())) != 1 or len(set(heldout.values())) != 1:
        raise RuntimeError("model-specific sampling drift")
    if set(train.values()) == set(heldout.values()):
        raise RuntimeError("training and heldout synthetic streams not isolated")
    # A separate warmup sampling stream cannot alter any fresh training stream.
    synthetic_cpu_stream_digest(train_path, 99999)
    if synthetic_cpu_stream_digest(train_path, 10000) != train[MODELS[0]]:
        raise RuntimeError("compile warmup polluted training sampler")
    return {
        **lock,
        "classification": "CONV7_SYNTHETIC_THREE_MODEL_STREAM_PARITY_PASS_NOT_EVIDENCE",
        "synthetic_cpu_only": True,
        "train_digests": train,
        "heldout_digests": heldout,
        "synthetic_seed_consumes_science": False,
        "gpu_allocated": False,
        "paid_run_authorized": False,
        "scientific_evidence": False,
    }
