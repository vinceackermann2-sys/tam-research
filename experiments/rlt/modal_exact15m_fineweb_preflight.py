from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import modal

EXPECTED_TRAIN_SHA256 = "0f9ea99680f532f990ca64fabbd02b46bba2c4f8077f868545a03f5e6907c907"
EXPECTED_VAL_SHA256 = "e6b3fb209a80a2cebe22924c3c8af26b30fc4afd322a1a49c71af8b968057db2"
EXPECTED_PARAMETERS = 15_129_344

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.10,<2.11",
        "numpy>=2.0,<3",
        "datasets>=4.0,<5",
        "transformers>=4.55,<5",
        "tokenizers>=0.21,<1",
        "huggingface-hub>=0.34,<1",
    )
    .add_local_python_source("tam_research", "experiments")
)
app = modal.App("tam-rlt-exact15m-fineweb-preflight-20260925-a")


def _encode(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode()


@app.function(image=image, cpu=4.0, memory=16384, timeout=25 * 60, retries=0)
def run_preflight_remote() -> str:
    import torch
    from experiments.rlt.model import RecurrentLoopedTransformer, parameter_count as rlt_parameter_count
    from experiments.rlt.train_exact15m_fineweb_4m_a import (
        rlt_config_exact15m,
        transformer_config_exact15m,
    )
    from tam_research.data import prepare_fineweb
    from tam_research.models import ResearchLM, parameter_count as transformer_parameter_count

    started = time.time()
    data_dir = Path("/tmp/rlt-exact15m-preflight-data")
    meta = prepare_fineweb(
        str(data_dir),
        train_tokens=6_000_000,
        val_tokens=500_000,
        seed=20_260_924,
    )

    def sha(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    train_sha = sha(data_dir / "train.bin")
    val_sha = sha(data_dir / "val.bin")
    if train_sha != EXPECTED_TRAIN_SHA256 or val_sha != EXPECTED_VAL_SHA256:
        raise RuntimeError(
            f"FineWeb D-byte mismatch: train={train_sha} val={val_sha}"
        )

    rlt = RecurrentLoopedTransformer(rlt_config_exact15m())
    transformer = ResearchLM(transformer_config_exact15m())
    rlt_params = rlt_parameter_count(rlt)
    transformer_params = transformer_parameter_count(transformer)
    if rlt_params != EXPECTED_PARAMETERS:
        raise RuntimeError(f"exact15m RLT params {rlt_params} != {EXPECTED_PARAMETERS}")
    if transformer_params != EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"exact15m Transformer params {transformer_params} != {EXPECTED_PARAMETERS}"
        )

    old_transformer = ResearchLM(
        transformer_config_exact15m().__class__(
            vocab_size=50_257,
            d_model=128,
            n_layers=3,
            n_heads=4,
            max_seq_len=128,
            ff_mult=4,
            architecture="transformer",
        )
    )
    old_count = transformer_parameter_count(old_transformer)
    if old_count != 7_040_896:
        raise RuntimeError(f"default ff_inner=None behavior drifted: {old_count}")

    with torch.no_grad():
        tiny = torch.tensor([[1, 2]], dtype=torch.long)
        rlt_shape = list(rlt(tiny).shape)
        transformer_shape = list(transformer(tiny).shape)
    if rlt_shape != [1, 2, 50_257] or transformer_shape != [1, 2, 50_257]:
        raise RuntimeError(f"forward shape drift: {rlt_shape} {transformer_shape}")

    result = {
        "schema": 1,
        "status": "pass",
        "classification": "RLT_EXACT15M_FINEWEB_ZERO_GPU_PARAMETER_AND_DATA_PREFLIGHT_PASS",
        "scientific_execution": False,
        "gpu_allocated": False,
        "started_unix": started,
        "finished_unix": time.time(),
        "runtime": {
            "torch_version": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "parameter_audit": {
            "rlt_parameters": rlt_params,
            "transformer_parameters": transformer_params,
            "parameter_delta": transformer_params - rlt_params,
            "expected_parameters": EXPECTED_PARAMETERS,
            "transformer_ff_inner": 938,
            "default_7m_transformer_parameters": old_count,
        },
        "data_audit": {
            "metadata": meta,
            "train_sha256": train_sha,
            "val_sha256": val_sha,
            "matches_prior_d_bytes": True,
        },
        "forward_shape_audit": {
            "rlt": rlt_shape,
            "transformer": transformer_shape,
        },
    }
    return _encode(result)


@app.local_entrypoint()
def main() -> None:
    result = run_preflight_remote.remote()
    if not isinstance(result, str):
        raise TypeError("exact15m preflight result must be base64 text")
    print("RLT_EXACT15M_PREFLIGHT_B64=" + result)
