from __future__ import annotations

from pathlib import Path
import re


def _component(value: str) -> str:
    value = value.strip().lower()
    if not re.fullmatch(r"[a-z0-9_-]+", value):
        raise ValueError(f"unsafe cache-key component: {value!r}")
    return value


def compiler_cache_dir(
    root: str,
    *,
    architecture: str,
    model_scale: str,
    seq_len: int,
    micro_batch_size: int,
    grad_accum_steps: int,
    accelerator: str = "h100",
) -> str:
    """Return a seed-independent cache directory for one compiled graph family.

    Seeds intentionally do not participate in the key: model weights differ by seed,
    while the compiled graph does not. Shape/execution settings do participate so an
    incompatible graph cannot be reused accidentally.
    """
    if seq_len <= 0 or micro_batch_size <= 0 or grad_accum_steps <= 0:
        raise ValueError("shape/batch settings must be positive")
    return str(
        Path(root)
        / _component(accelerator)
        / _component(model_scale)
        / _component(architecture)
        / f"ctx{seq_len}-micro{micro_batch_size}-accum{grad_accum_steps}"
    )


def compiler_cache_env(cache_dir: str) -> dict[str, str]:
    """Environment required for persistent PyTorch Inductor graph caches."""
    base = str(Path(cache_dir))
    return {
        "TORCHINDUCTOR_CACHE_DIR": base,
        "TRITON_CACHE_DIR": str(Path(base) / "triton"),
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
        "TORCHINDUCTOR_AUTOGRAD_CACHE": "1",
    }
