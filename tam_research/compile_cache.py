from __future__ import annotations

from pathlib import Path
import re


CACHE_SCHEMA_VERSION = "v2"
DEFAULT_COMPILE_MODE = "max-autotune-no-cudagraphs"


def _component(value: str) -> str:
    value = value.strip().lower()
    if not re.fullmatch(r"[a-z0-9_-]+", value):
        raise ValueError(f"unsafe cache-key component: {value!r}")
    return value


def _version_component(value: str) -> str:
    """Normalize package/compiler versions into a filesystem-safe key component."""
    normalized = re.sub(r"[^a-z0-9_-]+", "_", value.strip().lower()).strip("_")
    if not normalized:
        raise ValueError("empty compiler-version component")
    return normalized


def compiler_cache_dir(
    root: str,
    *,
    architecture: str,
    model_scale: str,
    seq_len: int,
    micro_batch_size: int,
    grad_accum_steps: int,
    accelerator: str = "h100",
    torch_build: str = "unknown",
    compile_mode: str = DEFAULT_COMPILE_MODE,
    cache_schema: str = CACHE_SCHEMA_VERSION,
) -> str:
    """Return a seed-independent cache directory for one compiled graph family.

    Seeds intentionally do not participate in the key: model weights differ by seed,
    while the compiled graph does not. Graph shape, accelerator, compile mode, cache
    schema, and PyTorch build do participate so incompatible artifacts cannot be
    reused accidentally after a compiler/runtime upgrade.
    """
    if seq_len <= 0 or micro_batch_size <= 0 or grad_accum_steps <= 0:
        raise ValueError("shape/batch settings must be positive")
    return str(
        Path(root)
        / _component(cache_schema)
        / f"torch-{_version_component(torch_build)}"
        / _component(compile_mode)
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
