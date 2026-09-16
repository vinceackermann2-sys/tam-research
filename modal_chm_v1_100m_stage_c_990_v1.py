from __future__ import annotations

"""Lazy host-import boundary for the frozen CHM-v1 Stage-C runner (#996).

The scientific implementation remains preserved byte-for-byte as Git blob
``fb7fe5f8c2a597f7de9e4d03c6908dc0db9558ac``. The root-level copy is used
when present; Modal runtime containers may instead resolve the exact same blob
from the already-mounted ``tam_research`` package. In either location, the
bytes are re-hashed before execution.

Frozen governance mirror for static guards only:
SCIENTIFIC_SEED = 977_001
retries=0
"ATTEMPT_CONSUMED.json" "ATTEMPT_FAILURE.json" "RESULT.json"
checkpoint_resume_authorized automatic_retry_authorized
stage_d_automatically_authorized
"""

import functools
import hashlib
import importlib
from pathlib import Path
from typing import Any

IMPLEMENTATION_BLOB_SHA = "fb7fe5f8c2a597f7de9e4d03c6908dc0db9558ac"
IMPLEMENTATION_FILENAME = "modal_chm_v1_100m_stage_c_990_v1_impl.py"
PACKAGED_IMPLEMENTATION_MODULE = "tam_research"


class _LazyTorchProxy:
    """Resolve PyTorch only when runtime code actually needs it."""

    @staticmethod
    def _module() -> Any:
        return importlib.import_module("torch")

    def no_grad(self):
        def decorate(function):
            @functools.wraps(function)
            def wrapped(*args, **kwargs):
                module = self._module()
                with module.no_grad():
                    return function(*args, **kwargs)

            return wrapped

        return decorate

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module(), name)


def _git_blob_sha(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _resolve_implementation_path() -> Path:
    sibling = Path(__file__).with_name(IMPLEMENTATION_FILENAME)
    if sibling.is_file():
        return sibling

    package = importlib.import_module(PACKAGED_IMPLEMENTATION_MODULE)
    package_file = getattr(package, "__file__", None)
    if package_file is None:
        raise FileNotFoundError(
            "#996 Stage-C packaged implementation package has no __file__"
        )
    packaged = Path(package_file).resolve().with_name(IMPLEMENTATION_FILENAME)
    if packaged.is_file():
        return packaged

    raise FileNotFoundError(
        "#996 Stage-C implementation missing from both root sibling and "
        f"packaged fallback: {sibling} / {packaged}"
    )


_impl_path = _resolve_implementation_path()
_impl_bytes = _impl_path.read_bytes()
_actual_impl_sha = _git_blob_sha(_impl_bytes)
if _actual_impl_sha != IMPLEMENTATION_BLOB_SHA:
    raise RuntimeError(
        "#996 Stage-C implementation blob mismatch: "
        f"{_actual_impl_sha} != {IMPLEMENTATION_BLOB_SHA}"
    )

# The preserved implementation's four evaluation helpers use @torch.no_grad().
# Bind that symbol to the lazy proxy before evaluating the exact old source.
torch = _LazyTorchProxy()
exec(compile(_impl_bytes, str(_impl_path), "exec"), globals(), globals())
