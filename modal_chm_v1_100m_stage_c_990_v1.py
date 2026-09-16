from __future__ import annotations

"""Lazy host-import boundary for the frozen CHM-v1 Stage-C runner (#993).

The scientific implementation is preserved byte-for-byte in the sibling
``_impl.py`` blob.  This shim exists only because Modal evaluates launcher
modules on the host before entering the image where PyTorch is installed.
It binds the implementation cryptographically and defers all real ``torch``
resolution until a decorated helper is actually called inside that image.

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


_impl_path = Path(__file__).with_name(IMPLEMENTATION_FILENAME)
_impl_bytes = _impl_path.read_bytes()
_actual_impl_sha = _git_blob_sha(_impl_bytes)
if _actual_impl_sha != IMPLEMENTATION_BLOB_SHA:
    raise RuntimeError(
        "#993 Stage-C implementation blob mismatch: "
        f"{_actual_impl_sha} != {IMPLEMENTATION_BLOB_SHA}"
    )

# The preserved implementation's four evaluation helpers use @torch.no_grad().
# Bind that symbol to the lazy proxy before evaluating the exact old source.
torch = _LazyTorchProxy()
exec(compile(_impl_bytes, str(_impl_path), "exec"), globals(), globals())
