from __future__ import annotations

"""CPU-only implementation repair for the frozen post-#316 residual recall audit.

The first CI-only implementation used ``torch.linalg.lstsq`` for the diagnostic
matrix-capacity upper bound.  On the pinned CPU PyTorch build, the default LAPACK
driver can truncate an exactly solvable underdetermined system.  This wrapper
changes no scientific protocol: it replaces only that numerical oracle helper
with an explicit Moore-Penrose pseudoinverse, then executes the already-frozen
#317 audit unchanged.
"""

import json

import torch

import aera_v21_residual_raw_recall_audit_cpu as base


def _least_squares_matrix(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Minimum-norm matrix-capacity oracle with explicit rank handling."""
    if k.ndim != 2 or v.ndim != 2 or k.size(0) != v.size(0):
        raise ValueError("K and V must be [bindings,dim] with matching rows")
    k32 = k.float()
    v32 = v.float()
    # Tight relative cutoff: the controlled audit has <=12 bindings in a 16-d
    # address space.  This keeps genuine learned rank while avoiding the CPU
    # lstsq driver's underdetermined truncation observed in PR #318 CI.
    pinv = torch.linalg.pinv(k32, atol=0.0, rtol=1e-7)
    return (pinv @ v32).to(v.dtype)


def run_audit() -> dict[str, object]:
    original = base._least_squares_matrix
    base._least_squares_matrix = _least_squares_matrix
    try:
        result = base.run_audit()
    finally:
        base._least_squares_matrix = original
    result["implementation_revision"] = "explicit-pseudoinverse-v2"
    return result


def main() -> None:
    result = run_audit()
    print(
        "AERA_V21_RESIDUAL_RAW_RECALL_AUDIT_RESULT_JSON="
        + json.dumps(result, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
