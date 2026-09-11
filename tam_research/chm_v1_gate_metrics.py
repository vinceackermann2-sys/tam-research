from __future__ import annotations

"""Preregistered CHM-v1 / EIEM gate metric helpers.

This module is measurement plumbing only. It does not alter model behavior,
training, data, retrieval, probes, thresholds, seeds, or execution authority.
It closes the stale-state accounting gap in issue #854 by reporting LOCAL and
EIEM overwrite stale-value errors under the already-frozen criterion.
"""

from collections.abc import Sequence
from typing import Any

from .chm_v1_long_memory_eval import EncodedProbe, summarize_probe_predictions


def summarize_preregistered_probe_metrics(
    probes: Sequence[EncodedProbe],
    local_predictions: Sequence[int],
    eiem_predictions: Sequence[int],
) -> dict[str, dict[str, Any]]:
    """Return existing probe metrics plus the frozen LOCAL-relative stale gate.

    Issue #854 requires EIEM overwrite stale-value error to be at least 50%
    lower relative to LOCAL, unless LOCAL stale error is already below 1%, in
    which case EIEM may be at most +0.5 percentage points worse.
    """
    summary: dict[str, dict[str, Any]] = {
        family: dict(metrics)
        for family, metrics in summarize_probe_predictions(
            probes, local_predictions, eiem_predictions
        ).items()
    }

    overwrite_rows = [i for i, probe in enumerate(probes) if probe.family == "overwrite"]
    if not overwrite_rows:
        return summary

    local_stale_hits = sum(
        local_predictions[i] in probes[i].stale_token_ids for i in overwrite_rows
    )
    eiem_stale_hits = sum(
        eiem_predictions[i] in probes[i].stale_token_ids for i in overwrite_rows
    )
    total = len(overwrite_rows)
    local_stale_error = local_stale_hits / total
    eiem_stale_error = eiem_stale_hits / total

    if local_stale_error < 0.01:
        stale_gate_pass = eiem_stale_error <= local_stale_error + 0.005
        relative_reduction = None
    else:
        stale_gate_pass = eiem_stale_error <= 0.5 * local_stale_error
        relative_reduction = 1.0 - (eiem_stale_error / local_stale_error)

    overwrite = summary.setdefault("overwrite", {})
    overwrite["local_stale_error"] = local_stale_error
    overwrite["eiem_stale_error"] = eiem_stale_error
    overwrite["stale_relative_reduction"] = relative_reduction
    overwrite["stale_gate_pass"] = stale_gate_pass
    overwrite["stale_gate_rule"] = (
        "eiem<=local+0.005_when_local<0.01"
        if local_stale_error < 0.01
        else "eiem<=0.5*local"
    )
    return summary
