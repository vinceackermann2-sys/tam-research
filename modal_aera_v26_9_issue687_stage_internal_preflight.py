from __future__ import annotations

"""Issue #687 zero-GPU preauthorization wrapper for the stage attribution preflight."""

import json

import modal_aera_v26_9_issue687_stage_internal_throughput_attribution as diagnostic

PREAUTH_MARKER = "AERA_V26_9_ISSUE687_STAGE_INTERNAL_PREAUTH_JSON="
app = diagnostic.app


@app.local_entrypoint()
def main() -> None:
    evidence = diagnostic.preflight.remote()
    print(PREAUTH_MARKER + json.dumps(evidence, sort_keys=True))
