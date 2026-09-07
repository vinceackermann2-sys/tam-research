from __future__ import annotations

"""Issue #690 zero-GPU preauthorization repair for #687 stage attribution.

This wrapper reuses the frozen #687 diagnostic app but registers a uniquely named
local entrypoint so importing the diagnostic's existing ``main`` entrypoint does
not collide.
"""

import json

import modal_aera_v26_9_issue687_stage_internal_throughput_attribution as diagnostic

PREAUTH_MARKER = "AERA_V26_9_ISSUE690_STAGE_INTERNAL_PREAUTH_JSON="
app = diagnostic.app


@app.local_entrypoint()
def preauth_main() -> None:
    evidence = diagnostic.preflight.remote()
    print(PREAUTH_MARKER + json.dumps(evidence, sort_keys=True))
