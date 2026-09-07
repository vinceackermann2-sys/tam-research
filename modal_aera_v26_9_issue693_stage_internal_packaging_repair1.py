from __future__ import annotations

"""Issue #693 infrastructure-only Modal source-packaging repair for frozen #687."""

import json

import modal_aera_v26_9_issue665_frozen_throughput_component_attribution as issue665

ISSUE665_LAUNCHER = "modal_aera_v26_9_issue665_frozen_throughput_component_attribution.py"
issue665.image = issue665.image.add_local_file(
    ISSUE665_LAUNCHER,
    f"/root/{ISSUE665_LAUNCHER}",
)

import modal_aera_v26_9_issue687_stage_internal_throughput_attribution as diagnostic

PREAUTH_MARKER = "AERA_V26_9_ISSUE693_STAGE_INTERNAL_PREAUTH_JSON="
app = diagnostic.app


@app.local_entrypoint()
def preauth_main() -> None:
    evidence = diagnostic.preflight.remote()
    print(PREAUTH_MARKER + json.dumps(evidence, sort_keys=True))


@app.local_entrypoint()
def l4_main() -> None:
    pre = diagnostic.preflight.remote()
    print(diagnostic.PRECHECK_MARKER + json.dumps(pre, sort_keys=True))
    print(
        diagnostic.L4_START_MARKER
        + json.dumps(
            {
                "research_issue": diagnostic.RESEARCH_ISSUE,
                "gpu": "L4",
                "max_gpu_seconds": diagnostic.MAX_GPU_SECONDS,
                "result_path": diagnostic.RESULT_PATH,
                "diagnostic_only": True,
            },
            sort_keys=True,
        )
    )
    summary = diagnostic.run_diagnostic.remote()
    print(diagnostic.SUMMARY_MARKER + json.dumps(summary, sort_keys=True))
