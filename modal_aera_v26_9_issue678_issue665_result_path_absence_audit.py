from __future__ import annotations

"""Issue #678 read-only audit of the frozen #665 diagnostic result path."""

import json

import modal_aera_v26_9_issue665_frozen_throughput_component_attribution as issue665

AUDIT_MARKER = "AERA_V26_9_ISSUE678_ISSUE665_RESULT_PATH_ABSENCE_AUDIT_JSON="
EXPECTED_RESULT_PATH = "/vol/aera-v26/issue665-frozen-throughput-component-attribution/result.json"
EXPECTED_SOURCE_RESULT_SHA256 = "914615db5267565563dcc9e82bfc31f444a656a68bd560f50447a8fd03588431"
EXPECTED_SOURCE_DECISION = "FAIL_FROZEN_E2E_SYSTEMS_GATE"
EXPECTED_FROZEN_BLOBS = {
    "scientific_adapter": "512572340cc09e2e7ad6729712258c12cb377ef2",
    "runtime_interface": "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7",
    "base_systems": "c9731cae7e386f09b2a190b045532591c4fa00be",
    "v26_9_backend": "b81cc209f5d95abbe1fb8bd620c78e87c067bc19",
}

app = issue665.app


@app.local_entrypoint()
def audit_main() -> None:
    if issue665.RESULT_PATH != EXPECTED_RESULT_PATH:
        raise RuntimeError("issue678 frozen issue665 result path drifted")

    evidence = issue665.preflight.remote()

    if evidence.get("research_issue") != 665:
        raise RuntimeError("issue678 frozen preflight research issue drifted")
    if evidence.get("result_absent") is not True:
        raise RuntimeError("issue678 result-path absence was not proven")
    if evidence.get("source_result_sha256") != EXPECTED_SOURCE_RESULT_SHA256:
        raise RuntimeError("issue678 immutable source result SHA256 drifted")
    if evidence.get("source_decision") != EXPECTED_SOURCE_DECISION:
        raise RuntimeError("issue678 immutable source decision drifted")
    if evidence.get("frozen_blobs") != EXPECTED_FROZEN_BLOBS:
        raise RuntimeError("issue678 frozen blob evidence drifted")

    for key in (
        "gpu_used",
        "model_constructed",
        "new_measurement_performed",
        "systems_pass_earned",
        "optimization_authorized",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "independent_replication_credit",
        "100m_authorized",
        "breakthrough_proven",
    ):
        if evidence.get(key) is not False:
            raise RuntimeError(f"issue678 forbidden preflight flag drifted: {key}")

    print(AUDIT_MARKER + json.dumps(evidence, sort_keys=True))
