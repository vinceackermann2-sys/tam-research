from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/aera-issue790-event-memory-seed37641-runner.yml"

V1_PREAUTH = "[aera-event-memory-repair-seed37641-preauth-v1]"
V1_L4 = "[aera-event-memory-repair-seed37641-l4-v1]"
V2_PREAUTH = "[aera-event-memory-repair-seed37641-preauth-v2]"
V2_L4 = "[aera-event-memory-repair-seed37641-l4-v2]"

FROZEN_BLOBS = {
    "tam_research/aera_issue790_event_memory_seed37641_scientific_adapter.py": "931ba460cae55ceb940b61d92288668f6e02ff53",
    "docs/aera_issue790_event_memory_seed37641_protocol.json": "9f865dff148aa46ffd9957259aeb4c0591595cad",
    "modal_aera_issue790_event_memory_seed37641_runner.py": "58bc3bbe16492fa8161456ea5103653dc320f6b6",
    "tests/test_aera_issue790_event_memory_seed37641_runner_cpu.py": "94670f0f77f5ee86ce58ac9e5b3af29be22f2b7d",
    "tam_research/aera_issue785_eval_materialization_bridge_cpu.py": "aa90b3ae3266786323b0517de8385135869c3653",
    "tam_research/aera_issue748_memory_capability_seed1_harness.py": "afc939a69633f68ded05eb585c95a599a8f5c981",
    "tam_research/aera_issue776_event_memory_scientific_adapter.py": "1489df753f2eee0f3acc5aba049c0eeb40fce10d",
    "tam_research/aera_issue770_integrated_event_memory_cpu.py": "b695b1b7b7476be3a16433c96dff32870f2e1f49",
    "tam_research/aera_issue748_memory_capability_seed1_harness_base.py": "40003b68987d026265b1d53fcb637f18277f9cac",
    "tam_research/aera_memory_capability_gate_v1.py": "981602432b684989f7a5011ff953e6965368c5e5",
    "tam_research/aera_hardware_core_v18.py": "97861a2407876f62665b13140c2135b4a11d4597",
    "tam_research/aera_delta_memory.py": "ec0b5d29b3d4ac27bd60fd9c152480b9f177c3e9",
}


def _git_hash(path: str) -> str:
    return subprocess.check_output(["git", "hash-object", path], cwd=ROOT, text=True).strip()


def _extract_bound_main(body: str) -> list[str]:
    return re.findall(r"^Bind main: `([0-9a-f]{40})`$", body, flags=re.MULTILINE)


def test_v2_is_the_only_executable_trigger_pair_and_v1_is_historical_only():
    text = WORKFLOW.read_text()
    job_if = text.split("runs-on:", 1)[0]
    assert V2_PREAUTH in job_if
    assert V2_L4 in job_if
    assert V1_PREAUTH not in job_if
    assert V1_L4 not in job_if
    assert f"if [[ \"${{title}}\" == '{V2_PREAUTH}'*" in text
    assert f"prefix='{V2_PREAUTH}'" in text
    assert f"prefix='{V2_L4}'" in text


def test_bind_main_parser_fails_closed_unless_exactly_one_valid_line():
    text = WORKFLOW.read_text()
    assert "sed -n 's/^Bind main: `\\([0-9a-f]\\{40\\}\\)`$/\\1/p'" in text
    assert "wc -l | tr -d ' ')\" = \"1\"" in text
    assert _extract_bound_main("no bind") == []
    assert _extract_bound_main("Bind main: `abc`") == []
    good = "Bind main: `39f7e55488fe4c3b03ae2664ef89365fed6d4ce7`"
    assert _extract_bound_main(good) == ["39f7e55488fe4c3b03ae2664ef89365fed6d4ce7"]
    assert len(_extract_bound_main(good + "\n" + good)) == 2


def test_v2_l4_requires_v2_preauth_and_separate_explicit_l4_authorization():
    text = WORKFLOW.read_text()
    assert "🔎 **AERA #790 seed37641 preauthorization evidence v2**" in text
    assert "## #790 repaired seed 37641 L4 scientific authorization v2" in text
    assert 'contains("Authorize main: `" + $bound + "`")' in text
    assert 'contains("Authorize seed: `37641`")' in text
    assert 'test "${preauth_count}" = "1"' in text
    assert 'test "${auth_count}" = "1"' in text


def test_v2_workflow_self_identity_is_bound_to_issue819_freeze():
    text = WORKFLOW.read_text()
    assert "## #819 pre-ref seed37641 v2 workflow correction freeze" in text
    assert "V2_WORKFLOW_BLOB=" in text
    assert "git hash-object .github/workflows/aera-issue790-event-memory-seed37641-runner.yml" in text
    assert '"${v2_workflow_blob}"' in text


def test_scientific_runtime_files_are_byte_identical():
    for path, expected in FROZEN_BLOBS.items():
        assert _git_hash(path) == expected


def test_scientific_namespace_and_seed_are_unchanged():
    protocol = (ROOT / "docs/aera_issue790_event_memory_seed37641_protocol.json").read_text()
    assert "/vol/aera-capability/event-memory-repair-v2-seed37641/result.json" in protocol
    assert "/vol/aera-capability/event-memory-repair-v2-seed37641/checkpoints" in protocol
    assert "37641" in protocol
    workflow_text = WORKFLOW.read_text()
    assert "modal_aera_issue790_event_memory_seed37641_runner.py::preauth_main" in workflow_text
    assert "modal_aera_issue790_event_memory_seed37641_runner.py::l4_main" in workflow_text


def test_cpu_static_contract_does_not_execute_modal_or_science():
    source = Path(__file__).read_text()
    assert "subprocess.check_output" in source
    assert "modal run" not in source
    assert "::preauth_main" not in source
    assert "::l4_main" not in source
