from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / ".github/workflows/aera-issue790-event-memory-seed37641-runner.yml"
V2 = ROOT / ".github/workflows/aera-issue829-event-memory-seed37641-v2.yml"


def test_v1_workflow_remains_byte_frozen() -> None:
    import subprocess
    got = subprocess.check_output(["git", "hash-object", str(V1)], text=True).strip()
    assert got == "f4357904b4316b6e2bc2dfcc6ef390a5317d2da5"


def test_v2_only_accepts_v2_trigger_pair() -> None:
    source = V2.read_text()
    assert "[aera-event-memory-repair-seed37641-preauth-v2]" in source
    assert "[aera-event-memory-repair-seed37641-l4-v2]" in source
    job_guard = source.split("runs-on:", 1)[0]
    assert "preauth-v1" not in job_guard
    assert "l4-v1" not in job_guard


def test_v2_requires_exactly_one_machine_readable_bind() -> None:
    source = V2.read_text()
    assert "sed -n 's/^Bind main:" in source
    pat = re.compile(r"^Bind main: `([0-9a-f]{40})`$", re.MULTILINE)
    good = "x\nBind main: `0123456789abcdef0123456789abcdef01234567`\ny"
    bad0 = "x\nBind main 0123456789abcdef0123456789abcdef01234567\ny"
    bad2 = good + "\nBind main: `89abcdef0123456789abcdef0123456789abcdef`"
    assert len(pat.findall(good)) == 1
    assert len(pat.findall(bad0)) == 0
    assert len(pat.findall(bad2)) == 2


def test_v2_preserves_frozen_runtime_hash_guards() -> None:
    source = V2.read_text()
    frozen = {
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
    for path, blob in frozen.items():
        assert path in source
        assert blob in source


def test_v2_preauth_evidence_is_distinct_and_l4_separately_authorized() -> None:
    source = V2.read_text()
    assert "AERA #790 seed37641 preauthorization evidence v2" in source
    assert "#790 repaired seed 37641 L4 scientific authorization v2" in source
    assert 'test "${preauth_count}" = "1"' in source
    assert 'test "${auth_count}" = "1"' in source


def test_v2_binds_to_issue829_freeze_and_self_hash() -> None:
    source = V2.read_text()
    assert "issues/829/comments" in source
    assert "#829 pre-ref additive v2 workflow freeze" in source
    assert 'git merge-base --is-ancestor "${source_main}" "${bound_main}"' in source
    assert 'test "$(git hash-object .github/workflows/aera-issue829-event-memory-seed37641-v2.yml)" = "${v2_workflow_blob}"' in source


def test_cpu_static_test_contains_no_execution_api_invocations() -> None:
    source = Path(__file__).read_text()
    forbidden = [
        "os." + "system(",
        "sub" + "process" + ".run(",
        "sub" + "process" + ".call(",
        "Po" + "pen(",
    ]
    for token in forbidden:
        assert token not in source
