from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "modal_cortex_s_100m_host_sync_profile_v3_result_readout.py"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "modal-cortex-s-100m-host-sync-profile-v3-result-readout-v1.yml"


def test_readout_runner_is_strictly_read_only_and_zero_gpu() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert 'PHASE = "host-sync-profile-v3-result-readout-v1"' in text
    assert 'TRIGGER_TITLE = "[modal-cortex-s-host-sync-profile-v3-result-readout-v1]"' in text
    assert 'ORIGINAL_SOURCE_SHA = "22ef4aac4735128219c720c839e6978c424114dc"' in text
    assert 'ORIGINAL_SOURCE_TREE = "0e89d152aba053b45dcf2c07b878781048f8c23e"' in text
    assert 'ORIGINAL_HARNESS_SHA = "13b3c3f8cc0f3d3cd60376099a945d6984cb320f"' in text
    assert 'ORIGINAL_ENGINEERING_SEED = 2_026_091_007' in text
    assert 'RESULT_ROOT = "/vol/cortex-s-v0/100m-2b/host-sync-profile-v3"' in text
    assert 'VOLUME_NAME = "tam-research-data"' in text
    assert 'volume.reload()' in text
    assert 'read_existing_result.remote(' in text
    assert 'CORTEX_S_V3_RESULT_READOUT=' in text

    forbidden = (
        'gpu=',
        'gpu =',
        'volume.commit',
        '_atomic_write',
        '.write_text(',
        '.write_bytes(',
        '.replace(',
        '.unlink(',
        '.mkdir(',
        'torch',
        'training_module',
        '_one_optimizer_step',
        '_compile_model',
        'train_full_2b',
        'optimizer.step',
    )
    for token in forbidden:
        assert token not in text, f"readout runner contains forbidden execution/mutation token: {token}"

    assert text.count('@app.function(') == 1
    function_block = text[text.index('@app.function('):text.index('@app.local_entrypoint()')]
    assert 'cpu=1' in function_block
    assert 'volumes={"/vol": volume}' in function_block


def test_readout_validates_consumed_attempt_and_forbids_authority_escalation() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert 'zero_gpu.get("engineering_seed_reserved") != ORIGINAL_ENGINEERING_SEED' in text
    assert 'dispatch.get("engineering_seed") != ORIGINAL_ENGINEERING_SEED' in text
    assert 'dispatch.get("h100_allocation_started") is not True' in text
    assert 'result.get("status") != "PROFILE_COMPLETE"' in text
    assert 'result.get("engineering_seed") != ORIGINAL_ENGINEERING_SEED' in text
    assert 'result.get("seed_8100_used") is not False' in text
    assert 'result.get("reserved_scientific_seeds_used") is not False' in text
    assert 'result.get("full_training_authorized") is not False' in text
    assert 'result.get("scientific_claim_authorized") is not False' in text
    assert '"read_only": True' in text
    assert '"gpu_allocated_by_readout": False' in text
    assert '"new_seed_consumed": False' in text
    assert '"full_training_authorized": False' in text
    assert '"scientific_claim_authorized": False' in text


def test_readout_workflow_is_owner_only_source_bound_and_gpu_free() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "github.event.issue.user.login == github.repository_owner" in text
    assert "github.event.issue.title == '[modal-cortex-s-host-sync-profile-v3-result-readout-v1]'" in text
    assert "modal_cortex_s_100m_host_sync_profile_v3_result_readout.py" in text
    assert "CORTEX_S_V3_RESULT_READOUT=" in text
    assert "22ef4aac4735128219c720c839e6978c424114dc" in text
    assert "0e89d152aba053b45dcf2c07b878781048f8c23e" in text
    assert "13b3c3f8cc0f3d3cd60376099a945d6984cb320f" in text

    forbidden = (
        'H100',
        'gpu=',
        'gpu =',
        'volume commit',
        'rerun',
        'retry',
        'train_full_2b',
    )
    for token in forbidden:
        assert token not in text, f"readout workflow contains forbidden escalation token: {token}"
