from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "modal_cortex_s_v10_durable_result_readout.py"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "modal-cortex-s-v10-durable-result-readout-v1.yml"


def test_v10_readout_runner_is_strictly_read_only_and_zero_gpu() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(text)

    assert 'PHASE = "preflight-v10-durable-result-readout-v1"' in text
    assert 'TRIGGER_TITLE = "[modal-cortex-s-v10-durable-result-readout-v1]"' in text
    assert 'ORIGINAL_SOURCE_SHA = "6e2b535a2eb8fdea6579fc27849091ee5125b9ea"' in text
    assert 'ORIGINAL_SOURCE_TREE = "2a019c7d5785e9c4b7e48eda15c9b6faec30e8f3"' in text
    assert 'ORIGINAL_HARNESS_SHA = "48c5d5a2ee7ff3688785f8e7943d537e5215358b"' in text
    assert 'ORIGINAL_ENGINEERING_SEED = 2_026_091_012' in text
    assert 'RESULT_ROOT = "/vol/cortex-s-v0/100m-2b/preflight-v10-autocast-ce-no-explicit-fp32"' in text
    assert 'VOLUME_NAME = "tam-research-data"' in text
    assert 'create_if_missing=False' in text
    assert 'volume.reload()' in text
    assert 'result_present' in text
    assert 'CORTEX_S_V10_DURABLE_READOUT=' in text

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
        'import torch',
        'training_module',
        '_one_optimizer_step',
        '_compile_model',
        'train_full_2b',
        'optimizer.step',
    )
    for token in forbidden:
        assert token not in text, f"readout runner contains forbidden execution/mutation token: {token}"

    gpu_keywords = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            gpu_keywords.extend(kw for kw in node.keywords if kw.arg == "gpu")
    assert gpu_keywords == []
    assert text.count('@app.function(') == 1
    function_block = text[text.index('@app.function('):text.index('@app.local_entrypoint()')]
    assert 'cpu=1' in function_block
    assert 'volumes={"/vol": volume}' in function_block


def test_v10_readout_validates_consumed_attempt_and_allows_result_absence() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert 'zero_gpu.get("engineering_seed_reserved") != ORIGINAL_ENGINEERING_SEED' in text
    assert 'dispatch.get("engineering_seed") != ORIGINAL_ENGINEERING_SEED' in text
    assert 'dispatch.get("status") != "H100_DISPATCH_CONSUMED"' in text
    assert 'return None, None' in text
    assert '"result_present": result is not None' in text
    assert '"new_seed_consumed": False' in text
    assert '"full_training_authorized": False' in text
    assert '"scientific_claim_authorized": False' in text


def test_v10_readout_workflow_is_owner_only_source_bound_and_gpu_free() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "github.event.issue.user.login == github.repository_owner" in text
    assert "github.event.issue.title == '[modal-cortex-s-v10-durable-result-readout-v1]'" in text
    assert "modal_cortex_s_v10_durable_result_readout.py" in text
    assert "CORTEX_S_V10_DURABLE_READOUT=" in text
    assert "6e2b535a2eb8fdea6579fc27849091ee5125b9ea" in text
    assert "2a019c7d5785e9c4b7e48eda15c9b6faec30e8f3" in text
    assert "48c5d5a2ee7ff3688785f8e7943d537e5215358b" in text
    assert "modal run" in text
    assert "modal-cortex-s-v10-durable-result-readout-v1" in text
    assert "for forbidden in (" in text
    assert "assert forbidden not in runner" in text
