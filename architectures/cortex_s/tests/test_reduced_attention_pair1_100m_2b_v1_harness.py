from __future__ import annotations

import ast
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "modal_cortex_reduced_attention_pair1_100m_2b_v1.py"
WORKFLOW = ROOT / ".github/workflows/modal-cortex-reduced-attention-pair1-100m-2b-v1.yml"


def _function(source: str, name: str) -> str:
    tree = ast.parse(source)
    lines = source.splitlines()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            return "\n".join(lines[start : node.end_lineno])
    raise AssertionError(name)


def _embedded_python(source: str) -> list[str]:
    lines = source.splitlines()
    out: list[str] = []
    index = 0
    while index < len(lines):
        if "python - <<'PY'" not in lines[index]:
            index += 1
            continue
        index += 1
        block: list[str] = []
        while index < len(lines) and lines[index].strip() != "PY":
            block.append(lines[index])
            index += 1
        assert index < len(lines)
        out.append(textwrap.dedent("\n".join(block)))
        index += 1
    return out


def test_pair1_runner_freezes_identity_geometry_and_authority() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    ast.parse(source)
    for needle in (
        'TRIGGER_TITLE = "[modal-cortex-reduced-attention-pair1-100m-2b-v1]"',
        "PAIR_SEED = 58_231",
        'RESULT_ROOT = "/vol/cortex-s-v0/100m-2b/reduced-attention-pair1-v1"',
        "TRAIN_TOKENS = 2_000_000_000",
        "VAL_TOKENS = 5_000_000",
        "SEQ_LEN = 512",
        "MICRO_BATCH_SIZE = 64",
        "GRAD_ACCUM_STEPS = 2",
        "TOTAL_OPTIMIZER_STEPS = 30_518",
        "EVAL_EVERY_TOKENS = 200_000_000",
        "CHECKPOINT_EVERY_TOKENS = 200_000_000",
        "FINAL_EVAL_BATCHES = 50",
        'COMPILE_MODE = "max-autotune-no-cudagraphs"',
        "HARD_MODEL_TIMEOUT_SECONDS = 10_000",
        "PAIR1_SCREEN_MAX_NLL_DELTA = 0.015",
    ):
        assert needle in source
    assert source.count('gpu="H100!"') == 1
    assert "retries=0" in source
    assert "torch.load(" not in source
    assert '"replication_seeds_authorized": False' in source
    assert '"250m_5b_authorized": False' in source
    assert '"architecture_superiority_claim_allowed": False' in source


def test_pair1_uses_shared_training_path_and_same_seed_data_order() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    training = _function(source, "_train_architecture")
    assert "seed_all(PAIR_SEED)" in training
    assert 'manual_seed(PAIR_SEED + 10_000)' in training
    assert "torch.optim.AdamW(" in training
    assert "betas=ADAMW_BETAS" in training
    assert "weight_decay=WEIGHT_DECAY" in training
    assert "fused=True" in training
    assert "torch.compile(" in training
    assert "mode=COMPILE_MODE" in training
    assert "fullgraph=False" in training
    assert 'torch.autocast(device_type="cuda", dtype=torch.bfloat16)' in source
    assert "logits.float().reshape(-1, logits.size(-1))" in source
    assert "torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)" in source
    assert 'architecture == "transformer"' in source
    assert "_build_transformer" in source
    assert "_build_reduced_attention" in source


def test_pair1_reservation_consumes_seed_before_any_h100_execution() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    reserve = _function(source, "reserve_pair_dispatch")
    gpu = _function(source, "h100_train_one")
    assert '"pair_seed_consumed": True' in reserve
    assert "_atomic_write(marker_path, marker)" in reserve
    assert "volume.commit()" in reserve
    assert 'status": "PAIR1_DISPATCH_RESERVED"' in reserve
    assert 'status": "ATTEMPT_STARTED"' in gpu
    assert '"h100_allocation_started": True' in gpu
    assert '"automatic_retry_authorized": False' in gpu
    assert '"resume_authorized": False' in gpu


def test_pair1_execution_order_is_transformer_then_candidate_and_fail_closed() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    main = _function(source, "main")
    transformer = main.index('h100_train_one.remote("transformer"')
    reduced = main.index('h100_train_one.remote("reduced_attention"')
    assert transformer < reduced
    assert 'transformer.get("status") == "COMPLETE"' in main
    assert '"reduced_attention_skipped": True' in main
    assert "finalize_pair.remote(" in main


def test_pair1_screen_is_exact_preregistered_rule() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    finalizer = _function(source, "finalize_pair")
    assert "r_nll - t_nll" in finalizer
    assert "delta <= PAIR1_SCREEN_MAX_NLL_DELTA" in finalizer
    assert '"PAIR1_SCREEN_PASS"' in finalizer
    assert '"PAIR1_SCREEN_FAIL"' in finalizer
    assert '"PAIR1_INCOMPLETE_NO_RETRY"' in finalizer
    assert '"replication_seeds_authorized": False' in finalizer


def test_pair1_checkpoints_are_evidence_only_never_resume_authority() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    save = _function(source, "_save_checkpoint")
    assert "torch.save(" in save
    assert '"resume_authorized": False' in save
    assert '"automatic_retry_authorized": False' in save
    assert "load_state_dict" not in source
    assert "resume_from" not in source


def test_pair1_workflow_is_exact_owner_only_one_shot_and_dual_account_bound() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    top = source.split("\njobs:", 1)[0]
    assert "workflow_dispatch:" not in top
    assert "issues:" in top and "types: [opened]" in top
    assert (
        "github.event.issue.title == "
        "'[modal-cortex-reduced-attention-pair1-100m-2b-v1]'"
        in source
    )
    assert "github.event.issue.user.login == github.repository_owner" in source
    assert 'test "$RUN_ATTEMPT" = "1"' in source
    assert "git merge-base --is-ancestor" in source
    assert "git diff --exit-code" in source
    assert "scripts/modal_select_account.py --required-volume tam-research-data" in source
    assert 'SELECTED" = "secondary"' in source
    assert 'SELECTED" = "primary"' in source
    assert "modal volume list --json" in source
    launch = source.split(
        "\n      - name: Launch exact one-shot Pair-1 scientific run", 1
    )[1]
    launch = launch.split("\n      - name: Record normal Modal return only", 1)[0]
    assert launch.count("modal run --detach --timestamps") == 1
    assert "modal_cortex_reduced_attention_pair1_100m_2b_v1.py" in launch


def test_pair1_workflow_embedded_python_parses() -> None:
    blocks = _embedded_python(WORKFLOW.read_text(encoding="utf-8"))
    assert blocks
    for block in blocks:
        ast.parse(block)
