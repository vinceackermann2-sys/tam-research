"""Zero GPU audit of the Conv7 engineering proposal cosine prefix."""
import json
import math
from pathlib import Path
from tam_research.train import cosine_lr

ROOT = Path(__file__).resolve().parents[1]
PROPOSAL = ROOT / "research/reduced_attention/attention8_conv7_engineering_proposal_v1.json"
REPORT = ROOT / "research/reduced_attention/attention8_conv7_lr_prefix_audit_v1.md"


def _pilot():
    proposal = json.loads(PROPOSAL.read_text(encoding="utf-8"))
    assert proposal["stage"] == "ZERO_GPU_PROPOSAL_NOT_PREREGISTERED"
    assert all(v is False for v in proposal["authority"].values())
    p = proposal["draft_engineering_pilot"]
    assert p["engineering_seed"] is None
    assert p["strict_max_gpu_spend_usd"] is None
    assert p["modal_account"] == "primary"
    return p


def test_lr_uses_full_2b_horizon_without_ending_at_200m():
    p = _pilot()
    n, total, peak = p["per_model_optimizer_steps"], p["full_horizon_optimizer_steps"], p["learning_rate"]
    warmup = int(total * p["warmup_ratio_full_horizon"])
    assert (n, total, warmup) == (3052, 30518, 610)
    assert p["warmup_steps_full_horizon"] == warmup
    lrs = [cosine_lr(i, total, warmup, peak) for i in range(n)]
    assert math.isclose(lrs[0], peak / 610, rel_tol=1e-12)
    assert math.isclose(lrs[609], peak, rel_tol=1e-12)
    assert math.isclose(lrs[610], peak, rel_tol=1e-12)
    assert math.isclose(lrs[-1], 0.0002955864947336454, rel_tol=1e-12)
    assert all(math.isfinite(x) and x > 0 for x in lrs)
    assert p["per_model_token_exposures"] == n * p["tokens_per_step"]
    assert p["full_horizon_token_exposures"] == total * p["tokens_per_step"]


def test_old_short_horizon_schedule_differs_substantially():
    p = _pilot()
    n, total, peak = p["per_model_optimizer_steps"], p["full_horizon_optimizer_steps"], p["learning_rate"]
    short_warmup = int(n * p["warmup_ratio_full_horizon"])
    correct = cosine_lr(n - 1, total, p["warmup_steps_full_horizon"], peak)
    incorrect = cosine_lr(n - 1, n, short_warmup, peak)
    assert short_warmup == 61
    assert math.isclose(incorrect, 0.000030000074468164976, rel_tol=1e-12)
    assert 9.8 < correct / incorrect < 10
    assert math.isclose(cosine_lr(0, n, short_warmup, peak) / cosine_lr(0, total, p["warmup_steps_full_horizon"], peak), 10, rel_tol=1e-12)


def test_draft_governance_and_evaluation_geometry():
    p = _pilot()
    assert p["comparators"] == ["fresh_transformer", "reduced_attention_dense_v2_attention8", "attention8_causal_depthwise_conv7"]
    assert p["shared_heldout_batch_stream"] is True
    assert p["same_training_seed_per_model"] is True
    assert p["evaluation_batch_size"] * p["sequence_length"] * p["final_evaluation_batches"] == 819200
    assert "no retries" in p["attempt_policy"]
    report = REPORT.read_text(encoding="utf-8")
    assert "not paid-run authorization" in report
    assert "No Modal runner" in report
