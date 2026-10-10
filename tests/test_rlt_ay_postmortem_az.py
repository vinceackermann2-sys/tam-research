"""Forensic tests consume recorded artifacts only: no torch, no training."""
from experiments.rlt.analysis.ay_postmortem_az import audit


def test_ay_negative_long_horizon_results_are_preserved():
    r = audit()
    assert r["ay_metrics_persisted"] and not r["new_training"] and not r["gpu_used"]
    carry = r["models"]["carry"]
    assert carry["accuracy_by_distance"]["4"] == 1.0
    for d in ("16", "64", "127"):
        assert carry["accuracy_by_distance"][d] == 0.5
        assert carry["long_distance"][d]["majority_class_collapse"]
        assert carry["long_distance"][d]["class0_accuracy"] == 0.0
        assert carry["long_distance"][d]["class1_accuracy"] == 1.0


def test_aw_is_not_mislabeled_as_reproduced_success():
    r = audit()
    assert r["AW_vs_AY_carry"]["16"]["percentage_point_change"] < -40
    assert r["AW_vs_AY_carry"]["64"]["percentage_point_change"] < -40
    assert 5.8 < r["transformer_to_carry_cpu_throughput_ratio"] < 6.1


def test_trained_checkpoint_preservation_defect_is_explicit():
    r = audit()
    assert r["pt_gitignore_rule_active"]
    assert r["checkpoints_in_repo"] is False
    assert len(r["missing_checkpoints"]) == 3
    assert r["checkpoint_recovery_by_rerun_prohibited"]


def test_report_retention_products_but_no_unfounded_causal_attribution():
    r = audit()
    for cls in (0, 1):
        assert 0 < r["AY_carry_direct_retention"]["127"][cls] < 0.04
        assert r["AY_carry_direct_retention"]["16"][cls] > 0.29
