from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tam_research.chm_v1_small_lm import LOCAL_WINDOW, _hidden
from tam_research.chm_v3_100m_daec import DecoderAlignedEpisodicCopy
from tam_research.chm_v3_100m_daec_stage_c import (
    REQUIRED_INTEGRITY_FLAGS,
    daec_hard_flat_final_logits_with_trace,
)
from tam_research.chm_v3_100m_daec_stage_c_forensic import (
    FROZEN_RESULT_COMMENT_ID,
    FROZEN_SCIENTIFIC_SEED,
    FROZEN_SOURCE_SHA,
    audit_archived_stage_c_result,
    parity_proof_status,
)

ROOT = Path(__file__).resolve().parents[1]
FROZEN_RUNNER = ROOT / "modal_chm_v3_100m_daec_stage_c_1323_v1.py"


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(17, 8)
        self.pos_emb = nn.Embedding(LOCAL_WINDOW, 8)
        self.blocks = nn.ModuleList()
        self.norm = nn.LayerNorm(8)
        self.lm_head = nn.Linear(8, 17, bias=False)
        self.lm_head.weight = self.token_emb.weight


class TinyDAEC(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = TinyBackbone()
        self.query_address = nn.Linear(8, 4, bias=False)
        self.key_address = nn.Linear(8, 4, bias=False)
        self.daec = DecoderAlignedEpisodicCopy(d_model=8, address_dim=4)

    def query_for(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.query_address(hidden), dim=-1)

    def key_for(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.key_address(hidden), dim=-1)


@torch.no_grad()
def independent_hard_flat_reference(
    model: TinyDAEC, prompt: list[int],
) -> tuple[torch.Tensor, tuple[int | None, int | None, int | None]]:
    """Separate explicit, position-by-position argmin reference.

    Uses Euclidean nearest position in normalized address space, and a
    probability-domain one-hot copy mixture (not the production logaddexp).
    All memory writes occur only after completed prior chunks.
    """
    chunks = [prompt[i:i+LOCAL_WINDOW] for i in range(0, len(prompt), LOCAL_WINDOW)]
    memory_keys: list[torch.Tensor] = []
    memory_ids: list[int] = []
    for completed in chunks[:-1]:
        x = torch.tensor([completed], dtype=torch.long)
        hidden = _hidden(model.backbone, x)
        keys = model.key_for(hidden)[0]
        memory_keys.extend([keys[j] for j in range(len(completed))])
        memory_ids.extend(completed)

    x = torch.tensor([chunks[-1]], dtype=torch.long)
    hidden = _hidden(model.backbone, x)
    final_hidden = hidden[0, -1]
    lm = torch.softmax(model.backbone.lm_head(final_hidden).float(), dim=-1)
    if not memory_ids:
        return lm.log(), (None, None, None)

    first_query = model.query_for(final_hidden)
    distances = [
        float(torch.sum((key - first_query).square()).item())
        for key in memory_keys
    ]
    first = min(range(len(distances)), key=lambda pos: distances[pos])
    first_embedding = model.backbone.token_emb(
        torch.tensor(memory_ids[first], dtype=torch.long)
    )
    second_query = model.daec.second_query(first_query, first_embedding)
    distances2 = [
        float(torch.sum((key - second_query).square()).item())
        for key in memory_keys
    ]
    second = min(range(len(distances2)), key=lambda pos: distances2[pos])
    copied_id = memory_ids[second]
    gate = model.daec.gate(final_hidden).float().reshape(())
    mixed = lm * (1.0 - gate)
    mixed[copied_id] += gate
    return mixed.log(), (first, second, copied_id)


@pytest.mark.parametrize("length", [7, 512, 513, 713, 1025])
def test_hard_flat_synthetic_independent_parity_and_causality(length: int) -> None:
    torch.manual_seed(1332 + length)
    model = TinyDAEC().eval()
    # Duplicate token IDs and differing final-chunk lengths are intentional.
    ids = [(3 * i + (i // 9)) % 17 for i in range(length)]
    actual, trace = daec_hard_flat_final_logits_with_trace(model, ids)
    expected, indices = independent_hard_flat_reference(model, ids)
    torch.testing.assert_close(actual.exp(), expected.exp(), rtol=2e-5, atol=2e-6)
    assert float(actual.exp().sum()) == pytest.approx(1.0, abs=2e-6)
    assert (
        trace["first_hop_index"],
        trace["second_hop_index"],
        trace["copied_token_id"],
    ) == indices
    assert trace["memory_count"] == ((length - 1) // LOCAL_WINDOW) * LOCAL_WINDOW


def test_parity_missing_is_distinct_from_measured_failure() -> None:
    assert "hard_flat_evaluation_parity_verified" in REQUIRED_INTEGRITY_FLAGS
    assert parity_proof_status({}) == "missing"
    assert parity_proof_status({"hard_flat_evaluation_parity_verified": False}) == "not_verified"
    assert parity_proof_status({"hard_flat_evaluation_parity_verified": True}) == "asserted_without_evidence"
    fake = {
        "source_sha": FROZEN_SOURCE_SHA,
        "trained_checkpoint_sha256": "1" * 64,
        "scored_probe_rows_sha256": "2" * 64,
        "independent_reference_impl_sha256": "3" * 64,
        "probes_compared": 512,
        "max_probability_absolute_error": 0.0,
        "first_hop_index_mismatches": 0,
        "second_hop_index_mismatches": 0,
    }
    assert parity_proof_status(
        {"hard_flat_evaluation_parity_verified": True}, fake
    ) == "independent_evidence_present_not_checkpoint_authenticated"
    assert parity_proof_status(
        {"hard_flat_evaluation_parity_verified": True}, {**fake, "first_hop_index_mismatches": 1}
    ) == "parity_mismatch"
    assert parity_proof_status(
        {"hard_flat_evaluation_parity_verified": True}, {**fake, "source_sha": "4" * 40}
    ) == "source_mismatch"


def test_frozen_scientific_runner_did_not_emit_parity_flag() -> None:
    source = FROZEN_RUNNER.read_text(encoding="utf-8")
    assert "def run_scientific(" in source
    integrity_start = source.index("        integrity = {", source.index("def run_scientific("))
    integrity_end = source.index("        decision = classify_stage_c(", integrity_start)
    original_mapping = source[integrity_start:integrity_end]
    assert '"hard_flat_evaluation_parity_verified"' not in original_mapping
    assert FROZEN_RESULT_COMMENT_ID == 6_058_454_735


def fake_archived_result() -> dict:
    rows = []
    for family_index, family in enumerate(("rare_fact", "overwrite", "two_hop", "local_negative")):
        for i in range(128):
            rows.append({
                "family": family,
                "case_id": family_index * 1000 + i,
                "local_correct": i < 16,
                "raw_correct": i < 16,
                "daec_correct": i < 16,
                "daec_copied_token_id": 7 if family != "local_negative" else None,
                "daec_copy_matches_answer": False,
            })
    return {
        "source_sha": FROZEN_SOURCE_SHA,
        "scientific_seed": FROZEN_SCIENTIFIC_SEED,
        "classification": "CHM_V3_100M_DAEC_STAGE_C_STOP",
        "passed": False,
        "scientific_seed_consumed": True,
        "integrity": {"finite_losses": True, "finite_parameters": True},
        "probe_rows": rows,
        "ordinary_language": {
            "local": {"nll": 5.1977433785796165},
            "raw_eiem": {"nll": 5.1769649013876915},
            "daec_eiem": {"nll": 5.094711102545261},
        },
        "metrics": {"aggregate_long_range": {
            "daec_vs_local": {"candidate_nll_benefit": -0.10808602316925923},
            "daec_vs_raw": {"candidate_nll_benefit": -0.1903133027565976},
        }},
    }


def test_archived_result_report_is_read_only_and_cannot_reclassify() -> None:
    original = fake_archived_result()
    snapshot = repr(original)
    report = audit_archived_stage_c_result(original)
    assert repr(original) == snapshot
    assert report["classification"] == "CHM_V3_1323_FORENSIC_READ_ONLY_REPORT"
    assert report["original_classification_unchanged"] == "CHM_V3_100M_DAEC_STAGE_C_STOP"
    assert report["can_certify_scientific_pass"] is False
    assert report["parity_proof_status"] == "missing"
    assert report["scientific_seed_consumed"] is True
    assert report["scientific_result_reinterpreted"] is False
    assert sum(x["memory_probe_count"] for x in report["family_rows"].values()) == 384
    assert sum(x["copy_targets_hit"] for x in report["family_rows"].values()) == 0
    for x in report["family_rows"].values():
        assert x["top1_correct"] == {"local": 16, "raw": 16, "daec": 16}
    assert report["ordinary_language_nll"]["daec"] < report["ordinary_language_nll"]["local"]


def test_postmortem_refuses_reclassification_seed_reuse_or_undergrounded_rows() -> None:
    original = fake_archived_result()
    for field, bad in (
        ("source_sha", "0" * 40),
        ("scientific_seed", 2013162),
        ("classification", "CHM_V3_100M_DAEC_POSITIVE_DEVELOPMENT_SCREEN"),
        ("passed", True),
        ("scientific_seed_consumed", False),
    ):
        with pytest.raises(ValueError):
            audit_archived_stage_c_result({**original, field: bad})
    with pytest.raises(ValueError, match="512"):
        audit_archived_stage_c_result({**original, "probe_rows": original["probe_rows"][:511]})


def test_forensic_helper_has_no_gpu_launcher_training_or_historical_mutation() -> None:
    import tam_research.chm_v3_100m_daec_stage_c_forensic as helper
    source = inspect.getsource(helper)
    assert "modal.App(" not in source
    assert "modal.run(" not in source
    assert "optimizer.step(" not in source
    assert "write_text(" not in source
    assert "torch.cuda" not in source
