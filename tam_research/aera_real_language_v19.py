from __future__ import annotations

from typing import Any

import torch

from . import aera_real_language_v18 as v18
from .aera_hardware_core_v19 import (
    HardwareAwareAERATextLMV19,
    memory_addressing_protocol,
)
from .aera_real_language import parameter_accounting

CPU_DIAGNOSTIC_SEED = 8361
CHUNK_SIZE = v18.CHUNK_SIZE
DENSE_WARMUP_STEPS = v18.DENSE_WARMUP_STEPS
ROUTER_CALIBRATION_END = v18.ROUTER_CALIBRATION_END
SPARSE_CALIBRATION_EVERY = v18.SPARSE_CALIBRATION_EVERY
STAGE_DIFFICULTY_WEIGHT = v18.STAGE_DIFFICULTY_WEIGHT
STAGE_BUDGET_WEIGHT = v18.STAGE_BUDGET_WEIGHT
STAGE_POLARIZATION_WEIGHT = v18.STAGE_POLARIZATION_WEIGHT

phase_for_step = v18.phase_for_step
route_mode_for_step = v18.route_mode_for_step
router_policy_for_step = v18.router_policy_for_step
per_chunk_language_loss = v18.per_chunk_language_loss
build_transformer = v18.build_transformer
aera_matched_loss = v18.aera_matched_loss
set_optional_stage_router_trainable = v18.set_optional_stage_router_trainable


def build_aera(device: torch.device) -> HardwareAwareAERATextLMV19:
    return HardwareAwareAERATextLMV19(v18.v12.aera_v12_config()).to(device)


def cpu_preflight() -> dict[str, Any]:
    torch.manual_seed(CPU_DIAGNOSTIC_SEED)
    reference = v18.build_aera(torch.device("cpu")).eval()
    torch.manual_seed(CPU_DIAGNOSTIC_SEED)
    candidate = build_aera(torch.device("cpu")).eval()

    if set(reference.state_dict()) != set(candidate.state_dict()):
        raise RuntimeError("v19 changed checkpoint/state-dict layout relative to v18")
    if sum(p.numel() for p in reference.parameters()) != sum(p.numel() for p in candidate.parameters()):
        raise RuntimeError("v19 changed stored parameter count relative to v18")
    for key, value in reference.state_dict().items():
        if not torch.equal(candidate.state_dict()[key], value):
            raise RuntimeError(f"v19 changed initialized parameter/buffer {key}")

    transformer = build_transformer(torch.device("cpu")).eval()
    counts = parameter_accounting(candidate, transformer)
    if abs(counts["stored_parameter_delta_fraction"]) > 0.05:
        raise RuntimeError(f"stored parameter mismatch exceeds 5%: {counts}")

    g = torch.Generator().manual_seed(CPU_DIAGNOSTIC_SEED + 1)
    tokens = torch.randint(0, candidate.cfg.vocab_size, (2, 2 * CHUNK_SIZE), generator=g)
    with torch.no_grad():
        ref_out = reference(tokens, hard=True, route_mode="hard_sparse", update_memory=False)
        got_out = candidate(tokens, hard=True, route_mode="hard_sparse", update_memory=False)
    ref_logits = ref_out["logits"]
    got_logits = got_out["logits"]
    assert isinstance(ref_logits, torch.Tensor) and isinstance(got_logits, torch.Tensor)
    if not torch.equal(ref_logits, got_logits):
        raise RuntimeError("v19 is not exactly v18-equivalent when fast-memory writes are disabled")

    return {
        "cpu_diagnostic_seed": CPU_DIAGNOSTIC_SEED,
        "version": "aera-v19-v18-memory-with-tokenwise-prior-state-reads",
        "gpu_authorized": False,
        "routing_schedule_changed": False,
        "routing_teacher_changed": False,
        "optional_stage_targets_changed": False,
        "hard_threshold_changed": False,
        "chunk_size": CHUNK_SIZE,
        "checkpoint_layout_changed": False,
        "stored_parameter_count_changed": False,
        "memory": memory_addressing_protocol(),
        "parameter_accounting": counts,
    }
