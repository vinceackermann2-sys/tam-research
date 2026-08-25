from __future__ import annotations

"""Frozen AERA-v25 FICEM real-language development binding (#366).

This module does not authorize a GPU run. It binds the exact merged v25 core to
all inherited real-language routing/training semantics and adds only the
preregistered decoder-aligned payload-token teaching term used by the v25
systems gate. Address auxiliary weight remains exactly zero.
"""

from typing import Any

import torch

from . import aera_real_language_v19 as v19
from . import aera_v25_systems as systems
from .aera_hardware_core_v25 import HardwareAwareAERATextLMV25
from .aera_real_language import GRAD_ACCUM

CPU_DIAGNOSTIC_SEED = 12_571
CHUNK_SIZE = systems.CHUNK_SIZE
DENSE_WARMUP_STEPS = v19.DENSE_WARMUP_STEPS
ROUTER_CALIBRATION_END = v19.ROUTER_CALIBRATION_END
SPARSE_CALIBRATION_EVERY = v19.SPARSE_CALIBRATION_EVERY
STAGE_DIFFICULTY_WEIGHT = v19.STAGE_DIFFICULTY_WEIGHT
STAGE_BUDGET_WEIGHT = v19.STAGE_BUDGET_WEIGHT
STAGE_POLARIZATION_WEIGHT = v19.STAGE_POLARIZATION_WEIGHT

PAYLOAD_TOKEN_WEIGHT = 1.0
ADDRESS_AUXILIARY_WEIGHT = 0.0
MAX_PAYLOAD_EVENTS_PER_MICROBATCH = systems.MAX_PAYLOAD_EVENTS_PER_MICROBATCH
MAX_PAYLOAD_EVENTS_PER_OPTIMIZER_STEP = systems.MAX_PAYLOAD_EVENTS_PER_OPTIMIZER_STEP

phase_for_step = v19.phase_for_step
route_mode_for_step = v19.route_mode_for_step
router_policy_for_step = v19.router_policy_for_step
per_chunk_language_loss = v19.per_chunk_language_loss
build_transformer = v19.build_transformer
build_aera = systems.build_aera
payload_teaching_terms = systems.payload_teaching_terms


def set_optional_stage_router_trainable(
    model: HardwareAwareAERATextLMV25,
    trainable: bool,
) -> None:
    model.set_optional_stage_routers_trainable(trainable)


def aera_matched_loss(
    model: HardwareAwareAERATextLMV25,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    step: int,
):
    """Inherited v19/v18 task+routing+stream loss plus payload CE weight 1."""
    inherited_total, terms, mode, phase = v19.aera_matched_loss(
        model,
        x,
        y,
        step=step,
    )
    payload = payload_teaching_terms(
        model,
        x,
        step=step,
        max_events=MAX_PAYLOAD_EVENTS_PER_MICROBATCH,
    )
    total = inherited_total + PAYLOAD_TOKEN_WEIGHT * payload["memory_payload_token_loss"]
    merged = dict(terms)
    merged.update(payload)
    merged["memory_address_auxiliary_weight"] = torch.zeros_like(total)
    merged["memory_payload_token_weight"] = torch.ones_like(total)
    merged["total"] = total
    return total, merged, mode, phase


def cpu_preflight() -> dict[str, Any]:
    inherited = systems.cpu_preflight()
    if MAX_PAYLOAD_EVENTS_PER_MICROBATCH != 256:
        raise RuntimeError("v25 real-language payload cap changed from 256/microbatch")
    if MAX_PAYLOAD_EVENTS_PER_OPTIMIZER_STEP != 1024 or GRAD_ACCUM != 4:
        raise RuntimeError("v25 real-language optimizer-step payload budget changed")
    if PAYLOAD_TOKEN_WEIGHT != 1.0 or ADDRESS_AUXILIARY_WEIGHT != 0.0:
        raise RuntimeError("v25 real-language memory teaching weights changed")
    return {
        **inherited,
        "version": "aera-v25-ficem-real-language-development-binding",
        "research_issue": 366,
        "gpu_authorized": False,
        "scientific_training_performed": False,
        "payload_token_weight": PAYLOAD_TOKEN_WEIGHT,
        "address_auxiliary_weight": ADDRESS_AUXILIARY_WEIGHT,
        "payload_events_per_microbatch": MAX_PAYLOAD_EVENTS_PER_MICROBATCH,
        "payload_events_per_optimizer_step": MAX_PAYLOAD_EVENTS_PER_OPTIMIZER_STEP,
        "routing_changed_from_v19": False,
        "stream_changed_from_v19": False,
        "architecture_changed_from_merged_v25": False,
        "100m_authorized": False,
    }
