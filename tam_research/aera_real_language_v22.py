from __future__ import annotations

"""Frozen AERA-v22 real-language development binding.

This module binds the controlled v22 interference-corrected dual-delta memory to
exactly the established v17/v19 real-language architecture/training schedule.  It
also carries over the conflict-free memory teaching signal that was required by
the controlled memory gate.  No GPU run is authorized by importing this module.
"""

from typing import Any

import torch
import torch.nn.functional as F

from . import aera_real_language_v12 as v12
from . import aera_real_language_v19 as v19
from .aera_hardware_core import HardwareAERAState
from .aera_hardware_core_v22 import (
    DualDeltaFastMemoryState,
    HardwareAwareAERATextLMV22,
    dual_delta_memory_protocol,
)
from .aera_real_language import parameter_accounting

CPU_DIAGNOSTIC_SEED = 8381
CHUNK_SIZE = v19.CHUNK_SIZE
DENSE_WARMUP_STEPS = v19.DENSE_WARMUP_STEPS
ROUTER_CALIBRATION_END = v19.ROUTER_CALIBRATION_END
SPARSE_CALIBRATION_EVERY = v19.SPARSE_CALIBRATION_EVERY
STAGE_DIFFICULTY_WEIGHT = v19.STAGE_DIFFICULTY_WEIGHT
STAGE_BUDGET_WEIGHT = v19.STAGE_BUDGET_WEIGHT
STAGE_POLARIZATION_WEIGHT = v19.STAGE_POLARIZATION_WEIGHT

ADDRESS_TEMPERATURE = 0.10
ADDRESS_CONTRASTIVE_WEIGHT = 1.0
PAYLOAD_TOKEN_WEIGHT = 1.0
LATENT_PAYLOAD_WEIGHT = 0.0
MAX_MEMORY_AUX_EVENTS = 1024
_SAMPLE_OFFSET_PRIME = 9973

phase_for_step = v19.phase_for_step
route_mode_for_step = v19.route_mode_for_step
router_policy_for_step = v19.router_policy_for_step
per_chunk_language_loss = v19.per_chunk_language_loss
build_transformer = v19.build_transformer


def build_aera(device: torch.device) -> HardwareAwareAERATextLMV22:
    """Use the existing ~25M real-language geometry, including memory_dim=50."""
    return HardwareAwareAERATextLMV22(v12.aera_v12_config()).to(device)


def set_optional_stage_router_trainable(
    model: HardwareAwareAERATextLMV22,
    trainable: bool,
) -> None:
    model.set_optional_stage_routers_trainable(trainable)


def _detached_stage0_event_pairs(
    model: HardwareAwareAERATextLMV22,
    tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if tokens.ndim != 2 or tokens.size(1) % CHUNK_SIZE:
        raise ValueError("tokens must be [batch,sequence] divisible by CHUNK_SIZE")
    chunks = tokens.view(tokens.size(0), -1, CHUNK_SIZE)
    pos = torch.arange(CHUNK_SIZE, device=tokens.device)
    events = model.token_emb(chunks) + model.local_pos(pos)[None, None, :, :]
    base_h = model.stages[0].norm(events).detach()
    address_tokens = chunks[:, :, :-1]
    positions = torch.arange(CHUNK_SIZE - 1, device=tokens.device)
    positions = positions.view(1, 1, -1).expand_as(address_tokens)
    identities = address_tokens * (CHUNK_SIZE - 1) + positions
    return (
        base_h[:, :, :-1],
        base_h[:, :, 1:],
        chunks[:, :, 1:],
        identities,
    )


def _stratified_indices(
    total: int,
    *,
    step: int,
    limit: int,
    device: torch.device,
) -> torch.Tensor:
    if total < 1 or limit < 1 or step < 0:
        raise ValueError("total/limit must be positive and step nonnegative")
    count = min(total, limit)
    # Deterministic label-independent coverage: evenly span the flattened event
    # list, then rotate the span each optimizer step.  No token/target value is
    # consulted when selecting examples.
    base = torch.div(
        torch.arange(count, device=device, dtype=torch.long) * total,
        count,
        rounding_mode="floor",
    )
    offset = (step * _SAMPLE_OFFSET_PRIME) % total
    return (base + offset) % total


def _multi_positive_contrastive_loss(
    q: torch.Tensor,
    k: torch.Tensor,
    identity: torch.Tensor,
    *,
    temperature: float = ADDRESS_TEMPERATURE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if q.shape != k.shape or q.ndim != 2:
        raise ValueError("q and k must match [events,dim]")
    if identity.shape != (q.size(0),):
        raise ValueError("identity must be [events]")
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    logits = (q @ k.transpose(0, 1)) / temperature
    positives = identity[:, None].eq(identity[None, :])
    if not bool(positives.any(dim=1).all()):
        raise RuntimeError("every sampled address must have a positive")
    neg_inf = torch.finfo(logits.dtype).min
    q_to_k = -(
        torch.logsumexp(logits.masked_fill(~positives, neg_inf), dim=1)
        - torch.logsumexp(logits, dim=1)
    ).mean()
    logits_t = logits.transpose(0, 1)
    k_to_q = -(
        torch.logsumexp(
            logits_t.masked_fill(~positives.transpose(0, 1), neg_inf), dim=1
        )
        - torch.logsumexp(logits_t, dim=1)
    ).mean()
    similarity = q @ k.transpose(0, 1)
    target_top1 = positives.gather(1, similarity.argmax(dim=-1, keepdim=True)).squeeze(1)
    # Descriptive separation against the best non-positive sampled address.
    best_positive = similarity.masked_fill(~positives, -torch.inf).max(dim=1).values
    has_negative = (~positives).any(dim=1)
    best_negative = similarity.masked_fill(positives, -torch.inf).max(dim=1).values
    margin = torch.where(
        has_negative,
        best_positive - best_negative,
        torch.zeros_like(best_positive),
    )
    return 0.5 * (q_to_k + k_to_q), target_top1.float().mean(), margin.mean()


def _decode_with_frozen_model_head(
    model: HardwareAwareAERATextLMV22,
    x: torch.Tensor,
) -> torch.Tensor:
    weight = model.norm.weight.detach() if model.norm.weight is not None else None
    bias = model.norm.bias.detach() if model.norm.bias is not None else None
    normalized = F.layer_norm(
        x,
        model.norm.normalized_shape,
        weight,
        bias,
        model.norm.eps,
    )
    head_weight = model.lm_head.weight.detach()
    head_bias = model.lm_head.bias.detach() if model.lm_head.bias is not None else None
    return F.linear(normalized, head_weight, head_bias)


def memory_auxiliary_terms(
    model: HardwareAwareAERATextLMV22,
    tokens: torch.Tensor,
    *,
    step: int,
    max_events: int = MAX_MEMORY_AUX_EVENTS,
) -> dict[str, torch.Tensor]:
    """Bounded conflict-free objective derived only from observed causal events."""
    address, payload, next_tokens, identities = _detached_stage0_event_pairs(model, tokens)
    flat_address = address.reshape(-1, address.size(-1))
    flat_payload = payload.reshape(-1, payload.size(-1))
    flat_next = next_tokens.reshape(-1)
    flat_ids = identities.reshape(-1)
    idx = _stratified_indices(
        flat_address.size(0),
        step=step,
        limit=max_events,
        device=tokens.device,
    )
    flat_address = flat_address.index_select(0, idx)
    flat_payload = flat_payload.index_select(0, idx)
    flat_next = flat_next.index_select(0, idx)
    flat_ids = flat_ids.index_select(0, idx)

    memory = model.stages[0].memory
    q = memory.q(flat_address)
    k = memory.k(flat_address)
    address_loss, address_top1, address_margin = _multi_positive_contrastive_loss(
        q,
        k,
        flat_ids,
        temperature=ADDRESS_TEMPERATURE,
    )

    payload_code = memory.out(torch.tanh(memory.v(flat_payload)))
    payload_logits = _decode_with_frozen_model_head(model, payload_code)
    payload_loss = F.cross_entropy(payload_logits.float(), flat_next)
    payload_accuracy = (payload_logits.argmax(dim=-1) == flat_next).float().mean()
    return {
        "memory_address_contrastive_loss": address_loss,
        "memory_payload_token_loss": payload_loss,
        "memory_payload_token_accuracy": payload_accuracy,
        "memory_address_sampled_top1": address_top1,
        "memory_address_sampled_margin": address_margin,
        "memory_aux_events": torch.tensor(float(idx.numel()), device=tokens.device),
    }


def aera_matched_loss(
    model: HardwareAwareAERATextLMV22,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    step: int,
):
    """Inherited v19 language/routing loss plus frozen conflict-free memory code."""
    inherited_total, terms, mode, phase = v19.aera_matched_loss(
        model,
        x,
        y,
        step=step,
    )
    memory_terms = memory_auxiliary_terms(model, x, step=step)
    total = (
        inherited_total
        + ADDRESS_CONTRASTIVE_WEIGHT * memory_terms["memory_address_contrastive_loss"]
        + PAYLOAD_TOKEN_WEIGHT * memory_terms["memory_payload_token_loss"]
    )
    merged = dict(terms)
    merged.update(memory_terms)
    merged["total"] = total
    return total, merged, mode, phase


def _common_initialization_audit(
    reference: torch.nn.Module,
    candidate: torch.nn.Module,
) -> dict[str, Any]:
    ref_state = reference.state_dict()
    got_state = candidate.state_dict()
    missing = sorted(set(ref_state) - set(got_state))
    if missing:
        raise RuntimeError(f"v22 lost inherited v19 state keys: {missing[:8]}")
    changed: list[str] = []
    for key, value in ref_state.items():
        if not torch.equal(got_state[key], value):
            changed.append(key)
    if changed:
        raise RuntimeError(f"v22 changed inherited initialized values: {changed[:8]}")
    added = sorted(set(got_state) - set(ref_state))
    allowed = all(".pair_write_gate." in key for key in added)
    if not allowed:
        raise RuntimeError(f"unexpected v22 checkpoint additions: {added[:8]}")
    return {"added_state_keys": added, "inherited_values_bit_exact": True}


def cpu_preflight() -> dict[str, Any]:
    torch.manual_seed(CPU_DIAGNOSTIC_SEED)
    reference = v19.build_aera(torch.device("cpu")).eval()
    torch.manual_seed(CPU_DIAGNOSTIC_SEED)
    candidate = build_aera(torch.device("cpu")).eval()
    initialization = _common_initialization_audit(reference, candidate)

    torch.manual_seed(CPU_DIAGNOSTIC_SEED)
    transformer = build_transformer(torch.device("cpu")).eval()
    counts = parameter_accounting(candidate, transformer)
    if abs(counts["stored_parameter_delta_fraction"]) > 0.05:
        raise RuntimeError(f"stored parameter mismatch exceeds 5%: {counts}")

    g = torch.Generator().manual_seed(CPU_DIAGNOSTIC_SEED + 1)
    tokens = torch.randint(
        0,
        candidate.cfg.vocab_size,
        (2, 2 * CHUNK_SIZE),
        generator=g,
    )
    with torch.no_grad():
        ref_out = reference(
            tokens,
            hard=True,
            route_mode="hard_sparse",
            update_memory=False,
        )
        got_out = candidate(
            tokens,
            hard=True,
            route_mode="hard_sparse",
            update_memory=False,
        )
    ref_logits = ref_out["logits"]
    got_logits = got_out["logits"]
    if not isinstance(ref_logits, torch.Tensor) or not isinstance(got_logits, torch.Tensor):
        raise TypeError("preflight expected tensor logits")
    torch.testing.assert_close(got_logits, ref_logits, atol=0.0, rtol=0.0)
    ref_state = ref_out.get("state")
    got_state = got_out.get("state")
    if not isinstance(ref_state, HardwareAERAState) or not isinstance(got_state, HardwareAERAState):
        raise RuntimeError("preflight missing hardware state")
    for old, new in zip(ref_state.stages, got_state.stages):
        torch.testing.assert_close(new.stream, old.stream, atol=0.0, rtol=0.0)
        torch.testing.assert_close(new.memory.matrix, old.memory.matrix, atol=0.0, rtol=0.0)
        if not isinstance(new.memory, DualDeltaFastMemoryState):
            raise RuntimeError("v22 real-language state lost inverse covariance")
        if new.memory.inverse_key_covariance.shape[-1] != candidate.cfg.memory_dim:
            raise RuntimeError("v22 inverse covariance dimension mismatch")

    # Isolated auxiliary-gradient audit.  The observed-event representations and
    # decoder are frozen by construction, while existing q/k/v/out must learn.
    candidate.train()
    candidate.zero_grad(set_to_none=True)
    aux = memory_auxiliary_terms(candidate, tokens[:, :CHUNK_SIZE], step=0, max_events=64)
    aux_total = aux["memory_address_contrastive_loss"] + aux["memory_payload_token_loss"]
    if not bool(torch.isfinite(aux_total)):
        raise RuntimeError("v22 memory auxiliary is non-finite")
    aux_total.backward()
    memory = candidate.stages[0].memory
    grad_norms: dict[str, float] = {}
    for name, param in (
        ("q", memory.q.weight),
        ("k", memory.k.weight),
        ("v", memory.v.weight),
        ("out", memory.out.weight),
    ):
        if param.grad is None or not bool(torch.isfinite(param.grad).all()):
            raise RuntimeError(f"missing/nonfinite isolated memory auxiliary gradient for {name}")
        norm = float(torch.linalg.vector_norm(param.grad.float()))
        if norm <= 0.0:
            raise RuntimeError(f"zero isolated memory auxiliary gradient for {name}")
        grad_norms[name] = norm
    tied_embedding_grad = candidate.token_emb.weight.grad
    if tied_embedding_grad is not None and float(tied_embedding_grad.abs().max()) != 0.0:
        raise RuntimeError("isolated memory auxiliary leaked gradient into frozen decoder/embedding")
    if candidate.stages[0].norm.weight.grad is not None and float(candidate.stages[0].norm.weight.grad.abs().max()) != 0.0:
        raise RuntimeError("isolated memory auxiliary leaked gradient into event backbone")
    candidate.zero_grad(set_to_none=True)

    return {
        "cpu_diagnostic_seed": CPU_DIAGNOSTIC_SEED,
        "version": "aera-v22-real-language-dual-delta-memory-development",
        "gpu_authorized": False,
        "routing_schedule_changed": False,
        "routing_teacher_changed": False,
        "optional_stage_targets_changed": False,
        "hard_threshold_changed": False,
        "chunk_size": CHUNK_SIZE,
        "memory_dim": candidate.cfg.memory_dim,
        "memory_dim_changed_from_prior_real_language": False,
        "controlled_diagnostic_memory_dim_was_16": True,
        "real_language_memory_dim_preserved_at_50": candidate.cfg.memory_dim == 50,
        "memory": dual_delta_memory_protocol(),
        "memory_auxiliary": {
            "address_temperature": ADDRESS_TEMPERATURE,
            "address_weight": ADDRESS_CONTRASTIVE_WEIGHT,
            "payload_token_weight": PAYLOAD_TOKEN_WEIGHT,
            "latent_payload_weight": LATENT_PAYLOAD_WEIGHT,
            "max_events_per_step": MAX_MEMORY_AUX_EVENTS,
            "sampling": "deterministic stratified flattened adjacent pairs plus step-dependent cyclic offset",
            "address_identity": "observed GPT-2 token id + local chunk position",
            "event_representation_detached": True,
            "decoder_weights_detached": True,
            "gradient_norms": grad_norms,
        },
        "initialization": initialization,
        "parameter_accounting": counts,
        "memory_off_logits_and_stream_bit_exact_v19": True,
    }


def frozen_protocol_summary() -> dict[str, Any]:
    return {
        "development_seed": 8391,
        "development_only": True,
        "memory_dim": 50,
        "thresholds_inherited_from_v18": True,
        "memory_auxiliary_weights": [ADDRESS_CONTRASTIVE_WEIGHT, PAYLOAD_TOKEN_WEIGHT],
        "memory_auxiliary_max_events": MAX_MEMORY_AUX_EVENTS,
        "gpu_authorized_by_module": False,
    }
