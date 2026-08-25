from __future__ import annotations

"""Systems-only helpers for the preregistered AERA-v25 L4 gate (#362).

No corpus is opened, no optimizer step is taken, and no checkpoint is written by
this module.  It binds the exact merged v25 FICEM core to the established ~25M
real-language geometry and exposes the frozen random-token training-shape loss
used only for runtime projection.
"""

from typing import Any

import torch
import torch.nn.functional as F

from . import aera_real_language_v12 as v12
from . import aera_real_language_v19 as v19
from .aera_hardware_core_v23 import sparse_write_budget
from .aera_hardware_core_v24 import episodic_state_bytes_per_session
from .aera_hardware_core_v25 import (
    FactorizedIdentityContextEpisodicMemoryStage,
    HardwareAwareAERATextLMV25,
    causal_identity_context,
    factorized_identity_context_protocol,
)
from .aera_real_language import GRAD_ACCUM, parameter_accounting

CHUNK_SIZE = v12.CHUNK_SIZE
MAX_PAYLOAD_EVENTS_PER_OPTIMIZER_STEP = 1024
if MAX_PAYLOAD_EVENTS_PER_OPTIMIZER_STEP % GRAD_ACCUM:
    raise RuntimeError("v25 systems payload budget must divide GRAD_ACCUM")
MAX_PAYLOAD_EVENTS_PER_MICROBATCH = MAX_PAYLOAD_EVENTS_PER_OPTIMIZER_STEP // GRAD_ACCUM
_SAMPLE_OFFSET_PRIME = 9973

phase_for_step = v19.phase_for_step
route_mode_for_step = v19.route_mode_for_step
router_policy_for_step = v19.router_policy_for_step
build_transformer = v19.build_transformer


def build_aera(device: torch.device) -> HardwareAwareAERATextLMV25:
    """Exact v25 core at the inherited chunk256/memory_dim50 ~25M geometry."""
    return HardwareAwareAERATextLMV25(v12.aera_v12_config()).to(device)


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
    base = torch.div(
        torch.arange(count, device=device, dtype=torch.long) * total,
        count,
        rounding_mode="floor",
    )
    offset = (step * _SAMPLE_OFFSET_PRIME) % total
    return (base + offset) % total


def _decode_with_frozen_head(
    model: HardwareAwareAERATextLMV25,
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


def payload_teaching_terms(
    model: HardwareAwareAERATextLMV25,
    tokens: torch.Tensor,
    *,
    step: int,
    max_events: int = MAX_PAYLOAD_EVENTS_PER_MICROBATCH,
) -> dict[str, torch.Tensor]:
    """Frozen v25 decoder-aligned payload CE; address auxiliary remains zero."""
    if tokens.ndim != 2 or tokens.size(1) % CHUNK_SIZE:
        raise ValueError("tokens must be [batch,sequence] divisible by chunk256")
    if max_events > MAX_PAYLOAD_EVENTS_PER_MICROBATCH:
        raise ValueError(
            "v25 payload auxiliary exceeds frozen per-microbatch cap: "
            f"{max_events} > {MAX_PAYLOAD_EVENTS_PER_MICROBATCH}"
        )

    chunks = tokens.view(tokens.size(0), -1, CHUNK_SIZE)
    pos = torch.arange(CHUNK_SIZE, device=tokens.device)
    events = model.token_emb(chunks) + model.local_pos(pos)[None, None, :, :]
    stage = model.stages[0]
    if not isinstance(stage, FactorizedIdentityContextEpisodicMemoryStage):
        raise TypeError("v25 systems helper expected FICEM stage0")
    base_h = stage.norm(events).detach()
    flat = base_h.reshape(-1, CHUNK_SIZE, base_h.size(-1))
    _, _, contextual = causal_identity_context(flat)
    payload_source = contextual[:, 1:].reshape(-1, contextual.size(-1))
    next_tokens = chunks[:, :, 1:].reshape(-1)
    idx = _stratified_indices(
        payload_source.size(0),
        step=step,
        limit=max_events,
        device=tokens.device,
    )
    payload = payload_source.index_select(0, idx)
    targets = next_tokens.index_select(0, idx)

    payload_code = stage.memory.out(torch.tanh(stage.memory.v(payload)))
    payload_logits = _decode_with_frozen_head(model, payload_code)
    loss = F.cross_entropy(payload_logits.float(), targets)
    accuracy = (payload_logits.argmax(dim=-1) == targets).float().mean()
    return {
        "memory_payload_token_loss": loss,
        "memory_payload_token_accuracy": accuracy,
        "sampled_payload_events": torch.tensor(
            float(idx.numel()), device=loss.device, dtype=loss.dtype
        ),
    }


def systems_matched_loss(
    model: HardwareAwareAERATextLMV25,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    step: int,
):
    """Inherited real-language task/routing loss + v25 payload CE weight 1.

    This exists only to time the frozen training shape.  It performs no optimizer
    step and supplies no address labels or address auxiliary.
    """
    inherited_total, terms, mode, phase = v19.aera_matched_loss(model, x, y, step=step)
    payload = payload_teaching_terms(model, x, step=step)
    total = inherited_total + payload["memory_payload_token_loss"]
    merged = dict(terms)
    merged.update(payload)
    merged["memory_address_auxiliary_weight"] = torch.zeros_like(total)
    merged["total"] = total
    return total, merged, mode, phase


def cpu_preflight() -> dict[str, Any]:
    torch.manual_seed(12525)
    model = build_aera(torch.device("cpu"))
    transformer = build_transformer(torch.device("cpu"))
    counts = parameter_accounting(model, transformer)
    if abs(counts["stored_parameter_delta_fraction"]) > 0.05:
        raise RuntimeError(f"stored parameter mismatch exceeds 5%: {counts}")
    if model.cfg.chunk_size != 256 or model.cfg.memory_dim != 50 or len(model.stages) != 4:
        raise RuntimeError("v25 systems production geometry drifted")
    state_bytes = episodic_state_bytes_per_session(n_stages=4, memory_dim=50)
    if state_bytes != 77_760:
        raise RuntimeError(f"v25 FICEM state bytes changed: {state_bytes}")
    if sparse_write_budget(255) != 16:
        raise RuntimeError("v25 production write budget is no longer 16/255")

    stage = model.stages[0]
    if not isinstance(stage, FactorizedIdentityContextEpisodicMemoryStage):
        raise RuntimeError("v25 production stage type mismatch")
    stage.memory.set_differentiable_pretraining(True)
    g = torch.Generator().manual_seed(12526)
    events = torch.randn(2, 256, model.cfg.d_model, generator=g, requires_grad=True)
    out, state, _ = stage.forward_chunk(
        events,
        None,
        hard=False,
        update_memory=True,
    )
    base_h = stage.norm(events)
    identity, context, _ = causal_identity_context(base_h)
    recall = stage.memory.read(identity, context, state.memory)
    loss = (
        out.float().square().mean()
        + recall.float().square().mean()
        + 0.01 * state.memory.values.float().square().mean()
    )
    loss.backward()
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("v25 production differentiable smoke is nonfinite")
    if stage.last_candidate_count != 255 or stage.last_selected_count != 16:
        raise RuntimeError("v25 production differentiable smoke did not execute 16/255")
    if stage.last_vectorized_update_calls != 1:
        raise RuntimeError("v25 production differentiable smoke did not use one block update")
    grads = {
        "identity_proj": stage.memory.identity_proj.weight.grad,
        "context_proj": stage.memory.context_proj.weight.grad,
        "v": stage.memory.v.weight.grad,
        "out": stage.memory.out.weight.grad,
        "pair_write_gate": stage.pair_write_gate.weight.grad,
    }
    grad_l1: dict[str, float] = {}
    for name, grad in grads.items():
        if grad is None or not bool(torch.isfinite(grad).all()) or float(grad.abs().sum()) <= 0.0:
            raise RuntimeError(f"v25 production smoke missing/nonfinite gradient: {name}")
        grad_l1[name] = float(grad.abs().sum())
    stage.memory.set_differentiable_pretraining(False)
    stage.zero_grad(set_to_none=True)

    snapshot = {name: p.detach().clone() for name, p in model.named_parameters()}
    with torch.no_grad():
        deployment_events = torch.randn(2, 256, model.cfg.d_model, generator=g)
        stage.forward_chunk(
            deployment_events,
            None,
            hard=True,
            update_memory=True,
        )
    for name, p in model.named_parameters():
        if not torch.equal(snapshot[name], p):
            raise RuntimeError(f"deployment local memory mutated base parameter: {name}")

    return {
        "version": "aera-v25-ficem-production-systems-preflight",
        "gpu_authorized": False,
        "scientific_training_performed": False,
        "optimizer_steps_performed": 0,
        "checkpoint_written": False,
        "corpus_reader_used": False,
        "chunk_size": model.cfg.chunk_size,
        "memory_dim": model.cfg.memory_dim,
        "n_stages": len(model.stages),
        "production_candidates": stage.last_candidate_count,
        "production_selected_writes": stage.last_selected_count,
        "vectorized_update_calls": stage.last_vectorized_update_calls,
        "state_bytes": state_bytes,
        "payload_events_per_microbatch": MAX_PAYLOAD_EVENTS_PER_MICROBATCH,
        "payload_events_per_optimizer_step": MAX_PAYLOAD_EVENTS_PER_OPTIMIZER_STEP,
        "parameter_accounting": counts,
        "gradient_l1": grad_l1,
        "protocol": factorized_identity_context_protocol(),
    }
