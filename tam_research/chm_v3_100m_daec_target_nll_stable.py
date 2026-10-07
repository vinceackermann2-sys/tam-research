from __future__ import annotations

"""Gradient-stable target-only DAEC training path (#1288).

This additive module preserves the frozen #1234 DAEC probability model and the
#1262 target-only objective exactly. It changes only how logarithms are
evaluated at structural zero probability so autograd never evaluates log(0).

The frozen DAEC architecture and the frozen #1262 helper remain unchanged.
This module contains no optimizer, GPU/Modal launcher, scientific seed, or
Stage-B/C/D authority.
"""

from typing import Any

import torch
import torch.nn.functional as F

from .chm_v1_small_lm import _hidden
from .chm_v3_100m_daec import (
    CHMV3100MDAECLM,
    RETRIEVAL_HOPS,
    RETRIEVAL_TEMPERATURE,
    DecoderAlignedEpisodicCopy,
)
from .chm_v3_100m_daec_target_nll import (
    _paired_chunks,
    _validate_targets,
    base_target_log_probability,
    target_copy_probability,
)

ISSUE = 1288
REFERENCE_DAEC_BLOB = "77e9ccbff4383be40e6e2865503e1c3926bbda74"
REFERENCE_TARGET_NLL_BLOB = "0b0a68f47186f154a48d1d75e3fe3b236188d69d"
CLASSIFICATION = "CHM_V3_100M_DAEC_TARGET_NLL_ZERO_MASS_GRADIENT_STABILIZATION_PASS"


def _log_positive_or_neg_inf(value: torch.Tensor) -> torch.Tensor:
    """log(value) for positive entries and -inf at structural zero, safely.

    The forward result is identical to:
        where(value > 0, log(value), -inf)
    but log is evaluated only on a strictly positive surrogate. The false
    branch therefore has zero derivative instead of an undefined log(0)
    derivative that can become NaN under autograd.
    """
    if bool((value < 0).any()):
        raise ValueError("probability values must be non-negative")
    positive = value > 0
    safe_value = torch.where(positive, value, torch.ones_like(value))
    logged = torch.log(safe_value)
    return torch.where(
        positive,
        logged,
        torch.full_like(logged, float("-inf")),
    )


def mix_target_log_probabilities_stable(
    base_target_log_probs: torch.Tensor,
    copy_target_probs: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Exact DAEC target mixture with finite gradients at structural zero."""
    if base_target_log_probs.shape != copy_target_probs.shape:
        raise ValueError("base and copy target tensors must have identical shape")
    if gate.shape == (*base_target_log_probs.shape, 1):
        gate = gate.squeeze(-1)
    if gate.shape != base_target_log_probs.shape:
        raise ValueError("gate must align to target-token probabilities")
    if bool((copy_target_probs < 0).any()):
        raise ValueError("copy target probabilities must be non-negative")

    base = base_target_log_probs.float()
    copy = copy_target_probs.float()
    gate_f = gate.float()

    if bool((gate_f < 0).any()) or bool((gate_f > 1).any()):
        raise ValueError("gate probabilities must lie in [0,1]")

    log_copy = _log_positive_or_neg_inf(copy)
    log_g = _log_positive_or_neg_inf(gate_f)
    log_one_minus_g = _log_positive_or_neg_inf(1.0 - gate_f)
    return torch.logaddexp(base + log_one_minus_g, log_copy + log_g)


def two_hop_copy_target_log_probs_stable(
    core: DecoderAlignedEpisodicCopy,
    *,
    first_query: torch.Tensor,
    hidden: torch.Tensor,
    memory_keys: torch.Tensor,
    memory_token_ids: torch.Tensor,
    token_embedding_weight: torch.Tensor,
    base_logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = RETRIEVAL_TEMPERATURE,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Target-only DAEC read with exact forward semantics and safe zero grads."""
    if float(temperature) != RETRIEVAL_TEMPERATURE:
        raise RuntimeError(
            f"#1288 retrieval temperature must remain frozen at {RETRIEVAL_TEMPERATURE}"
        )
    if RETRIEVAL_HOPS != 2:
        raise RuntimeError("#1288 DAEC requires exactly two retrieval hops")
    if first_query.ndim != 3 or memory_keys.ndim != 3:
        raise ValueError("first_query and memory_keys must be rank-3")
    if first_query.shape[0] != memory_keys.shape[0]:
        raise ValueError("query/key batch mismatch")
    if first_query.shape[-1] != memory_keys.shape[-1]:
        raise ValueError("query/key address dimension mismatch")
    if memory_keys.shape[1] <= 0:
        raise ValueError("DAEC requires non-empty completed prior memory")
    if memory_token_ids.shape != memory_keys.shape[:2]:
        raise ValueError("memory token IDs must align with memory keys")
    if token_embedding_weight.ndim != 2:
        raise ValueError("token_embedding_weight must be [vocab,d_model]")
    if token_embedding_weight.shape[1] != core.d_model:
        raise ValueError("token embedding dimension mismatch")
    if hidden.shape[:-1] != first_query.shape[:-1]:
        raise ValueError("hidden/query leading dimensions mismatch")
    if targets.shape != hidden.shape[:-1]:
        raise ValueError("targets must align with hidden/query positions")
    _validate_targets(base_logits=base_logits, targets=targets)
    if base_logits.shape[-1] != token_embedding_weight.shape[0]:
        raise ValueError("base-logit vocabulary mismatch")

    scores1 = torch.einsum("btd,bsd->bts", first_query, memory_keys) / RETRIEVAL_TEMPERATURE
    weights1 = torch.softmax(scores1.float(), dim=-1).to(hidden.dtype)

    memory_token_embeddings = F.embedding(memory_token_ids, token_embedding_weight)
    hop_embedding = torch.matmul(weights1, memory_token_embeddings)
    second_query = core.second_query(first_query, hop_embedding)

    scores2 = torch.einsum("btd,bsd->bts", second_query, memory_keys) / RETRIEVAL_TEMPERATURE
    weights2 = torch.softmax(scores2.float(), dim=-1).to(hidden.dtype)

    copy_target_probs = target_copy_probability(weights2, memory_token_ids, targets)
    gate = core.gate(hidden)
    base_target_log_probs = base_target_log_probability(base_logits, targets)
    target_log_probs = mix_target_log_probabilities_stable(
        base_target_log_probs,
        copy_target_probs,
        gate,
    )
    return target_log_probs, {
        "first_hop_weights": weights1,
        "hop_embedding": hop_embedding,
        "second_query": second_query,
        "second_hop_weights": weights2,
        "copy_target_probs": copy_target_probs,
        "copy_gate": gate,
        "base_target_log_probs": base_target_log_probs,
    }


def daec_flat_training_session_target_log_probs_stable(
    model: CHMV3100MDAECLM,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    *,
    temperature: float = RETRIEVAL_TEMPERATURE,
) -> torch.Tensor:
    """Frozen DAEC target log probabilities with safe structural-zero backward."""
    if float(temperature) != RETRIEVAL_TEMPERATURE:
        raise RuntimeError(
            f"#1288 retrieval temperature must remain frozen at {RETRIEVAL_TEMPERATURE}"
        )

    memory_keys: torch.Tensor | None = None
    memory_token_ids: torch.Tensor | None = None
    outputs: list[torch.Tensor] = []

    for chunk, target_chunk in _paired_chunks(tokens, targets):
        hidden = _hidden(model.backbone, chunk)
        base_logits = model.backbone.lm_head(hidden)
        current_keys = model.key_for(hidden)

        if memory_keys is None:
            outputs.append(base_target_log_probability(base_logits, target_chunk))
        else:
            assert memory_token_ids is not None
            first_query = model.query_for(hidden)
            target_log_probs, _ = two_hop_copy_target_log_probs_stable(
                model.daec,
                first_query=first_query,
                hidden=hidden,
                memory_keys=memory_keys,
                memory_token_ids=memory_token_ids,
                token_embedding_weight=model.backbone.token_emb.weight,
                base_logits=base_logits,
                targets=target_chunk,
                temperature=temperature,
            )
            outputs.append(target_log_probs)

        memory_keys = (
            current_keys
            if memory_keys is None
            else torch.cat((memory_keys, current_keys), dim=1)
        )
        memory_token_ids = (
            chunk
            if memory_token_ids is None
            else torch.cat((memory_token_ids, chunk), dim=1)
        )

    return torch.cat(outputs, dim=1)


def daec_flat_training_session_nll_stable(
    model: CHMV3100MDAECLM,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    *,
    temperature: float = RETRIEVAL_TEMPERATURE,
) -> torch.Tensor:
    target_log_probs = daec_flat_training_session_target_log_probs_stable(
        model,
        tokens,
        targets,
        temperature=temperature,
    )
    return -target_log_probs.mean()


def stabilization_manifest() -> dict[str, Any]:
    return {
        "classification": CLASSIFICATION,
        "issue": ISSUE,
        "reference_daec_blob": REFERENCE_DAEC_BLOB,
        "reference_target_nll_blob": REFERENCE_TARGET_NLL_BLOB,
        "retrieval_hops": RETRIEVAL_HOPS,
        "retrieval_temperature": RETRIEVAL_TEMPERATURE,
        "forward_probability_model_changed": False,
        "architecture_changed": False,
        "parameters_changed": False,
        "objective_changed": False,
        "dense_copy_distribution_allocated": False,
        "structural_zero_log_domain_guard": True,
        "gpu_authorized": False,
        "paid_compute_authorized": False,
        "replacement_stage_b_attempt_authorized": False,
        "scientific_seed_authorized": False,
        "stage_c_authorized": False,
        "stage_d_authorized": False,
    }
