from __future__ import annotations

"""Training-only target NLL path for CHM-v3 DAEC (#1262).

This module is a systems optimization only. It preserves the frozen #1234
DAEC probability model exactly while avoiding materialization of a dense
[batch, query, vocab] copy distribution during ordinary next-token training.

The reference dense implementation remains in chm_v3_100m_daec.py and is
unchanged. No architecture parameters, retrieval equations, gate semantics,
objective, optimizer, data, or scientific protocol are defined here.
"""

from typing import Any

import torch
import torch.nn.functional as F

from .chm_v1_small_lm import LOCAL_WINDOW, _hidden
from .chm_v3_100m_daec import (
    CHMV3100MDAECLM,
    RETRIEVAL_HOPS,
    RETRIEVAL_TEMPERATURE,
    DecoderAlignedEpisodicCopy,
)

ISSUE = 1262
REFERENCE_DAEC_BLOB = "77e9ccbff4383be40e6e2865503e1c3926bbda74"
CLASSIFICATION = "CHM_V3_100M_DAEC_TARGET_NLL_OPTIMIZATION_EQUIVALENCE_PASS"


def _validate_targets(
    *,
    base_logits: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    if base_logits.ndim != targets.ndim + 1:
        raise ValueError("base_logits must have exactly one vocabulary dimension beyond targets")
    if base_logits.shape[:-1] != targets.shape:
        raise ValueError("base_logits leading dimensions must match targets")
    if targets.dtype != torch.long:
        raise ValueError("targets must be torch.long")
    vocab_size = int(base_logits.shape[-1])
    if vocab_size <= 1:
        raise ValueError("vocabulary must contain more than one token")
    if targets.numel() and (int(targets.min()) < 0 or int(targets.max()) >= vocab_size):
        raise ValueError("target token ID outside vocabulary")


def base_target_log_probability(
    base_logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Return log p_lm(y) without retaining a dense log-softmax tensor."""
    _validate_targets(base_logits=base_logits, targets=targets)
    vocab_size = int(base_logits.shape[-1])
    nll = F.cross_entropy(
        base_logits.float().reshape(-1, vocab_size),
        targets.reshape(-1),
        reduction="none",
    )
    return (-nll).reshape(targets.shape)


def target_copy_probability(
    retrieval_weights: torch.Tensor,
    memory_token_ids: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Return p_copy(y) directly from memory-position weights.

    This is exactly the target-token component of the frozen dense
    copy_distribution scatter-add, but has shape [batch, query] rather than
    [batch, query, vocab].
    """
    if retrieval_weights.ndim != 3:
        raise ValueError("retrieval_weights must be [batch, query, memory]")
    if memory_token_ids.ndim != 2:
        raise ValueError("memory_token_ids must be [batch, memory]")
    if targets.ndim != 2:
        raise ValueError("targets must be [batch, query]")
    batch, queries, memory = retrieval_weights.shape
    if memory <= 0:
        raise ValueError("DAEC copy branch requires non-empty completed prior memory")
    if memory_token_ids.shape != (batch, memory):
        raise ValueError("memory token IDs must align with retrieval memory dimension")
    if targets.shape != (batch, queries):
        raise ValueError("targets must align with retrieval query dimension")
    if memory_token_ids.dtype != torch.long or targets.dtype != torch.long:
        raise ValueError("memory_token_ids and targets must be torch.long")

    matches = memory_token_ids[:, None, :].eq(targets[:, :, None])
    return (retrieval_weights * matches.to(dtype=retrieval_weights.dtype)).sum(dim=-1)


def mix_target_log_probabilities(
    base_target_log_probs: torch.Tensor,
    copy_target_probs: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Return log((1-g)p_lm(y) + g p_copy(y)) for the observed target only."""
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

    neg_inf = torch.full_like(copy, float("-inf"))
    log_copy = torch.where(copy > 0, torch.log(copy), neg_inf)
    log_g = torch.where(
        gate_f > 0,
        torch.log(gate_f),
        torch.full_like(gate_f, float("-inf")),
    )
    log_one_minus_g = torch.where(
        gate_f < 1,
        torch.log1p(-gate_f),
        torch.full_like(gate_f, float("-inf")),
    )
    return torch.logaddexp(base + log_one_minus_g, log_copy + log_g)


def two_hop_copy_target_log_probs(
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
    """Exact target-only equivalent of frozen two_hop_copy_log_probs."""
    if float(temperature) != RETRIEVAL_TEMPERATURE:
        raise RuntimeError(
            f"#1262 retrieval temperature must remain frozen at {RETRIEVAL_TEMPERATURE}"
        )
    if RETRIEVAL_HOPS != 2:
        raise RuntimeError("#1262 DAEC requires exactly two retrieval hops")
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
    target_log_probs = mix_target_log_probabilities(
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


def _paired_chunks(tokens: torch.Tensor, targets: torch.Tensor):
    if tokens.ndim != 2 or targets.ndim != 2:
        raise ValueError("tokens and targets must be [batch, sequence]")
    if tokens.shape != targets.shape:
        raise ValueError("tokens and targets must have identical shape")
    if tokens.shape[1] <= 0 or tokens.shape[1] % LOCAL_WINDOW != 0:
        raise ValueError(f"sequence length must be a positive multiple of {LOCAL_WINDOW}")
    for start in range(0, tokens.shape[1], LOCAL_WINDOW):
        stop = start + LOCAL_WINDOW
        yield tokens[:, start:stop], targets[:, start:stop]


def daec_flat_training_session_target_log_probs(
    model: CHMV3100MDAECLM,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    *,
    temperature: float = RETRIEVAL_TEMPERATURE,
) -> torch.Tensor:
    """Training target log probabilities with frozen DAEC semantics."""
    if float(temperature) != RETRIEVAL_TEMPERATURE:
        raise RuntimeError(
            f"#1262 retrieval temperature must remain frozen at {RETRIEVAL_TEMPERATURE}"
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
            target_log_probs, _ = two_hop_copy_target_log_probs(
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

        memory_keys = current_keys if memory_keys is None else torch.cat((memory_keys, current_keys), dim=1)
        memory_token_ids = chunk if memory_token_ids is None else torch.cat((memory_token_ids, chunk), dim=1)

    return torch.cat(outputs, dim=1)


def daec_flat_training_session_nll(
    model: CHMV3100MDAECLM,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    *,
    temperature: float = RETRIEVAL_TEMPERATURE,
) -> torch.Tensor:
    """Mean next-token NLL for the exact frozen DAEC probability model."""
    target_log_probs = daec_flat_training_session_target_log_probs(
        model,
        tokens,
        targets,
        temperature=temperature,
    )
    return -target_log_probs.mean()


def optimization_manifest() -> dict[str, Any]:
    return {
        "classification": CLASSIFICATION,
        "issue": ISSUE,
        "reference_daec_blob": REFERENCE_DAEC_BLOB,
        "retrieval_hops": RETRIEVAL_HOPS,
        "retrieval_temperature": RETRIEVAL_TEMPERATURE,
        "dense_copy_distribution_allocated": False,
        "architecture_changed": False,
        "parameters_changed": False,
        "objective_changed": False,
        "gpu_authorized": False,
        "paid_compute_authorized": False,
        "replacement_stage_b_attempt_authorized": False,
        "scientific_seed_authorized": False,
        "stage_c_authorized": False,
        "stage_d_authorized": False,
    }
