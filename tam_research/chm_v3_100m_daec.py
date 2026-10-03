from __future__ import annotations

"""CHM-v3 ~100M Decoder-Aligned Episodic Copy (DAEC), Stage A (#1234).

Stage-A implementation only. DAEC replaces opaque hidden-state memory payload
integration with a memory-causal copy distribution in the tied decoder
vocabulary space. This module contains no optimizer, corpus loader, training
loop, Modal/GPU launcher, scientific seed, or Stage-B/C/D execution authority.
"""

from dataclasses import asdict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .chm_v1_100m_scale import (
    ADDRESS_DIM,
    D_MODEL,
    EXPECTED_EIEM_PARAMETERS,
    EXPECTED_LOCAL_PARAMETERS,
    PARAMETER_MISMATCH_LIMIT,
    CHMV1100MEIEMLM,
    CHMV1100MLocalLM,
    transformer_100m_config,
)
from .chm_v1_small_lm import LOCAL_WINDOW, _hidden
from .models import parameter_count

ISSUE = 1234
IMPLEMENTATION_BASE_MAIN_SHA = "7402157959c5b2a9a7f048dca3bed34c18c190d3"
IMPLEMENTATION_BASE_TREE_SHA = "d2186bd187fff42d7f401d0106f6c1f83515da9c"
ENGINEERING_SEED = 1_234_001

RETRIEVAL_HOPS = 2
RETRIEVAL_TEMPERATURE = 0.10
COPY_GATE_INIT = -4.0

CONSUMED_OR_RESERVED_SCIENTIFIC_SEEDS = (
    977_001,
    2_011_121,
    2_011_371,
    2_011_431,
    2_011_761,
)

DAEC_HOP_UPDATE_PARAMETERS = D_MODEL * ADDRESS_DIM
DAEC_COPY_GATE_PARAMETERS = D_MODEL + 1
DAEC_ADDED_PARAMETERS = DAEC_HOP_UPDATE_PARAMETERS + DAEC_COPY_GATE_PARAMETERS
EXPECTED_DAEC_PARAMETERS = EXPECTED_EIEM_PARAMETERS + DAEC_ADDED_PARAMETERS
EXPECTED_DAEC_MINUS_LOCAL = EXPECTED_DAEC_PARAMETERS - EXPECTED_LOCAL_PARAMETERS
EXPECTED_DAEC_DELTA_FRACTION = EXPECTED_DAEC_MINUS_LOCAL / EXPECTED_LOCAL_PARAMETERS

STAGE_A_CLASSIFICATION = "CHM_V3_100M_DAEC_STAGE_A_IMPLEMENTATION_PASS"


class DecoderAlignedEpisodicCopy(nn.Module):
    """Parameter-light decision-space memory readout frozen by #1234.

    The first retrieval hop produces an expected tied-token embedding used only
    to update the second-hop address. The second-hop weights are scattered
    directly into vocabulary probability space. No retrieved hidden-state
    residual is injected into the decoder stream.
    """

    def __init__(
        self,
        *,
        d_model: int,
        address_dim: int,
        gate_init: float = COPY_GATE_INIT,
    ) -> None:
        super().__init__()
        d_model = int(d_model)
        address_dim = int(address_dim)
        if d_model <= 0 or address_dim <= 0:
            raise ValueError("d_model and address_dim must be positive")
        if float(gate_init) != COPY_GATE_INIT:
            raise RuntimeError(f"#1234 copy gate init is frozen at {COPY_GATE_INIT}")

        self.d_model = d_model
        self.address_dim = address_dim
        self.hidden_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.hop_embedding_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.hop_update = nn.Linear(d_model, address_dim, bias=False)
        self.copy_gate = nn.Linear(d_model, 1, bias=True)

        nn.init.zeros_(self.hop_update.weight)
        nn.init.zeros_(self.copy_gate.weight)
        nn.init.constant_(self.copy_gate.bias, COPY_GATE_INIT)

    def second_query(
        self,
        first_query: torch.Tensor,
        hop_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if first_query.shape[:-1] != hop_embedding.shape[:-1]:
            raise ValueError("first_query and hop_embedding leading dimensions must match")
        if first_query.shape[-1] != self.address_dim:
            raise ValueError(
                f"first_query last dimension must be {self.address_dim}, got {first_query.shape[-1]}"
            )
        if hop_embedding.shape[-1] != self.d_model:
            raise ValueError(
                f"hop_embedding last dimension must be {self.d_model}, got {hop_embedding.shape[-1]}"
            )
        update = self.hop_update(self.hop_embedding_norm(hop_embedding))
        return F.normalize(first_query + update, dim=-1)

    def gate(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.shape[-1] != self.d_model:
            raise ValueError(
                f"hidden last dimension must be {self.d_model}, got {hidden.shape[-1]}"
            )
        return torch.sigmoid(self.copy_gate(self.hidden_norm(hidden)))


class CHMV3100MDAECLM(CHMV1100MEIEMLM):
    """CHM-v1 addressing + decoder-aligned episodic token-copy payload."""

    def __init__(self, *, address_dim: int = ADDRESS_DIM) -> None:
        if int(address_dim) != ADDRESS_DIM:
            raise RuntimeError(f"#1234 address_dim is frozen at {ADDRESS_DIM}")
        super().__init__(address_dim=address_dim)
        self.daec = DecoderAlignedEpisodicCopy(
            d_model=D_MODEL,
            address_dim=ADDRESS_DIM,
            gate_init=COPY_GATE_INIT,
        )

    def _integrate(self, hidden: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        del hidden, memory
        raise RuntimeError(
            "#1234 DAEC forbids hidden-state memory residual integration; "
            "memory may affect output only through decoder-aligned episodic copy"
        )

    def second_query(
        self,
        first_query: torch.Tensor,
        hop_embedding: torch.Tensor,
    ) -> torch.Tensor:
        return self.daec.second_query(first_query, hop_embedding)

    def copy_gate_for(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.daec.gate(hidden)


def validate_engineering_seed(seed: int) -> int:
    value = int(seed)
    if value in CONSUMED_OR_RESERVED_SCIENTIFIC_SEEDS:
        raise RuntimeError("#1234 refuses all historical/reserved CHM scientific seeds")
    if value != ENGINEERING_SEED:
        raise RuntimeError(f"#1234 Stage A accepts only engineering seed {ENGINEERING_SEED}")
    return value


def copy_distribution(
    retrieval_weights: torch.Tensor,
    memory_token_ids: torch.Tensor,
    *,
    vocab_size: int,
) -> torch.Tensor:
    """Scatter normalized memory-position mass into vocabulary space."""
    if retrieval_weights.ndim != 3:
        raise ValueError("retrieval_weights must be [batch, query, memory]")
    if memory_token_ids.ndim != 2:
        raise ValueError("memory_token_ids must be [batch, memory]")
    batch, queries, memory = retrieval_weights.shape
    if memory <= 0:
        raise ValueError("DAEC copy branch requires non-empty completed prior memory")
    if memory_token_ids.shape != (batch, memory):
        raise ValueError(
            f"memory token shape mismatch: {tuple(memory_token_ids.shape)} != {(batch, memory)}"
        )
    vocab_size = int(vocab_size)
    if vocab_size <= 1:
        raise ValueError("vocab_size must exceed one")
    if memory_token_ids.dtype != torch.long:
        raise ValueError("memory_token_ids must be torch.long")
    if int(memory_token_ids.min()) < 0 or int(memory_token_ids.max()) >= vocab_size:
        raise ValueError("memory token ID outside vocabulary")

    output = retrieval_weights.new_zeros((batch, queries, vocab_size))
    scatter_index = memory_token_ids[:, None, :].expand(batch, queries, memory)
    output.scatter_add_(dim=-1, index=scatter_index, src=retrieval_weights)
    return output


def mix_lm_and_copy_log_probs(
    base_logits: torch.Tensor,
    copy_probs: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Return log p where p=(1-g)p_lm + g p_copy."""
    if base_logits.shape != copy_probs.shape:
        raise ValueError("base_logits and copy_probs must have identical shape")
    if gate.shape != (*base_logits.shape[:-1], 1):
        raise ValueError("gate must be [...,1] aligned to logits")
    if bool((copy_probs < 0).any()):
        raise ValueError("copy probabilities must be non-negative")

    base_log_probs = F.log_softmax(base_logits.float(), dim=-1)
    copy = copy_probs.float()
    gate_f = gate.float()

    neg_inf = torch.full_like(copy, float("-inf"))
    log_copy = torch.where(copy > 0, torch.log(copy), neg_inf)

    gate_full = gate_f.expand_as(base_log_probs)
    log_g = torch.where(
        gate_full > 0,
        torch.log(gate_full),
        torch.full_like(gate_full, float("-inf")),
    )
    log_one_minus_g = torch.where(
        gate_full < 1,
        torch.log1p(-gate_full),
        torch.full_like(gate_full, float("-inf")),
    )
    return torch.logaddexp(base_log_probs + log_one_minus_g, log_copy + log_g)


def no_memory_log_probs(base_logits: torch.Tensor) -> torch.Tensor:
    """Exact probability-equivalent LOCAL behavior when prior memory is absent."""
    return F.log_softmax(base_logits.float(), dim=-1)


def two_hop_copy_log_probs(
    core: DecoderAlignedEpisodicCopy,
    *,
    first_query: torch.Tensor,
    hidden: torch.Tensor,
    memory_keys: torch.Tensor,
    memory_token_ids: torch.Tensor,
    token_embedding_weight: torch.Tensor,
    base_logits: torch.Tensor,
    temperature: float = RETRIEVAL_TEMPERATURE,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Differentiable two-hop DAEC read with no hidden-residual payload path."""
    if float(temperature) != RETRIEVAL_TEMPERATURE:
        raise RuntimeError(
            f"#1234 retrieval temperature is frozen at {RETRIEVAL_TEMPERATURE}"
        )
    if RETRIEVAL_HOPS != 2:
        raise RuntimeError("#1234 DAEC requires exactly two retrieval hops")
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
    if base_logits.shape[:-1] != hidden.shape[:-1]:
        raise ValueError("base_logits/hidden leading dimensions mismatch")
    if base_logits.shape[-1] != token_embedding_weight.shape[0]:
        raise ValueError("base-logit vocabulary mismatch")

    scores1 = torch.einsum("btd,bsd->bts", first_query, memory_keys) / RETRIEVAL_TEMPERATURE
    weights1 = torch.softmax(scores1.float(), dim=-1).to(hidden.dtype)

    memory_token_embeddings = F.embedding(memory_token_ids, token_embedding_weight)
    hop_embedding = torch.matmul(weights1, memory_token_embeddings)
    second_query = core.second_query(first_query, hop_embedding)

    scores2 = torch.einsum("btd,bsd->bts", second_query, memory_keys) / RETRIEVAL_TEMPERATURE
    weights2 = torch.softmax(scores2.float(), dim=-1).to(hidden.dtype)
    copy_probs = copy_distribution(
        weights2,
        memory_token_ids,
        vocab_size=int(token_embedding_weight.shape[0]),
    )
    gate = core.gate(hidden)
    log_probs = mix_lm_and_copy_log_probs(base_logits, copy_probs, gate)
    return log_probs, {
        "first_hop_weights": weights1,
        "hop_embedding": hop_embedding,
        "second_query": second_query,
        "second_hop_weights": weights2,
        "copy_probs": copy_probs,
        "copy_gate": gate,
    }


def _chunked(tokens: torch.Tensor):
    if tokens.ndim != 2:
        raise ValueError("tokens must be [batch, sequence]")
    if tokens.shape[1] <= 0 or tokens.shape[1] % LOCAL_WINDOW != 0:
        raise ValueError(f"sequence length must be a positive multiple of {LOCAL_WINDOW}")
    for start in range(0, tokens.shape[1], LOCAL_WINDOW):
        yield tokens[:, start : start + LOCAL_WINDOW]


def daec_flat_training_session_log_probs(
    model: CHMV3100MDAECLM,
    tokens: torch.Tensor,
    *,
    temperature: float = RETRIEVAL_TEMPERATURE,
) -> torch.Tensor:
    """Causal differentiable DAEC session path over completed prior chunks only."""
    if float(temperature) != RETRIEVAL_TEMPERATURE:
        raise RuntimeError(
            f"#1234 retrieval temperature is frozen at {RETRIEVAL_TEMPERATURE}"
        )

    memory_keys: torch.Tensor | None = None
    memory_token_ids: torch.Tensor | None = None
    outputs: list[torch.Tensor] = []

    for chunk in _chunked(tokens):
        hidden = _hidden(model.backbone, chunk)
        base_logits = model.backbone.lm_head(hidden)
        current_keys = model.key_for(hidden)

        if memory_keys is None:
            outputs.append(no_memory_log_probs(base_logits))
        else:
            assert memory_token_ids is not None
            first_query = model.query_for(hidden)
            log_probs, _ = two_hop_copy_log_probs(
                model.daec,
                first_query=first_query,
                hidden=hidden,
                memory_keys=memory_keys,
                memory_token_ids=memory_token_ids,
                token_embedding_weight=model.backbone.token_emb.weight,
                base_logits=base_logits,
                temperature=temperature,
            )
            outputs.append(log_probs)

        # Causality: current chunk becomes readable only after its outputs exist.
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


def analytical_parameter_accounting() -> dict[str, int | float | bool]:
    daec = EXPECTED_EIEM_PARAMETERS + DAEC_ADDED_PARAMETERS
    delta = (daec - EXPECTED_LOCAL_PARAMETERS) / EXPECTED_LOCAL_PARAMETERS
    return {
        "local_parameters": EXPECTED_LOCAL_PARAMETERS,
        "inherited_eiem_parameters": EXPECTED_EIEM_PARAMETERS,
        "hop_update_parameters": DAEC_HOP_UPDATE_PARAMETERS,
        "copy_gate_parameters": DAEC_COPY_GATE_PARAMETERS,
        "daec_added_parameters": DAEC_ADDED_PARAMETERS,
        "daec_parameters": daec,
        "daec_minus_local_parameters": daec - EXPECTED_LOCAL_PARAMETERS,
        "delta_fraction_vs_local": delta,
        "within_point_one_percent": abs(delta) <= PARAMETER_MISMATCH_LIMIT,
    }


def instantiated_parameter_accounting() -> dict[str, int | float | bool]:
    model = CHMV3100MDAECLM()
    total = parameter_count(model)
    result = analytical_parameter_accounting()
    return {
        **result,
        "instantiated_daec_parameters": total,
        "instantiated_matches_analytical": total == int(result["daec_parameters"]),
    }


def build_engineering_pair(
    *,
    seed: int = ENGINEERING_SEED,
    device: torch.device | None = None,
) -> tuple[CHMV1100MLocalLM, CHMV3100MDAECLM]:
    validate_engineering_seed(seed)
    target = torch.device("cpu") if device is None else device

    torch.manual_seed(seed)
    local = CHMV1100MLocalLM().to(target)
    torch.manual_seed(seed)
    daec = CHMV3100MDAECLM().to(target)

    local_backbone = local.backbone.state_dict()
    daec_backbone = daec.backbone.state_dict()
    if local_backbone.keys() != daec_backbone.keys():
        raise RuntimeError("#1234 paired backbone state-key drift")
    for name, value in local_backbone.items():
        if not torch.equal(value, daec_backbone[name]):
            raise RuntimeError(f"#1234 paired backbone initialization mismatch at {name}")

    return local, daec


def protocol_manifest() -> dict[str, Any]:
    return {
        "classification": "CHM_V3_100M_DAEC_STAGE_A_PREREGISTRATION_NO_GPU_AUTHORITY",
        "issue": ISSUE,
        "implementation_base_main_sha": IMPLEMENTATION_BASE_MAIN_SHA,
        "implementation_base_tree_sha": IMPLEMENTATION_BASE_TREE_SHA,
        "engineering_seed": ENGINEERING_SEED,
        "configuration": asdict(transformer_100m_config()),
        "local_window": LOCAL_WINDOW,
        "address_dim": ADDRESS_DIM,
        "retrieval_hops": RETRIEVAL_HOPS,
        "retrieval_temperature": RETRIEVAL_TEMPERATURE,
        "copy_gate_init": COPY_GATE_INIT,
        "payload": "past_token_id",
        "decoder_basis": "tied_token_embedding",
        "first_hop_effect": "second_hop_address_update_only",
        "second_hop_effect": "vocabulary_copy_distribution",
        "hidden_memory_residual": False,
        "recency_bias": False,
        "parameter_accounting": analytical_parameter_accounting(),
        "gpu_authorized": False,
        "modal_authorized": False,
        "paid_compute_authorized": False,
        "training_execution_authorized": False,
        "scientific_seed_reserved": False,
        "stage_b_authorized": False,
        "stage_c_authorized": False,
        "stage_d_authorized": False,
    }


def stage_a_preflight(*, instantiate: bool = False) -> dict[str, Any]:
    manifest = protocol_manifest()
    accounting = (
        instantiated_parameter_accounting()
        if instantiate
        else analytical_parameter_accounting()
    )
    if int(accounting["daec_parameters"]) != EXPECTED_DAEC_PARAMETERS:
        raise RuntimeError(f"#1234 analytical DAEC parameter drift: {accounting}")
    if int(accounting["daec_minus_local_parameters"]) != EXPECTED_DAEC_MINUS_LOCAL:
        raise RuntimeError(f"#1234 DAEC-vs-LOCAL parameter drift: {accounting}")
    if not bool(accounting["within_point_one_percent"]):
        raise RuntimeError(f"#1234 parameter fairness gate failed: {accounting}")
    if instantiate and not bool(accounting["instantiated_matches_analytical"]):
        raise RuntimeError(f"#1234 instantiated parameter accounting mismatch: {accounting}")
    if RETRIEVAL_HOPS != 2 or RETRIEVAL_TEMPERATURE != 0.10:
        raise RuntimeError("#1234 frozen retrieval contract drift")
    if D_MODEL != 512 or ADDRESS_DIM != 32 or LOCAL_WINDOW != 512:
        raise RuntimeError("#1234 frozen geometry drift")

    return {
        **manifest,
        "classification": STAGE_A_CLASSIFICATION,
        "parameter_accounting": accounting,
        "interpretation_ceiling": (
            "Stage-A engineering evidence only; no modeling result, GPU result, "
            "scientific comparison, or breakthrough claim."
        ),
    }
