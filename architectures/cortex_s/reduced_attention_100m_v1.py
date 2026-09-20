from __future__ import annotations

import torch
from torch import nn

from tam_research.models import CausalSelfAttention


VOCAB_SIZE = 50_257
D_MODEL = 512
N_LAYERS = 24
N_HEADS = 16
MAX_SEQ_LEN = 1_024
ATTENTION_EVERY = 6
FF_HIDDEN = 2_902

EXPECTED_TRANSFORMER_PARAMETERS = 101_803_520
EXPECTED_REDUCED_ATTENTION_PARAMETERS = 101_799_424
MAX_PARAMETER_GAP_FRACTION = 0.005


def attention_layer_indices() -> tuple[int, ...]:
    return tuple(
        index
        for index in range(N_LAYERS)
        if (index + 1) % ATTENTION_EVERY == 0
    )


class DenseFeedForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_proj = nn.Linear(D_MODEL, FF_HIDDEN, bias=False)
        self.activation = nn.GELU()
        self.out_proj = nn.Linear(FF_HIDDEN, D_MODEL, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.activation(self.in_proj(x)))


class ReducedAttentionBlock(nn.Module):
    def __init__(self, index: int) -> None:
        super().__init__()
        if not 0 <= index < N_LAYERS:
            raise ValueError(f"block index {index} is outside [0, {N_LAYERS})")
        self.index = index
        self.has_attention = (index + 1) % ATTENTION_EVERY == 0

        if self.has_attention:
            self.norm_attention: nn.LayerNorm | None = nn.LayerNorm(D_MODEL)
            self.attention: CausalSelfAttention | None = CausalSelfAttention(
                D_MODEL,
                N_HEADS,
            )
        else:
            self.norm_attention = None
            self.attention = None

        self.norm_ff = nn.LayerNorm(D_MODEL)
        self.feedforward = DenseFeedForward()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.attention is not None:
            assert self.norm_attention is not None
            x = x + self.attention(self.norm_attention(x))
        return x + self.feedforward(self.norm_ff(x))


class ReducedAttentionDenseLM(nn.Module):
    """Parameter-matched 100M reduced-attention quality candidate.

    This candidate intentionally contains no recurrent/world-state mechanism and
    no sparse MoE/router. Full causal attention appears only in layers
    6, 12, 18 and 24; every block retains a dense feed-forward network.
    """

    def __init__(self) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Embedding(MAX_SEQ_LEN, D_MODEL)
        self.blocks = nn.ModuleList(
            ReducedAttentionBlock(index) for index in range(N_LAYERS)
        )
        self.norm = nn.LayerNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, sequence]")
        batch, length = tokens.shape
        if batch <= 0 or length <= 0:
            raise ValueError("tokens must have positive batch and sequence dimensions")
        if length > MAX_SEQ_LEN:
            raise ValueError(
                f"sequence length {length} exceeds max_seq_len={MAX_SEQ_LEN}"
            )

        positions = torch.arange(length, device=tokens.device)
        x = self.token_emb(tokens) + self.pos_emb(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))


def expected_parameter_count() -> int:
    common = (
        VOCAB_SIZE * D_MODEL
        + MAX_SEQ_LEN * D_MODEL
        + 2 * D_MODEL
    )
    attention_layers = len(attention_layer_indices())
    attention = attention_layers * 4 * D_MODEL * D_MODEL
    layer_norms = (N_LAYERS + attention_layers) * 2 * D_MODEL
    feedforward = N_LAYERS * 2 * D_MODEL * FF_HIDDEN
    return common + attention + layer_norms + feedforward


def parameter_contract() -> dict[str, float | int]:
    candidate = expected_parameter_count()
    if candidate != EXPECTED_REDUCED_ATTENTION_PARAMETERS:
        raise RuntimeError(
            "reduced-attention parameter formula drifted: "
            f"{candidate:,} != {EXPECTED_REDUCED_ATTENTION_PARAMETERS:,}"
        )
    gap = abs(candidate - EXPECTED_TRANSFORMER_PARAMETERS)
    gap_fraction = gap / EXPECTED_TRANSFORMER_PARAMETERS
    if gap_fraction > MAX_PARAMETER_GAP_FRACTION:
        raise RuntimeError(
            f"parameter gap {gap_fraction:.6%} exceeds "
            f"{MAX_PARAMETER_GAP_FRACTION:.2%}"
        )
    return {
        "transformer_parameters": EXPECTED_TRANSFORMER_PARAMETERS,
        "reduced_attention_parameters": candidate,
        "absolute_gap": gap,
        "gap_fraction": gap_fraction,
        "attention_layers": len(attention_layer_indices()),
        "attention_fraction": len(attention_layer_indices()) / N_LAYERS,
    }


def architecture_contract() -> dict[str, object]:
    params = parameter_contract()
    return {
        "name": "reduced_attention_dense_v1",
        "vocab_size": VOCAB_SIZE,
        "d_model": D_MODEL,
        "n_layers": N_LAYERS,
        "n_heads": N_HEADS,
        "max_seq_len": MAX_SEQ_LEN,
        "attention_every": ATTENTION_EVERY,
        "attention_layer_indices_zero_based": list(attention_layer_indices()),
        "attention_layer_numbers_one_based": [
            index + 1 for index in attention_layer_indices()
        ],
        "ff_hidden": FF_HIDDEN,
        "world_state": False,
        "moe": False,
        "router": False,
        "parameter_contract": params,
        "scientific_claim_authorized": False,
        "training_authorized": False,
    }
