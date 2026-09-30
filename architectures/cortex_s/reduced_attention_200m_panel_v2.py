from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch import nn

from tam_research.models import CausalSelfAttention, ModelConfig, ResearchLM


VOCAB_SIZE: Final = 50_257
D_MODEL: Final = 512
N_LAYERS: Final = 24
N_HEADS: Final = 16
MAX_SEQ_LEN: Final = 1_024

EXPECTED_TRANSFORMER_PARAMETERS: Final = 101_803_520
PARAMETER_TOLERANCE_FRACTION: Final = 0.005


@dataclass(frozen=True)
class PanelVariant:
    name: str
    attention_layers_one_based: tuple[int, ...]
    ff_hidden: int
    expected_parameters: int


V1_REPLICATE = PanelVariant(
    name="v1_replicate",
    attention_layers_one_based=(6, 12, 18, 24),
    ff_hidden=2_902,
    expected_parameters=101_799_424,
)

EARLY_4 = PanelVariant(
    name="early_4",
    attention_layers_one_based=(1, 8, 16, 24),
    ff_hidden=2_902,
    expected_parameters=101_799_424,
)

ATTENTION_8 = PanelVariant(
    name="attention_8",
    attention_layers_one_based=(3, 6, 9, 12, 15, 18, 21, 24),
    ff_hidden=2_731,
    expected_parameters=101_795_328,
)

PANEL_VARIANTS: Final = (V1_REPLICATE, EARLY_4, ATTENTION_8)
PANEL_VARIANTS_BY_NAME: Final = {variant.name: variant for variant in PANEL_VARIANTS}


class DenseFeedForward(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        if hidden <= 0:
            raise ValueError("hidden must be positive")
        self.in_proj = nn.Linear(D_MODEL, hidden, bias=False)
        self.activation = nn.GELU()
        self.out_proj = nn.Linear(hidden, D_MODEL, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.activation(self.in_proj(x)))


class ScheduledAttentionBlock(nn.Module):
    def __init__(self, *, index: int, variant: PanelVariant) -> None:
        super().__init__()
        if not 0 <= index < N_LAYERS:
            raise ValueError(f"block index {index} is outside [0, {N_LAYERS})")
        layer_number = index + 1
        self.index = index
        self.layer_number = layer_number
        self.has_attention = layer_number in variant.attention_layers_one_based

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
        self.feedforward = DenseFeedForward(variant.ff_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.attention is not None:
            assert self.norm_attention is not None
            x = x + self.attention(self.norm_attention(x))
        return x + self.feedforward(self.norm_ff(x))


class ScheduledAttentionDenseLM(nn.Module):
    """Parameter-matched 100M dense-FFN LM with a frozen attention schedule."""

    def __init__(self, variant: PanelVariant) -> None:
        super().__init__()
        validate_variant(variant)
        self.variant = variant
        self.token_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Embedding(MAX_SEQ_LEN, D_MODEL)
        self.blocks = nn.ModuleList(
            ScheduledAttentionBlock(index=index, variant=variant)
            for index in range(N_LAYERS)
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


def build_fresh_transformer() -> ResearchLM:
    return ResearchLM(
        ModelConfig(
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            n_layers=N_LAYERS,
            n_heads=N_HEADS,
            max_seq_len=MAX_SEQ_LEN,
            ff_mult=4,
            architecture="transformer",
        )
    )


def build_panel_variant(name: str) -> ScheduledAttentionDenseLM:
    try:
        variant = PANEL_VARIANTS_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(f"unknown panel variant: {name!r}") from exc
    return ScheduledAttentionDenseLM(variant)


def expected_parameter_count(variant: PanelVariant) -> int:
    common = (
        VOCAB_SIZE * D_MODEL
        + MAX_SEQ_LEN * D_MODEL
        + 2 * D_MODEL
    )
    attention_layers = len(variant.attention_layers_one_based)
    attention = attention_layers * 4 * D_MODEL * D_MODEL
    layer_norms = (N_LAYERS + attention_layers) * 2 * D_MODEL
    feedforward = N_LAYERS * 2 * D_MODEL * variant.ff_hidden
    return common + attention + layer_norms + feedforward


def validate_variant(variant: PanelVariant) -> None:
    layers = variant.attention_layers_one_based
    if not layers:
        raise ValueError("panel variant must contain at least one attention layer")
    if tuple(sorted(set(layers))) != layers:
        raise ValueError("attention layers must be unique and sorted")
    if layers[0] < 1 or layers[-1] > N_LAYERS:
        raise ValueError("attention layers must be within [1, 24]")
    actual = expected_parameter_count(variant)
    if actual != variant.expected_parameters:
        raise RuntimeError(
            f"{variant.name} parameter formula drifted: "
            f"{actual:,} != {variant.expected_parameters:,}"
        )
    gap = abs(actual - EXPECTED_TRANSFORMER_PARAMETERS)
    if gap / EXPECTED_TRANSFORMER_PARAMETERS > PARAMETER_TOLERANCE_FRACTION:
        raise RuntimeError(
            f"{variant.name} parameter gap exceeds "
            f"{PARAMETER_TOLERANCE_FRACTION:.2%}"
        )


def architecture_contract() -> dict[str, object]:
    for variant in PANEL_VARIANTS:
        validate_variant(variant)
    return {
        "panel_name": "reduced_attention_v2_200m_engineering_panel",
        "fresh_transformer": {
            "attention_layers_one_based": list(range(1, N_LAYERS + 1)),
            "ff_hidden": 4 * D_MODEL,
            "expected_parameters": EXPECTED_TRANSFORMER_PARAMETERS,
        },
        "variants": {
            variant.name: {
                "attention_layers_one_based": list(
                    variant.attention_layers_one_based
                ),
                "ff_hidden": variant.ff_hidden,
                "expected_parameters": variant.expected_parameters,
                "attention_fraction": (
                    len(variant.attention_layers_one_based) / N_LAYERS
                ),
            }
            for variant in PANEL_VARIANTS
        },
        "d_model": D_MODEL,
        "n_layers": N_LAYERS,
        "n_heads": N_HEADS,
        "max_seq_len": MAX_SEQ_LEN,
        "world_state": False,
        "moe": False,
        "router": False,
        "gpu_authorized": False,
        "training_authorized": False,
    }
