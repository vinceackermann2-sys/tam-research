from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from architectures.cortex_s.grouped_moe_v4 import (
    MemoryLeanPhysicalPaddedGroupedSparseMoE,
)
from architectures.cortex_s.language_model import CausalSelfAttention
from architectures.cortex_s.production_scan_integration_v1 import (
    ProductionScanPersistentWorldState,
)


VOCAB_SIZE = 50_257
D_MODEL = 512
N_LAYERS = 24
N_HEADS = 16
MAX_SEQ_LEN = 16_384
STATE_SIZE = 128
NUM_EXPERTS = 8
TOP_K = 2
ATTENTION_EVERY = 6
PARAMETER_TOLERANCE = 0.005

# Mechanically solved widths that keep all systems-only ablations within 0.5% of
# the 16K Transformer parameter target. These are not training hyperparameters and
# do not modify the completed CORTEX-S v11 model.
TRANSFORMER_FF_HIDDEN = 2_048
WORLD_ONLY_FF_HIDDEN = 1_855
REDUCED_ATTENTION_FF_HIDDEN = 2_902
MOE_ONLY_EXPERT_HIDDEN = 255
FULL_CORTEX_EXPERT_HIDDEN = 338


FeedForwardKind = Literal["dense", "moe"]


@dataclass(frozen=True)
class SystemsVariant:
    name: str
    use_world: bool
    attention_every: int
    feedforward_kind: FeedForwardKind
    dense_hidden: int | None = None
    expert_hidden: int | None = None

    def validate(self) -> None:
        if self.attention_every <= 0 or N_LAYERS % self.attention_every:
            raise ValueError("attention_every must divide the frozen 24-layer depth")
        if self.feedforward_kind == "dense":
            if self.dense_hidden is None or self.expert_hidden is not None:
                raise ValueError("dense variants require only dense_hidden")
        elif self.feedforward_kind == "moe":
            if self.expert_hidden is None or self.dense_hidden is not None:
                raise ValueError("MoE variants require only expert_hidden")
        else:  # pragma: no cover - Literal + construction guards this in normal use.
            raise ValueError(f"unknown feedforward kind: {self.feedforward_kind}")


VARIANTS: dict[str, SystemsVariant] = {
    "transformer": SystemsVariant(
        name="transformer",
        use_world=False,
        attention_every=1,
        feedforward_kind="dense",
        dense_hidden=TRANSFORMER_FF_HIDDEN,
    ),
    "world_only": SystemsVariant(
        name="world_only",
        use_world=True,
        attention_every=1,
        feedforward_kind="dense",
        dense_hidden=WORLD_ONLY_FF_HIDDEN,
    ),
    "reduced_attention_only": SystemsVariant(
        name="reduced_attention_only",
        use_world=False,
        attention_every=ATTENTION_EVERY,
        feedforward_kind="dense",
        dense_hidden=REDUCED_ATTENTION_FF_HIDDEN,
    ),
    "moe_only": SystemsVariant(
        name="moe_only",
        use_world=False,
        attention_every=1,
        feedforward_kind="moe",
        expert_hidden=MOE_ONLY_EXPERT_HIDDEN,
    ),
    "full_cortex": SystemsVariant(
        name="full_cortex",
        use_world=True,
        attention_every=ATTENTION_EVERY,
        feedforward_kind="moe",
        expert_hidden=FULL_CORTEX_EXPERT_HIDDEN,
    ),
}

VARIANT_ORDER = tuple(VARIANTS)


class DenseFeedForward(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.in_proj = nn.Linear(D_MODEL, hidden, bias=False)
        self.activation = nn.GELU()
        self.out_proj = nn.Linear(hidden, D_MODEL, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.activation(self.in_proj(x)))


class SystemsAblationBlock(nn.Module):
    def __init__(self, variant: SystemsVariant, index: int):
        super().__init__()
        variant.validate()
        self.variant = variant
        self.has_attention = (index + 1) % variant.attention_every == 0

        if variant.use_world:
            self.norm_world = nn.LayerNorm(D_MODEL)
            # Use the exact production scan class so CUDA dispatches through the
            # same Triton scan seam as CORTEX-S v11.
            self.world = ProductionScanPersistentWorldState(D_MODEL, STATE_SIZE)
        else:
            self.norm_world = None
            self.world = None

        if self.has_attention:
            self.norm_attention = nn.LayerNorm(D_MODEL)
            self.attention = CausalSelfAttention(D_MODEL, N_HEADS)
        else:
            self.norm_attention = None
            self.attention = None

        self.norm_ff = nn.LayerNorm(D_MODEL)
        if variant.feedforward_kind == "dense":
            assert variant.dense_hidden is not None
            self.feedforward: nn.Module = DenseFeedForward(variant.dense_hidden)
        else:
            assert variant.expert_hidden is not None
            # Use the exact production memory-lean grouped backend. On H100 this
            # executes grouped BF16 GEMMs; unsupported environments retain its
            # guarded reference path for CPU tests.
            self.feedforward = MemoryLeanPhysicalPaddedGroupedSparseMoE(
                d_model=D_MODEL,
                num_experts=NUM_EXPERTS,
                top_k=TOP_K,
                hidden=variant.expert_hidden,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.world is not None:
            assert self.norm_world is not None
            world_delta, _, _ = self.world(self.norm_world(x), None)
            x = x + world_delta
        if self.attention is not None:
            assert self.norm_attention is not None
            x = x + self.attention(self.norm_attention(x))
        x = x + self.feedforward(self.norm_ff(x))
        return x


class LongContextSystemsLM(nn.Module):
    """Untrained 100M-class LM used only for systems scaling falsification.

    Every variant shares vocabulary, width, depth, heads, learned 16K position
    table, tied token/output embedding and full-vocabulary LM head. Differences are
    limited to the preregistered recurrent-state, attention-frequency and sparse-MoE
    ablations above.
    """

    def __init__(self, variant: SystemsVariant):
        super().__init__()
        variant.validate()
        self.variant = variant
        self.token_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Embedding(MAX_SEQ_LEN, D_MODEL)
        self.blocks = nn.ModuleList(
            SystemsAblationBlock(variant, index) for index in range(N_LAYERS)
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
        batch, length = tokens.shape
        if batch <= 0 or length <= 0:
            raise ValueError("tokens must have positive batch and sequence dimensions")
        if length > MAX_SEQ_LEN:
            raise ValueError(
                f"sequence length {length} exceeds systems max_seq_len={MAX_SEQ_LEN}"
            )
        positions = torch.arange(length, device=tokens.device)
        x = self.token_emb(tokens) + self.pos_emb(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))


def build_variant(name: str) -> LongContextSystemsLM:
    try:
        variant = VARIANTS[name]
    except KeyError as exc:
        raise ValueError(f"unknown systems variant: {name}") from exc
    return LongContextSystemsLM(variant)


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _common_parameter_count() -> int:
    # lm_head is tied to token_emb and therefore is not counted twice by
    # module.parameters().
    return VOCAB_SIZE * D_MODEL + MAX_SEQ_LEN * D_MODEL + 2 * D_MODEL


def _attention_parameter_count() -> int:
    return 4 * D_MODEL * D_MODEL


def _world_parameter_count() -> int:
    # candidate(no bias) + keep(with bias) + output(no bias)
    return (
        D_MODEL * STATE_SIZE
        + D_MODEL * STATE_SIZE
        + STATE_SIZE
        + STATE_SIZE * D_MODEL
    )


def _dense_parameter_count(hidden: int) -> int:
    return 2 * D_MODEL * hidden


def _moe_parameter_count(hidden: int) -> int:
    router = D_MODEL * NUM_EXPERTS + NUM_EXPERTS
    experts = NUM_EXPERTS * 2 * D_MODEL * hidden
    return router + experts


def expected_parameter_count(name: str) -> int:
    variant = VARIANTS[name]
    common = _common_parameter_count()
    attention_layers = N_LAYERS // variant.attention_every
    total = common + attention_layers * _attention_parameter_count()

    # Every active component has its own LayerNorm, exactly matching the forward
    # structure above. Each LayerNorm contributes weight+bias = 2*d_model.
    norm_count = N_LAYERS  # feed-forward norm in every block
    if variant.use_world:
        total += N_LAYERS * _world_parameter_count()
        norm_count += N_LAYERS
    norm_count += attention_layers
    total += norm_count * 2 * D_MODEL

    if variant.feedforward_kind == "dense":
        assert variant.dense_hidden is not None
        total += N_LAYERS * _dense_parameter_count(variant.dense_hidden)
    else:
        assert variant.expert_hidden is not None
        total += N_LAYERS * _moe_parameter_count(variant.expert_hidden)
    return total


def parameter_contract() -> dict[str, object]:
    counts = {name: expected_parameter_count(name) for name in VARIANT_ORDER}
    target = counts["transformer"]
    gaps = {name: abs(count - target) / target for name, count in counts.items()}
    if any(gap > PARAMETER_TOLERANCE for gap in gaps.values()):
        raise RuntimeError(f"systems ablation parameter mismatch: {gaps}")
    return {
        "target": target,
        "counts": counts,
        "gaps_fraction": gaps,
        "tolerance_fraction": PARAMETER_TOLERANCE,
    }


def architecture_contract() -> dict[str, object]:
    parameter_info = parameter_contract()
    return {
        "classification": "ENGINEERING_SYSTEMS_ABLATION_ONLY",
        "vocab_size": VOCAB_SIZE,
        "d_model": D_MODEL,
        "n_layers": N_LAYERS,
        "n_heads": N_HEADS,
        "max_seq_len": MAX_SEQ_LEN,
        "state_size": STATE_SIZE,
        "num_experts": NUM_EXPERTS,
        "top_k": TOP_K,
        "attention_every": ATTENTION_EVERY,
        "variant_order": list(VARIANT_ORDER),
        "variants": {
            name: {
                "use_world": spec.use_world,
                "attention_every": spec.attention_every,
                "attention_layers": N_LAYERS // spec.attention_every,
                "feedforward_kind": spec.feedforward_kind,
                "dense_hidden": spec.dense_hidden,
                "expert_hidden": spec.expert_hidden,
                "expected_parameters": parameter_info["counts"][name],
            }
            for name, spec in VARIANTS.items()
        },
        "parameter_contract": parameter_info,
        "trained_checkpoint_used": False,
        "quality_claim_authorized": False,
        "scientific_claim_authorized": False,
    }
