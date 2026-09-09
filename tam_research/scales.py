from __future__ import annotations

from dataclasses import dataclass

from .models import ModelConfig


@dataclass(frozen=True)
class ScaleSpec:
    name: str
    d_model: int
    n_layers: int
    n_heads: int

    @property
    def state_size(self) -> int:
        return self.d_model // 4

    @property
    def tamv3_attention_inner(self) -> int:
        return 13 * self.d_model // 16


SCALE_SPECS: dict[str, ScaleSpec] = {
    "25m": ScaleSpec("25m", d_model=256, n_layers=15, n_heads=8),
    "50m": ScaleSpec("50m", d_model=384, n_layers=17, n_heads=12),
    "100m": ScaleSpec("100m", d_model=512, n_layers=24, n_heads=16),
    # Post-100M candidates. These remain self-similar: 64-dim Transformer heads,
    # state width=d_model/4, and TAM v3 attention width=13*d_model/16.
    "300m": ScaleSpec("300m", d_model=768, n_layers=36, n_heads=12),
    "1b": ScaleSpec("1b", d_model=1280, n_layers=48, n_heads=20),
}


def analytical_parameter_count(
    architecture: str,
    scale: str,
    *,
    max_seq_len: int = 1024,
    vocab_size: int = 50_257,
    ff_mult: int = 4,
) -> int:
    """Exact parameter count without instantiating large models.

    This mirrors ResearchLM's tied token/output embedding, learned position
    embedding, LayerNorm parameters, attention/world-state mixer, and MLP.
    It is used by CI for 300M/1B candidates so tests do not materialize huge
    parameter tensors merely to verify matching.
    """
    try:
        spec = SCALE_SPECS[scale.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown model scale: {scale!r}") from exc

    d = spec.d_model
    layers = spec.n_layers
    shared = vocab_size * d + max_seq_len * d + 2 * d  # tied embedding + pos + final norm
    ff = 2 * ff_mult * d * d
    norms = 4 * d

    if architecture == "transformer":
        mixer = 4 * d * d
    elif architecture in {"tamv3", "tamv3_fixed"}:
        inner = spec.tamv3_attention_inner
        state = spec.state_size
        attention = 4 * d * inner
        world = 3 * d * state + state  # candidate + keep(+bias) + state output
        mixer = attention + world + 1  # scalar TAM v3 gate
    else:
        raise ValueError("analytical scaling counts support transformer/tamv3 only")

    return shared + layers * (mixer + ff + norms)


def model_config_for_scale(
    architecture: str,
    scale: str,
    *,
    max_seq_len: int,
) -> ModelConfig:
    try:
        spec = SCALE_SPECS[scale.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown model scale: {scale!r}") from exc

    if scale.lower() != "25m" and architecture not in {"transformer", "tamv3", "tamv3_fixed"}:
        raise ValueError(
            "scaled runs above 25M are restricted to Transformer and TAM v3 variants"
        )

    return ModelConfig(
        architecture=architecture,
        d_model=spec.d_model,
        n_layers=spec.n_layers,
        n_heads=spec.n_heads,
        max_seq_len=max_seq_len,
        tamv2_state_size=spec.state_size,
        tamv3_attn_inner=spec.tamv3_attention_inner,
    )