from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FutureScaleSpec:
    """Self-similar post-100M scale candidate.

    These specs are intentionally NOT wired into the Modal scaling launcher.
    They are preserved here so the architecture plan is version-controlled while
    compute remains gated behind the preregistered 100M result and a fresh budget.
    """

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


FUTURE_SCALE_SPECS: dict[str, FutureScaleSpec] = {
    "300m": FutureScaleSpec("300m", d_model=768, n_layers=36, n_heads=12),
    "1b": FutureScaleSpec("1b", d_model=1280, n_layers=48, n_heads=20),
}


def analytical_parameter_count(
    architecture: str,
    scale: str,
    *,
    max_seq_len: int = 1024,
    vocab_size: int = 50_257,
    ff_mult: int = 4,
) -> int:
    """Exact future-scale parameter count without allocating a giant model.

    Mirrors the current ResearchLM parameterization: tied token/output embedding,
    learned positional embedding, final LayerNorm, two LayerNorms per block,
    bias-free MLP, and either full causal attention or TAM v3 attention +
    recurrent world-state + scalar gate.
    """

    try:
        spec = FUTURE_SCALE_SPECS[scale.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown future scale: {scale!r}") from exc

    d = spec.d_model
    layers = spec.n_layers

    shared = vocab_size * d + max_seq_len * d + 2 * d
    norms_per_block = 4 * d
    mlp_per_block = 2 * ff_mult * d * d

    if architecture == "transformer":
        mixer_per_block = 4 * d * d
    elif architecture == "tamv3":
        inner = spec.tamv3_attention_inner
        state = spec.state_size
        attention = 4 * d * inner
        world = 3 * d * state + state
        mixer_per_block = attention + world + 1
    else:
        raise ValueError("architecture must be transformer or tamv3")

    return shared + layers * (norms_per_block + mlp_per_block + mixer_per_block)
