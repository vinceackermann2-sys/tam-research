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
}


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
            "50m/100m scaling runs are restricted to Transformer and TAM v3 variants"
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
