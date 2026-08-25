import torch

from aera_v19_memory_necessity_cpu import diagnostic_config
from tam_research.aera_hardware_core_v24 import (
    ContextualEpisodicMemoryState,
    HardwareAwareAERATextLMV24,
    episodic_state_bytes_per_session,
)


def test_v24_reported_state_bytes_match_allocated_tensor_dtypes_at_production_shape():
    cfg = diagnostic_config()
    cfg = type(cfg)(**{**cfg.__dict__, 'memory_dim': 50})
    model = HardwareAwareAERATextLMV24(cfg)
    state = model.stages[0].memory.empty_state(1, torch.device('cpu'), torch.float32)
    assert isinstance(state, ContextualEpisodicMemoryState)
    per_stage = sum(t.numel() * t.element_size() for t in (
        state.keys, state.values, state.strengths, state.valid
    ))
    assert per_stage * 4 == episodic_state_bytes_per_session(n_stages=4, memory_dim=50)
    assert per_stage * 4 == 77_760
