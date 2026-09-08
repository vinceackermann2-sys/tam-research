from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from architectures.cortex_s.language_model import CortexSLM, CortexSLMConfig, parameter_count
from architectures.cortex_s.systems_candidate_model import convert_to_grouped_moe_candidate
from architectures.cortex_s.systems_optimization import GroupedSparseMoECandidate


def _tiny_config() -> CortexSLMConfig:
    return CortexSLMConfig(
        vocab_size=64,
        d_model=32,
        n_layers=4,
        n_heads=4,
        max_seq_len=32,
        state_size=8,
        num_experts=4,
        top_k=2,
        expert_hidden=24,
        attention_every=2,
    )


def test_full_model_conversion_preserves_outputs_state_and_parameters() -> None:
    torch.manual_seed(77501)
    baseline = CortexSLM(_tiny_config()).eval()
    candidate = deepcopy(baseline)
    before = parameter_count(candidate)

    rng_before = torch.random.get_rng_state().clone()
    convert_to_grouped_moe_candidate(candidate)
    rng_after = torch.random.get_rng_state()

    assert torch.equal(rng_before, rng_after)
    assert parameter_count(candidate) == before == parameter_count(baseline)
    assert all(isinstance(block.moe, GroupedSparseMoECandidate) for block in candidate.blocks)

    tokens = torch.randint(0, 64, (2, 13))
    with torch.no_grad():
        baseline_logits, baseline_state = baseline(tokens, return_state=True)
        candidate_logits, candidate_state = candidate(tokens, return_state=True)

    torch.testing.assert_close(candidate_logits, baseline_logits, atol=3e-6, rtol=3e-6)
    assert len(candidate_state) == len(baseline_state)
    for observed, expected in zip(candidate_state, baseline_state):
        torch.testing.assert_close(observed, expected, atol=3e-6, rtol=3e-6)


def test_full_model_conversion_preserves_causal_chunk_continuation() -> None:
    torch.manual_seed(77502)
    baseline = CortexSLM(_tiny_config()).eval()
    candidate = convert_to_grouped_moe_candidate(deepcopy(baseline)).eval()
    tokens = torch.randint(0, 64, (2, 14))

    with torch.no_grad():
        _, baseline_state = baseline(tokens[:, :8], return_state=True)
        _, candidate_state = candidate(tokens[:, :8], return_state=True)
        baseline_next = baseline(tokens[:, 8:12], state=baseline_state)
        candidate_next = candidate(tokens[:, 8:12], state=candidate_state)

    torch.testing.assert_close(candidate_next, baseline_next, atol=3e-6, rtol=3e-6)
