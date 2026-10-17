"""Split-protection unit fixtures. No benchmark promotion authority."""
import pytest
import torch

from tam_research.si_pcma_v1_ood_world import OODWorldConfig
from tam_research.si_pcma_v1_ood_development import (
    collect_development_factual_data,
)
from tam_research.si_pcma_v1_factual_learning import causal_tokens


def test_training_builder_only_allows_development_rules():
    with pytest.raises(ValueError, match="heldout"):
        collect_development_factual_data(
            [OODWorldConfig(2, "heldout")],
            exploration_fixture_seed=17,
        )
    with pytest.raises(ValueError, match="heldout"):
        collect_development_factual_data(
            [OODWorldConfig(2, "development"),
             OODWorldConfig(3, "heldout")],
            exploration_fixture_seed=17,
        )


def test_training_builder_cannot_reuse_same_world_seed():
    with pytest.raises(ValueError, match="duplicate"):
        collect_development_factual_data(
            [OODWorldConfig(2, "development"),
             OODWorldConfig(2, "development", "shift")],
            exploration_fixture_seed=17,
        )


def test_development_factual_data_is_deterministic_with_no_query_label():
    configs = [OODWorldConfig(1, "development"),
               OODWorldConfig(2, "development", "noisy")]
    a = collect_development_factual_data(configs, exploration_fixture_seed=20)
    b = collect_development_factual_data(configs, exploration_fixture_seed=20)
    assert a == b
    assert len(a.records) == len(a.samples) == 24
    assert a.world_count == 2 and a.sessions_per_world == 2
    assert len(set(e.provenance for e in a.records)) == 24
    assert all(s.prior == () for i, s in enumerate(a.samples) if i in (0, 6, 12, 18))
    for sample in a.samples:
        obs, action, modality = causal_tokens(
            sample.observation, sample.action, sample.prior,
            logical_time=sample.logical_time,
        )
        assert obs[0, -1, 2].item() == 0
        assert action.shape[-1] == 4
        assert modality[0, -1].item() == 1


def test_reject_invalid_development_collection_configuration():
    with pytest.raises(ValueError):
        collect_development_factual_data([], exploration_fixture_seed=1)
    with pytest.raises(ValueError):
        collect_development_factual_data(
            [OODWorldConfig(1, "development")],
            exploration_fixture_seed=1, sessions=0)
    with pytest.raises(ValueError):
        collect_development_factual_data(
            [OODWorldConfig(1, "development")],
            exploration_fixture_seed=-1)
