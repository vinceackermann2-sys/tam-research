"""Original SI hybrid versus Transformer on public ten-arm developer data.

No scientific train/eval seeds, hidden arm probabilities, oracle,
architecture-specific data, or paid model training.
"""
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from tam_research.si_pcma_v1_external_bandit import (
    ARMS, HORIZON, PublicBanditFeedback,
)
from tam_research.si_pcma_v1_bandit_neural_adapter import (
    BanditPublicPayoutPredictor, PublicBanditExample,
    ENGINEERING_TRAIN_EPISODES, ENGINEERING_VALIDATION_EPISODES,
    TOKEN_WINDOW, DEVELOPER_UPDATES, bandit_tokens, stack_factual_batch,
    development_bandit_examples, development_neural_forecast_screen,
)
from tam_research.si_original_hybrid_v0 import (
    OriginalSIHybrid, TransformerControl,
)
from tam_research.si_pcma_v1_science_preflight import (
    CANONICAL_ORIGINAL_HYBRID, CANONICAL_ORIGINAL_PATH, git_blob,
)


def _feedback(step: int, action: int = 2, payout: int = 1):
    return PublicBanditFeedback(
        action=action, observed_payout=payout,
        reward=float((2*payout-1)/HORIZON), step=step,
    )


def test_original_source_remains_immutable():
    root = Path(__file__).resolve().parents[1]
    assert git_blob(root / CANONICAL_ORIGINAL_PATH, repo_root=root) == (
        CANONICAL_ORIGINAL_HYBRID)


def test_ten_action_expansion_reuses_exact_original_hybrid_and_transformer():
    torch.manual_seed(1)
    hybrid = BanditPublicPayoutPredictor("hybrid")
    trans = BanditPublicPayoutPredictor("transformer")
    assert isinstance(hybrid.backbone, OriginalSIHybrid)
    assert isinstance(trans.backbone, TransformerControl)
    assert hybrid.backbone.config.action_dim == ARMS == 10
    assert trans.backbone.config.action_dim == ARMS
    assert hybrid.backbone.config.obs_dim == 3
    assert trans.backbone.config.obs_dim == 3
    assert hybrid.backbone.config.max_seq_len == 16
    assert type(hybrid.payout_head) is type(trans.payout_head)
    assert sum(p.numel() for p in hybrid.parameters()) > 0
    assert sum(p.numel() for p in trans.parameters()) > 0
    with pytest.raises(ValueError):
        BanditPublicPayoutPredictor("oracle")


def test_query_has_no_label_and_exact_same_public_shapes():
    history = tuple(_feedback(t, action=t % ARMS,
                              payout=t % 2) for t in range(1, 9))
    obs, act, modality = bandit_tokens(history, 9, step=9)
    assert obs.shape == (1, TOKEN_WINDOW, 3)
    assert act.shape == (1, TOKEN_WINDOW, ARMS)
    assert modality.shape == (1, TOKEN_WINDOW)
    assert obs[0,-1,0].item() == 0.0
    assert obs[0,-1,2].item() == 1.0
    assert int(modality[0,-1]) == 1
    assert act[0,-1,9].item() == 1.0
    assert obs[0,-2,0].item() == float(history[-1].observed_payout)
    assert all(x == 0 for x in modality[0,:-1].tolist())
    # Only prior observed feedback is available. Altering the current
    # sample's target must not alter the model's input.
    record = PublicBanditExample(history, 9, 9, 1)
    alternate = replace(record, observed_payout=0)
    tokens_a = stack_factual_batch([record])
    tokens_b = stack_factual_batch([alternate])
    assert all(torch.equal(x, y) for x,y in zip(tokens_a[:3],tokens_b[:3]))
    assert not torch.equal(tokens_a[3], tokens_b[3])


def test_future_outcomes_duplicates_and_malformed_reward_rejected():
    one = (_feedback(1), _feedback(2))
    with pytest.raises(ValueError):
        bandit_tokens(one+(_feedback(2),), 3, 3)
    with pytest.raises(ValueError):
        bandit_tokens(one, 3, 2)
    with pytest.raises(ValueError):
        bandit_tokens(one, 3, 4)  # Gap in last factual feedback
    with pytest.raises(ValueError):
        bandit_tokens((_feedback(1), _feedback(2, payout=0)), 11, 3)
    with pytest.raises(ValueError):
        bandit_tokens((_feedback(1), replace(_feedback(2), reward=float("nan"))),
                      3, 3)
    with pytest.raises(ValueError):
        bandit_tokens(one, 10, 3)
    with pytest.raises(ValueError):
        bandit_tokens(one, True, 3)


def test_batch_and_all_causal_models_accept_same_actual_public_features():
    torch.set_num_threads(2)
    history = tuple(_feedback(i, i%ARMS, i%2) for i in range(1, 9))
    records = (
        PublicBanditExample(history, 0, 9, 0),
        PublicBanditExample(history, 5, 9, 1),
    )
    tensors = stack_factual_batch(records)
    assert tensors[0].shape == (2, TOKEN_WINDOW, 3)
    assert tensors[1].shape == (2, TOKEN_WINDOW, ARMS)
    assert tensors[2].shape == (2, TOKEN_WINDOW)
    for kind in ("hybrid", "transformer"):
        model = BanditPublicPayoutPredictor(kind)
        with torch.no_grad():
            values = model(*tensors[:3])
        assert values.shape == (2,)
        assert torch.isfinite(values).all()


def test_development_collection_uses_fixed_nonoverlapping_real_worlds():
    train = development_bandit_examples(ENGINEERING_TRAIN_EPISODES)
    valid = development_bandit_examples(ENGINEERING_VALIDATION_EPISODES)
    assert not set(ENGINEERING_TRAIN_EPISODES) & set(ENGINEERING_VALIDATION_EPISODES)
    assert len(train) == 39*len(ENGINEERING_TRAIN_EPISODES)
    assert len(valid) == 39*len(ENGINEERING_VALIDATION_EPISODES)
    assert all(x.step % 5 == 0 and x.step >= 8 for x in train+valid)
    assert all(x.observed_payout in (0, 1) for x in train+valid)
    assert all(x.action == (x.step-1) % ARMS for x in train+valid)
    with pytest.raises(ValueError):
        development_bandit_examples((101, 201, 101))
    with pytest.raises(ValueError):
        development_bandit_examples((999999,))
    with pytest.raises(ValueError):
        development_bandit_examples((True,))


def test_fixed_cpu_smoke_performs_identical_updates_no_architecture_claim():
    # This checks engineering feasibility, NOT scientific parameter/FLOP
    # parity. Repeatability, positive losses and disjoint dev IDs are gates.
    report = development_neural_forecast_screen()
    assert report.status == "SMALL_CPU_DEVELOPMENT_FORECAST_ONLY"
    assert report.fixed_training_episodes == ENGINEERING_TRAIN_EPISODES
    assert report.fixed_validation_episodes == ENGINEERING_VALIDATION_EPISODES
    assert not report.reward_policy_evaluated
    assert not report.strong_thompson_comparison_run
    assert report.science_seeds_reserved == report.heldout_worlds_evaluated == 0
    assert not report.may_claim_architecture_advantage
    assert {row.kind for row in report.results} == {"hybrid", "transformer"}
    assert all(row.optimizer_updates == DEVELOPER_UPDATES == 4
               and row.training_examples == 156
               and row.validation_examples == 78
               and row.model_parameters > 10000
               and row.last_training_bce > 0
               and 0 < row.factual_validation_bce < 10
               and row.training_cpu_wall_seconds > 0
               and not row.matched_active_flops
               for row in report.results)


def test_no_privileged_bandit_probability_is_encoded_or_exposed():
    import tam_research.si_pcma_v1_bandit_neural_adapter as adapter
    assert not hasattr(adapter, "get_state")
    assert not hasattr(adapter, "bandits")
    assert not hasattr(adapter, "oracle")
    assert not hasattr(adapter, "counterfactual")
    assert [f.name for f in adapter.PublicBanditExample.__dataclass_fields__.values()] == [
        "previous", "action", "step", "observed_payout",
    ]
