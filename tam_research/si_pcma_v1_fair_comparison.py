"""PCMA-v1 fair-information engineering harness (not a scientific study).

This module intentionally refuses heldout-rule panels: they must be run only
after an independent frozen protocol and unique attempt reservations.
Repository CI exercises correctness using development fixtures, never promotion
metrics. Private Python attributes are NOT an adversarial security boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Callable, Mapping, Protocol

import torch
from torch import nn

from tam_research.si_pcma_v1 import FactualTransition, PublicObservation
from tam_research.si_pcma_v1_factual_learning import (
    CausalRewardSample, FrozenNeuralFactualPolicy, causal_tokens,
)
from tam_research.si_pcma_v1_ood_world import (
    OODWorldConfig, PublicPolicy, run_ood_sessions,
)


class PCMAGRURewardPredictor(nn.Module):
    """Smaller, deliberately UNMATCHED GRU baseline with same causal tokens.

    Shared [obs(3), action(4)] inputs and public modality as hybrid/Transformer;
    unlike the other two, this model's architecture and parameter count differ.
    Its parameter count and measured cost MUST be reported separately.
    """
    def __init__(self, *, hidden: int = 40, layers: int = 3):
        super().__init__()
        if hidden < 1 or layers < 1:
            raise ValueError("invalid GRU configuration")
        self.kind = "gru_unmatched"
        self.modality = nn.Embedding(4, 7)
        self.gru = nn.GRU(7, hidden, num_layers=layers, batch_first=True)
        self.prediction = nn.Linear(hidden, 3)
        self.reward_head = nn.Linear(3, 1)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor,
                modality: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 3 or obs.shape[-1] != 3:
            raise ValueError("expected three public observation features")
        if actions.shape != (*obs.shape[:2], 4):
            raise ValueError("expected four action features")
        if modality.shape != obs.shape[:2]:
            raise ValueError("bad public modality shape")
        x = torch.cat([obs, actions], dim=-1) + self.modality(modality)
        y, _ = self.gru(x)
        return self.reward_head(self.prediction(y[:, -1])).flatten()

    def predict_sample(self, sample: CausalRewardSample) -> torch.Tensor:
        obs, action, mode = causal_tokens(
            sample.observation, sample.action, sample.prior,
            logical_time=sample.logical_time)
        return self(obs, action, mode)


class MeasuredFactualPolicy:
    """Exact-feedback, one-action-at-a-time wrapper with runtime accounting.

    Nothing here inspects the world's hidden state, even when the world is
    executed in the same Python process. Not a secure sandbox for hostile code.
    """
    def __init__(self, policy: PublicPolicy):
        self.policy = policy
        self._pending: tuple[PublicObservation, int] | None = None
        self._last_time = 0
        self._seen_provenance: set[str] = set()
        self.choose_ns = 0
        self.feedback_ns = 0
        self.calls = 0
        self.feedback_events = 0

    def choose(self, observation: PublicObservation) -> int:
        if type(observation) is not PublicObservation:
            raise TypeError("public observation required")
        if self._pending is not None:
            raise RuntimeError("previous chosen action not resolved")
        start = perf_counter_ns()
        action = self.policy.choose(observation)
        self.choose_ns += perf_counter_ns() - start
        if type(action) is not int or action not in range(4):
            raise ValueError("policy must return one of four integer actions")
        self._pending = (observation, action)
        self.calls += 1
        return action

    def observe(self, event: FactualTransition) -> None:
        if type(event) is not FactualTransition or self._pending is None:
            raise RuntimeError("unexpected or missing executed-action feedback")
        before, action = self._pending
        if event.before != before or event.action != action:
            raise ValueError("feedback mismatches action actually chosen")
        if event.time <= self._last_time or event.provenance in self._seen_provenance:
            raise ValueError("replayed or unordered factual transition")
        if (event.fee != (0.1 if action == 3 else 0.0) or
                event.reward not in (0.0, 1.0) or
                (action == 3 and event.reward != 0.0)):
            raise ValueError("invalid executed reward or fee")
        start = perf_counter_ns()
        self.policy.observe(event)
        self.feedback_ns += perf_counter_ns() - start
        self.feedback_events += 1
        self._last_time = event.time
        self._seen_provenance.add(event.provenance)
        self._pending = None

    def restart_session(self, *, erase_persistent: bool = False) -> None:
        if self._pending is not None:
            raise RuntimeError("cannot restart with unresolved factual action")
        self.policy.restart_session(erase_persistent=erase_persistent)


@dataclass(frozen=True)
class PublicAccount:
    """Per-policy factual trace accounting, not an SI advancement gate."""
    model: str
    world_count: int
    actions: int
    probes: int
    gross_reward: float
    total_fee: float
    net_reward: float
    net_per_action: float
    decision_ns: int
    feedback_ns: int
    param_count: int | None
    memory_entries: int | None
    adapter_updates: int | None


def count_trainable_and_frozen_parameters(policy: object) -> int | None:
    """Report total deployed neural parameters, including frozen weights."""
    predictor = getattr(policy, "predictor", None)
    if predictor is None:
        return None
    if not isinstance(predictor, nn.Module):
        return None
    return sum(p.numel() for p in predictor.parameters())


def account_run(name: str, wrapper: MeasuredFactualPolicy,
                traces: tuple[tuple[FactualTransition, ...], ...],
                *, worlds: int = 1) -> PublicAccount:
    if not traces or not all(traces) or worlds < 1:
        raise ValueError("need factual terminal traces")
    records = [e for session in traces for e in session]
    if not all(session[-1].done for session in traces):
        raise ValueError("nonterminal trace")
    if wrapper.feedback_events != len(records) or wrapper.calls != len(records):
        raise ValueError("unmatched action / observation budget")
    if len({e.time for e in records}) != len(records):
        raise ValueError("replayed logical time")
    gross = sum(e.reward for e in records)
    fee = sum(e.fee for e in records)
    state = getattr(wrapper.policy, "state", None)
    memory_entries = None
    adapter_updates = None
    if state is not None:
        memory = getattr(state, "memory", None)
        adapter = getattr(state, "adapter", None)
        if memory is not None and hasattr(memory, "items"):
            memory_entries = len(memory.items)
        if adapter is not None and hasattr(adapter, "total_updates"):
            adapter_updates = adapter.total_updates
    return PublicAccount(
        name, worlds, len(records), sum(e.action == 3 for e in records),
        float(gross), float(fee), float(gross - fee),
        float((gross - fee) / len(records)),
        wrapper.choose_ns, wrapper.feedback_ns,
        count_trainable_and_frozen_parameters(wrapper.policy),
        memory_entries, adapter_updates,
    )


def development_comparison_contract(
    factories: Mapping[str, Callable[[], PublicPolicy]],
    configs: tuple[OODWorldConfig, ...],
    *, sessions: int = 2, erase_persistent: bool = False,
) -> dict[str, PublicAccount]:
    """Runnable DEVELOPMENT-ONLY harness; no heldout evaluation or tuning.

    New policy object for each world prevents cross-world memorization. Session
    persistence within each world is configurable. Wall timing is diagnostic,
    NOT equal-active-FLOPs evidence. The method intentionally returns no
    cross-world p-values, confidence intervals or performance promotions.
    """
    if not factories or not configs:
        raise ValueError("empty development comparison")
    if any(config.split != "development" for config in configs):
        raise ValueError("heldout evaluation prohibited by engineering harness")
    if len({(c.seed, c.scenario) for c in configs}) != len(configs):
        raise ValueError("duplicate development scenario")
    results: dict[str, PublicAccount] = {}
    for name, factory in factories.items():
        if not name or not callable(factory):
            raise ValueError("invalid control factory")
        accounts: list[PublicAccount] = []
        for config in configs:
            actor = MeasuredFactualPolicy(factory())
            traces = run_ood_sessions(actor, config, sessions=sessions,
                                      erase_persistent=erase_persistent)
            accounts.append(account_run(name, actor, traces))
        total_actions = sum(a.actions for a in accounts)
        total_net = sum(a.net_reward for a in accounts)
        mem = [a.memory_entries for a in accounts if a.memory_entries is not None]
        updates = [a.adapter_updates for a in accounts if a.adapter_updates is not None]
        params = {a.param_count for a in accounts}
        if len(params) != 1:
            raise ValueError("factory did not create same architecture across worlds")
        results[name] = PublicAccount(
            name, len(accounts), total_actions, sum(a.probes for a in accounts),
            sum(a.gross_reward for a in accounts),
            sum(a.total_fee for a in accounts),
            total_net, total_net / total_actions,
            sum(a.decision_ns for a in accounts),
            sum(a.feedback_ns for a in accounts),
            next(iter(params)), sum(mem) if mem else None,
            sum(updates) if updates else None,
        )
    return results


def build_frozen_gru_policy(*, pcma_weight: float = 0.5,
                            memory_enabled: bool = True,
                            adaptation_enabled: bool = True,
                            ) -> FrozenNeuralFactualPolicy:
    """Untrained GRU ENGINEERING adapter, NOT valid scientific comparator yet.

    For scientifically meaningful comparison, train from the exact same
    development-only factual examples as hybrid and Transformer; freeze the
    checkpoint and use a separate reserved confirmation protocol.
    """
    learner = PCMAGRURewardPredictor()
    return FrozenNeuralFactualPolicy(
        learner,
        pcma_weight=pcma_weight,
        memory_enabled=memory_enabled,
        adaptation_enabled=adaptation_enabled,
    )
