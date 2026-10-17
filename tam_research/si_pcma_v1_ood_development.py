"""Development-only factual data bridge for the separate PCMA-v1 OOD world.

NO training or held-out scientific results are run by this module. A future
study must preregister independent seeds, budgets, thresholds and controls.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence

from tam_research.si_pcma_v1 import FactualTransition
from tam_research.si_pcma_v1_factual_learning import (
    CausalRewardSample, causal_samples,
)
from tam_research.si_pcma_v1_ood_world import (
    OODRuleWorld, OODWorldConfig,
)


@dataclass(frozen=True)
class DevelopmentDataset:
    """Only public executed-action data, with train rule split validation."""
    records: tuple[FactualTransition, ...]
    samples: tuple[CausalRewardSample, ...]
    world_count: int
    sessions_per_world: int


def collect_development_factual_data(
    configs: Sequence[OODWorldConfig],
    *, exploration_fixture_seed: int,
    sessions: int = 2,
) -> DevelopmentDataset:
    """Deny held-out environments even if caller mistakenly supplies them.

    Each generated sample's current outcome is ONLY its separate target.
    On-policy OOD evaluation will be handled by a separate runner, not used
    for fitting/tuning. Deterministic fixture RNG is NOT a scientific seed.
    """
    if not configs or type(sessions) is not int or not 1 <= sessions <= 32:
        raise ValueError("invalid development collection budget")
    if type(exploration_fixture_seed) is not int or exploration_fixture_seed < 0:
        raise ValueError("invalid development fixture seed")
    if any(c.split != "development" for c in configs):
        raise ValueError("heldout rules cannot supply training data")
    if len({c.seed for c in configs}) != len(configs):
        raise ValueError("duplicate development world seed")

    rng = random.Random(exploration_fixture_seed)
    records: list[FactualTransition] = []
    for index, cfg in enumerate(configs):
        world = OODRuleWorld(cfg)
        for session in range(sessions):
            if session:
                world.restart_session()
            for _ in range(cfg.session_horizon):
                action = rng.randrange(4)
                actual = world.step(action)
                # Strictly increasing logical times across *worlds*, with
                # fresh session boundaries from actual.done.
                records.append(FactualTransition(
                    before=actual.before, action=actual.action,
                    after=actual.after, reward=actual.reward, fee=actual.fee,
                    time=index * 1_000_000 + actual.time,
                    provenance=f"dev-world-{index}/{actual.provenance}",
                    done=actual.done,
                ))
    return DevelopmentDataset(
        records=tuple(records),
        samples=causal_samples(records),
        world_count=len(configs),
        sessions_per_world=sessions,
    )
