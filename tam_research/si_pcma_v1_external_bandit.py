"""Independent published action-dependent acquisition task: POPGym bandit.

Bounded nonconfirmatory DEVELOPMENT screen only; no SI training/scores.
Only actual chosen action, public binary payout and normalized reward enter
controllers. Never pass reset/step `info` (contains privileged true arm
probabilities!) to a controller, and never read .bandits, .get_state(), or
an unchosen action's reward.

POPGym 1.0.7 MultiarmedBanditEasy has 10 arms, 200 decisions and a newly
sampled hidden payout probability for each arm on every episode. This is a
useful cheap-baseline gate for PCMA's information acquisition story, not a
proof of long-horizon planning, artificial superintelligence or neural gain.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
import math
import random
from typing import Literal

PACKAGE_VERSION = "1.0.7"
FIXTURE_EPISODES = tuple(range(41, 49))
HORIZON = 200
ARMS = 10
Method = Literal["fixed", "round_robin", "ucb1", "beta_thompson"]
METHODS: tuple[Method, ...] = (
    "fixed", "round_robin", "ucb1", "beta_thompson"
)


@dataclass(frozen=True)
class PublicBanditFeedback:
    """ONE actually executed arm's outcome; no alternate-arm information."""
    action: int
    observed_payout: int
    reward: float
    step: int


class FactualOnlineBanditPolicy:
    """Transparent, frozen classical acquisition baselines.

    Counts and wins are built ONLY from the feedback actually delivered by
    the published environment. Each policy resets per independent world;
    Thompson's RNG is separate from the environment's RNG. No supervision
    from simulator-private probabilities or optimal-action labels.
    """

    def __init__(self, method: Method, *, fixture_rng_seed: int = 0):
        if method not in METHODS:
            raise ValueError("unrecognized locked classical baseline")
        if type(fixture_rng_seed) is not int or fixture_rng_seed < 0:
            raise ValueError("invalid public algorithm fixture RNG")
        self.method = method
        self._rng = random.Random(fixture_rng_seed)
        self._pending: int | None = None
        self._time = 0
        self.counts = [0] * ARMS
        self.wins = [0] * ARMS
        self.history: list[PublicBanditFeedback] = []

    def choose(self, public_observation: int) -> int:
        if type(public_observation) is not int or public_observation not in (0, 1):
            raise ValueError("only published binary public observation")
        if self._pending is not None:
            raise RuntimeError("previous chosen action needs feedback")
        if self._time >= HORIZON:
            raise RuntimeError("fixed developer episode complete")
        if self.method == "fixed":
            action = 0
        elif self.method == "round_robin":
            action = self._time % ARMS
        elif self.method == "ucb1":
            unseen = [a for a, count in enumerate(self.counts) if count == 0]
            if unseen:
                action = unseen[0]
            else:
                t = self._time
                action = max(
                    range(ARMS),
                    key=lambda a: (
                        self.wins[a]/self.counts[a]
                        + math.sqrt(2*math.log(t)/self.counts[a]),
                        -a,
                    ),
                )
        else:
            action = max(
                range(ARMS),
                key=lambda a: (
                    self._rng.betavariate(1+self.wins[a],
                                        1+self.counts[a]-self.wins[a]),
                    -a,
                ),
            )
        self._pending = action
        return action

    def observe(self, feedback: PublicBanditFeedback) -> None:
        if type(feedback) is not PublicBanditFeedback or self._pending is None:
            raise ValueError("only pending factual feedback may update controller")
        if (type(feedback.action) is not int or feedback.action != self._pending
                or type(feedback.observed_payout) is not int
                or feedback.observed_payout not in (0, 1)
                or type(feedback.step) is not int
                or feedback.step != self._time + 1
                or type(feedback.reward) is not float
                or not math.isfinite(feedback.reward)
                or abs(feedback.reward - (
                    (2*feedback.observed_payout - 1)/HORIZON
                )) > 1e-12):
            raise ValueError("bad, future or nonfactual payoff feedback")
        arm = feedback.action
        self.counts[arm] += 1
        self.wins[arm] += feedback.observed_payout
        self.history.append(feedback)
        self._time += 1
        self._pending = None


@dataclass(frozen=True)
class DeveloperBanditEpisode:
    method: Method
    episode_id: int
    actions: int
    total_normalized_return: float
    distinct_arms_sampled: int
    feedback_count: int


@dataclass(frozen=True)
class DeveloperBanditSummary:
    method: Method
    episodes: int
    mean_episode_return: float
    min_episode_return: float
    max_episode_return: float
    mean_distinct_arms_sampled: float


@dataclass(frozen=True)
class DeveloperBanditReadout:
    status: str
    package_version: str
    published_environment: str
    training_models_evaluated: int
    science_seeds_reserved: int
    heldout_worlds_evaluated: int
    episode_count_per_control: int
    action_opportunities_per_episode: int
    summaries: tuple[DeveloperBanditSummary, ...]
    original_v26_verdict: str
    pcma_verdict: str
    may_claim_architecture_gain: bool


def run_development_bandit_episode(
    method: Method, episode_id: int,
) -> DeveloperBanditEpisode:
    if method not in METHODS or type(episode_id) is not int or (
            episode_id not in FIXTURE_EPISODES):
        raise ValueError("only fixed development episode IDs allowed")
    if version("popgym") != PACKAGE_VERSION:
        raise RuntimeError("published benchmark version not pinned")
    import popgym
    world = popgym.envs.MultiarmedBanditEasy()
    policy = FactualOnlineBanditPolicy(
        method, fixture_rng_seed=1000 + episode_id)
    try:
        obs, _discard_private_reset_info = world.reset(seed=episode_id)
        score = 0.0
        for step in range(1, HORIZON + 1):
            # NEVER give the policy the env or its info dictionary, which
            # reveals all actual private arm probabilities in this task.
            action = policy.choose(int(obs))
            obs, reward, terminated, truncated, _discard_private_step_info = (
                world.step(action))
            payout = int(obs)
            policy.observe(PublicBanditFeedback(
                action=action, observed_payout=payout,
                reward=float(reward), step=step,
            ))
            score += float(reward)
            if step < HORIZON and (terminated or truncated):
                raise RuntimeError("published task terminated early")
            if step == HORIZON and not (terminated or truncated):
                raise RuntimeError("published task did not end at horizon")
        if not math.isfinite(score):
            raise RuntimeError("nonfinite developer reward")
        if policy._pending is not None or sum(policy.counts) != HORIZON:
            raise RuntimeError("feedback accounting lost an executed action")
        return DeveloperBanditEpisode(
            method=method, episode_id=episode_id, actions=HORIZON,
            total_normalized_return=score,
            distinct_arms_sampled=sum(x > 0 for x in policy.counts),
            feedback_count=len(policy.history),
        )
    finally:
        world.close()


def published_bandit_development_screen() -> DeveloperBanditReadout:
    """Descriptive independent episode means, no counterfactual simulator info.

    The same reset integer gives each policy the same *initial world*
    distribution, not identical latent random draws under divergent choices.
    No causal per-action comparisons, confidence claims or automatic GO gates.
    """
    rows: dict[str, tuple[DeveloperBanditEpisode, ...]] = {}
    for method in METHODS:
        rows[method] = tuple(
            run_development_bandit_episode(method, seed)
            for seed in FIXTURE_EPISODES
        )
    stats: list[DeveloperBanditSummary] = []
    for method in METHODS:
        data = rows[method]
        returns = [r.total_normalized_return for r in data]
        stats.append(DeveloperBanditSummary(
            method=method, episodes=len(data),
            mean_episode_return=sum(returns)/len(returns),
            min_episode_return=min(returns),
            max_episode_return=max(returns),
            mean_distinct_arms_sampled=(
                sum(r.distinct_arms_sampled for r in data)/len(data)
            ),
        ))
    return DeveloperBanditReadout(
        status="NONCONFIRMATORY_PUBLISHED_DEVELOPMENT_SCREEN",
        package_version=PACKAGE_VERSION,
        published_environment="POPGym MultiarmedBanditEasy",
        training_models_evaluated=0, science_seeds_reserved=0,
        heldout_worlds_evaluated=0,
        episode_count_per_control=len(FIXTURE_EPISODES),
        action_opportunities_per_episode=HORIZON,
        summaries=tuple(stats),
        original_v26_verdict="NO_GO",
        pcma_verdict="UNKNOWN",
        may_claim_architecture_gain=False,
    )


def main() -> None:
    print("EXTERNALLY AUTHORED ACTION-DEPENDENT ACQUISITION DEVELOPMENT SCREEN")
    print("NO SI MODEL TRAINING / NO HELDOUT / NO SCIENTIFIC CLAIM")
    print(json.dumps(asdict(published_bandit_development_screen()),
                     sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
