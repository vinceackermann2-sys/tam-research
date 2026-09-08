from __future__ import annotations

"""CPU-only fixtures for the #746 bounded persistent-memory capability gate."""

from dataclasses import asdict, dataclass
import math
import random
from typing import Sequence

PROTOCOL_VERSION = "aera-memory-capability-v1"
RESEARCH_ISSUE = 746
SOURCE_MAIN = "7c6d3dd187ccb80bb93eade170dfb717549b8eeb"
SOURCE_TREE = "91979a36c56d36e61d5a4e7bcc798f56ef4b61b4"
VOCAB_SIZE = 50_257
CHUNK_SIZE = 256
PAD, WRITE, UPDATE, QUERY, ANSWER, RESET, SEP, UNKNOWN = range(8)
KEY_BASE, KEY_COUNT = 1_000, 128
VALUE_BASE, VALUE_COUNT = 2_000, 128
DISTRACTOR_KEY_BASE, DISTRACTOR_KEY_COUNT = 3_000, 128
DISTRACTOR_VALUE_BASE, DISTRACTOR_VALUE_COUNT = 4_000, 128
NOISE_BASE, NOISE_COUNT = 5_000, 1_024
PAIR_SPLIT_MODULUS = 5
EVAL_PAIR_RESIDUE = 4
TRAIN_RETENTION_DISTANCES = (1, 2, 4, 8)
EVAL_RETENTION_DISTANCES = (2, 8, 32)
TRAIN_CONCURRENT_FACTS = (1, 2, 4)
EVAL_CONCURRENT_FACTS = (1, 4, 8)
TRAIN_DISTRACTOR_RECORDS = (0, 4, 16)
EVAL_DISTRACTOR_RECORDS = (0, 16, 32)
TRAIN_DATA_SEED = 746_101
HELDOUT_DATA_SEED = 746_201
BOOTSTRAP_SEED = 746_301
SCIENTIFIC_MODEL_SEEDS = (17_641, 27_641, 37_641)
CPU_TEST_SEED = 746_999
TOKEN_BUDGET_PER_TRAINED_VARIANT = 16_777_216
CHECKPOINT_INTERVAL_TOKENS = 2_097_152
MAX_GPU_SECONDS_PER_VARIANT_SEED = 3_600
MAX_TOTAL_TRAINED_VARIANT_SEED_RUNS = 12
TRAINED_VARIANTS = ("A_backbone", "B_backbone_plus_memory", "C_backbone_plus_routing", "D_combined")
LONG_DISTANCE_ACCURACY_GAIN_MIN = 0.10
PAIRED_CI_LOWER_ACCURACY_GAIN_MIN = 0.05
CORRECTION_ACCURACY_GAIN_MIN = 0.10
STALE_VALUE_ERROR_REDUCTION_MIN = 0.10
ROUTING_CONTROL_ACCURACY_MARGIN_MIN = 0.05
SESSION_RESET_LEAKAGE_MAX = 0.01
SIMPLE_RETRIEVAL_DOMINANCE_MARGIN = 0.05
MEMORY_INFERENCE_LATENCY_MULTIPLIER_MAX = 2.0
MEMORY_TRAIN_TIME_MULTIPLIER_MAX = 1.5
PARAMETER_DELTA_FRACTION_MAX = 0.05
BOOTSTRAP_RESAMPLES = 10_000
GPU_AUTHORIZED = False
SCIENTIFIC_TRAINING_AUTHORIZED = False
FRESH_SCIENTIFIC_SEED_AUTHORIZED = False
ARCHITECTURE_FREEZE_AUTHORIZED = False
SCALING_AUTHORIZED = False
BREAKTHROUGH_PROVEN = False

@dataclass(frozen=True)
class VariantSpec:
    name: str
    persistent_learned_memory: bool
    adaptive_routing: bool
    recurrent_state: bool
    simple_retrieval_cache: bool
    fixed_depth: bool
    latent_reasoning: bool = False
    multimodality: bool = False
    replay: bool = False
    block_generation: bool = False

VARIANTS = (
    VariantSpec("A_backbone", False, False, False, False, True),
    VariantSpec("B_backbone_plus_memory", True, False, False, False, True),
    VariantSpec("C_backbone_plus_routing", False, True, False, False, False),
    VariantSpec("D_combined", True, True, False, False, False),
    VariantSpec("E_simple_retrieval", False, False, False, True, True),
    VariantSpec("F_backbone_plus_recurrent_state", False, False, True, False, True),
)

@dataclass(frozen=True)
class MemoryCase:
    split: str
    sample_index: int
    retention_distance_chunks: int
    concurrent_facts: int
    distractor_records: int
    correction: bool
    reset_before_query: bool
    chunks: tuple[tuple[int, ...], ...]
    target_key: int
    original_value: int
    latest_value: int
    expected_answer: int
    answer_chunk_index: int
    answer_token_index: int
    stale_value: int | None

    @property
    def flat_tokens(self) -> tuple[int, ...]:
        return tuple(token for chunk in self.chunks for token in chunk)


def _pair_is_eval(key_index: int, value_index: int) -> bool:
    return (key_index + value_index) % PAIR_SPLIT_MODULUS == EVAL_PAIR_RESIDUE


def pair_split(key_token: int, value_token: int) -> str:
    ki = key_token - KEY_BASE
    vi = value_token - VALUE_BASE
    if not 0 <= ki < KEY_COUNT or not 0 <= vi < VALUE_COUNT:
        raise ValueError("pair outside capability ranges")
    return "eval" if _pair_is_eval(ki, vi) else "train"


def _sample_pair(rng: random.Random, split: str) -> tuple[int, int]:
    if split not in {"train", "eval"}:
        raise ValueError("split must be train or eval")
    for _ in range(10_000):
        ki, vi = rng.randrange(KEY_COUNT), rng.randrange(VALUE_COUNT)
        if _pair_is_eval(ki, vi) == (split == "eval"):
            return KEY_BASE + ki, VALUE_BASE + vi
    raise RuntimeError("failed to sample pair")


def _sample_new_value_for_key(rng: random.Random, *, key: int, split: str, not_value: int) -> int:
    ki = key - KEY_BASE
    for _ in range(10_000):
        vi = rng.randrange(VALUE_COUNT)
        value = VALUE_BASE + vi
        if value != not_value and _pair_is_eval(ki, vi) == (split == "eval"):
            return value
    raise RuntimeError("failed to sample corrected value")


def _noise(rng: random.Random) -> int:
    return NOISE_BASE + rng.randrange(NOISE_COUNT)


def _pack_chunk(records: Sequence[Sequence[int]], rng: random.Random) -> tuple[int, ...]:
    out: list[int] = []
    for record in records:
        gap = rng.randrange(8)
        if len(out) + gap + len(record) > CHUNK_SIZE:
            raise ValueError("records exceed chunk capacity")
        out.extend(_noise(rng) for _ in range(gap))
        out.extend(int(x) for x in record)
    out.extend(_noise(rng) for _ in range(CHUNK_SIZE - len(out)))
    return tuple(out)


def _derive_rng_seed(seed: int, sample_index: int, distance: int) -> int:
    return ((seed * 1_000_003) ^ (sample_index * 97_409) ^ (distance * 65_537) ^ 0x746746) & ((1 << 63) - 1)


def generate_case(*, split: str, seed: int, sample_index: int, retention_distance_chunks: int,
                  concurrent_facts: int, distractor_records: int, correction: bool,
                  reset_before_query: bool) -> MemoryCase:
    if split not in {"train", "eval"}:
        raise ValueError("split must be train or eval")
    if retention_distance_chunks < 1:
        raise ValueError("retention distance must be >=1")
    if not 1 <= concurrent_facts <= 8:
        raise ValueError("concurrent_facts must be in [1,8]")
    if not 0 <= distractor_records <= 32:
        raise ValueError("distractor_records must be in [0,32]")
    if reset_before_query and retention_distance_chunks < 2:
        raise ValueError("reset requires distance >=2")
    rng = random.Random(_derive_rng_seed(seed, sample_index, retention_distance_chunks))
    pairs: list[tuple[int, int]] = []
    used_keys: set[int] = set()
    while len(pairs) < concurrent_facts:
        pair = _sample_pair(rng, split)
        if pair[0] not in used_keys:
            used_keys.add(pair[0]); pairs.append(pair)
    target_key, original_value = pairs[0]
    latest_value, stale_value = original_value, None
    n_chunks = retention_distance_chunks + 1
    records: list[list[list[int]]] = [[] for _ in range(n_chunks)]
    for key, value in pairs:
        records[0].append([WRITE, key, value, SEP])
    if correction:
        stale_value = original_value
        latest_value = _sample_new_value_for_key(rng, key=target_key, split=split, not_value=original_value)
        correction_chunk = min(max(1, retention_distance_chunks // 2), retention_distance_chunks - 1)
        records[correction_chunk].append([UPDATE, target_key, latest_value, SEP])
    available = list(range(1, retention_distance_chunks)) or [0]
    for i in range(distractor_records):
        ci = available[i % len(available)]
        dk = DISTRACTOR_KEY_BASE + rng.randrange(DISTRACTOR_KEY_COUNT)
        dv = DISTRACTOR_VALUE_BASE + rng.randrange(DISTRACTOR_VALUE_COUNT)
        records[ci].append([WRITE, dk, dv, SEP])
    if reset_before_query:
        records[retention_distance_chunks - 1].append([RESET, SEP])
        expected = UNKNOWN
    else:
        expected = latest_value
    records[retention_distance_chunks].append([QUERY, target_key, ANSWER, expected, SEP])
    chunks = tuple(_pack_chunk(rs, rng) for rs in records)
    final = chunks[-1]
    qpos = [i for i, token in enumerate(final) if token == QUERY]
    if len(qpos) != 1:
        raise RuntimeError("expected exactly one query marker")
    answer_index = qpos[0] + 3
    if answer_index >= CHUNK_SIZE or final[answer_index] != expected:
        raise RuntimeError("query answer mismatch")
    return MemoryCase(split, sample_index, retention_distance_chunks, concurrent_facts,
                      distractor_records, correction, reset_before_query, chunks,
                      target_key, original_value, latest_value, expected,
                      retention_distance_chunks, answer_index, stale_value)


def exact_match_accuracy(gold: Sequence[int], pred: Sequence[int]) -> float:
    if len(gold) != len(pred) or not gold:
        raise ValueError("gold/pred must be non-empty and equal length")
    return sum(int(g == p) for g, p in zip(gold, pred)) / len(gold)


def mean_answer_nll(gold_log_probabilities: Sequence[float]) -> float:
    if not gold_log_probabilities or any(not math.isfinite(x) or x > 1e-9 for x in gold_log_probabilities):
        raise ValueError("expected finite log probabilities <=0")
    return -sum(gold_log_probabilities) / len(gold_log_probabilities)


def stale_value_error_rate(predictions: Sequence[int], stale_values: Sequence[int | None]) -> float:
    if len(predictions) != len(stale_values) or not predictions:
        raise ValueError("arrays must be non-empty and equal length")
    eligible = [(p, s) for p, s in zip(predictions, stale_values) if s is not None]
    return 0.0 if not eligible else sum(int(p == s) for p, s in eligible) / len(eligible)


def session_reset_leakage_rate(predictions: Sequence[int], pre_reset_values: Sequence[int]) -> float:
    if len(predictions) != len(pre_reset_values) or not predictions:
        raise ValueError("arrays must be non-empty and equal length")
    return sum(int(p == v) for p, v in zip(predictions, pre_reset_values)) / len(predictions)


def paired_bootstrap_accuracy_delta(a_correct: Sequence[int], b_correct: Sequence[int], *,
                                    seed: int = BOOTSTRAP_SEED,
                                    resamples: int = BOOTSTRAP_RESAMPLES) -> tuple[float, float, float]:
    if len(a_correct) != len(b_correct) or not a_correct or resamples < 40:
        raise ValueError("invalid paired bootstrap input")
    n = len(a_correct)
    delta = sum(b_correct) / n - sum(a_correct) / n
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        draws.append(sum(b_correct[i] - a_correct[i] for i in (rng.randrange(n) for _ in range(n))) / n)
    draws.sort()
    return delta, draws[int(0.025 * (resamples - 1))], draws[int(0.975 * (resamples - 1))]


def protocol_snapshot() -> dict[str, object]:
    return {
        "version": PROTOCOL_VERSION, "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN, "source_tree": SOURCE_TREE,
        "task": {"vocab_size": VOCAB_SIZE, "chunk_size": CHUNK_SIZE,
                 "pair_split": "(key_index + value_index) % 5 == 4 => eval",
                 "train_retention_distances": list(TRAIN_RETENTION_DISTANCES),
                 "eval_retention_distances": list(EVAL_RETENTION_DISTANCES),
                 "train_concurrent_facts": list(TRAIN_CONCURRENT_FACTS),
                 "eval_concurrent_facts": list(EVAL_CONCURRENT_FACTS),
                 "train_distractor_records": list(TRAIN_DISTRACTOR_RECORDS),
                 "eval_distractor_records": list(EVAL_DISTRACTOR_RECORDS),
                 "correction_cases_required": True, "session_reset_negative_control_required": True,
                 "heldout_compositional_pairs_required": True},
        "seeds": {"train_data_seed": TRAIN_DATA_SEED, "heldout_data_seed": HELDOUT_DATA_SEED,
                  "bootstrap_seed": BOOTSTRAP_SEED, "scientific_model_seeds": list(SCIENTIFIC_MODEL_SEEDS),
                  "cpu_test_seed": CPU_TEST_SEED, "scientific_model_seeds_consumed": False},
        "variants": [asdict(v) for v in VARIANTS],
        "training_budget": {"tokens_per_trained_variant_seed": TOKEN_BUDGET_PER_TRAINED_VARIANT,
                            "checkpoint_interval_tokens": CHECKPOINT_INTERVAL_TOKENS,
                            "max_gpu_seconds_per_variant_seed": MAX_GPU_SECONDS_PER_VARIANT_SEED,
                            "max_total_trained_variant_seed_runs": MAX_TOTAL_TRAINED_VARIANT_SEED_RUNS,
                            "trained_variants": list(TRAINED_VARIANTS), "seed1_fail_fast_screen": True,
                            "seeds_2_and_3_only_after_seed1_primary_gate": True,
                            "equal_token_comparison_required": True, "equal_gpu_time_comparison_required": True,
                            "equal_gpu_time_checkpoint_rule": "latest checkpoint not exceeding minimum cumulative GPU time shared by A-D",
                            "parameter_delta_fraction_max": PARAMETER_DELTA_FRACTION_MAX},
        "primary_thresholds": {"long_distance_accuracy_gain_B_minus_A_min": LONG_DISTANCE_ACCURACY_GAIN_MIN,
                               "paired_95ci_lower_accuracy_gain_min": PAIRED_CI_LOWER_ACCURACY_GAIN_MIN,
                               "correction_accuracy_gain_B_minus_A_min": CORRECTION_ACCURACY_GAIN_MIN,
                               "stale_value_error_reduction_A_minus_B_min": STALE_VALUE_ERROR_REDUCTION_MIN,
                               "B_minus_C_long_accuracy_margin_min": ROUTING_CONTROL_ACCURACY_MARGIN_MIN,
                               "session_reset_leakage_max": SESSION_RESET_LEAKAGE_MAX,
                               "simple_retrieval_dominance_margin": SIMPLE_RETRIEVAL_DOMINANCE_MARGIN,
                               "memory_inference_latency_multiplier_max": MEMORY_INFERENCE_LATENCY_MULTIPLIER_MAX,
                               "memory_train_time_multiplier_max": MEMORY_TRAIN_TIME_MULTIPLIER_MAX,
                               "bootstrap_resamples": BOOTSTRAP_RESAMPLES},
        "decision_tree": {
            "screen_fail": "after seed1 stop if B-A long-distance accuracy gain <0.10 or paired 95% CI lower bound <=0.05",
            "memory_specificity_fail": "stop if B-C long-distance accuracy margin <0.05 after capability screen passes",
            "safety_fail": "stop if B session-reset leakage exceeds 0.01",
            "confirmation": "only after seed1 clears gates, consume seeds2/3; require pooled primary gate plus positive B-A long-distance gain on each seed",
            "simpler_control_dominates": "learned memory not justified if E exceeds B by >=0.05 on both long-distance and correction accuracy while E latency <= B",
            "costly_positive": "scientifically positive but not systems-preservation positive if B/A inference latency >2.0 or equal-token training time >1.5",
            "preserve_for_systems": "requires confirmed memory utility, no reset failure, no simpler-control dominance, and cost multipliers within frozen limits"},
        "metrics": {"primary": "answer-token exact-match at eval distances 8 and 32 pooled equally by distance",
                    "answer_token_nll": True, "retention_curve": True, "correction_accuracy": True,
                    "stale_value_error_rate": True, "distractor_interference_error_rate": True,
                    "session_reset_leakage_rate": True, "concurrent_fact_curve": True,
                    "paired_bootstrap_ci": True, "training_gpu_time": True, "inference_latency": True,
                    "parameter_count": True, "peak_memory": True},
        "authority": {"gpu_authorized": GPU_AUTHORIZED,
                      "scientific_training_authorized": SCIENTIFIC_TRAINING_AUTHORIZED,
                      "fresh_scientific_seed_authorized": FRESH_SCIENTIFIC_SEED_AUTHORIZED,
                      "architecture_freeze_authorized": ARCHITECTURE_FREEZE_AUTHORIZED,
                      "scaling_authorized": SCALING_AUTHORIZED, "breakthrough_proven": BREAKTHROUGH_PROVEN},
    }
