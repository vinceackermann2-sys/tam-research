from __future__ import annotations

"""#750 correction wrapper for the frozen #748 seed1 capability harness.

The scientific/model/training implementation remains byte-identical in the
internal base module. This wrapper repairs only the preregistered evaluation
materialization defect and frozen future-execution namespace metadata.
"""

from tam_research import aera_issue748_memory_capability_seed1_harness_base as _base

# Re-export the complete frozen base surface, including private helpers used by
# the unchanged CPU contract. Dunder module identity fields stay wrapper-local.
for _name, _value in vars(_base).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

BRANCH = "research/aera-issue748-memory-capability-seed1-harness"
SEED1_PREAUTH_PREFIX = "[aera-issue748-memory-capability-seed1-preauth]"
SEED1_RUN_PREFIX = "[aera-issue748-memory-capability-seed1-l4]"
SEED1_GPU_PREFIX = SEED1_RUN_PREFIX
SEED1_RESULT_PATH = "/vol/aera-capability/issue748-memory-capability-seed1/result.json"
SEED1_CHECKPOINT_DIR = "/vol/aera-capability/issue748-memory-capability-seed1/checkpoints"

_MAX_EVAL_SAMPLE_ATTEMPTS = 4096
_CAPACITY_ERROR = "records exceed chunk capacity"


def _first_valid_eval_case(*, used_sample_ids: set[int], sample_index: int, **kwargs):
    candidate = int(sample_index)
    for _ in range(_MAX_EVAL_SAMPLE_ATTEMPTS):
        if candidate in used_sample_ids:
            candidate += 1
            continue
        try:
            case = _base.gate.generate_case(sample_index=candidate, **kwargs)
        except ValueError as exc:
            if type(exc) is not ValueError or str(exc) != _CAPACITY_ERROR:
                raise
            candidate += 1
            continue
        used_sample_ids.add(candidate)
        return case
    raise RuntimeError("bounded evaluation sample-id retry exhausted")


def evaluation_cases():
    cases = []
    used_sample_ids: set[int] = set()
    combos = [
        (concurrent, distractors, correction)
        for concurrent in _base.gate.EVAL_CONCURRENT_FACTS
        for distractors in _base.gate.EVAL_DISTRACTOR_RECORDS
        for correction in (False, True)
    ]

    reps = _base.EVAL_NONRESET_CASES_PER_DISTANCE // len(combos)
    if reps * len(combos) != _base.EVAL_NONRESET_CASES_PER_DISTANCE:
        raise RuntimeError("non-reset evaluator count does not balance frozen grid")
    for distance in _base.gate.EVAL_RETENTION_DISTANCES:
        for combo_index, (concurrent, distractors, correction) in enumerate(combos):
            for rep in range(reps):
                sample_index = 1_000_000 + distance * 10_000 + combo_index * reps + rep
                cases.append(
                    _first_valid_eval_case(
                        used_sample_ids=used_sample_ids,
                        sample_index=sample_index,
                        split="eval",
                        seed=_base.gate.HELDOUT_DATA_SEED,
                        retention_distance_chunks=distance,
                        concurrent_facts=concurrent,
                        distractor_records=distractors,
                        correction=correction,
                        reset_before_query=False,
                    )
                )

    reset_reps = _base.EVAL_RESET_CASES_PER_LONG_DISTANCE // len(combos)
    if reset_reps * len(combos) != _base.EVAL_RESET_CASES_PER_LONG_DISTANCE:
        raise RuntimeError("reset evaluator count does not balance frozen grid")
    for distance in (8, 32):
        for combo_index, (concurrent, distractors, correction) in enumerate(combos):
            for rep in range(reset_reps):
                sample_index = 2_000_000 + distance * 10_000 + combo_index * reset_reps + rep
                cases.append(
                    _first_valid_eval_case(
                        used_sample_ids=used_sample_ids,
                        sample_index=sample_index,
                        split="eval",
                        seed=_base.gate.HELDOUT_DATA_SEED,
                        retention_distance_chunks=distance,
                        concurrent_facts=concurrent,
                        distractor_records=distractors,
                        correction=correction,
                        reset_before_query=True,
                    )
                )

    return tuple(cases)


# Existing base functions resolve globals in the base module. Mirror only the
# corrected metadata/evaluator into that runtime namespace so indirect callers
# (notably paired_primary_delta) use the same corrected held-out case set.
_base.BRANCH = BRANCH
_base.SEED1_PREAUTH_PREFIX = SEED1_PREAUTH_PREFIX
_base.SEED1_RUN_PREFIX = SEED1_RUN_PREFIX
_base.SEED1_GPU_PREFIX = SEED1_GPU_PREFIX
_base.SEED1_RESULT_PATH = SEED1_RESULT_PATH
_base.SEED1_CHECKPOINT_DIR = SEED1_CHECKPOINT_DIR
_base.evaluation_cases = evaluation_cases


def harness_protocol_snapshot():
    return _base.harness_protocol_snapshot()
