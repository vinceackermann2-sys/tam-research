from tam_research.pretrain_mixture import (
    ASSEMBLY_VERSION,
    PRETRAIN_SOURCES,
    TOTAL_TRAIN_TOKENS,
    TRAIN_INTERLEAVE_CHUNK_TOKENS,
    weighted_interleave_schedule,
)


def test_production_schedule_preserves_exact_2b_quotas():
    counts = [source.train_tokens for source in PRETRAIN_SOURCES]
    schedule = weighted_interleave_schedule(counts, TRAIN_INTERLEAVE_CHUNK_TOKENS)
    assert ASSEMBLY_VERSION == 2
    assert len(schedule) * TRAIN_INTERLEAVE_CHUNK_TOKENS == TOTAL_TRAIN_TOKENS
    for index, count in enumerate(counts):
        assert schedule.count(index) * TRAIN_INTERLEAVE_CHUNK_TOKENS == count


def test_weighted_schedule_distributes_small_domains_early_and_late():
    counts = [9, 3, 2]
    schedule = weighted_interleave_schedule(counts, 1)
    assert len(schedule) == 14
    assert [schedule.count(i) for i in range(3)] == counts
    # Every source appears before the first quarter is over and again after halfway.
    for source in range(3):
        positions = [i for i, value in enumerate(schedule) if value == source]
        assert positions[0] < len(schedule) // 4 + 1
        assert positions[-1] >= len(schedule) // 2


def test_schedule_rejects_non_integral_chunks():
    try:
        weighted_interleave_schedule([10, 5], 4)
    except ValueError as exc:
        assert "multiple" in str(exc)
    else:
        raise AssertionError("expected ValueError")
