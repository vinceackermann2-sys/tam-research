from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from tam_research.posttrain100_data import _last_user_assistant, _stack_inputs
from tam_research.pretrain_mixture import (
    PRETRAIN_SOURCES,
    TOTAL_TRAIN_TOKENS,
    TOTAL_VAL_TOKENS,
)


def test_pretrain_mixture_is_exactly_two_billion_tokens() -> None:
    assert TOTAL_TRAIN_TOKENS == 2_000_000_000
    assert TOTAL_VAL_TOKENS == 5_000_000
    assert sum(s.train_tokens for s in PRETRAIN_SOURCES) == TOTAL_TRAIN_TOKENS
    assert {s.name for s in PRETRAIN_SOURCES} == {
        "fineweb_edu",
        "finemath_4plus",
        "stackv2_edu_open",
        "cosmopedia_openstax",
        "cosmopedia_stanford",
        "arxiv_open_papers",
    }


def test_code_source_has_extra_quality_floor() -> None:
    stack = next(s for s in PRETRAIN_SOURCES if s.name == "stackv2_edu_open")
    assert stack.min_score == 3.0


def test_posttrain_input_storage_is_cuda_indexable() -> None:
    arrays = [np.asarray([1, 2, 3], dtype=np.uint16), np.asarray([4, 5, 6], dtype=np.uint16)]
    stacked = _stack_inputs(arrays)
    assert stacked.dtype == np.int32
    assert stacked.shape == (2, 3)


def test_smol_turn_extraction_uses_last_user_before_last_assistant() -> None:
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "latest"},
        {"role": "assistant", "content": "new answer"},
    ]
    assert _last_user_assistant(messages) == ("latest", "new answer")


def test_full100m_budget_is_fixed_and_not_cli_overridable() -> None:
    source = Path("modal_full100m_app.py").read_text()
    tree = ast.parse(source)
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"MAX_GPU_SECONDS", "PRETRAIN_TOKENS"}
    }
    assert assignments["MAX_GPU_SECONDS"] == 11_700
    assert assignments["PRETRAIN_TOKENS"] == 2_000_000_000
    assert "max_gpu_seconds" not in source.split("def main(", 1)[1].split("):", 1)[0]
