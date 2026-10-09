from __future__ import annotations

"""#1377: isolated tiny CPU-only Transformer and two-hop pointer training arms.

NOT an LLM or a scientific 100M experiment. Model forward accepts solely the
three model-visible fields of SealedModelView; hidden labels and gold pointers
are available ONLY to the TRAIN loss/evaluator outside forward().
"""

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from .chm_v3_counterfactual_memory_suite_1358 import ANSWER_IDS
from .chm_v3_counterfactual_model_view_1365 import (
    PairedEpisode, SealedModelView, oracle_read, seal_model_view,
)
from .chm_v3_counterfactual_multiset_suite_1373 import generate_episode

VOCAB_SIZE = 8192
DIM = 32
N_HEAD = 4
FFN_DIM = 64
MAX_TOKENS = 384
INIT_SEED = 1_377_005
SCIENTIFIC_SEED_CONSUMED = 2_013_161
GPU_AUTHORIZED = False
ORIGINAL_SCIENTIFIC_STATUS = "CHM_V3_100M_DAEC_STAGE_C_STOP"
# Hashing avoids train/test vocabulary fitting, but hash collisions remain a
# possible confound, measured in the forthcoming one-shot CPU report.
TOKEN_PATTERN = re.compile(
    r"CODE-[0-9]+|V3-[A-Za-z0-9-]+|\[[0-9]{4}\]|[A-Za-z]+|[0-9]+|[^\s]"
)


@dataclass(frozen=True)
class EncodedView:
    token_ids: tuple[int, ...]
    memory_length: int
    # Each CODE token is discovered from the visible memory TEXT, not oracle.
    code_tokens: tuple[tuple[int, int], ...]
    # Physical memory record line index -> token position from TEXT, not gold.
    line_anchors: tuple[tuple[int, int], ...]


def _token_id(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return 2 + int.from_bytes(digest[:8], "big") % (VOCAB_SIZE - 2)


def encode_sealed_view(view: SealedModelView) -> EncodedView:
    if not isinstance(view, SealedModelView):
        raise TypeError("model forward accepts only SealedModelView")
    if view.answer_options != tuple(f"CODE-{x}" for x in ANSWER_IDS):
        raise ValueError("unexpected fixed answer options")
    if not view.query or not view.memory_text or len(view.memory_text) > 12000:
        raise ValueError("invalid sealed query/memory text")
    tokens: list[str] = []
    codes: list[tuple[int, int]] = []
    anchors: list[tuple[int, int]] = []
    for line in view.memory_text.splitlines():
        parts = TOKEN_PATTERN.findall(line)
        if not parts:
            continue
        first = re.fullmatch(r"\[([0-9]{4})\]", parts[0])
        if first is None:
            raise ValueError("memory record lacks serialized position")
        anchor = int(first.group(1))
        if any(pos == anchor for pos, _ in anchors):
            raise ValueError("duplicate visible memory position")
        anchors.append((anchor, len(tokens)))
        for part in parts:
            if part.startswith("CODE-") and part[5:].isdigit():
                codes.append((len(tokens), int(part[5:])))
            tokens.append(part)
    n_mem = len(tokens)
    if not anchors or not codes:
        raise ValueError("no parseable completed-memory code values")
    tokens.extend(("QUESTION", *TOKEN_PATTERN.findall(view.query), "ANSWER"))
    if n_mem >= len(tokens) or len(tokens) > MAX_TOKENS:
        raise ValueError("encoded memory/query exceeds CPU limit")
    return EncodedView(
        token_ids=tuple(_token_id(token) for token in tokens),
        memory_length=n_mem, code_tokens=tuple(codes), line_anchors=tuple(anchors),
    )


def _positional_encoding(count: int, d_model: int) -> torch.Tensor:
    pos = torch.arange(count, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    result = torch.zeros(count, d_model)
    result[:, 0::2] = torch.sin(pos * div)
    result[:, 1::2] = torch.cos(pos * div)
    return result


class CausalBackbone(nn.Module):
    """Identical conventional decoder-only token encoder for all three arms."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, DIM)
        layer = nn.TransformerEncoderLayer(
            d_model=DIM, nhead=N_HEAD, dim_feedforward=FFN_DIM,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=False,
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(DIM)
        self.answer_head = nn.Linear(DIM, len(ANSWER_IDS) + 1)

    def forward(self, data: EncodedView) -> torch.Tensor:
        ids = torch.tensor(data.token_ids, dtype=torch.long)
        n = len(data.token_ids)
        hidden = self.embedding(ids) + _positional_encoding(n, DIM)
        causal_mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        hidden = self.decoder(hidden.unsqueeze(0), mask=causal_mask).squeeze(0)
        return self.final_norm(hidden)


@dataclass
class TinyOutput:
    answer_logits: torch.Tensor
    # Empty for pure decoder baseline; never original oracle positions.
    first_scores: torch.Tensor | None
    second_scores: torch.Tensor | None
    first_candidate_token_positions: tuple[int, ...]
    second_candidate_token_positions: tuple[int, ...]


class TinyDecoderOnly(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(INIT_SEED)
            self.base = CausalBackbone()

    def forward(self, view: SealedModelView) -> TinyOutput:
        encoded = encode_sealed_view(view)
        hidden = self.base(encoded)
        return TinyOutput(
            answer_logits=self.base.answer_head(hidden[-1]),
            first_scores=None, second_scores=None,
            first_candidate_token_positions=(),
            second_candidate_token_positions=(),
        )


class TinyTwoHopPointer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(INIT_SEED)
            self.base = CausalBackbone()
            self.query_projection = nn.Linear(DIM, DIM, bias=False)
            self.key_projection = nn.Linear(DIM, DIM, bias=False)

    def forward(self, view: SealedModelView) -> TinyOutput:
        encoded = encode_sealed_view(view)
        h = self.base(encoded)
        memory = h[:encoded.memory_length]
        query = h[-1]
        keys = self.key_projection(memory)
        first_scores = (keys @ self.query_projection(query)) / math.sqrt(DIM)
        first_weights = torch.softmax(first_scores, dim=0)
        context = first_weights @ memory
        # Second hop depends on the model's *own predicted* soft first hop;
        # NO gold pointer/payload enters either forward or hard evaluation.
        second_query = self.query_projection(query + context)
        memory_second = (keys @ second_query) / math.sqrt(DIM)
        code_positions = tuple(pos for pos, _ in encoded.code_tokens)
        code_ids = tuple(token for _, token in encoded.code_tokens)
        code_scores = memory_second[list(code_positions)]
        weights = torch.softmax(code_scores, dim=0)
        option_index = torch.tensor(
            [ANSWER_IDS.index(code) if code in ANSWER_IDS else 0 for code in code_ids],
            dtype=torch.long,
        )
        valid_mask = torch.tensor(
            [1.0 if code in ANSWER_IDS else 0.0 for code in code_ids],
            dtype=weights.dtype,
        )
        answer_weights = torch.zeros(len(ANSWER_IDS), dtype=weights.dtype)
        answer_weights = answer_weights.scatter_add(
            0, option_index, weights * valid_mask,
        )
        copy_log = torch.log(answer_weights + 1e-3)
        # Center evidence so it doesn't automatically promote abstention.
        copy_log = copy_log - copy_log.mean()
        logits = self.base.answer_head(query)
        logits = torch.cat((logits[:8] + copy_log, logits[8:]), dim=0)
        return TinyOutput(
            answer_logits=logits,
            first_scores=first_scores, second_scores=code_scores,
            first_candidate_token_positions=tuple(range(encoded.memory_length)),
            second_candidate_token_positions=code_positions,
        )


def train_only_targets(ep: PairedEpisode) -> tuple[int, int | None, int | None]:
    """EVALUATOR-HELD targets. Never pass this output to model.forward()."""
    encoded = encode_sealed_view(seal_model_view(ep))
    true_code, relation_line, answer_line = oracle_read(ep)
    expected_code = ep.gold_answer
    if true_code != expected_code:
        raise ValueError("oracle does not match gold; reject training input")
    if expected_code is None:
        return len(ANSWER_IDS), None, None
    first_line = relation_line if relation_line is not None else answer_line
    line_to_token = dict(encoded.line_anchors)
    if first_line not in line_to_token or answer_line not in line_to_token:
        raise ValueError("training source not visible in completed memory")
    first_target = line_to_token[first_line]
    code_candidates = [tok for tok, code in encoded.code_tokens
                       if code == expected_code and tok >= line_to_token[answer_line]]
    # Need the matching code on the actual authoritative line, not a decoy.
    sorted_lines = sorted(encoded.line_anchors)
    next_line_starts = sorted(p for p, _ in encoded.line_anchors if p > answer_line)
    token_at_start = line_to_token[answer_line]
    token_end = min(
        (tok for _, tok in encoded.line_anchors if tok > token_at_start),
        default=encoded.memory_length,
    )
    matching = [i for i, (tok, code) in enumerate(encoded.code_tokens)
                if token_at_start <= tok < token_end and code == expected_code]
    if len(matching) != 1:
        raise ValueError("gold code isn't uniquely bound to authoritative line")
    return ANSWER_IDS.index(expected_code), first_target, matching[0]


def learning_objective(
    model: nn.Module, ep: PairedEpisode, *,
    pointer_supervision: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    view = seal_model_view(ep)
    result: TinyOutput = model(view)
    gold, first, second = train_only_targets(ep)
    answer_loss = F.cross_entropy(
        result.answer_logits.unsqueeze(0), torch.tensor([gold])
    )
    total = answer_loss
    address = answer_loss.new_zeros(())
    if pointer_supervision and first is not None and second is not None:
        if result.first_scores is None or result.second_scores is None:
            raise ValueError("pointer supervision requires a pointer model")
        first_loss = F.cross_entropy(
            result.first_scores.unsqueeze(0), torch.tensor([first])
        )
        second_loss = F.cross_entropy(
            result.second_scores.unsqueeze(0), torch.tensor([second])
        )
        address = first_loss + second_loss
        total = total + 0.4 * address
    if not torch.isfinite(total):
        raise FloatingPointError("non-finite CPU synthetic loss")
    return total, {
        "answer_nll": float(answer_loss.detach().item()),
        "address_ce": float(address.detach().item()),
    }


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def matched_initial_models() -> tuple[TinyDecoderOnly, TinyTwoHopPointer, TinyTwoHopPointer]:
    baseline = TinyDecoderOnly()
    supervised = TinyTwoHopPointer()
    ablated = TinyTwoHopPointer()
    for a, b in zip(baseline.base.parameters(), supervised.base.parameters()):
        assert torch.equal(a, b)
    for a, b in zip(supervised.parameters(), ablated.parameters()):
        assert torch.equal(a, b)
    n_a, n_b = trainable_parameter_count(baseline), trainable_parameter_count(supervised)
    if abs(n_b - n_a) / max(n_b, n_a) > 0.01:
        raise RuntimeError("parameter mismatch above 1%; matched quality claim forbidden")
    return baseline, supervised, ablated


def training_episode(step: int) -> PairedEpisode:
    if type(step) is not int or not 0 <= step < 48:
        raise ValueError("training step outside precommitted 48-step envelope")
    family = ("direct", "overwrite", "two_hop", "no_match")[step % 4]
    entity_index = (step // 4) % 8
    variant = (step * 5 // 4) % 8
    return generate_episode("train", family, entity_index, variant)


def train_cpu_models(*, steps: int = 4) -> tuple[
    TinyDecoderOnly, TinyTwoHopPointer, TinyTwoHopPointer, dict[str, object]
]:
    """Small explicit CPU routine only; no score against dev/test data."""
    if type(steps) is not int or not 1 <= steps <= 48:
        raise ValueError("training cap 1..48 steps per arm")
    models = matched_initial_models()
    counts = [trainable_parameter_count(m) for m in models]
    metadata: dict[str, object] = {
        "classification": "CHM_V3_1377_TINY_CPU_ENGINEERING_ONLY",
        "steps_per_arm": steps,
        "training_example_count_per_arm": steps,
        "parameters": dict(zip(("transformer", "pointer_ce", "pointer_no_ce"), counts)),
        "training_token_count_per_arm": sum(
            len(encode_sealed_view(seal_model_view(training_episode(i))).token_ids)
            for i in range(steps)
        ),
        "no_gpu": True, "new_scientific_attempt": False,
        "historical_scientific_status": ORIGINAL_SCIENTIFIC_STATUS,
    }
    optims = [
        torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.0)
        for model in models
    ]
    for step in range(steps):
        ep = training_episode(step)
        for index, (model, optimizer) in enumerate(zip(models, optims)):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = learning_objective(
                model, ep, pointer_supervision=(index == 1),
            )
            loss.backward()
            optimizer.step()
    for model in models:
        model.eval()
    return models[0], models[1], models[2], metadata


@torch.no_grad()
def predict_from_sealed_view(model: nn.Module, view: SealedModelView) -> dict[str, object]:
    """Predictions and hard attention indexes, without any gold label input."""
    result: TinyOutput = model(view)
    probs = torch.softmax(result.answer_logits, dim=0)
    pred = int(torch.argmax(probs).item())
    first = (
        result.first_candidate_token_positions[int(torch.argmax(result.first_scores).item())]
        if result.first_scores is not None else None
    )
    second = (
        result.second_candidate_token_positions[int(torch.argmax(result.second_scores).item())]
        if result.second_scores is not None else None
    )
    return {
        "predicted_answer_id": None if pred == 8 else ANSWER_IDS[pred],
        "abstained": pred == 8,
        "confidence": float(probs[pred].item()),
        "first_selected_visible_token_index": first,
        "second_selected_visible_token_index": second,
    }


__all__ = (
    "EncodedView", "TinyDecoderOnly", "TinyTwoHopPointer", "encode_sealed_view",
    "train_only_targets", "learning_objective", "training_episode",
    "trainable_parameter_count", "matched_initial_models",
    "train_cpu_models", "predict_from_sealed_view",
)
