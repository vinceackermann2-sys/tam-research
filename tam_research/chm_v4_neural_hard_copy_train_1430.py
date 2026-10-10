from __future__ import annotations

"""#1430 CPU-only TRAIN engineering: neural address -> literal hard CODE copy.

This is a prototype with GOLD ADDRESS CE in TRAIN only, NOT a matched-compute
architecture comparison or evidence from development/test entities. A soft
first-hop mixture conditions second-hop logits; only OUTPUT decoding hard-copies.
No hidden episode/oracle metadata is available to model.forward().
"""

from collections import Counter
from dataclasses import dataclass
import math
import re

import torch
from torch import nn
from torch.nn import functional as F

from .chm_v3_matched_tiny_models_1377 import (
    CausalBackbone, DIM, EncodedView, INIT_SEED, MAX_TOKENS,
    _token_id, trainable_parameter_count,
)
from .chm_v4_independent_binding_benchmark_1421 import (
    CODES, STALE_CODE, Episode, SealedInput,
    OLD_100M_STOP, OLD_SCIENTIFIC_SEED_CONSUMED,
    oracle_read, seal_input, train_counterfactual_group,
)

CLASSIFICATION = "CHM_V4_1430_TRAIN_ONLY_NEURAL_HARD_COPY_GRADIENT_SMOKE"
GPU_AUTHORIZED = False
HELDOUT_SCORING_AUTHORIZED = False
NEW_SCIENTIFIC_ATTEMPT = False
MAX_TRAIN_OPTIMIZER_STEPS = 1

V4_TOKEN_PATTERN = re.compile(
    r"CODE-[0-9]+|V4-[A-Za-z0-9-]+|\[[0-9]{4}\]|[A-Za-z]+|[0-9]+|[^\s]"
)
SOURCE_ANCHOR = re.compile(r"\[([0-9]{4})\]")
CODE_TOKEN = re.compile(r"CODE-(\d+)")


@dataclass(frozen=True)
class EncodedV4:
    model_tokens: EncodedView
    code_candidate_positions: tuple[int, ...]
    code_candidate_ids: tuple[int, ...]
    code_candidate_source_physical: tuple[int, ...]
    first_source_physical: tuple[int, ...]


@dataclass
class PointerV4Output:
    presence_logits: torch.Tensor
    first_line_logits: torch.Tensor
    second_code_logits: torch.Tensor
    first_source_physical: tuple[int, ...]
    code_source_physical: tuple[int, ...]
    code_candidate_ids: tuple[int, ...]


@dataclass(frozen=True)
class HardCopyPrediction:
    predicted_answer_id: int | None
    first_source_physical: int
    payload_source_physical: int
    first_line_choice: int
    code_choice: int
    did_abstain: bool


def encode_visible_v4(view: SealedInput) -> EncodedV4:
    """Read exactly three visible input fields, never evaluator side channels."""
    if not isinstance(view, SealedInput):
        raise TypeError("neural forward accepts only SealedInput")
    if view.answer_options != tuple(f"CODE-{n}" for n in CODES):
        raise ValueError("unexpected V4 candidate options")
    if not view.query or not view.memory_text or len(view.memory_text) > 12000:
        raise ValueError("invalid visible question or completed memory")
    tokens: list[str] = []
    codes: list[tuple[int,int]] = []
    anchors: list[tuple[int,int]] = []
    for line in view.memory_text.splitlines():
        parts = V4_TOKEN_PATTERN.findall(line)
        if not parts or SOURCE_ANCHOR.fullmatch(parts[0]) is None:
            raise ValueError("missing visible physical source line")
        physical = int(SOURCE_ANCHOR.fullmatch(parts[0]).group(1))
        if physical >= 128 or any(p == physical for p,_ in anchors):
            raise ValueError("invalid or duplicate physical memory source")
        if anchors and physical <= anchors[-1][0]:
            raise ValueError("nonmonotonic completed memory source order")
        anchors.append((physical, len(tokens)))
        for part in parts:
            m = CODE_TOKEN.fullmatch(part)
            if m:
                codes.append((len(tokens), int(m.group(1))))
            tokens.append(part)
    memory_length = len(tokens)
    tokens.extend(("QUESTION", *V4_TOKEN_PATTERN.findall(view.query), "ANSWER"))
    if (not codes or len(tokens) > MAX_TOKENS or
        memory_length >= len(tokens) or not anchors):
        raise ValueError("invalid input sequence/context budget")
    selected = [(p,c) for p,c in codes if c in CODES]
    if Counter(code for _,code in selected) != Counter(CODES):
        raise ValueError("visible CODE candidates must form a unique eight-way bag")
    if any(code not in (*CODES,STALE_CODE) for _,code in codes):
        raise ValueError("unknown visible source CODE")
    if sum(code == STALE_CODE for _,code in codes) > 1:
        raise ValueError("unsupported multiple stale noncandidate CODE rows")
    if any(not re.fullmatch(r"V4-[A-Za-z0-9-]+",s)
           for s in V4_TOKEN_PATTERN.findall(view.query)
           if s.startswith("V4-")):
        raise ValueError("malformed V4 identity in query")
    firsts = tuple(physical for physical,_ in anchors)
    physical_for_code: list[int] = []
    for position, _ in selected:
        parents = [physical for physical,start in anchors if start <= position]
        if not parents:
            raise ValueError("candidate CODE appears outside completed source")
        physical_for_code.append(parents[-1])
    return EncodedV4(
        model_tokens=EncodedView(
            token_ids=tuple(_token_id(t) for t in tokens),
            memory_length=memory_length,
            code_tokens=tuple(codes),
            line_anchors=tuple(anchors),
        ),
        code_candidate_positions=tuple(p for p,_ in selected),
        code_candidate_ids=tuple(c for _,c in selected),
        code_candidate_source_physical=tuple(physical_for_code),
        first_source_physical=firsts,
    )


class TinyV4HardCopyPointer(nn.Module):
    """Soft first-hop *training representation*, discrete hard CODE output."""

    def __init__(self) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(INIT_SEED)
            self.base = CausalBackbone()
            self.first_query = nn.Linear(DIM,DIM,bias=False)
            self.first_key = nn.Linear(DIM,DIM,bias=False)
            self.second_query = nn.Linear(DIM,DIM,bias=False)
            self.gate_head = nn.Linear(DIM,2)

    def forward(self, view: SealedInput) -> PointerV4Output:
        encoded = encode_visible_v4(view)
        h = self.base(encoded.model_tokens)
        memory = h[:encoded.model_tokens.memory_length]
        question = h[-1]
        line_index = tuple(pos for _,pos in encoded.model_tokens.line_anchors)
        line_hidden = memory[list(line_index)]
        keys = self.first_key(line_hidden)
        first_logits = (keys @ self.first_query(question)) / math.sqrt(DIM)
        first_soft = torch.softmax(first_logits, dim=0)
        context = first_soft @ line_hidden
        query_second = self.second_query(question + context)
        code_hidden = memory[list(encoded.code_candidate_positions)]
        second_logits = (code_hidden @ query_second) / math.sqrt(DIM)
        return PointerV4Output(
            presence_logits=self.gate_head(question),
            first_line_logits=first_logits,
            second_code_logits=second_logits,
            first_source_physical=encoded.first_source_physical,
            code_source_physical=encoded.code_candidate_source_physical,
            code_candidate_ids=encoded.code_candidate_ids,
        )


def hard_copy_prediction(output: PointerV4Output) -> HardCopyPrediction:
    """Hard-copy EXACT selected visible CODE; never feed gold back to output."""
    first = int(output.first_line_logits.argmax().item())
    second = int(output.second_code_logits.argmax().item())
    present = int(output.presence_logits.argmax().item()) == 1
    answer = output.code_candidate_ids[second] if present else None
    return HardCopyPrediction(
        predicted_answer_id=answer,
        first_source_physical=output.first_source_physical[first],
        payload_source_physical=output.code_source_physical[second],
        first_line_choice=first, code_choice=second,
        did_abstain=not present,
    )


def train_only_supervised_loss(
    out: PointerV4Output, episode: Episode,
) -> tuple[torch.Tensor, dict[str,float]]:
    """ONLY training/evaluator side may inspect structured gold addresses."""
    if not isinstance(episode, Episode) or episode.split != "train":
        raise ValueError("train-only source address supervision")
    gold, relation, payload_source = oracle_read(episode)
    if gold != episode.gold_answer:
        raise ValueError("training oracle/gold mismatch")
    target_present = 0 if gold is None else 1
    gate_loss = F.cross_entropy(
        out.presence_logits.unsqueeze(0),
        torch.tensor([target_present],dtype=torch.long),
    )
    first_loss = gate_loss.new_zeros(())
    second_loss = gate_loss.new_zeros(())
    if gold is not None:
        first_physical = relation if relation is not None else payload_source
        first_indices = [
            k for k,pos in enumerate(out.first_source_physical)
            if pos == first_physical
        ]
        second_indices = [
            k for k,(source,code) in enumerate(zip(
                out.code_source_physical,out.code_candidate_ids
            )) if source == payload_source and code == gold
        ]
        if len(first_indices) != 1 or len(second_indices) != 1:
            raise ValueError("oracle source not uniquely visible in model text")
        first_loss = F.cross_entropy(
            out.first_line_logits.unsqueeze(0),
            torch.tensor(first_indices,dtype=torch.long),
        )
        second_loss = F.cross_entropy(
            out.second_code_logits.unsqueeze(0),
            torch.tensor(second_indices,dtype=torch.long),
        )
    objective = gate_loss + .4*(first_loss+second_loss)
    if not bool(torch.isfinite(objective).item()):
        raise FloatingPointError("train-only hard-copy loss non-finite")
    return objective, {
        "presence_ce": float(gate_loss.detach().item()),
        "first_address_ce": float(first_loss.detach().item()),
        "payload_address_ce": float(second_loss.detach().item()),
    }


def cpu_one_train_step_smoke(*, steps: int = 1) -> dict[str,object]:
    """A single positive TRAIN-only gradient update; NO dev/test scoring."""
    if type(steps) is not int or steps != MAX_TRAIN_OPTIMIZER_STEPS:
        raise ValueError("only exactly one TRAIN optimizer step permitted")
    torch.set_num_threads(1)
    ep = train_counterfactual_group("two_hop",0)[0]
    if ep.split != "train":
        raise RuntimeError("nontrain sample in bounded smoke")
    view = seal_input(ep)
    model = TinyV4HardCopyPointer()
    optimizer = torch.optim.AdamW(model.parameters(),lr=.005,weight_decay=0.)
    optimizer.zero_grad(set_to_none=True)
    output = model(view)
    before = hard_copy_prediction(output)
    loss, components = train_only_supervised_loss(output,ep)
    loss.backward()
    named = {
        "first_query": model.first_query.weight,
        "first_key": model.first_key.weight,
        "second_query": model.second_query.weight,
        "gate_head": model.gate_head.weight,
        "base_embeddings": model.base.embedding.weight,
    }
    grad_abs = {}
    for name,parameter in named.items():
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all().item()):
            raise RuntimeError(f"missing/nonfinite TRAIN gradient on {name}")
        grad_abs[name] = float(parameter.grad.abs().sum().item())
        if grad_abs[name] <= 0:
            raise RuntimeError(f"zero TRAIN gradient on {name}")
    optimizer.step()
    return {
        "classification": CLASSIFICATION,
        "train_only_steps": steps,
        "training_examples_seen": 1,
        "model_parameters": trainable_parameter_count(model),
        "visible_input_token_count": len(encode_visible_v4(view).model_tokens.token_ids),
        "supervision": "TRAIN-only first physical source / payload CODE / abstention",
        "loss_components": components,
        "abs_grad_sum": grad_abs,
        "preupdate_output_is_hard_copy_of_visible_code_or_abstention": (
            before.predicted_answer_id is None or before.predicted_answer_id in CODES
        ),
        "soft_first_hop_then_hard_final_copy": True,
        "genuinely_hard_conditioned_second_hop": False,
        "scored_dev_test_cases_constructed": False,
        "gpu_used": False,
        "new_scientific_attempt": False,
        "historical_scientific_status": OLD_100M_STOP,
        "historical_scientific_seed_consumed": OLD_SCIENTIFIC_SEED_CONSUMED,
        "interpretation_ceiling": (
            "One TRAIN-only optimizer step establishes gradient connectivity, "
            "NOT correctness, generalization, matched training, or breakthrough. "
            "First-hop is SOFT-conditioned; only output is hard-copied. "
            "Extra oracle address CE is TRAIN-only supervision."
        ),
    }


__all__ = (
    "EncodedV4", "PointerV4Output", "HardCopyPrediction",
    "encode_visible_v4", "TinyV4HardCopyPointer", "hard_copy_prediction",
    "train_only_supervised_loss", "cpu_one_train_step_smoke",
)
