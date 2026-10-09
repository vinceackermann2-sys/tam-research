from __future__ import annotations

"""#1357 fresh CPU-only DAEC-A role/soft-hard ablation; NOT 100M science.

The learned-role arm observes a NOISY SYNTHETIC SLOT-TYPE CUE. The cue is
not an oracle role mask, but neither is it inferred from natural-language
evidence: no such capability is demonstrated by this toy.
"""

from dataclasses import dataclass
from typing import Literal
import time

import torch
from torch import nn
from torch.nn import functional as F

from .chm_v3_daec_a_address_target_toy_1350 import (
    DIM, ROLE_OTHER, ROLE_RELATION, ROLE_RECORD,
    NO_MATCH_SIMILARITY, SCORE_SCALE, ToyEpisode, PointerAddressToy,
    _unit_random, _key_view, _query_view,
)

SPLIT_SEEDS = {"train": 1_357_111, "development": 1_357_222, "test": 1_357_333}
INITIALIZATION_SEED = 1_357_005
MEMORY_LENGTHS = (128, 256, 512, 1024)
TRAIN_STEPS = 96
TEST_CASES_PER_SPLIT_LENGTH = 48
TRAIN_MEMORY_LENGTH = 128
LEARNING_RATE = 0.025
ROLE_CE_WEIGHT = 0.5
ROLE_LOGIT_WEIGHT = 12.0
MODES = ("oracle", "learned", "soft")
HISTORICAL_SCIENTIFIC_SEED_CONSUMED = 2_013_161
PAID_GPU_AUTHORIZED = False


@dataclass(frozen=True)
class AblationEpisode:
    base: ToyEpisode
    observed_role_cues: torch.Tensor

    @property
    def memory_length(self) -> int:
        return self.base.memory_length


def make_episode(
    split: Literal["train", "development", "test"],
    index: int,
    *,
    memory_length: int,
) -> AblationEpisode:
    if split not in SPLIT_SEEDS:
        raise ValueError("#1357 unexpected synthetic split")
    limit = TRAIN_STEPS if split == "train" else TEST_CASES_PER_SPLIT_LENGTH
    if not isinstance(index, int) or not 0 <= index < limit:
        raise ValueError("#1357 episode index outside frozen split")
    if memory_length not in MEMORY_LENGTHS:
        raise ValueError("#1357 unsupported memory length")

    rng = torch.Generator(device="cpu")
    rng.manual_seed(SPLIT_SEEDS[split] + index * 1009 + memory_length * 11)
    num_pairs = 16
    overwritten = index % 4 == 1
    negative = index % 5 == 0
    entities = _unit_random(rng, (num_pairs, DIM))
    documents = _unit_random(rng, (num_pairs, DIM))
    keys = _key_view(_unit_random(rng, (memory_length, DIM)))
    roles = torch.full((memory_length,), ROLE_OTHER, dtype=torch.long)
    payload = torch.zeros(memory_length, DIM)
    answers = torch.full((memory_length,), -1, dtype=torch.long)

    allocated = torch.randperm(memory_length, generator=rng)[:2 * num_pairs + int(overwritten)].tolist()
    relations = allocated[:num_pairs]
    records = allocated[num_pairs:2 * num_pairs]
    for j in range(num_pairs):
        rel, record = relations[j], records[j]
        roles[rel], roles[record] = ROLE_RELATION, ROLE_RECORD
        keys[rel], keys[record] = _key_view(entities[j]), _key_view(documents[j])
        payload[rel] = documents[j]
        answers[record] = 8000 + j

    chosen = (index // 2) % num_pairs
    first = relations[chosen]
    second = records[chosen]
    if overwritten:
        older, newer = sorted((second, allocated[-1]))
        roles[older], roles[newer] = ROLE_RECORD, ROLE_RECORD
        keys[older], keys[newer] = _key_view(documents[chosen]), _key_view(documents[chosen])
        answers[older], answers[newer] = 7000 + chosen, 8000 + chosen
        second = newer

    query_value = _unit_random(rng, (1, DIM))[0] if negative else entities[chosen]
    if negative:
        first, second, answer = None, None, None
    else:
        answer = int(answers[second].item())
    base = ToyEpisode(
        split=split, index=index, query_view=_query_view(query_value),
        key_views=keys, role=roles, relation_payload=payload,
        memory_answer_ids=answers, target_relation_pos=first,
        target_answer_pos=second, correct_answer_id=answer,
        overwritten=overwritten,
    )

    # A learned slot-role predictor can SEE noisy observable role cues but not
    # the gold integer role mask. The marker is synthetic, not natural language.
    centers = torch.tensor([[-0.8, -0.8], [0.9, -0.4], [-0.4, 0.9]])
    noise = 0.18 * torch.randn(memory_length, 2, generator=rng)
    cues = centers.index_select(0, roles) + noise
    return AblationEpisode(base=base, observed_role_cues=cues)


class AblationArm(nn.Module):
    def __init__(self, mode: Literal["oracle", "learned", "soft"]) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError("#1357 unknown arm")
        self.mode = mode
        self.pointer = PointerAddressToy(initialization_seed=INITIALIZATION_SEED)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(INITIALIZATION_SEED + 1)
            self.slot_type_classifier = nn.Linear(2, 3)

    def _role_logits(self, ep: AblationEpisode) -> torch.Tensor:
        return self.slot_type_classifier(ep.observed_role_cues)

    def _scores(self, ep: AblationEpisode, query: torch.Tensor, role: int) -> torch.Tensor:
        if self.mode == "oracle":
            return self.pointer._logits(query, ep.base, role)
        q = F.normalize(self.pointer.query_projection(query), dim=-1)
        k = F.normalize(self.pointer.key_projection(ep.base.key_views), dim=-1)
        scores = SCORE_SCALE * (k @ q)
        if self.mode == "learned":
            role_log_probs = F.log_softmax(self._role_logits(ep), dim=-1)
            scores = scores + ROLE_LOGIT_WEIGHT * role_log_probs[:, role]
        return scores

    def first_logits(self, ep: AblationEpisode) -> torch.Tensor:
        scores = self._scores(ep, ep.base.query_view, ROLE_RELATION)
        return torch.cat((scores, scores.new_tensor([SCORE_SCALE * NO_MATCH_SIMILARITY])),dim=0)

    def second_logits(self, ep: AblationEpisode, doc_payload: torch.Tensor) -> torch.Tensor:
        scores = self._scores(ep, _query_view(doc_payload), ROLE_RECORD)
        recency = torch.arange(ep.memory_length, dtype=scores.dtype) * 1e-4
        return scores + recency


def loss_for_arm(arm: AblationArm, ep: AblationEpisode) -> torch.Tensor:
    first_target = ep.memory_length if ep.base.target_relation_pos is None else ep.base.target_relation_pos
    first_logits = arm.first_logits(ep)
    first_ce = F.cross_entropy(first_logits.unsqueeze(0),torch.tensor([first_target]))
    if arm.mode == "soft":
        if ep.base.target_relation_pos is None:
            return first_ce
        # Unmasked differentiable two-hop copy NLL; FIRST hop is not teacher forced.
        weights1 = F.softmax(first_logits,dim=0)[:ep.memory_length]
        doc = weights1 @ ep.base.relation_payload
        weights2 = F.softmax(arm.second_logits(ep, doc),dim=0)
        valid = ep.base.memory_answer_ids.eq(int(ep.base.correct_answer_id))
        return -torch.log(weights2[valid].sum().clamp_min(1e-12))

    if ep.base.target_relation_pos is None:
        pointer_loss = first_ce
    else:
        # Gold first-hop payload is TRAIN ONLY, never used for hard evaluation.
        doc = ep.base.relation_payload[ep.base.target_relation_pos]
        second_ce = F.cross_entropy(
            arm.second_logits(ep, doc).unsqueeze(0),
            torch.tensor([int(ep.base.target_answer_pos)]),
        )
        pointer_loss = first_ce + second_ce
    if arm.mode == "learned":
        role_ce = F.cross_entropy(arm._role_logits(ep), ep.base.role)
        pointer_loss = pointer_loss + ROLE_CE_WEIGHT * role_ce
    return pointer_loss


@torch.no_grad()
def hard_read(arm: AblationArm, ep: AblationEpisode) -> dict[str, object]:
    """Hard evaluation uses only its own selected first-hop payload."""
    selected_first = int(torch.argmax(arm.first_logits(ep)).item())
    if selected_first == ep.memory_length:
        selected_second, answer = None, None
    else:
        selected_second = int(
            torch.argmax(arm.second_logits(ep, ep.base.relation_payload[selected_first])).item()
        )
        answer = int(ep.base.memory_answer_ids[selected_second].item())
    target_first = ep.base.target_relation_pos
    target_second = ep.base.target_answer_pos
    correct_negative = target_first is None
    stale = bool(ep.base.overwritten and answer is not None and 7000 <= answer < 8000)
    prediction = {
        "first": None if selected_first == ep.memory_length else selected_first,
        "second": selected_second,
        "answer": answer,
        "first_correct": (selected_first == (ep.memory_length if correct_negative else target_first)),
        "second_correct": (selected_second == target_second) if not correct_negative else False,
        "answer_correct": answer == ep.base.correct_answer_id,
        "negative": correct_negative,
        "overwrite": bool(ep.base.overwritten),
        "false_positive_on_negative": bool(correct_negative and answer is not None),
        "stale_on_overwrite": stale,
        "role_correct_count": None,
        "role_count": None,
    }
    if arm.mode == "learned":
        role_prediction = torch.argmax(arm._role_logits(ep),dim=-1)
        prediction["role_correct_count"] = int(role_prediction.eq(ep.base.role).sum().item())
        prediction["role_count"] = ep.memory_length
    return prediction


@torch.no_grad()
def soft_target_probability(arm: AblationArm, ep: AblationEpisode) -> float:
    """A soft-read diagnostic; independent of hard answer correctness."""
    w1 = F.softmax(arm.first_logits(ep),dim=0)
    if ep.base.target_relation_pos is None:
        return float(w1[-1].item())
    payload = w1[:ep.memory_length] @ ep.base.relation_payload
    w2 = F.softmax(arm.second_logits(ep,payload),dim=0)
    mask = ep.base.memory_answer_ids.eq(int(ep.base.correct_answer_id))
    return float(w2[mask].sum().item())


def train_arms() -> tuple[dict[str, AblationArm], dict[str, object]]:
    arms = {name:AblationArm(name) for name in MODES}
    qk = tuple(arms["oracle"].pointer.parameters())
    for name in ("learned","soft"):
        other=tuple(arms[name].pointer.parameters())
        if not all(torch.equal(a,b) for a,b in zip(qk,other)):
            raise RuntimeError("#1357 identical address initialization violated")
    optimizers={
        name:torch.optim.AdamW(arm.parameters(),lr=LEARNING_RATE,weight_decay=0.0)
        for name,arm in arms.items()
    }
    times={name:0.0 for name in MODES}
    final_losses={name:None for name in MODES}
    for index in range(TRAIN_STEPS):
        ep=make_episode("train",index,memory_length=TRAIN_MEMORY_LENGTH)
        for name,arm in arms.items():
            before=time.perf_counter()
            optimizers[name].zero_grad(set_to_none=True)
            loss=loss_for_arm(arm,ep)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("#1357 nonfinite CPU ablation loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(arm.parameters(),max_norm=1.0)
            optimizers[name].step()
            times[name]+=time.perf_counter()-before
            final_losses[name]=float(loss.detach().item())
    for arm in arms.values():
        arm.eval()
    return arms,{
        "train_examples_per_arm":TRAIN_STEPS,
        "training_memory_length":TRAIN_MEMORY_LENGTH,
        "matching_address_init":True,
        "train_seconds_by_arm":times,
        "final_loss_by_arm":final_losses,
        "parameters_by_arm":{name:sum(p.numel() for p in arm.parameters()) for name,arm in arms.items()},
    }


@torch.no_grad()
def evaluate_panel(
    arm: AblationArm,
    split: Literal["development", "test"],
    length: int,
    *,
    cases: int = TEST_CASES_PER_SPLIT_LENGTH,
) -> dict[str, object]:
    if split not in ("development", "test") or length not in MEMORY_LENGTHS:
        raise ValueError("#1357 invalid panel")
    if not 1 <= cases <= TEST_CASES_PER_SPLIT_LENGTH:
        raise ValueError("#1357 invalid case count")
    started=time.perf_counter()
    rows=[]
    for idx in range(cases):
        ep=make_episode(split,idx,memory_length=length)
        item=hard_read(arm,ep)
        item["soft_target_probability"]=soft_target_probability(arm,ep)
        rows.append(item)
    positive=[r for r in rows if not r["negative"]]
    negative=[r for r in rows if r["negative"]]
    overwrite=[r for r in rows if r["overwrite"] and not r["negative"]]
    return {
        "mode":arm.mode,"split":split,"memory_length":length,"cases":cases,
        "positive_count":len(positive),"negative_count":len(negative),
        "overwrite_positive_count":len(overwrite),
        "first_pointer_correct":sum(bool(r["first_correct"]) for r in rows),
        "second_pointer_correct_on_positives":sum(bool(r["second_correct"]) for r in positive),
        "hard_answer_correct":sum(bool(r["answer_correct"]) for r in rows),
        "hard_answer_correct_on_positives":sum(bool(r["answer_correct"]) for r in positive),
        "negative_abstention_correct":sum(bool(r["answer_correct"]) for r in negative),
        "negative_false_positive_count":sum(bool(r["false_positive_on_negative"]) for r in negative),
        "overwrite_correct":sum(bool(r["answer_correct"]) for r in overwrite),
        "overwrite_stale_choice_count":sum(bool(r["stale_on_overwrite"]) for r in overwrite),
        "role_correct":sum(int(r["role_correct_count"] or 0) for r in rows) if arm.mode=="learned" else None,
        "role_count":sum(int(r["role_count"] or 0) for r in rows) if arm.mode=="learned" else None,
        "mean_soft_target_probability":sum(float(r["soft_target_probability"]) for r in rows)/cases,
        "cpu_wall_seconds":time.perf_counter()-started,
    }


__all__ = (
    "SPLIT_SEEDS","MEMORY_LENGTHS","MODES","TRAIN_STEPS","PAID_GPU_AUTHORIZED",
    "HISTORICAL_SCIENTIFIC_SEED_CONSUMED","AblationEpisode","AblationArm",
    "make_episode","loss_for_arm","hard_read","soft_target_probability",
    "train_arms","evaluate_panel",
)
