from __future__ import annotations

"""CHM-v3 DAEC-A #1350: CPU-only supervised two-hop pointer toy.

Synthetic structurally separate episodes, NOT original aligned-v4 probes,
not the 100M checkpoint, no Modal/GPU, no scientific execution authority.
Explicit relation/record role masks are oracle-provided toy metadata; results
must never be interpreted as learned real-language relation understanding.
"""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

DIM = 24
ROLE_OTHER = 0
ROLE_RELATION = 1
ROLE_RECORD = 2
NO_MATCH_SIMILARITY = 0.50
SCORE_SCALE = 14.0
SPLIT_SEEDS = {"train": 1_350_011, "development": 1_350_101, "test": 1_350_201}
TRAIN_EXAMPLES = 256
DEVELOPMENT_EXAMPLES = 64
TEST_EXAMPLES = 64
ALLOWED_MEMORY_LENGTHS = (128, 256, 512, 1024)
GENERATING_SCIENTIFIC_ALIGNED_V4_PROBES = False
HISTORICAL_SCIENTIFIC_SEED_CONSUMED = 2_013_161
PAID_GPU_AUTHORIZED = False


@dataclass(frozen=True)
class ToyEpisode:
    split: str
    index: int
    query_view: torch.Tensor
    key_views: torch.Tensor
    role: torch.Tensor
    relation_payload: torch.Tensor
    memory_answer_ids: torch.Tensor
    target_relation_pos: int | None
    target_answer_pos: int | None
    correct_answer_id: int | None
    overwritten: bool

    @property
    def memory_length(self) -> int:
        return int(self.key_views.shape[0])


def _unit_random(generator: torch.Generator, shape: tuple[int, ...]) -> torch.Tensor:
    return F.normalize(torch.randn(*shape, generator=generator), dim=-1)


def _query_view(x: torch.Tensor) -> torch.Tensor:
    # Fixed, globally consistent observation transformation, NOT a label.
    signs = torch.tensor([1.0 if i % 3 else -1.0 for i in range(DIM)])
    return torch.roll(x, shifts=5, dims=-1) * signs


def _key_view(x: torch.Tensor) -> torch.Tensor:
    signs = torch.tensor([-1.0 if i % 4 else 1.0 for i in range(DIM)])
    return torch.roll(x, shifts=-3, dims=-1) * signs


def generate_toy_episode(
    split: Literal["train", "development", "test"],
    index: int,
    *,
    memory_length: int = 128,
) -> ToyEpisode:
    """Train/development/test use disjoint generator seeds and episode IDs.

    Every relation carries a document identifier; matching record carries the
    answer token. Overwrite episodes place a duplicate record key at a later
    position, with the later record authoritative. Negative episodes query a
    novel identifier that is absent from relation slots.
    """
    if split not in SPLIT_SEEDS:
        raise ValueError("split must be one of the frozen three split names")
    cap = {"train": TRAIN_EXAMPLES, "development": DEVELOPMENT_EXAMPLES,
           "test": TEST_EXAMPLES}[split]
    if not isinstance(index, int) or not 0 <= index < cap:
        raise ValueError("episode index outside its frozen split")
    if memory_length not in ALLOWED_MEMORY_LENGTHS:
        raise ValueError("memory length outside #1350 CPU-only envelope")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(SPLIT_SEEDS[split] + index * 1009 + memory_length * 11)
    num_pairs = min(16, memory_length // 12)
    overwritten = index % 4 == 1
    negative = index % 5 == 0
    entities = _unit_random(gen, (num_pairs, DIM))
    documents = _unit_random(gen, (num_pairs, DIM))
    filler = _unit_random(gen, (memory_length, DIM))
    key_views = _key_view(filler)
    role = torch.full((memory_length,), ROLE_OTHER, dtype=torch.long)
    relation_payload = torch.zeros((memory_length, DIM))
    memory_answer_ids = torch.full((memory_length,), -1, dtype=torch.long)

    count = 2 * num_pairs + int(overwritten)
    positions = torch.randperm(memory_length, generator=gen)[:count].tolist()
    relation_positions = positions[:num_pairs]
    answer_positions = positions[num_pairs:2 * num_pairs]
    for j in range(num_pairs):
        rel, ans = relation_positions[j], answer_positions[j]
        role[rel] = ROLE_RELATION
        key_views[rel] = _key_view(entities[j])
        relation_payload[rel] = documents[j]
        role[ans] = ROLE_RECORD
        key_views[ans] = _key_view(documents[j])
        memory_answer_ids[ans] = 8_000 + j

    chosen = (index // 2) % num_pairs
    target_relation = relation_positions[chosen]
    target_answer = answer_positions[chosen]
    if overwritten:
        # Both records have the identical document key. Hard-read tie breaking
        # must select the newest (highest positional index) record.
        spare = positions[-1]
        older, newer = sorted((target_answer, spare))
        key_views[spare] = _key_view(documents[chosen])
        role[spare] = ROLE_RECORD
        key_views[older] = _key_view(documents[chosen])
        memory_answer_ids[older] = 7_000 + chosen  # stale value
        memory_answer_ids[newer] = 8_000 + chosen  # authoritative value
        target_answer = newer

    query_vector = (
        _unit_random(gen, (1, DIM))[0] if negative else entities[chosen]
    )
    if negative:
        target_relation = None
        target_answer = None
        answer_id = None
    else:
        answer_id = int(memory_answer_ids[target_answer].item())
    return ToyEpisode(
        split=split,index=index,query_view=_query_view(query_vector),
        key_views=key_views,role=role,relation_payload=relation_payload,
        memory_answer_ids=memory_answer_ids,
        target_relation_pos=target_relation,target_answer_pos=target_answer,
        correct_answer_id=answer_id,overwritten=overwritten,
    )


class PointerAddressToy(nn.Module):
    """Same learned query/key transforms in both hops; no LM or hidden residual."""

    def __init__(self, *, initialization_seed: int = 1_350_005) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(initialization_seed)
            self.query_projection = nn.Linear(DIM, DIM, bias=False)
            self.key_projection = nn.Linear(DIM, DIM, bias=False)

    def _logits(self, query_view: torch.Tensor, ep: ToyEpisode, role: int) -> torch.Tensor:
        projected_query = F.normalize(self.query_projection(query_view), dim=-1)
        projected_keys = F.normalize(self.key_projection(ep.key_views), dim=-1)
        similarities = projected_keys @ projected_query
        logits = SCORE_SCALE * similarities
        return logits.masked_fill(ep.role != role, -1.0e6)

    def first_logits(self, ep: ToyEpisode) -> torch.Tensor:
        logits = self._logits(ep.query_view, ep, ROLE_RELATION)
        # Explicit no-match option, not a hidden answer-oracle selection.
        no_match = logits.new_tensor([SCORE_SCALE * NO_MATCH_SIMILARITY])
        return torch.cat((logits, no_match), dim=0)

    def second_logits(self, ep: ToyEpisode, payload: torch.Tensor) -> torch.Tensor:
        scores = self._logits(_query_view(payload), ep, ROLE_RECORD)
        # Stable latest-record tie break, independent of answer token ID.
        recency = torch.arange(ep.memory_length, dtype=scores.dtype)
        return scores + recency * 1.0e-4


def pointer_supervision_loss(model: PointerAddressToy, ep: ToyEpisode) -> torch.Tensor:
    """Positive first/second position CE, with gold first-hop payload in TRAIN.

    Teacher forcing here is explicit and must never be used in hard evaluation.
    """
    first_target = (
        ep.memory_length if ep.target_relation_pos is None
        else ep.target_relation_pos
    )
    first_loss = F.cross_entropy(
        model.first_logits(ep).unsqueeze(0), torch.tensor([first_target])
    )
    if ep.target_relation_pos is None:
        return first_loss
    gold_doc = ep.relation_payload[ep.target_relation_pos]
    second_loss = F.cross_entropy(
        model.second_logits(ep, gold_doc).unsqueeze(0),
        torch.tensor([ep.target_answer_pos]),
    )
    return first_loss + second_loss


@torch.no_grad()
def evaluate_hard_read(model: PointerAddressToy, ep: ToyEpisode) -> dict[str, object]:
    """Never uses training labels until *after* predicted hard positions."""
    first = int(torch.argmax(model.first_logits(ep)).item())
    if first == ep.memory_length:
        predicted_answer = None
        second = None
    else:
        payload = ep.relation_payload[first]
        second = int(torch.argmax(model.second_logits(ep, payload)).item())
        predicted_answer = int(ep.memory_answer_ids[second].item())
    return {
        "first_selected": None if first == ep.memory_length else first,
        "second_selected": second,
        "predicted_answer_id": predicted_answer,
        "first_correct": first == (
            ep.memory_length if ep.target_relation_pos is None else ep.target_relation_pos
        ),
        "answer_correct": predicted_answer == ep.correct_answer_id,
        "negative": ep.target_relation_pos is None,
        "overwritten": ep.overwritten,
    }


def train_cpu_pointer_toy(
    *,
    steps: int = 96,
    learning_rate: float = 0.025,
    memory_length: int = 128,
) -> tuple[PointerAddressToy, PointerAddressToy]:
    if not 1 <= steps <= TRAIN_EXAMPLES or not 0 < learning_rate <= 0.05:
        raise ValueError("CPU-only pointer training envelope exceeded")
    if memory_length not in ALLOWED_MEMORY_LENGTHS:
        raise ValueError("unsupported CPU-only memory length")
    baseline = PointerAddressToy()
    model = PointerAddressToy()
    assert all(torch.equal(a, b) for a,b in zip(model.parameters(),baseline.parameters()))
    opt = torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=0.0)
    model.train()
    for i in range(steps):
        ep = generate_toy_episode("train",i,memory_length=memory_length)
        opt.zero_grad(set_to_none=True)
        loss = pointer_supervision_loss(model,ep)
        if not bool(torch.isfinite(loss.detach()).item()):
            raise FloatingPointError("#1350 nonfinite CPU pointer toy loss")
        loss.backward()
        opt.step()
    model.eval()
    baseline.eval()
    return baseline,model


@torch.no_grad()
def score_split(
    model: PointerAddressToy,
    split: Literal["development","test"],
    *,
    cases: int = 32,
    memory_length: int = 128,
) -> dict[str, object]:
    if split not in ("development","test") or not 1 <= cases <= 64:
        raise ValueError("only frozen development/test split may be evaluated")
    episodes = [generate_toy_episode(split,i,memory_length=memory_length) for i in range(cases)]
    outputs = [evaluate_hard_read(model,ep) for ep in episodes]
    return {
        "split":split,"cases":cases,"memory_length":memory_length,
        "hard_answer_top1":sum(bool(x["answer_correct"]) for x in outputs)/cases,
        "hard_first_pointer_top1":sum(bool(x["first_correct"]) for x in outputs)/cases,
        "positive_cases":sum(not bool(x["negative"]) for x in outputs),
        "negative_cases":sum(bool(x["negative"]) for x in outputs),
        "overwrite_cases":sum(bool(x["overwritten"]) for x in outputs),
        "no_gpu":True,"scientific_evidence":False,
    }


__all__ = (
    "ALLOWED_MEMORY_LENGTHS", "ToyEpisode", "PointerAddressToy",
    "generate_toy_episode", "pointer_supervision_loss", "evaluate_hard_read",
    "train_cpu_pointer_toy", "score_split",
)
