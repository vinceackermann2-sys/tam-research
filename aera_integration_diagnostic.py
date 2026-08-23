from __future__ import annotations

import json
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from tam_research.aera_full import FullAERAConfig
from tam_research.aera_integrated import IntegratedAERATextLM


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


class DenseDiagnosticLM(nn.Module):
    """Small causal dense control for validating the synthetic task itself."""

    def __init__(self, vocab: int = 31, d_model: int = 32, heads: int = 4, length: int = 20):
        super().__init__()
        self.vocab = vocab
        self.token = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(length, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.token.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        t = tokens.size(1)
        x = self.token(tokens) + self.pos(torch.arange(t, device=tokens.device))[None]
        h = self.norm1(x)
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=tokens.device), diagonal=1)
        a, _ = self.attn(h, h, h, attn_mask=causal, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return self.head(self.norm3(x))


def second_order(batch: int, length: int, vocab: int) -> torch.Tensor:
    a = torch.randint(0, vocab, (batch,))
    b = torch.randint(0, vocab, (batch,))
    seq = [a, b]
    for _ in range(2, length):
        seq.append((seq[-1] + 2 * seq[-2] + 1) % vocab)
    return torch.stack(seq, dim=1)


def first_order(batch: int, length: int, vocab: int) -> torch.Tensor:
    a = torch.randint(0, vocab, (batch,))
    offsets = torch.arange(length)[None]
    return (a[:, None] + offsets) % vocab


def score_logits(logits: torch.Tensor, tokens: torch.Tensor) -> dict[str, float]:
    nll = float(
        F.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.size(-1)),
            tokens[:, 1:].reshape(-1),
        )
    )
    acc = float((logits[:, :-1].argmax(dim=-1) == tokens[:, 1:]).float().mean())
    return {"nll": nll, "accuracy": acc}


def train_dense(task: str, steps: int = 300) -> dict[str, float]:
    seed_all(4101 if task == "second" else 4102)
    vocab, length = 31, 20
    make = second_order if task == "second" else first_order
    model = DenseDiagnosticLM(vocab=vocab, length=length)
    opt = torch.optim.AdamW(model.parameters(), lr=0.004, weight_decay=0.01)
    fixed = make(512, length, vocab)
    with torch.no_grad():
        initial = score_logits(model(fixed), fixed)
    for _ in range(steps):
        tok = make(64, length, vocab)
        logits = model(tok)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab), tok[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    with torch.no_grad():
        final = score_logits(model(fixed), fixed)
    return {
        "initial_nll": initial["nll"],
        "final_nll": final["nll"],
        "final_accuracy": final["accuracy"],
    }


def aera_cfg() -> FullAERAConfig:
    return FullAERAConfig(
        vocab_size=31,
        d_model=32,
        n_stages=1,
        n_heads=4,
        local_window=8,
        max_seq_len=20,
        n_experts=4,
        top_k_experts=1,
        expert_mult=2,
        memory_dim=8,
        max_reason_steps=2,
        block_size=3,
    )


def train_aera(task: str, *, full_objective: bool, steps: int = 300) -> dict[str, object]:
    seed_all((4201 if task == "second" else 4202) + (10 if full_objective else 0))
    cfg = aera_cfg()
    make = second_order if task == "second" else first_order
    model = IntegratedAERATextLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=0.004, weight_decay=0.01)
    fixed = make(512, cfg.max_seq_len, cfg.vocab_size)
    with torch.no_grad():
        out = model(fixed, return_block_logits=full_objective)
        initial = score_logits(out["logits"], fixed)
    for _ in range(steps):
        tok = make(64, cfg.max_seq_len, cfg.vocab_size)
        out = model(tok, return_block_logits=full_objective)
        if full_objective:
            loss = model.objective(
                tok,
                out,
                event_weight=0.03,
                compute_weight=0.001,
                balance_weight=0.01,
                block_weight=0.35,
            )["total"]
        else:
            logits = out["logits"]
            assert isinstance(logits, torch.Tensor)
            loss = F.cross_entropy(
                logits[:, :-1].float().reshape(-1, cfg.vocab_size),
                tok[:, 1:].reshape(-1),
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    with torch.no_grad():
        out = model(fixed, hard=False, update_memory=False, return_block_logits=full_objective)
        logits = out["logits"]
        assert isinstance(logits, torch.Tensor)
        final = score_logits(logits, fixed)
        model(fixed[:16], hard=True, update_memory=False)
        stats = model.stats()
    return {
        "initial_nll": initial["nll"],
        "final_nll": final["nll"],
        "final_accuracy": final["accuracy"],
        "stats": stats,
    }


def main() -> None:
    result = {
        "scope": "diagnostic_only_no_scale_authorization",
        "chance_nll": math.log(31),
        "dense_second_order": train_dense("second"),
        "dense_first_order": train_dense("first"),
        "aera_second_order_lm_only": train_aera("second", full_objective=False),
        "aera_first_order_lm_only": train_aera("first", full_objective=False),
        "aera_second_order_full_objective": train_aera("second", full_objective=True),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
