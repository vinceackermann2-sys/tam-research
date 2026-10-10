"""PGW-v3 Stage-2B0: frozen CPU-only latent prediction derivative probe.

No optimizer, training step, scientific seed, GPU allocation, or historical
model edits. This helper reuses Stage-2A encoder and Stage-0 predictor weights.
It does not modify the detached routing scores in Stage-0.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from tam_research.pgw_v3_stage1b.oracle import QUERY, READ, WRITE
from tam_research.pgw_v3_stage2a.model import PGWV3Stage2A


EVENT_CHUNKS = 8
CHUNK_SIZE = 8
WIDTH = 32
N_PREDICTIONS_PER_CHUNK = CHUNK_SIZE - 1


def _validate_cpu_event_prefix(
    model: PGWV3Stage2A, tokens: torch.Tensor, anchors: torch.Tensor
) -> tuple[int, int]:
    if not isinstance(model, PGWV3Stage2A):
        raise TypeError("expected frozen Stage-2A PGW model")
    if not isinstance(tokens, torch.Tensor) or tokens.dtype != torch.long or tokens.ndim != 2:
        raise ValueError("tokens must be a LongTensor[B,T]")
    if tokens.device.type != "cpu":
        raise ValueError("structural probe is CPU only")
    batch, length = tokens.shape
    # 8 completed WRITE chunks + at least one delay chunk + final READ chunk.
    if batch < 1 or length < (EVENT_CHUNKS + 2) * CHUNK_SIZE or length % CHUNK_SIZE:
        raise ValueError("expected eight WRITE event chunks, delay, and complete READ")
    if bool(((tokens < 0) | (tokens >= 256)).any()):
        raise ValueError("input token outside 256-symbol vocabulary")
    if not isinstance(anchors, torch.Tensor) or anchors.dtype != torch.long or anchors.ndim != 1:
        raise ValueError("anchors must be LongTensor[B]")
    if anchors.device.type != "cpu" or anchors.shape[0] != batch:
        raise ValueError("anchors must be CPU LongTensor[B]")
    expected_anchor = length - CHUNK_SIZE + 2
    if bool((anchors != expected_anchor).any()):
        raise ValueError("QUERY anchor must be in final READ chunk at offset two")
    if bool((tokens[:, -CHUNK_SIZE] != READ).any()) or bool((tokens[:, -CHUNK_SIZE + 2] != QUERY).any()):
        raise ValueError("final READ/QUERY marker missing")
    chunks = tokens[:, : EVENT_CHUNKS * CHUNK_SIZE].reshape(
        batch, EVENT_CHUNKS, CHUNK_SIZE
    )
    if bool((chunks[:, :, 0] != WRITE).any()):
        raise ValueError("first eight complete event chunks must all start with WRITE")
    if model.cfg.chunk_size != CHUNK_SIZE or model.cfg.d_model != WIDTH:
        raise ValueError("Stage-2A model geometry differs from frozen auxiliary contract")
    return batch, length


def causal_event_hidden(
    model: PGWV3Stage2A, tokens: torch.Tensor, anchors: torch.Tensor
) -> torch.Tensor:
    """H[B,8,8,32] through the SAME shared local encoder as Stage-2A.

    READ and all intervening delay chunks are intentionally outside this loss.
    Chunk-local causal attention is inherited unchanged from Stage-2A.
    """
    batch, _ = _validate_cpu_event_prefix(model, tokens, anchors)
    event_tokens = tokens[:, : EVENT_CHUNKS * CHUNK_SIZE]
    positions = torch.arange(CHUNK_SIZE, device=tokens.device).repeat(EVENT_CHUNKS)
    h = model.token_embedding(event_tokens) + model.local_position(positions).unsqueeze(0)
    h = model.local(h.reshape(batch * EVENT_CHUNKS, CHUNK_SIZE, WIDTH))
    return h.reshape(batch, EVENT_CHUNKS, CHUNK_SIZE, WIDTH)


def predictor_step_errors(
    model: PGWV3Stage2A, causal_h: torch.Tensor
) -> torch.Tensor:
    """S[B,8,7]: squared norm to STOP-GRAD latent target, positions p=1..7.

    Input H is a causal local hidden tensor (not token IDs), so this function
    also allows a direct target-gradient isolation test with an H leaf.
    """
    if not isinstance(model, PGWV3Stage2A):
        raise TypeError("expected Stage-2A PGW model")
    if not isinstance(causal_h, torch.Tensor) or causal_h.ndim != 4:
        raise ValueError("expected causal hidden [B,8,8,32]")
    if causal_h.shape[0] < 1 or tuple(causal_h.shape[1:]) != (8, 8, 32):
        raise ValueError("expected causal hidden [B,8,8,32]")
    if causal_h.device.type != "cpu" or not causal_h.is_floating_point():
        raise ValueError("expected floating-point CPU causal hidden")
    previous = causal_h[:, :, :-1, :]
    current = causal_h[:, :, 1:, :]
    predicted = model.workspace.predict_up(
        F.gelu(model.workspace.predict_down(previous))
    )
    pred_norm = F.layer_norm(predicted.float(), (WIDTH,), eps=1e-5)
    # Detach BEFORE normalization so labels cannot backpropagate via target.
    target_norm = F.layer_norm(current.detach().float(), (WIDTH,), eps=1e-5)
    return (pred_norm - target_norm).square().sum(dim=-1)


def predictor_auxiliary_loss(
    model: PGWV3Stage2A, tokens: torch.Tensor, anchors: torch.Tensor
) -> torch.Tensor:
    """Mean_{batch,8 WRITE chunks,7 within-chunk steps} ||z-target||^2."""
    return predictor_step_errors(model, causal_event_hidden(model, tokens, anchors)).mean()
