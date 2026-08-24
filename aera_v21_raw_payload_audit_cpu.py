from __future__ import annotations

"""CPU-only raw-payload audit after the terminal AERA-v21 memory failure.

This deterministically reproduces the already-failed v21 synthetic checkpoint only
for mechanistic diagnosis. It is not a new architecture version or independent
replication evidence.
"""

from contextlib import ExitStack, contextmanager
import json
from types import MethodType
from typing import Any, Iterator

import torch
import torch.nn.functional as F

from aera_v19_memory_necessity_cpu import (
    CHUNK_SIZE,
    EVAL_SEED,
    N_VALUES,
    QUERY,
    VALUE_START,
    WRITE,
    _query_logits,
    make_batch,
)
from aera_v21_memory_pipeline_audit_cpu import (
    _oracle_pair_position_one,
    train_deterministic_reproduction,
)
from tam_research.aera_hardware_core_v21 import HardwareAwareAERATextLMV21

RAW_HIGH = 0.80
DECODED_HIGH = 0.80
MATERIAL_GAIN = 0.15
SCALED_MEMORY_FACTOR = 4.0


def _value_tokens(device: torch.device) -> torch.Tensor:
    return torch.arange(VALUE_START, VALUE_START + N_VALUES, device=device)


def _value_codebook(model: HardwareAwareAERATextLMV21) -> torch.Tensor:
    """Exact stage-0 v21 payload targets for legal values at write position 2."""
    stage = model.stages[0]
    device = model.token_emb.weight.device
    values = _value_tokens(device)
    pos = model.local_pos(torch.tensor(2, device=device))
    h = stage.norm(model.token_emb(values) + pos[None, :])
    return torch.tanh(stage.memory.v(h))


def _raw_recall(
    model: HardwareAwareAERATextLMV21,
    key_tokens: torch.Tensor,
    memory_state: Any,
    *,
    use_k: bool,
) -> torch.Tensor:
    stage = model.stages[0]
    device = model.token_emb.weight.device
    pos = model.local_pos(torch.tensor(1, device=device))
    h = stage.norm(model.token_emb(key_tokens.to(device)) + pos[None, :])
    projection = stage.memory.k if use_k else stage.memory.q
    query = F.normalize(projection(h), dim=-1)
    return torch.einsum("bd,bdm->bm", query, memory_state.matrix)


def _raw_metrics(
    model: HardwareAwareAERATextLMV21,
    raw: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    codebook = _value_codebook(model)
    raw_n = F.normalize(raw.float(), dim=-1)
    code_n = F.normalize(codebook.float(), dim=-1)
    similarity = raw_n @ code_n.transpose(0, 1)
    target_index = targets.to(raw.device) - VALUE_START
    predicted = similarity.argmax(dim=-1)
    target_score = similarity.gather(1, target_index[:, None]).squeeze(1)
    wrong = similarity.clone()
    wrong.scatter_(1, target_index[:, None], float("-inf"))
    best_wrong = wrong.max(dim=-1).values

    decoded = model.stages[0].memory.out(raw)
    decoded_logits = model.lm_head(model.norm(decoded))
    value_logits = decoded_logits[:, VALUE_START : VALUE_START + N_VALUES]
    decoded_prediction = value_logits.argmax(dim=-1)

    return {
        "raw_value_accuracy": float((predicted == target_index).float().mean()),
        "raw_target_cosine_mean": float(target_score.mean()),
        "raw_best_wrong_cosine_mean": float(best_wrong.mean()),
        "raw_target_margin_mean": float((target_score - best_wrong).mean()),
        "decoded_value_accuracy": float((decoded_prediction == target_index).float().mean()),
        "raw_norm_mean": float(raw.float().norm(dim=-1).mean()),
        "decoded_norm_mean": float(decoded.float().norm(dim=-1).mean()),
    }


def collect_raw_payload_metrics(
    model: HardwareAwareAERATextLMV21,
    batch: Any,
    *,
    use_k: bool = False,
    oracle_pair: bool = False,
) -> dict[str, float]:
    model.eval()
    model.set_memory_pretraining_mode(False)
    state = None
    raws: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with ExitStack() as stack:
        if oracle_pair:
            stack.enter_context(_oracle_pair_position_one(model))
        for start in range(0, batch.tokens.size(1), CHUNK_SIZE):
            chunk = batch.tokens[:, start : start + CHUNK_SIZE]
            if state is not None and bool(chunk[:, 0].eq(QUERY).all()):
                raws.append(
                    _raw_recall(
                        model,
                        chunk[:, 1],
                        state.stages[0].memory,
                        use_k=use_k,
                    )
                )
                targets.append(chunk[:, 2].to(model.token_emb.weight.device))
            update = True
            if oracle_pair:
                update = bool(chunk[:, 0].eq(WRITE).all())
            out = model(
                chunk,
                state=state,
                hard=True,
                route_mode="hard_sparse",
                update_memory=update,
                return_block_logits=False,
            )
            state = out["state"]
    if not raws:
        raise RuntimeError("no delayed query recalls collected")
    return _raw_metrics(model, torch.cat(raws, dim=0), torch.cat(targets, dim=0))


@contextmanager
def _memory_context_mode(
    model: HardwareAwareAERATextLMV21,
    mode: str,
    *,
    scale: float = SCALED_MEMORY_FACTOR,
) -> Iterator[None]:
    """Temporary diagnostic intervention on token-wise memory residual injection."""
    if mode not in {"removed", "normal", "scaled"}:
        raise ValueError(mode)
    originals: list[tuple[torch.nn.Module, object]] = []
    if mode == "normal":
        yield
        return
    for stage in model.stages:
        original = stage._tokenwise_context
        originals.append((stage, original))

        def patched(this: torch.nn.Module, h: torch.Tensor, state: Any, start_control: dict[str, torch.Tensor], _mode=mode):
            memory_read = this.memory.read(h, state.memory)
            carried = this.state_to_chunk(state.stream)
            if _mode == "removed":
                context = carried[:, None, :]
            else:
                context = carried[:, None, :] + scale * memory_read
            return context, memory_read

        stage._tokenwise_context = MethodType(patched, stage)
    try:
        yield
    finally:
        for stage, original in originals:
            stage._tokenwise_context = original  # type: ignore[method-assign]


def _collect_query_logits(
    model: HardwareAwareAERATextLMV21,
    batch: Any,
    *,
    mode: str,
) -> torch.Tensor:
    model.eval()
    model.set_memory_pretraining_mode(False)
    with _memory_context_mode(model, mode):
        out = model(
            batch.tokens,
            hard=True,
            route_mode="hard_sparse",
            update_memory=True,
            return_block_logits=False,
        )
    logits = out["logits"]
    if not isinstance(logits, torch.Tensor):
        raise TypeError("expected logits tensor")
    return _query_logits(logits, batch.query_positions)


@torch.no_grad()
def logit_sensitivity(model: HardwareAwareAERATextLMV21, batch: Any) -> dict[str, float]:
    removed = _collect_query_logits(model, batch, mode="removed").float()
    normal = _collect_query_logits(model, batch, mode="normal").float()
    scaled = _collect_query_logits(model, batch, mode="scaled").float()
    target = batch.query_targets.to(removed.device)
    rows = torch.arange(removed.size(0), device=removed.device)[:, None]
    cols = torch.arange(removed.size(1), device=removed.device)[None, :]

    def target_logits(x: torch.Tensor) -> torch.Tensor:
        return x[rows, cols, target]

    def kl_from_removed(x: torch.Tensor) -> float:
        p = F.log_softmax(removed, dim=-1)
        q = F.log_softmax(x, dim=-1)
        return float(F.kl_div(q, p.exp(), reduction="batchmean"))

    return {
        "normal_target_logit_delta_vs_removed": float((target_logits(normal) - target_logits(removed)).mean()),
        "scaled_target_logit_delta_vs_removed": float((target_logits(scaled) - target_logits(removed)).mean()),
        "normal_logits_l2_vs_removed": float((normal - removed).norm(dim=-1).mean()),
        "scaled_logits_l2_vs_removed": float((scaled - removed).norm(dim=-1).mean()),
        "normal_kl_vs_removed": kl_from_removed(normal),
        "scaled_kl_vs_removed": kl_from_removed(scaled),
        "normal_query_accuracy": float((normal.argmax(dim=-1).cpu() == batch.query_targets).float().mean()),
        "removed_query_accuracy": float((removed.argmax(dim=-1).cpu() == batch.query_targets).float().mean()),
        "scaled_query_accuracy": float((scaled.argmax(dim=-1).cpu() == batch.query_targets).float().mean()),
    }


def diagnose(raw: dict[str, dict[str, float]], sensitivity: dict[str, float]) -> str:
    normal = raw["normal_q"]
    oracle_k = raw["oracle_k"]
    oracle_q = raw["oracle_q"]
    if normal["raw_value_accuracy"] >= RAW_HIGH and normal["decoded_value_accuracy"] < DECODED_HIGH:
        return "memory_out_or_decoder_alignment_bottleneck"
    if normal["decoded_value_accuracy"] >= DECODED_HIGH and sensitivity["normal_query_accuracy"] < RAW_HIGH:
        return "downstream_residual_injection_or_objective_bottleneck"
    if oracle_q["raw_value_accuracy"] - normal["raw_value_accuracy"] >= MATERIAL_GAIN:
        return "write_interference_or_selectivity_bottleneck"
    if oracle_k["raw_value_accuracy"] <= (1.0 / N_VALUES + 0.10):
        return "payload_representation_not_recoverably_stored"
    return "mixed_payload_decoder_or_objective_bottleneck"


def run_audit(*, steps: int | None = None) -> dict[str, Any]:
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    kwargs = {} if steps is None else {"steps": steps}
    model, gradients = train_deterministic_reproduction(**kwargs)
    eval_batch = make_batch(24, EVAL_SEED)
    raw = {
        "normal_q": collect_raw_payload_metrics(model, eval_batch, use_k=False, oracle_pair=False),
        "normal_k": collect_raw_payload_metrics(model, eval_batch, use_k=True, oracle_pair=False),
        "oracle_q": collect_raw_payload_metrics(model, eval_batch, use_k=False, oracle_pair=True),
        "oracle_k": collect_raw_payload_metrics(model, eval_batch, use_k=True, oracle_pair=True),
    }
    sensitivity = logit_sensitivity(model, eval_batch)
    diagnosis = diagnose(raw, sensitivity)
    return {
        "scope": "aera_v21_raw_recalled_payload_decodability_cpu",
        "diagnostic_reproduction_only": True,
        "independent_evidence": False,
        "raw_payload": raw,
        "logit_sensitivity": sensitivity,
        "diagnosis": diagnosis,
        "final_gradient_snapshot": gradients[-1] if gradients else {},
        "claims": {
            "production_architecture_changed": False,
            "gpu_authorized": False,
            "v22_authorized": False,
            "architecture_freeze_authorized": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        },
    }


def main() -> None:
    print("AERA_V21_RAW_PAYLOAD_AUDIT_RESULT_JSON=" + json.dumps(run_audit(), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
