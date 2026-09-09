from __future__ import annotations

from contextlib import nullcontext
import gc
import math
import time
from typing import Any, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch import _dynamo as dynamo

from architectures.cortex_s.language_model import CortexSLM, TrulySparseMoE, parameter_count
from architectures.cortex_s.systems_optimization import pack_topk_assignments
from tam_research.data import TokenBin

from .protocol import (
    CORTEX_100M_CONFIG,
    DATA_DIR,
    EXPECTED_CORTEX_PARAMS,
    GRAD_ACCUM_STEPS,
    LEARNING_RATE,
    MICRO_BATCH_SIZE,
    RESERVED_FRESH_SEEDS,
    SEQ_LEN,
    TOKENS_PER_OPTIMIZER_STEP,
)
from .train import _compile_model, _make_optimizer, _one_optimizer_step, build_cortex_100m, seed_all
from .systems_microbench_v1_repair3 import _benchmark_scan


CLASSIFICATION = "ENGINEERING_SYSTEMS_MICROBENCH_ONLY"
ENGINEERING_SEED = 2_026_090_904
CONSUMED_ENGINEERING_SEEDS = (910_001, 2_026_090_901, 2_026_090_902, 2_026_090_903)
FORBIDDEN_SCIENTIFIC_SEEDS = (8_100, *RESERVED_FRESH_SEEDS)
TRIGGER_TITLE = "[modal-cortex-s-100m-systems-microbench-v1-repair4]"
RESULT_ROOT = "/vol/cortex-s-v0/100m-systems-microbench-v1-repair4"
PARENT_REPAIR3_SOURCE_SHA = "1c0ecd321a530daee32a4603a9cd6fb201be2d8a"
PARENT_REPAIR3_RUN_ID = 34_335_827_718
PARENT_REPAIR3_JOB_ID = 102_414_857_556

MEASURED_STEPS = 20
ADDITIONAL_WARMUP_STEPS = 3
MIN_PROMISING_SPEEDUP = 1.20
STOP_GROUPED_BELOW_SPEEDUP = 1.10
MAX_PEAK_VRAM_GIB = 70.0
MAX_SEMANTIC_LOSS_DELTA = 0.02
GROUPED_MM_ALIGNMENT_BYTES = 16


def _aligned_hidden_width(hidden: int, *, element_bytes: int = 2) -> int:
    elements = GROUPED_MM_ALIGNMENT_BYTES // element_bytes
    return ((hidden + elements - 1) // elements) * elements


class LogicalPaddedBF16GroupedSparseMoE(nn.Module):
    """Top-k MoE whose grouped compute remains logically aligned end-to-end.

    Repair2/3 kept a logical width of 338 after GELU while relying on padded
    physical storage. Under torch.compile/FakeTensor, grouped_mm still observed
    the second lhs row stride as 338 and rejected it. Repair4 instead carries a
    logical compute width of 344 through both grouped GEMMs. The six extra
    channels are runtime zeros derived by padding the existing 338-wide weights;
    they are not trainable Parameters and therefore do not alter parameter count.
    """

    def __init__(self, d_model: int, num_experts: int, top_k: int, hidden: int):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden = hidden
        self.compute_hidden = _aligned_hidden_width(hidden)
        self.router = nn.Linear(d_model, num_experts, bias=True)
        self.expert_w1 = nn.Parameter(torch.empty(num_experts, hidden, d_model))
        self.expert_w2 = nn.Parameter(torch.empty(num_experts, hidden, d_model))
        self.reset_parameters()
        self.last_counts: Optional[torch.Tensor] = None

    def reset_parameters(self) -> None:
        self.router.reset_parameters()
        nn.init.normal_(self.expert_w1, mean=0.0, std=0.02)
        nn.init.normal_(self.expert_w2, mean=0.0, std=0.02)

    @torch.no_grad()
    def copy_from_legacy(self, legacy: TrulySparseMoE) -> "LogicalPaddedBF16GroupedSparseMoE":
        if legacy.num_experts != self.num_experts or legacy.top_k != self.top_k:
            raise ValueError("legacy routing dimensions do not match repair4 candidate")
        self.router.weight.copy_(legacy.router.weight)
        self.router.bias.copy_(legacy.router.bias)
        for index, expert in enumerate(legacy.experts):
            first = expert[0]
            second = expert[2]
            self.expert_w1[index].copy_(first.weight)
            self.expert_w2[index].copy_(second.weight.T)
        return self

    def _w1_compute(self) -> torch.Tensor:
        logical = self.expert_w1.transpose(-2, -1)  # [E, d_model, hidden]
        pad = self.compute_hidden - self.hidden
        return F.pad(logical, (0, pad)) if pad else logical

    def _w2_compute(self) -> torch.Tensor:
        pad = self.compute_hidden - self.hidden
        return F.pad(self.expert_w2, (0, 0, 0, pad)) if pad else self.expert_w2

    @staticmethod
    def grouped_cuda_available(x: torch.Tensor) -> bool:
        return bool(
            hasattr(F, "grouped_mm")
            and x.is_cuda
            and torch.cuda.get_device_capability(x.device) >= (8, 0)
        )

    def _reference_grouped_mm(
        self,
        packed: torch.Tensor,
        matrices: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        start = 0
        for expert_index, end_tensor in enumerate(offsets.cpu()):
            end = int(end_tensor)
            outputs.append(packed[start:end] @ matrices[expert_index])
            start = end
        return torch.cat(outputs, dim=0)

    def _grouped_mm(
        self,
        packed: torch.Tensor,
        matrices: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        if not self.grouped_cuda_available(packed):
            return self._reference_grouped_mm(packed, matrices, offsets)
        lhs = packed if packed.dtype == torch.bfloat16 else packed.to(torch.bfloat16)
        rhs = matrices if matrices.dtype == torch.bfloat16 else matrices.to(torch.bfloat16)
        return F.grouped_mm(lhs, rhs, offs=offsets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])
        router_logits = self.router(flat)
        top_values, top_indices = router_logits.topk(self.top_k, dim=-1)
        assignment_weights = F.softmax(top_values.float(), dim=-1).to(flat.dtype)
        plan = pack_topk_assignments(top_indices, assignment_weights, num_experts=self.num_experts)
        packed = flat.index_select(0, plan.token_indices)

        hidden = self._grouped_mm(packed, self._w1_compute(), plan.offsets)
        hidden = F.gelu(hidden)
        expert_output = self._grouped_mm(hidden, self._w2_compute(), plan.offsets)

        weighted = expert_output * plan.assignment_weights[:, None]
        output = torch.zeros_like(flat)
        output.index_add_(0, plan.token_indices, weighted)
        self.last_counts = plan.counts.detach()
        return output.view(original_shape)

    @property
    def theoretical_executed_fraction(self) -> float:
        return self.top_k / self.num_experts


def convert_to_logical_padded_grouped(model: CortexSLM) -> CortexSLM:
    before = parameter_count(model)
    for block in model.blocks:
        legacy = block.moe
        if not isinstance(legacy, TrulySparseMoE):
            raise TypeError("expected production TrulySparseMoE before repair4 conversion")
        first = legacy.experts[0][0]
        candidate = LogicalPaddedBF16GroupedSparseMoE(
            d_model=first.in_features,
            num_experts=legacy.num_experts,
            top_k=legacy.top_k,
            hidden=first.out_features,
        ).to(device=legacy.router.weight.device, dtype=legacy.router.weight.dtype)
        candidate.copy_from_legacy(legacy)
        block.moe = candidate
    after = parameter_count(model)
    if before != after:
        raise RuntimeError(f"repair4 changed parameter count: {before} -> {after}")
    return model


def zero_gpu_repair4_contract_probe() -> dict[str, Any]:
    hidden = CORTEX_100M_CONFIG.expert_hidden
    candidate = LogicalPaddedBF16GroupedSparseMoE(
        CORTEX_100M_CONFIG.d_model,
        CORTEX_100M_CONFIG.num_experts,
        CORTEX_100M_CONFIG.top_k,
        hidden,
    ).to(dtype=torch.bfloat16)
    w1 = candidate._w1_compute()
    w2 = candidate._w2_compute()
    if candidate.compute_hidden != 344:
        raise RuntimeError(f"repair4 expected compute hidden 344, got {candidate.compute_hidden}")
    if w1.shape[-1] != 344 or w2.shape[-2] != 344:
        raise RuntimeError("repair4 compute matrices do not preserve aligned logical width")
    if w1.stride(-1) != 1 or w1.stride(-2) % 8 != 0:
        raise RuntimeError(f"repair4 W1 compute layout invalid: {w1.stride()}")
    if w2.stride(-1) != 1 or w2.stride(-2) % 8 != 0:
        raise RuntimeError(f"repair4 W2 compute layout invalid: {w2.stride()}")

    legacy = TrulySparseMoE(32, 4, 2, 10)
    probe = LogicalPaddedBF16GroupedSparseMoE(32, 4, 2, 10).copy_from_legacy(legacy)
    x1 = torch.randn(3, 5, 32, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    y1 = legacy(x1)
    y2 = probe(x2)
    max_output_delta = float((y1 - y2).abs().max())
    loss1 = y1.square().mean()
    loss2 = y2.square().mean()
    loss1.backward()
    loss2.backward()
    max_input_grad_delta = float((x1.grad - x2.grad).abs().max())
    if max_output_delta > 1e-5 or max_input_grad_delta > 1e-5:
        raise RuntimeError(
            f"repair4 CPU semantics mismatch: output={max_output_delta}, grad={max_input_grad_delta}"
        )
    return {
        "status": "PASS",
        "logical_hidden": hidden,
        "compute_hidden": candidate.compute_hidden,
        "padding_channels": candidate.compute_hidden - hidden,
        "compute_overhead_ratio": candidate.compute_hidden / hidden,
        "w1_compute_shape": list(w1.shape),
        "w1_compute_stride": list(w1.stride()),
        "w2_compute_shape": list(w2.shape),
        "w2_compute_stride": list(w2.stride()),
        "no_slice_between_grouped_mm": True,
        "parameter_count_changed": False,
        "cpu_output_max_abs_delta": max_output_delta,
        "cpu_input_grad_max_abs_delta": max_input_grad_delta,
    }


def validate_microbench_protocol() -> dict[str, Any]:
    forbidden = set(CONSUMED_ENGINEERING_SEEDS) | set(FORBIDDEN_SCIENTIFIC_SEEDS)
    if ENGINEERING_SEED in forbidden:
        raise RuntimeError("repair4 engineering seed collides with consumed/forbidden seed")
    if MICRO_BATCH_SIZE != 64 or SEQ_LEN != 512 or GRAD_ACCUM_STEPS != 2:
        raise RuntimeError("repair4 drifted from production batch shape")
    if CORTEX_100M_CONFIG.expert_hidden != 338:
        raise RuntimeError("repair4 must preserve production hidden width 338")
    return {
        "classification": CLASSIFICATION,
        "engineering_seed": ENGINEERING_SEED,
        "consumed_engineering_seeds": list(CONSUMED_ENGINEERING_SEEDS),
        "forbidden_seeds": sorted(forbidden),
        "trigger_title": TRIGGER_TITLE,
        "result_root": RESULT_ROOT,
        "parent_repair3_source_sha": PARENT_REPAIR3_SOURCE_SHA,
        "parent_repair3_run_id": PARENT_REPAIR3_RUN_ID,
        "parent_repair3_job_id": PARENT_REPAIR3_JOB_ID,
        "measured_steps": MEASURED_STEPS,
        "tokens_per_optimizer_step": TOKENS_PER_OPTIMIZER_STEP,
        "primary_speedup_gate": MIN_PROMISING_SPEEDUP,
        "stop_below_speedup": STOP_GROUPED_BELOW_SPEEDUP,
        "full_training_authorized": False,
        "next_stage_authorized": False,
    }


def _autocast(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


@torch.no_grad()
def _semantic_probe(device: torch.device) -> dict[str, Any]:
    seed_all(ENGINEERING_SEED)
    legacy = build_cortex_100m().to(device).eval()
    seed_all(ENGINEERING_SEED)
    candidate = convert_to_logical_padded_grouped(build_cortex_100m().to(device)).eval()
    tokens = torch.randint(
        0,
        CORTEX_100M_CONFIG.vocab_size,
        (2, 64),
        generator=torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 7),
    ).to(device)
    targets = torch.randint(
        0,
        CORTEX_100M_CONFIG.vocab_size,
        (2, 64),
        generator=torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 9),
    ).to(device)
    features = torch.randn(
        257,
        CORTEX_100M_CONFIG.d_model,
        generator=torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 8),
    ).to(device)
    routing_exact = bool(torch.equal(
        legacy.blocks[0].moe.router(features).topk(2, dim=-1).indices,
        candidate.blocks[0].moe.router(features).topk(2, dim=-1).indices,
    ))
    with _autocast(device):
        legacy_logits = legacy(tokens)
        candidate_logits = candidate(tokens)
        legacy_loss = F.cross_entropy(legacy_logits.float().reshape(-1, legacy_logits.size(-1)), targets.reshape(-1))
        candidate_loss = F.cross_entropy(candidate_logits.float().reshape(-1, candidate_logits.size(-1)), targets.reshape(-1))
    loss_delta = abs(float(legacy_loss) - float(candidate_loss))
    result = {
        "routing_exact": routing_exact,
        "legacy_loss": float(legacy_loss),
        "candidate_loss": float(candidate_loss),
        "absolute_loss_delta": loss_delta,
        "max_absolute_logit_delta": float((legacy_logits.float() - candidate_logits.float()).abs().max()),
        "mean_absolute_logit_delta": float((legacy_logits.float() - candidate_logits.float()).abs().mean()),
        "loss_delta_gate": loss_delta <= MAX_SEMANTIC_LOSS_DELTA,
    }
    del legacy, candidate, tokens, targets, features, legacy_logits, candidate_logits
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _benchmark_full_model_variant(*, label: str, train_data: TokenBin, device: torch.device, grouped: bool) -> dict[str, Any]:
    seed_all(ENGINEERING_SEED)
    model = build_cortex_100m().to(device)
    if grouped:
        model = convert_to_logical_padded_grouped(model)
    if parameter_count(model) != EXPECTED_CORTEX_PARAMS:
        raise RuntimeError(f"{label} parameter count drift")
    optimizer = _make_optimizer(model)
    generator = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 10_000)
    runner: torch.nn.Module | None = None
    try:
        dynamo.reset()
        compile_start = time.perf_counter()
        runner = _compile_model(model)
        last_loss = _one_optimizer_step(
            model=model,
            runner=runner,
            optimizer=optimizer,
            train_data=train_data,
            generator=generator,
            device=device,
            lr=LEARNING_RATE,
        )
        torch.cuda.synchronize(device)
        compile_seconds = time.perf_counter() - compile_start
    except Exception as exc:
        result = {
            "label": label,
            "status": "COMPILE_OR_FIRST_STEP_FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "parameter_count": parameter_count(model),
        }
        del runner, optimizer, model
        gc.collect(); torch.cuda.empty_cache()
        return result

    try:
        for _ in range(ADDITIONAL_WARMUP_STEPS):
            last_loss = _one_optimizer_step(
                model=model, runner=runner, optimizer=optimizer, train_data=train_data,
                generator=generator, device=device, lr=LEARNING_RATE,
            )
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for _ in range(MEASURED_STEPS):
            last_loss = _one_optimizer_step(
                model=model, runner=runner, optimizer=optimizer, train_data=train_data,
                generator=generator, device=device, lr=LEARNING_RATE,
            )
        torch.cuda.synchronize(device)
    except Exception as exc:
        result = {
            "label": label,
            "status": "WARMUP_OR_MEASURED_STEP_FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "compile_seconds": compile_seconds,
            "parameter_count": parameter_count(model),
        }
        del runner, optimizer, model
        gc.collect(); torch.cuda.empty_cache()
        return result

    measured_seconds = max(time.perf_counter() - started, 1e-9)
    measured_tokens = MEASURED_STEPS * TOKENS_PER_OPTIMIZER_STEP
    result = {
        "label": label,
        "status": "PASS" if math.isfinite(float(last_loss)) else "NONFINITE",
        "compile_seconds": compile_seconds,
        "measured_steps": MEASURED_STEPS,
        "measured_tokens": measured_tokens,
        "measured_seconds": measured_seconds,
        "training_tokens_per_second": measured_tokens / measured_seconds,
        "peak_vram_gib": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
        "last_loss": float(last_loss),
        "finite_loss": math.isfinite(float(last_loss)),
        "parameter_count": parameter_count(model),
    }
    del runner, optimizer, model
    gc.collect(); torch.cuda.empty_cache()
    return result


def run_h100_systems_microbenchmark(*, source_sha: str, data_dir: str = DATA_DIR) -> dict[str, Any]:
    protocol = validate_microbench_protocol()
    if not torch.cuda.is_available():
        raise RuntimeError("repair4 systems microbenchmark requires CUDA")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    if torch.cuda.get_device_capability(device) < (9, 0):
        raise RuntimeError("repair4 requires H100-class SM90+")
    if not hasattr(F, "grouped_mm"):
        raise RuntimeError("installed PyTorch does not expose grouped_mm")

    contract = zero_gpu_repair4_contract_probe()
    train_data = TokenBin(f"{data_dir}/train.bin")
    cache_start = time.perf_counter()
    train_data._device_tokens(device)
    torch.cuda.synchronize(device)
    data_cache_seconds = time.perf_counter() - cache_start

    semantic = _semantic_probe(device)
    legacy = _benchmark_full_model_variant(label="legacy", train_data=train_data, device=device, grouped=False)
    grouped = _benchmark_full_model_variant(label="grouped_bf16_repair4", train_data=train_data, device=device, grouped=True)
    scan = _benchmark_scan(device)

    speedup: float | None = None
    if legacy.get("status") == "PASS" and grouped.get("status") == "PASS":
        speedup = float(grouped["training_tokens_per_second"]) / max(float(legacy["training_tokens_per_second"]), 1e-9)

    gates = {
        "contract": contract.get("status") == "PASS",
        "semantic_routing_exact": bool(semantic.get("routing_exact")),
        "semantic_loss_delta": bool(semantic.get("loss_delta_gate")),
        "legacy_pass": legacy.get("status") == "PASS",
        "grouped_pass": grouped.get("status") == "PASS",
        "parameter_count_exact": grouped.get("parameter_count") == EXPECTED_CORTEX_PARAMS,
        "peak_vram": grouped.get("status") == "PASS" and float(grouped.get("peak_vram_gib", float("inf"))) <= MAX_PEAK_VRAM_GIB,
    }
    if grouped.get("status") != "PASS" or legacy.get("status") != "PASS":
        status = "ENGINEERING_RUNTIME_FAIL"
    elif not all(gates.values()):
        status = "ENGINEERING_GATE_FAIL"
    elif speedup is None or speedup < STOP_GROUPED_BELOW_SPEEDUP:
        status = "STOP_GROUPED_PATH"
    elif speedup < MIN_PROMISING_SPEEDUP:
        status = "INSUFFICIENT_SYSTEMS_GAIN"
    else:
        status = "PROMISING_SYSTEMS_CANDIDATE"

    return {
        "status": status,
        "scientific_status": CLASSIFICATION,
        "repair_namespace": "microbench-v1-repair4",
        "source_sha": source_sha,
        "engineering_seed": ENGINEERING_SEED,
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": str(torch.__version__),
        "cuda_capability": list(torch.cuda.get_device_capability(device)),
        "data_cache_seconds": data_cache_seconds,
        "repair4_contract": contract,
        "semantic_probe": semantic,
        "legacy": legacy,
        "grouped_bf16_repair4": grouped,
        "grouped_full_model_speedup": speedup,
        "scan_diagnostic": scan,
        "gates": gates,
        "protocol": protocol,
        "full_training_authorized": False,
        "next_stage_authorized": False,
        "note": "Engineering systems evidence only; repair4 cannot directly authorize the 2B run.",
    }
