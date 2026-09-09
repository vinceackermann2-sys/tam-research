from __future__ import annotations

from contextlib import nullcontext
import gc
import math
import time
from typing import Any, Optional

import torch
from torch import nn
import torch.nn.functional as F

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
from .systems_microbench_v1_repair2 import _grouped_mm_layout_ok, _require_grouped_mm_layout


CLASSIFICATION = "ENGINEERING_SYSTEMS_MICROBENCH_ONLY"
ENGINEERING_SEED = 2_026_090_904
CONSUMED_ENGINEERING_SEEDS = (910_001, 2_026_090_901, 2_026_090_902, 2_026_090_903)
FORBIDDEN_SCIENTIFIC_SEEDS = (8_100, *RESERVED_FRESH_SEEDS)
TRIGGER_TITLE = "[modal-cortex-s-100m-systems-microbench-v1-repair4]"
RESULT_ROOT = "/vol/cortex-s-v0/100m-systems-microbench-v1-repair4"
PARENT_REPAIR3_SOURCE_SHA = "1c0ecd321a530daee32a4603a9cd6fb201be2d8a"
PARENT_REPAIR3_RUN_ID = 34_335_827_718
PARENT_REPAIR3_JOB_ID = 102_414_857_556
PARENT_REPAIR3_ISSUE = 791
PARENT_REPAIR3_GROUPED_FAILURE = (
    "RuntimeError: Expected mat_a stride along 1 dim to be multiple of 16 bytes, got 338."
)

COMPILE_TRIGGER_STEPS = 1
ADDITIONAL_WARMUP_STEPS = 3
MEASURED_STEPS = 20
MIN_PROMISING_SPEEDUP = 1.20
STOP_GROUPED_BELOW_SPEEDUP = 1.10
MAX_PEAK_VRAM_GIB = 70.0
MAX_SEMANTIC_LOSS_DELTA = 0.02
GROUPED_MM_ALIGNMENT_BYTES = 16
BF16_ALIGNMENT_ELEMENTS = GROUPED_MM_ALIGNMENT_BYTES // 2
OPERATOR_PROBE_ASSIGNMENTS_PER_EXPERT = 8


def _padded_hidden(hidden: int) -> int:
    return ((hidden + BF16_ALIGNMENT_ELEMENTS - 1) // BF16_ALIGNMENT_ELEMENTS) * BF16_ALIGNMENT_ELEMENTS


class PhysicalPaddedBF16GroupedSparseMoE(nn.Module):
    """Grouped-MoE candidate with a real aligned runtime hidden dimension.

    Repair2/3 used padded backing storage followed by a logical 338-wide view.
    Inductor canonicalized that view and grouped-mm backward still saw row stride
    338. Repair4 carries six zero channels physically through W1 -> GELU -> W2:
    trainable parameters remain logical width 338, runtime width is 344, and the
    padded channels contribute exactly zero because W1/W2 padding is zero and
    GELU(0)=0.
    """

    def __init__(self, d_model: int, num_experts: int, top_k: int, hidden: int):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden = hidden
        self.physical_hidden = _padded_hidden(hidden)
        self.router = nn.Linear(d_model, num_experts, bias=True)
        self.expert_w1 = nn.Parameter(torch.empty(num_experts, hidden, d_model))
        self.expert_w2 = nn.Parameter(torch.empty(num_experts, hidden, d_model))
        self.reset_parameters()
        self.last_counts: Optional[torch.Tensor] = None

    def reset_parameters(self) -> None:
        self.router.reset_parameters()
        for expert in range(self.num_experts):
            nn.init.normal_(self.expert_w1[expert], mean=0.0, std=0.02)
            nn.init.normal_(self.expert_w2[expert], mean=0.0, std=0.02)

    @torch.no_grad()
    def copy_from_legacy(self, legacy: TrulySparseMoE) -> "PhysicalPaddedBF16GroupedSparseMoE":
        if legacy.num_experts != self.num_experts or legacy.top_k != self.top_k:
            raise ValueError("legacy routing dimensions do not match repair4 candidate")
        self.router.weight.copy_(legacy.router.weight)
        self.router.bias.copy_(legacy.router.bias)
        for index, expert in enumerate(legacy.experts):
            first = expert[0]
            second = expert[2]
            if first.weight.shape != self.expert_w1[index].shape:
                raise ValueError("legacy first projection shape mismatch")
            if second.weight.T.shape != self.expert_w2[index].shape:
                raise ValueError("legacy second projection shape mismatch")
            self.expert_w1[index].copy_(first.weight)
            self.expert_w2[index].copy_(second.weight.T)
        return self

    @staticmethod
    def grouped_cuda_available(x: torch.Tensor) -> bool:
        return bool(
            hasattr(F, "grouped_mm")
            and x.is_cuda
            and torch.cuda.get_device_capability(x.device) >= (8, 0)
        )

    def _physical_matrices(self) -> tuple[torch.Tensor, torch.Tensor]:
        pad = self.physical_hidden - self.hidden
        w1 = self.expert_w1.transpose(-2, -1)
        w1_physical = F.pad(w1, (0, pad)).contiguous() if pad else w1.contiguous()
        w2_physical = (
            F.pad(self.expert_w2, (0, 0, 0, pad)).contiguous()
            if pad
            else self.expert_w2.contiguous()
        )
        return w1_physical, w2_physical

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
        packed_bf16 = packed if packed.dtype == torch.bfloat16 else packed.to(dtype=torch.bfloat16)
        matrices_bf16 = matrices if matrices.dtype == torch.bfloat16 else matrices.to(dtype=torch.bfloat16)
        _require_grouped_mm_layout(packed_bf16, "repair4 grouped_mm lhs")
        _require_grouped_mm_layout(matrices_bf16, "repair4 grouped_mm rhs")
        if offsets.dtype != torch.int32:
            raise RuntimeError(f"repair4 grouped_mm offsets must be int32, got {offsets.dtype}")
        return F.grouped_mm(packed_bf16, matrices_bf16, offs=offsets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])
        router_logits = self.router(flat)
        top_values, top_indices = router_logits.topk(self.top_k, dim=-1)
        assignment_weights = F.softmax(top_values.float(), dim=-1).to(flat.dtype)
        plan = pack_topk_assignments(top_indices, assignment_weights, num_experts=self.num_experts)
        packed = flat.index_select(0, plan.token_indices)

        w1_physical, w2_physical = self._physical_matrices()
        hidden_physical = self._grouped_mm(packed, w1_physical, plan.offsets)
        hidden_physical = F.gelu(hidden_physical)
        expert_output = self._grouped_mm(hidden_physical, w2_physical, plan.offsets)

        weighted = expert_output * plan.assignment_weights[:, None]
        output = torch.zeros_like(flat)
        output.index_add_(0, plan.token_indices, weighted)
        self.last_counts = plan.counts.detach()
        return output.view(original_shape)

    @property
    def theoretical_executed_fraction(self) -> float:
        return self.top_k / self.num_experts


def convert_to_physical_padded_grouped(model: CortexSLM) -> CortexSLM:
    before = parameter_count(model)
    for block in model.blocks:
        legacy = block.moe
        if not isinstance(legacy, TrulySparseMoE):
            raise TypeError("expected production TrulySparseMoE before repair4 conversion")
        first = legacy.experts[0][0]
        device = legacy.router.weight.device
        dtype = legacy.router.weight.dtype
        cuda_devices: list[int] = []
        if device.type == "cuda":
            cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=cuda_devices):
            candidate = PhysicalPaddedBF16GroupedSparseMoE(
                d_model=first.in_features,
                num_experts=legacy.num_experts,
                top_k=legacy.top_k,
                hidden=first.out_features,
            ).to(device=device, dtype=dtype)
        candidate.copy_from_legacy(legacy)
        block.moe = candidate
    after = parameter_count(model)
    if before != after:
        raise RuntimeError(f"repair4 systems conversion changed parameter count: {before} -> {after}")
    return model


def zero_gpu_physical_padding_contract_probe() -> dict[str, Any]:
    hidden = CORTEX_100M_CONFIG.expert_hidden
    d_model = CORTEX_100M_CONFIG.d_model
    experts = CORTEX_100M_CONFIG.num_experts
    candidate = PhysicalPaddedBF16GroupedSparseMoE(
        d_model, experts, CORTEX_100M_CONFIG.top_k, hidden
    ).to(dtype=torch.bfloat16)
    w1, w2 = candidate._physical_matrices()
    activation = torch.zeros(31, candidate.physical_hidden, dtype=torch.bfloat16)

    if candidate.physical_hidden != 344:
        raise RuntimeError(f"repair4 expected physical hidden 344, got {candidate.physical_hidden}")
    if w1.shape != (experts, d_model, 344):
        raise RuntimeError(f"repair4 W1 physical shape drift: {tuple(w1.shape)}")
    if w2.shape != (experts, 344, d_model):
        raise RuntimeError(f"repair4 W2 physical shape drift: {tuple(w2.shape)}")
    if not all(_grouped_mm_layout_ok(t) for t in (w1, w2, activation)):
        raise RuntimeError("repair4 physical padding still violates grouped_mm layout")

    return {
        "status": "PASS",
        "logical_hidden": hidden,
        "physical_hidden": candidate.physical_hidden,
        "padding_channels": candidate.physical_hidden - hidden,
        "w1_shape": list(w1.shape),
        "w1_stride": list(w1.stride()),
        "w2_shape": list(w2.shape),
        "w2_stride": list(w2.stride()),
        "activation_shape": list(activation.shape),
        "activation_stride": list(activation.stride()),
        "all_grouped_layouts_valid": True,
        "runtime_padding_adds_parameters": False,
        "parent_repair3_failed_stride": 338,
    }


def validate_microbench_protocol() -> dict[str, Any]:
    forbidden = set(CONSUMED_ENGINEERING_SEEDS) | set(FORBIDDEN_SCIENTIFIC_SEEDS)
    if ENGINEERING_SEED in forbidden:
        raise RuntimeError("repair4 engineering seed collides with a consumed/forbidden seed")
    if MICRO_BATCH_SIZE != 64 or SEQ_LEN != 512 or GRAD_ACCUM_STEPS != 2:
        raise RuntimeError("repair4 drifted from production batch shape")
    if CORTEX_100M_CONFIG.top_k != 2 or CORTEX_100M_CONFIG.num_experts != 8:
        raise RuntimeError("repair4 router semantics drifted from top-2/8")
    if CORTEX_100M_CONFIG.expert_hidden != 338:
        raise RuntimeError("repair4 must preserve logical production hidden width 338")
    if _padded_hidden(CORTEX_100M_CONFIG.expert_hidden) != 344:
        raise RuntimeError("repair4 BF16 physical hidden contract drift")
    if not (1.0 < STOP_GROUPED_BELOW_SPEEDUP < MIN_PROMISING_SPEEDUP):
        raise RuntimeError("repair4 speedup gates are not ordered")
    return {
        "classification": CLASSIFICATION,
        "engineering_seed": ENGINEERING_SEED,
        "consumed_engineering_seeds": list(CONSUMED_ENGINEERING_SEEDS),
        "trigger_title": TRIGGER_TITLE,
        "result_root": RESULT_ROOT,
        "parent_repair3_source_sha": PARENT_REPAIR3_SOURCE_SHA,
        "parent_repair3_run_id": PARENT_REPAIR3_RUN_ID,
        "parent_repair3_job_id": PARENT_REPAIR3_JOB_ID,
        "parent_repair3_issue": PARENT_REPAIR3_ISSUE,
        "parent_repair3_grouped_failure": PARENT_REPAIR3_GROUPED_FAILURE,
        "logical_hidden": CORTEX_100M_CONFIG.expert_hidden,
        "physical_hidden": _padded_hidden(CORTEX_100M_CONFIG.expert_hidden),
        "compile_trigger_steps": COMPILE_TRIGGER_STEPS,
        "additional_warmup_steps": ADDITIONAL_WARMUP_STEPS,
        "measured_steps": MEASURED_STEPS,
        "tokens_per_optimizer_step": TOKENS_PER_OPTIMIZER_STEP,
        "primary_speedup_gate": MIN_PROMISING_SPEEDUP,
        "stop_below_speedup": STOP_GROUPED_BELOW_SPEEDUP,
        "max_peak_vram_gib": MAX_PEAK_VRAM_GIB,
        "max_semantic_loss_delta": MAX_SEMANTIC_LOSS_DELTA,
        "forbidden_seeds": sorted(forbidden),
        "full_training_authorized": False,
        "next_stage_authorized": False,
    }


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def _semantic_probe(device: torch.device) -> dict[str, Any]:
    seed_all(ENGINEERING_SEED)
    legacy = build_cortex_100m().to(device).eval()
    seed_all(ENGINEERING_SEED)
    candidate = convert_to_physical_padded_grouped(build_cortex_100m().to(device)).eval()
    if parameter_count(legacy) != EXPECTED_CORTEX_PARAMS or parameter_count(candidate) != EXPECTED_CORTEX_PARAMS:
        raise RuntimeError("repair4 semantic probe parameter count mismatch")

    token_generator = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 7)
    tokens = torch.randint(0, CORTEX_100M_CONFIG.vocab_size, (2, 64), generator=token_generator).to(device)
    targets = torch.randint(0, CORTEX_100M_CONFIG.vocab_size, (2, 64), generator=token_generator).to(device)
    feature_generator = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 8)
    features = torch.randn(257, CORTEX_100M_CONFIG.d_model, generator=feature_generator).to(device)
    legacy_router = legacy.blocks[0].moe.router(features).topk(CORTEX_100M_CONFIG.top_k, dim=-1).indices
    candidate_router = candidate.blocks[0].moe.router(features).topk(CORTEX_100M_CONFIG.top_k, dim=-1).indices
    routing_exact = bool(torch.equal(legacy_router, candidate_router))

    with _autocast(device):
        legacy_logits = legacy(tokens)
        candidate_logits = candidate(tokens)
        legacy_loss = F.cross_entropy(
            legacy_logits.float().reshape(-1, legacy_logits.size(-1)), targets.reshape(-1)
        )
        candidate_loss = F.cross_entropy(
            candidate_logits.float().reshape(-1, candidate_logits.size(-1)), targets.reshape(-1)
        )
    loss_delta = abs(float(legacy_loss) - float(candidate_loss))
    max_logit_delta = float((legacy_logits.float() - candidate_logits.float()).abs().max())
    mean_logit_delta = float((legacy_logits.float() - candidate_logits.float()).abs().mean())

    del legacy, candidate, tokens, targets, features
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "routing_exact": routing_exact,
        "legacy_loss": float(legacy_loss),
        "candidate_loss": float(candidate_loss),
        "absolute_loss_delta": loss_delta,
        "max_absolute_logit_delta": max_logit_delta,
        "mean_absolute_logit_delta": mean_logit_delta,
        "loss_delta_gate": loss_delta <= MAX_SEMANTIC_LOSS_DELTA,
    }


def _compiled_grouped_operator_probe(device: torch.device) -> dict[str, Any]:
    """Compile forward+backward at physical hidden=344 before full-model work."""

    experts = CORTEX_100M_CONFIG.num_experts
    d_model = CORTEX_100M_CONFIG.d_model
    hidden = _padded_hidden(CORTEX_100M_CONFIG.expert_hidden)
    n = experts * OPERATOR_PROBE_ASSIGNMENTS_PER_EXPERT
    gen = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 9_000)
    packed = (
        torch.randn(n, d_model, generator=gen, dtype=torch.float32)
        .to(device=device, dtype=torch.bfloat16)
        .requires_grad_(True)
    )
    w1 = (
        torch.randn(experts, d_model, hidden, generator=gen, dtype=torch.float32)
        .to(device=device, dtype=torch.bfloat16)
        .requires_grad_(True)
    )
    w2 = (
        torch.randn(experts, hidden, d_model, generator=gen, dtype=torch.float32)
        .to(device=device, dtype=torch.bfloat16)
        .requires_grad_(True)
    )
    # Match production pack_topk_assignments exactly: grouped_mm requires int32 offs.
    offsets = torch.arange(
        OPERATOR_PROBE_ASSIGNMENTS_PER_EXPERT,
        n + 1,
        OPERATOR_PROBE_ASSIGNMENTS_PER_EXPERT,
        device=device,
        dtype=torch.int32,
    )
    for label, tensor in (("packed", packed), ("w1", w1), ("w2", w2)):
        _require_grouped_mm_layout(tensor, f"repair4 operator probe {label}")

    def objective(
        a: torch.Tensor,
        first: torch.Tensor,
        second: torch.Tensor,
        offs: torch.Tensor,
    ) -> torch.Tensor:
        h = F.grouped_mm(a, first, offs=offs)
        h = F.gelu(h)
        out = F.grouped_mm(h, second, offs=offs)
        return out.float().square().mean()

    from torch import _dynamo as dynamo

    dynamo.reset()
    started = time.perf_counter()
    try:
        compiled = torch.compile(objective)
        loss = compiled(packed, w1, w2, offsets)
        loss.backward()
        torch.cuda.synchronize(device)
    except Exception as exc:
        del packed, w1, w2, offsets
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "status": "COMPILE_OR_BACKWARD_FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "logical_hidden": CORTEX_100M_CONFIG.expert_hidden,
            "physical_hidden": hidden,
            "offsets_dtype": "int32",
        }

    elapsed = time.perf_counter() - started
    finite = bool(torch.isfinite(loss).item())
    grad_finite = all(
        tensor.grad is not None and bool(torch.isfinite(tensor.grad).all().item())
        for tensor in (packed, w1, w2)
    )
    result = {
        "status": "PASS" if finite and grad_finite else "NONFINITE",
        "compile_and_backward_seconds": elapsed,
        "loss": float(loss),
        "finite_loss": finite,
        "finite_gradients": grad_finite,
        "logical_hidden": CORTEX_100M_CONFIG.expert_hidden,
        "physical_hidden": hidden,
        "offsets_dtype": "int32",
        "packed_stride": list(packed.stride()),
        "w1_stride": list(w1.stride()),
        "w2_stride": list(w2.stride()),
    }
    del packed, w1, w2, offsets
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _benchmark_full_model_variant(
    *,
    label: str,
    train_data: TokenBin,
    device: torch.device,
    grouped: bool,
) -> dict[str, Any]:
    seed_all(ENGINEERING_SEED)
    model = build_cortex_100m().to(device)
    if grouped:
        model = convert_to_physical_padded_grouped(model)
    if parameter_count(model) != EXPECTED_CORTEX_PARAMS:
        raise RuntimeError(f"{label} parameter count drift")
    optimizer = _make_optimizer(model)
    generator = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 10_000)
    runner: torch.nn.Module | None = None

    try:
        from torch import _dynamo as dynamo

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
        gc.collect()
        torch.cuda.empty_cache()
        return result

    try:
        for _ in range(ADDITIONAL_WARMUP_STEPS):
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
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for _ in range(MEASURED_STEPS):
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
    except Exception as exc:
        result = {
            "label": label,
            "status": "WARMUP_OR_MEASURED_STEP_FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "compile_seconds": compile_seconds,
            "parameter_count": parameter_count(model),
        }
        del runner, optimizer, model
        gc.collect()
        torch.cuda.empty_cache()
        return result

    measured_seconds = max(time.perf_counter() - started, 1e-9)
    measured_tokens = MEASURED_STEPS * TOKENS_PER_OPTIMIZER_STEP
    tps = measured_tokens / measured_seconds
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    finite = math.isfinite(float(last_loss))
    result = {
        "label": label,
        "status": "PASS" if finite else "NONFINITE",
        "compile_seconds": compile_seconds,
        "measured_steps": MEASURED_STEPS,
        "measured_tokens": measured_tokens,
        "measured_seconds": measured_seconds,
        "training_tokens_per_second": tps,
        "peak_vram_gib": peak_vram,
        "last_loss": float(last_loss),
        "finite_loss": finite,
        "parameter_count": parameter_count(model),
    }
    del runner, optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_h100_systems_microbenchmark(*, source_sha: str, data_dir: str = DATA_DIR) -> dict[str, Any]:
    protocol = validate_microbench_protocol()
    if not torch.cuda.is_available():
        raise RuntimeError("repair4 systems microbenchmark requires CUDA")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    if torch.cuda.get_device_capability(device) < (9, 0):
        raise RuntimeError("repair4 systems benchmark requires H100-class SM90+")
    if not hasattr(F, "grouped_mm"):
        raise RuntimeError("installed PyTorch does not expose torch.nn.functional.grouped_mm")

    layout_contract = zero_gpu_physical_padding_contract_probe()
    train_data = TokenBin(f"{data_dir}/train.bin")
    cache_start = time.perf_counter()
    train_data._device_tokens(device)
    torch.cuda.synchronize(device)
    data_cache_seconds = time.perf_counter() - cache_start

    operator_probe = _compiled_grouped_operator_probe(device)
    if operator_probe.get("status") != "PASS":
        return {
            "status": "STOP_GROUPED_PATH",
            "scientific_status": CLASSIFICATION,
            "repair_namespace": "microbench-v1-repair4",
            "source_sha": source_sha,
            "engineering_seed": ENGINEERING_SEED,
            "device_name": torch.cuda.get_device_name(device),
            "torch_version": str(torch.__version__),
            "cuda_capability": list(torch.cuda.get_device_capability(device)),
            "data_cache_seconds": data_cache_seconds,
            "physical_padding_contract": layout_contract,
            "compiled_grouped_operator_probe": operator_probe,
            "semantic_probe": None,
            "legacy": None,
            "grouped_bf16_repair4": None,
            "grouped_full_model_speedup": None,
            "gates": {"physical_padding_contract": True, "compiled_operator_probe": False},
            "protocol": protocol,
            "full_training_authorized": False,
            "next_stage_authorized": False,
            "note": "Compiled physical-padding operator probe failed, so repair4 stopped before full-model benchmarking.",
        }

    semantic = _semantic_probe(device)
    legacy = _benchmark_full_model_variant(
        label="legacy", train_data=train_data, device=device, grouped=False
    )
    grouped = _benchmark_full_model_variant(
        label="grouped_bf16_repair4", train_data=train_data, device=device, grouped=True
    )

    gates = {
        "physical_padding_contract": layout_contract.get("status") == "PASS",
        "compiled_operator_probe": operator_probe.get("status") == "PASS",
        "semantic_routing_exact": bool(semantic.get("routing_exact")),
        "semantic_loss_delta": bool(semantic.get("loss_delta_gate")),
        "legacy_pass": legacy.get("status") == "PASS",
        "grouped_pass": grouped.get("status") == "PASS",
        "parameter_count_exact": grouped.get("parameter_count") == EXPECTED_CORTEX_PARAMS,
        "peak_vram": float(grouped.get("peak_vram_gib", float("inf"))) <= MAX_PEAK_VRAM_GIB,
    }
    speedup: float | None = None
    if legacy.get("status") == "PASS" and grouped.get("status") == "PASS":
        speedup = float(grouped["training_tokens_per_second"]) / max(
            float(legacy["training_tokens_per_second"]), 1e-9
        )

    if not all(gates.values()) or speedup is None or speedup < STOP_GROUPED_BELOW_SPEEDUP:
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
        "physical_padding_contract": layout_contract,
        "compiled_grouped_operator_probe": operator_probe,
        "semantic_probe": semantic,
        "legacy": legacy,
        "grouped_bf16_repair4": grouped,
        "grouped_full_model_speedup": speedup,
        "gates": gates,
        "protocol": protocol,
        "full_training_authorized": False,
        "next_stage_authorized": False,
        "note": "Engineering systems evidence only. No repair4 result directly authorizes the 2B run.",
    }
