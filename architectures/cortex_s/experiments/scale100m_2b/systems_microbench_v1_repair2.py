from __future__ import annotations

from contextlib import nullcontext
import gc
import math
import time
from typing import Any, Optional

import torch
from torch import nn
import torch.nn.functional as F

from architectures.cortex_s.language_model import CortexSLM, TrulySparseMoE, affine_scan, parameter_count
from architectures.cortex_s.systems_optimization import affine_scan_slice_candidate, pack_topk_assignments
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


CLASSIFICATION = "ENGINEERING_SYSTEMS_MICROBENCH_ONLY"
ENGINEERING_SEED = 2_026_090_902
CONSUMED_ENGINEERING_SEEDS = (910_001, 2_026_090_901)
FORBIDDEN_SCIENTIFIC_SEEDS = (8_100, *RESERVED_FRESH_SEEDS)
TRIGGER_TITLE = "[modal-cortex-s-100m-systems-microbench-v1-repair2]"
RESULT_ROOT = "/vol/cortex-s-v0/100m-systems-microbench-v1-repair2"

COMPILE_TRIGGER_STEPS = 1
ADDITIONAL_WARMUP_STEPS = 3
MEASURED_STEPS = 20
MIN_PROMISING_SPEEDUP = 1.20
STOP_GROUPED_BELOW_SPEEDUP = 1.10
MAX_PEAK_VRAM_GIB = 70.0
MAX_SEMANTIC_LOSS_DELTA = 0.02
SCAN_WARMUP_ITERS = 3
SCAN_MEASURED_ITERS = 10
GROUPED_MM_ALIGNMENT_BYTES = 16


def _grouped_mm_layout_ok(tensor: torch.Tensor) -> bool:
    """Mirror PyTorch GroupedMMUtils stride contract without allocating a GPU."""

    if tensor.dim() not in (2, 3):
        return False
    alignment = GROUPED_MM_ALIGNMENT_BYTES // tensor.element_size()
    row_stride = tensor.stride(-2)
    col_stride = tensor.stride(-1)
    rows = tensor.size(-2)
    cols = tensor.size(-1)
    transposed = row_stride == 1 and col_stride >= max(1, rows) and col_stride % alignment == 0
    row_major = col_stride == 1 and row_stride >= max(1, cols) and row_stride % alignment == 0
    return bool(transposed or row_major)


def _require_grouped_mm_layout(tensor: torch.Tensor, label: str) -> None:
    if not _grouped_mm_layout_ok(tensor):
        raise RuntimeError(
            f"{label} violates grouped_mm 16-byte stride contract: "
            f"shape={tuple(tensor.shape)}, strides={tuple(tensor.stride())}, "
            f"element_size={tensor.element_size()}"
        )
    if tensor.is_cuda and tensor.data_ptr() % GROUPED_MM_ALIGNMENT_BYTES != 0:
        raise RuntimeError(f"{label} data_ptr is not 16-byte aligned")


def _aligned_row_view(tensor: torch.Tensor) -> torch.Tensor:
    """Preserve logical width while giving a 2-D tensor a 16-byte row stride.

    PyTorch grouped_mm itself returns padded-stride outputs on CUDA, but GELU
    materializes the logical hidden width contiguously. CORTEX-S uses hidden=338,
    so BF16 GELU output has a 676-byte row stride, which violates grouped_mm's
    16-byte contract. Padding physical storage to 344 BF16 values and slicing back
    to logical width 338 keeps semantics/parameter count unchanged while restoring
    an aligned 688-byte row stride. Autograd flows through pad/slice normally.
    """

    if tensor.dim() != 2:
        raise ValueError("aligned grouped rows require a 2-D tensor")
    if _grouped_mm_layout_ok(tensor):
        return tensor
    alignment = GROUPED_MM_ALIGNMENT_BYTES // tensor.element_size()
    width = tensor.size(-1)
    padded_width = ((width + alignment - 1) // alignment) * alignment
    if padded_width == width:
        aligned = tensor.contiguous()
    else:
        storage = F.pad(tensor, (0, padded_width - width))
        aligned = storage[..., :width]
    _require_grouped_mm_layout(aligned, "aligned activation")
    return aligned


class StrideAlignedBF16GroupedSparseMoE(nn.Module):
    """Repair2 candidate preserving exact MoE values with grouped_mm-valid layouts.

    The consumed repair1 stored W1 in multiplication orientation [E, in, out].
    With expert_hidden=338, contiguous BF16 W1 therefore had row stride 338 * 2
    = 676 bytes and H100 grouped_mm rejected it. Repair2 stores W1 in the native
    Linear orientation [E, out, in] and passes a transpose *view*. Its logical
    multiplication shape is still [E, in, out], but the non-unit stride is 512,
    which is 16-byte aligned. W2 remains [E, hidden, d_model] contiguous because
    its logical output width is 512 and already aligned.
    """

    def __init__(self, d_model: int, num_experts: int, top_k: int, hidden: int):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden = hidden
        self.router = nn.Linear(d_model, num_experts, bias=True)
        # W1 native Linear storage [expert, out=hidden, in=d_model].
        self.expert_w1 = nn.Parameter(torch.empty(num_experts, hidden, d_model))
        # W2 multiplication storage [expert, in=hidden, out=d_model].
        self.expert_w2 = nn.Parameter(torch.empty(num_experts, hidden, d_model))
        self.reset_parameters()
        self.last_counts: Optional[torch.Tensor] = None

    def reset_parameters(self) -> None:
        self.router.reset_parameters()
        for expert in range(self.num_experts):
            nn.init.normal_(self.expert_w1[expert], mean=0.0, std=0.02)
            nn.init.normal_(self.expert_w2[expert], mean=0.0, std=0.02)

    @torch.no_grad()
    def copy_from_legacy(self, legacy: TrulySparseMoE) -> "StrideAlignedBF16GroupedSparseMoE":
        if legacy.num_experts != self.num_experts or legacy.top_k != self.top_k:
            raise ValueError("legacy routing dimensions do not match candidate")
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
        # torch.to preserves a dense transpose view's strides; W1 intentionally
        # arrives here as that transpose view. W2 is contiguous/aligned already.
        matrices_bf16 = matrices if matrices.dtype == torch.bfloat16 else matrices.to(dtype=torch.bfloat16)
        _require_grouped_mm_layout(packed_bf16, "grouped_mm lhs")
        _require_grouped_mm_layout(matrices_bf16, "grouped_mm rhs")
        return F.grouped_mm(packed_bf16, matrices_bf16, offs=offsets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])
        router_logits = self.router(flat)
        top_values, top_indices = router_logits.topk(self.top_k, dim=-1)
        assignment_weights = F.softmax(top_values.float(), dim=-1).to(flat.dtype)
        plan = pack_topk_assignments(top_indices, assignment_weights, num_experts=self.num_experts)
        packed = flat.index_select(0, plan.token_indices)

        # Logical W1 multiplication shape [E, d_model, hidden] via an aligned
        # transpose view of native [E, hidden, d_model] storage.
        hidden = self._grouped_mm(packed, self.expert_w1.transpose(-2, -1), plan.offsets)
        hidden = F.gelu(hidden)
        if self.grouped_cuda_available(hidden):
            hidden = _aligned_row_view(hidden)
        expert_output = self._grouped_mm(hidden, self.expert_w2, plan.offsets)

        weighted = expert_output * plan.assignment_weights[:, None]
        output = torch.zeros_like(flat)
        output.index_add_(0, plan.token_indices, weighted)
        self.last_counts = plan.counts.detach()
        return output.view(original_shape)

    @property
    def theoretical_executed_fraction(self) -> float:
        return self.top_k / self.num_experts


def convert_to_stride_aligned_grouped(model: CortexSLM) -> CortexSLM:
    before = parameter_count(model)
    for block in model.blocks:
        legacy = block.moe
        if not isinstance(legacy, TrulySparseMoE):
            raise TypeError("expected production TrulySparseMoE before repair2 conversion")
        first = legacy.experts[0][0]
        device = legacy.router.weight.device
        dtype = legacy.router.weight.dtype
        cuda_devices: list[int] = []
        if device.type == "cuda":
            cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=cuda_devices):
            candidate = StrideAlignedBF16GroupedSparseMoE(
                d_model=first.in_features,
                num_experts=legacy.num_experts,
                top_k=legacy.top_k,
                hidden=first.out_features,
            ).to(device=device, dtype=dtype)
        candidate.copy_from_legacy(legacy)
        block.moe = candidate
    after = parameter_count(model)
    if before != after:
        raise RuntimeError(f"repair2 systems conversion changed parameter count: {before} -> {after}")
    return model


def zero_gpu_layout_contract_probe() -> dict[str, Any]:
    """CPU-only regression for the exact H100 stride failure seen in repair1."""

    hidden = CORTEX_100M_CONFIG.expert_hidden
    d_model = CORTEX_100M_CONFIG.d_model
    experts = CORTEX_100M_CONFIG.num_experts
    candidate = StrideAlignedBF16GroupedSparseMoE(d_model, experts, CORTEX_100M_CONFIG.top_k, hidden)

    old_w1 = torch.empty(experts, d_model, hidden, dtype=torch.bfloat16)
    repaired_w1 = candidate.expert_w1.detach().to(torch.bfloat16).transpose(-2, -1)
    repaired_w2 = candidate.expert_w2.detach().to(torch.bfloat16)
    gelu_like = torch.zeros(31, hidden, dtype=torch.bfloat16)
    repaired_activation = _aligned_row_view(gelu_like)

    old_w1_ok = _grouped_mm_layout_ok(old_w1)
    w1_ok = _grouped_mm_layout_ok(repaired_w1)
    w2_ok = _grouped_mm_layout_ok(repaired_w2)
    activation_ok = _grouped_mm_layout_ok(repaired_activation)
    if old_w1_ok:
        raise RuntimeError("repair2 regression fixture no longer reproduces the repair1 invalid W1 stride")
    if not all((w1_ok, w2_ok, activation_ok)):
        raise RuntimeError("repair2 layout contract probe failed")
    return {
        "status": "PASS",
        "old_repair1_w1_stride": list(old_w1.stride()),
        "old_repair1_w1_valid": old_w1_ok,
        "repair2_w1_logical_stride": list(repaired_w1.stride()),
        "repair2_w1_valid": w1_ok,
        "repair2_post_gelu_stride": list(repaired_activation.stride()),
        "repair2_post_gelu_valid": activation_ok,
        "repair2_w2_stride": list(repaired_w2.stride()),
        "repair2_w2_valid": w2_ok,
        "logical_hidden": hidden,
        "physical_hidden_row_stride": repaired_activation.stride(-2),
        "parameter_count_changed": False,
    }


def validate_microbench_protocol() -> dict[str, Any]:
    forbidden = set(CONSUMED_ENGINEERING_SEEDS) | set(FORBIDDEN_SCIENTIFIC_SEEDS)
    if ENGINEERING_SEED in forbidden:
        raise RuntimeError("repair2 engineering seed collides with a consumed/forbidden seed")
    if MICRO_BATCH_SIZE != 64 or SEQ_LEN != 512 or GRAD_ACCUM_STEPS != 2:
        raise RuntimeError("repair2 drifted from production batch shape")
    if CORTEX_100M_CONFIG.top_k != 2 or CORTEX_100M_CONFIG.num_experts != 8:
        raise RuntimeError("repair2 router semantics drifted from top-2/8")
    if CORTEX_100M_CONFIG.expert_hidden != 338:
        raise RuntimeError("repair2 must test the exact production hidden width that exposed the stride bug")
    if not (1.0 < STOP_GROUPED_BELOW_SPEEDUP < MIN_PROMISING_SPEEDUP):
        raise RuntimeError("speedup gates are not ordered")
    return {
        "classification": CLASSIFICATION,
        "engineering_seed": ENGINEERING_SEED,
        "consumed_engineering_seeds": list(CONSUMED_ENGINEERING_SEEDS),
        "trigger_title": TRIGGER_TITLE,
        "result_root": RESULT_ROOT,
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
    candidate = convert_to_stride_aligned_grouped(build_cortex_100m().to(device)).eval()
    if parameter_count(legacy) != EXPECTED_CORTEX_PARAMS or parameter_count(candidate) != EXPECTED_CORTEX_PARAMS:
        raise RuntimeError("repair2 semantic probe parameter count mismatch")

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
        legacy_loss = F.cross_entropy(legacy_logits.float().reshape(-1, legacy_logits.size(-1)), targets.reshape(-1))
        candidate_loss = F.cross_entropy(candidate_logits.float().reshape(-1, candidate_logits.size(-1)), targets.reshape(-1))
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


def _benchmark_full_model_variant(*, label: str, train_data: TokenBin, device: torch.device, grouped: bool) -> dict[str, Any]:
    seed_all(ENGINEERING_SEED)
    model = build_cortex_100m().to(device)
    if grouped:
        model = convert_to_stride_aligned_grouped(model)
    if parameter_count(model) != EXPECTED_CORTEX_PARAMS:
        raise RuntimeError(f"{label} parameter count drift")
    optimizer = _make_optimizer(model)
    generator = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 10_000)
    runner: torch.nn.Module | None = None

    try:
        import torch._dynamo
        torch._dynamo.reset()
        compile_start = time.perf_counter()
        runner = _compile_model(model)
        last_loss = _one_optimizer_step(
            model=model, runner=runner, optimizer=optimizer, train_data=train_data,
            generator=generator, device=device, lr=LEARNING_RATE,
        )
        torch.cuda.synchronize(device)
        compile_seconds = time.perf_counter() - compile_start
    except Exception as exc:
        result = {"label": label, "status": "COMPILE_OR_FIRST_STEP_FAIL", "error": f"{type(exc).__name__}: {exc}", "parameter_count": parameter_count(model)}
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
        result = {"label": label, "status": "WARMUP_OR_MEASURED_STEP_FAIL", "error": f"{type(exc).__name__}: {exc}", "compile_seconds": compile_seconds, "parameter_count": parameter_count(model)}
        del runner, optimizer, model
        gc.collect(); torch.cuda.empty_cache()
        return result

    measured_seconds = max(time.perf_counter() - started, 1e-9)
    measured_tokens = MEASURED_STEPS * TOKENS_PER_OPTIMIZER_STEP
    tps = measured_tokens / measured_seconds
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    finite = math.isfinite(last_loss)
    result = {
        "label": label,
        "status": "PASS" if finite else "NONFINITE",
        "compile_seconds": compile_seconds,
        "measured_steps": MEASURED_STEPS,
        "measured_tokens": measured_tokens,
        "measured_seconds": measured_seconds,
        "training_tokens_per_second": tps,
        "peak_vram_gib": peak_vram,
        "last_loss": last_loss,
        "finite_loss": finite,
        "parameter_count": parameter_count(model),
    }
    del runner, optimizer, model
    gc.collect(); torch.cuda.empty_cache()
    return result


def _scan_once(fn, a: torch.Tensor, b: torch.Tensor, initial: torch.Tensor):
    for tensor in (a, b, initial):
        if tensor.grad is not None:
            tensor.grad = None
    out = fn(a, b, initial)
    loss = out.float().square().mean()
    loss.backward()
    grads = (a.grad.detach().clone(), b.grad.detach().clone(), initial.grad.detach().clone())
    return out.detach(), grads


def _benchmark_scan(device: torch.device) -> dict[str, Any]:
    gen = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 30_000)
    a0 = torch.sigmoid(torch.randn(64, 512, 128, generator=gen)).to(device=device, dtype=torch.bfloat16)
    b0 = (0.1 * torch.randn(64, 512, 128, generator=gen)).to(device=device, dtype=torch.bfloat16)
    i0 = torch.randn(64, 128, generator=gen).to(device=device, dtype=torch.bfloat16)

    def fresh():
        return a0.detach().clone().requires_grad_(True), b0.detach().clone().requires_grad_(True), i0.detach().clone().requires_grad_(True)

    la, lb, li = fresh(); ca, cb, ci = fresh()
    legacy_out, legacy_grads = _scan_once(affine_scan, la, lb, li)
    candidate_out, candidate_grads = _scan_once(affine_scan_slice_candidate, ca, cb, ci)
    output_max_delta = float((legacy_out.float() - candidate_out.float()).abs().max())
    grad_max_delta = max(float((left.float() - right.float()).abs().max()) for left, right in zip(legacy_grads, candidate_grads))
    del la, lb, li, ca, cb, ci, legacy_out, candidate_out, legacy_grads, candidate_grads

    timings: dict[str, float] = {}
    for name, fn in (("legacy", affine_scan), ("slice_candidate", affine_scan_slice_candidate)):
        for _ in range(SCAN_WARMUP_ITERS):
            a, b, initial = fresh(); _scan_once(fn, a, b, initial)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(SCAN_MEASURED_ITERS):
            a, b, initial = fresh(); _scan_once(fn, a, b, initial)
        torch.cuda.synchronize(device)
        timings[name] = (time.perf_counter() - started) / SCAN_MEASURED_ITERS

    return {
        "shape": [64, 512, 128],
        "dtype": "bfloat16",
        "forward_backward_seconds_per_iter": timings,
        "slice_speedup": timings["legacy"] / max(timings["slice_candidate"], 1e-9),
        "output_max_abs_delta": output_max_delta,
        "gradient_max_abs_delta": grad_max_delta,
    }


def run_h100_systems_microbenchmark(*, source_sha: str, data_dir: str = DATA_DIR) -> dict[str, Any]:
    protocol = validate_microbench_protocol()
    if not torch.cuda.is_available():
        raise RuntimeError("repair2 systems microbenchmark requires CUDA")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    if torch.cuda.get_device_capability(device) < (9, 0):
        raise RuntimeError("repair2 systems benchmark requires H100-class SM90+")
    if not hasattr(F, "grouped_mm"):
        raise RuntimeError("installed PyTorch does not expose torch.nn.functional.grouped_mm")

    # Re-run the exact stride regression on the installed runtime before touching
    # the full model. This is cheap and fail-closed.
    layout_contract = zero_gpu_layout_contract_probe()

    train_data = TokenBin(f"{data_dir}/train.bin")
    cache_start = time.perf_counter()
    train_data._device_tokens(device)
    torch.cuda.synchronize(device)
    data_cache_seconds = time.perf_counter() - cache_start

    semantic = _semantic_probe(device)
    legacy = _benchmark_full_model_variant(label="legacy", train_data=train_data, device=device, grouped=False)
    grouped = _benchmark_full_model_variant(label="grouped_bf16_repair2", train_data=train_data, device=device, grouped=True)
    scan = _benchmark_scan(device)

    gates = {
        "layout_contract": layout_contract.get("status") == "PASS",
        "semantic_routing_exact": bool(semantic.get("routing_exact")),
        "semantic_loss_delta": bool(semantic.get("loss_delta_gate")),
        "legacy_pass": legacy.get("status") == "PASS",
        "grouped_pass": grouped.get("status") == "PASS",
        "parameter_count_exact": grouped.get("parameter_count") == EXPECTED_CORTEX_PARAMS,
        "peak_vram": float(grouped.get("peak_vram_gib", float("inf"))) <= MAX_PEAK_VRAM_GIB,
    }
    speedup: float | None = None
    if legacy.get("status") == "PASS" and grouped.get("status") == "PASS":
        speedup = float(grouped["training_tokens_per_second"]) / max(float(legacy["training_tokens_per_second"]), 1e-9)

    if not all(gates.values()) or speedup is None or speedup < STOP_GROUPED_BELOW_SPEEDUP:
        status = "STOP_GROUPED_PATH"
    elif speedup < MIN_PROMISING_SPEEDUP:
        status = "INSUFFICIENT_SYSTEMS_GAIN"
    else:
        status = "PROMISING_SYSTEMS_CANDIDATE"

    return {
        "status": status,
        "scientific_status": CLASSIFICATION,
        "repair_namespace": "microbench-v1-repair2",
        "source_sha": source_sha,
        "engineering_seed": ENGINEERING_SEED,
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_capability": list(torch.cuda.get_device_capability(device)),
        "data_cache_seconds": data_cache_seconds,
        "layout_contract": layout_contract,
        "semantic_probe": semantic,
        "legacy": legacy,
        "grouped_bf16_repair2": grouped,
        "grouped_full_model_speedup": speedup,
        "scan_diagnostic": scan,
        "gates": gates,
        "protocol": protocol,
        "full_training_authorized": False,
        "next_stage_authorized": False,
        "note": "Engineering systems evidence only. No result from this repair2 benchmark directly authorizes the 2B run.",
    }
