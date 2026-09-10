from __future__ import annotations

import json
from pathlib import Path
import time

import modal


APP_NAME = "cortex-s-affine-scan-triton-integrated-cuda-equivalence-v1"
VOLUME_NAME = "tam-research-data"
RESULT_ROOT = "/vol/cortex-s-v0/affine-scan-triton-integrated-cuda-equivalence-v1"
ENGINEERING_SEED = 2_026_091_001
H100_TIMEOUT_SECONDS = 10 * 60

FROZEN_SOURCE_SHA = "46b29efb5cc0d975b4f4e6b3c397248723cacf3f"
FROZEN_SOURCE_TREE = "dc485e134eba99b709fcfa302a69616928286d7e"
INTEGRATION_BLOB = "c3d05f8b4d1e3e8ea7bcbad5fdeca5d7091651c7"
TRITON_CANDIDATE_BLOB = "5c6bf7df95f49c306b4ffadd6a8a09b9f71344f9"
PRODUCTION_SHAPE = (64, 512, 512, 128)
FP32_PROBE_SHAPE = (4, 64, 64, 16)
BF16_ATOL = 0.05
BF16_RTOL = 0.05
FP32_ATOL = 5e-5
FP32_RTOL = 5e-4
EXPECTED_CORTEX_PARAMS = 101_778_112

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11")
    .add_local_python_source("architectures")
)


def protocol() -> dict[str, object]:
    return {
        "classification": "ENGINEERING_INTEGRATED_CUDA_EQUIVALENCE_ONLY",
        "engineering_seed": ENGINEERING_SEED,
        "frozen_source_sha": FROZEN_SOURCE_SHA,
        "frozen_source_tree": FROZEN_SOURCE_TREE,
        "integration_blob": INTEGRATION_BLOB,
        "triton_candidate_blob": TRITON_CANDIDATE_BLOB,
        "production_shape": list(PRODUCTION_SHAPE),
        "fp32_probe_shape": list(FP32_PROBE_SHAPE),
        "bf16_tolerance": {"atol": BF16_ATOL, "rtol": BF16_RTOL},
        "fp32_tolerance": {"atol": FP32_ATOL, "rtol": FP32_RTOL},
        "expected_cortex_params": EXPECTED_CORTEX_PARAMS,
        "result_root": RESULT_ROOT,
        "gpu_dispatch_authorized": False,
        "production_preflight_authorized": False,
        "full_training_authorized": False,
        "scientific_claim_authorized": False,
    }


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=10 * 60,
    volumes={"/vol": volume},
)
def verify_zero_gpu(source_sha: str) -> str:
    if source_sha != FROZEN_SOURCE_SHA:
        raise RuntimeError("source SHA is not the frozen PR #821 integration head")

    volume.reload()
    root = Path(RESULT_ROOT)
    gate_path = root / "ZERO_GPU_GATE.json"
    dispatch_path = root / "H100_DISPATCH_CONSUMED.json"
    result_path = root / "RESULT.json"
    if gate_path.exists() or dispatch_path.exists() or result_path.exists():
        raise RuntimeError("integrated CUDA-equivalence namespace is already consumed")

    result = {
        "status": "PASS",
        "classification": "ZERO_GPU_PROTOCOL_GATE_ONLY",
        "source_sha": source_sha,
        "engineering_seed": ENGINEERING_SEED,
        "protocol": protocol(),
        "gpu_allocated": False,
        "production_preflight_authorized": False,
        "full_training_authorized": False,
    }
    root.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    volume.commit()
    print(json.dumps({"zero_gpu_gate": result}, indent=2, sort_keys=True), flush=True)
    return json.dumps(result, sort_keys=True)


@app.function(
    image=image,
    gpu="H100!",
    cpu=4,
    memory=16384,
    timeout=H100_TIMEOUT_SECONDS,
    volumes={"/vol": volume},
)
def h100_integrated_cuda_equivalence(source_sha: str) -> str:
    import copy

    import torch

    import architectures.cortex_s.production_scan_integration_v1 as integration
    from architectures.cortex_s.grouped_moe_v4 import (
        MemoryLeanPhysicalPaddedGroupedSparseMoE,
    )
    from architectures.cortex_s.language_model import CortexSLM, CortexSLMConfig, parameter_count

    def comparison(left: torch.Tensor, right: torch.Tensor, *, atol: float, rtol: float) -> dict[str, object]:
        left_f = left.detach().float()
        right_f = right.detach().float()
        delta = (left_f - right_f).abs()
        return {
            "allclose": bool(torch.allclose(left_f, right_f, atol=atol, rtol=rtol)),
            "finite_left": bool(torch.isfinite(left_f).all().item()),
            "finite_right": bool(torch.isfinite(right_f).all().item()),
            "max_abs_delta": float(delta.max().item()) if delta.numel() else 0.0,
            "mean_abs_delta": float(delta.mean().item()) if delta.numel() else 0.0,
        }

    def run_world_probe(
        *,
        batch: int,
        length: int,
        d_model: int,
        state_size: int,
        dtype: torch.dtype,
        atol: float,
        rtol: float,
        seed_offset: int,
    ) -> dict[str, object]:
        if d_model % 8:
            raise RuntimeError("probe d_model must be divisible by 8")

        cfg = CortexSLMConfig(
            vocab_size=64,
            d_model=d_model,
            n_layers=1,
            n_heads=8,
            max_seq_len=length,
            state_size=state_size,
            num_experts=2,
            top_k=1,
            expert_hidden=16,
            attention_every=97,
        )
        device = torch.device("cuda")

        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(ENGINEERING_SEED + seed_offset)
            torch.cuda.manual_seed_all(ENGINEERING_SEED + seed_offset)
            base = CortexSLM(cfg).to(device=device, dtype=dtype)

        reference_model = copy.deepcopy(base)
        integrated_model = copy.deepcopy(base)
        del base

        cpu_rng_before = torch.random.get_rng_state().clone()
        cuda_rng_before = torch.cuda.get_rng_state().clone()
        integrated_model = integration.convert_world_state_to_triton_scan(integrated_model)
        cpu_rng_after = torch.random.get_rng_state()
        cuda_rng_after = torch.cuda.get_rng_state()
        rng_preserved = bool(
            torch.equal(cpu_rng_before, cpu_rng_after)
            and torch.equal(cuda_rng_before, cuda_rng_after)
        )

        reference_world = reference_model.blocks[0].world
        integrated_world = integrated_model.blocks[0].world
        if not isinstance(integrated_world, integration.ProductionScanPersistentWorldState):
            raise RuntimeError("conversion did not install ProductionScanPersistentWorldState")

        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(ENGINEERING_SEED + seed_offset + 1)
            torch.cuda.manual_seed_all(ENGINEERING_SEED + seed_offset + 1)
            x = torch.randn((batch, length, d_model), device=device, dtype=dtype)
            initial = torch.randn((batch, state_size), device=device, dtype=dtype)
            upstream_out = torch.randn((batch, length, d_model), device=device, dtype=dtype)
            upstream_states = torch.randn((batch, length, state_size), device=device, dtype=dtype)
            upstream_carry = torch.randn((batch, state_size), device=device, dtype=dtype)

        x_ref = x.detach().clone().requires_grad_(True)
        x_int = x.detach().clone().requires_grad_(True)
        state_ref = initial.detach().clone().requires_grad_(True)
        state_int = initial.detach().clone().requires_grad_(True)

        if not integration._should_use_triton(x_int):
            raise RuntimeError("production dispatcher predicate rejected a real CUDA tensor")

        call_counter = {"count": 0}
        original_candidate = integration.affine_scan_triton_candidate

        def counted_candidate(a, b, initial_state=None):
            call_counter["count"] += 1
            return original_candidate(a, b, initial_state)

        integration.affine_scan_triton_candidate = counted_candidate
        try:
            int_out, int_carry, int_states = integrated_world(x_int, state_int)
        finally:
            integration.affine_scan_triton_candidate = original_candidate

        ref_out, ref_carry, ref_states = reference_world(x_ref, state_ref)

        ref_loss = (
            (ref_out.float() * upstream_out.float()).mean()
            + (ref_states.float() * upstream_states.float()).mean()
            + (ref_carry.float() * upstream_carry.float()).mean()
        )
        int_loss = (
            (int_out.float() * upstream_out.float()).mean()
            + (int_states.float() * upstream_states.float()).mean()
            + (int_carry.float() * upstream_carry.float()).mean()
        )
        ref_loss.backward()
        int_loss.backward()

        metrics: dict[str, object] = {
            "world_output": comparison(ref_out, int_out, atol=atol, rtol=rtol),
            "full_states": comparison(ref_states, int_states, atol=atol, rtol=rtol),
            "final_carry": comparison(ref_carry, int_carry, atol=atol, rtol=rtol),
            "grad_input": comparison(x_ref.grad, x_int.grad, atol=atol, rtol=rtol),
            "grad_initial": comparison(state_ref.grad, state_int.grad, atol=atol, rtol=rtol),
        }

        ref_params = dict(reference_world.named_parameters())
        int_params = dict(integrated_world.named_parameters())
        if ref_params.keys() != int_params.keys():
            raise RuntimeError("world parameter names differ after conversion")
        parameter_metrics: dict[str, object] = {}
        for name in ref_params:
            left_grad = ref_params[name].grad
            right_grad = int_params[name].grad
            if left_grad is None or right_grad is None:
                raise RuntimeError(f"missing parameter gradient for {name}")
            parameter_metrics[name] = comparison(left_grad, right_grad, atol=atol, rtol=rtol)
        metrics["parameter_grads"] = parameter_metrics

        metric_passes = [
            bool(value["allclose"] and value["finite_left"] and value["finite_right"])
            for key, value in metrics.items()
            if key != "parameter_grads"
        ]
        metric_passes.extend(
            bool(value["allclose"] and value["finite_left"] and value["finite_right"])
            for value in parameter_metrics.values()
        )
        passed = bool(
            call_counter["count"] == 1
            and rng_preserved
            and all(metric_passes)
        )

        return {
            "shape": [batch, length, d_model, state_size],
            "dtype": str(dtype),
            "dispatcher_selected": True,
            "candidate_call_count": call_counter["count"],
            "conversion_rng_preserved": rng_preserved,
            "metrics": metrics,
            "pass": passed,
        }

    if source_sha != FROZEN_SOURCE_SHA:
        raise RuntimeError("source SHA is not the frozen PR #821 integration head")

    volume.reload()
    root = Path(RESULT_ROOT)
    gate_path = root / "ZERO_GPU_GATE.json"
    dispatch_path = root / "H100_DISPATCH_CONSUMED.json"
    result_path = root / "RESULT.json"
    if not gate_path.exists():
        raise RuntimeError("zero-GPU gate is missing")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS" or gate.get("source_sha") != source_sha:
        raise RuntimeError("zero-GPU gate does not bind this exact source")
    if dispatch_path.exists() or result_path.exists():
        raise RuntimeError("integrated CUDA-equivalence namespace is already consumed")

    dispatch = {
        "source_sha": source_sha,
        "engineering_seed": ENGINEERING_SEED,
        "dispatched_unix": time.time(),
        "scientific_seed_used": False,
        "production_preflight_authorized": False,
        "full_training_authorized": False,
    }
    dispatch_path.write_text(json.dumps(dispatch, indent=2, sort_keys=True), encoding="utf-8")
    volume.commit()
    print(json.dumps({"h100_dispatch": dispatch}, indent=2, sort_keys=True), flush=True)

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        device_name = torch.cuda.get_device_name()
        capability = torch.cuda.get_device_capability()
        if capability[0] < 9:
            raise RuntimeError(f"SM90+ required, got compute capability {capability}")
        if "H100" not in device_name.upper():
            raise RuntimeError(f"H100-class device required, got {device_name}")
        if not integration.triton_available():
            raise RuntimeError("Triton is not importable in the integrated source")

        bf16 = run_world_probe(
            batch=PRODUCTION_SHAPE[0],
            length=PRODUCTION_SHAPE[1],
            d_model=PRODUCTION_SHAPE[2],
            state_size=PRODUCTION_SHAPE[3],
            dtype=torch.bfloat16,
            atol=BF16_ATOL,
            rtol=BF16_RTOL,
            seed_offset=10,
        )
        if not bf16["pass"]:
            raise RuntimeError("BF16 integrated production-shape equivalence failed")

        fp32 = run_world_probe(
            batch=FP32_PROBE_SHAPE[0],
            length=FP32_PROBE_SHAPE[1],
            d_model=FP32_PROBE_SHAPE[2],
            state_size=FP32_PROBE_SHAPE[3],
            dtype=torch.float32,
            atol=FP32_ATOL,
            rtol=FP32_RTOL,
            seed_offset=20,
        )
        if not fp32["pass"]:
            raise RuntimeError("FP32 integrated CUDA equivalence failed")

        model_100m = integration.build_memory_lean_grouped_triton_scan_cortex_100m()
        count_100m = parameter_count(model_100m)
        worlds_integrated = all(
            isinstance(block.world, integration.ProductionScanPersistentWorldState)
            for block in model_100m.blocks
        )
        grouped_v4 = all(
            isinstance(block.moe, MemoryLeanPhysicalPaddedGroupedSparseMoE)
            for block in model_100m.blocks
        )
        grouped_widths = all(
            getattr(block.moe, "hidden", None) == 338
            and getattr(block.moe, "physical_hidden", None) == 344
            for block in model_100m.blocks
        )
        builder_pass = bool(
            count_100m == EXPECTED_CORTEX_PARAMS
            and worlds_integrated
            and grouped_v4
            and grouped_widths
        )
        if not builder_pass:
            raise RuntimeError("integrated 100M builder invariant check failed")

        result = {
            "status": "INTEGRATED_CUDA_EQUIVALENCE_PASS",
            "classification": "ENGINEERING_INTEGRATED_CUDA_EQUIVALENCE_ONLY",
            "source_sha": source_sha,
            "engineering_seed": ENGINEERING_SEED,
            "device_name": device_name,
            "cuda_capability": list(capability),
            "torch_version": torch.__version__,
            "bf16_production_world": bf16,
            "fp32_probe_world": fp32,
            "builder_100m": {
                "parameter_count": count_100m,
                "expected_parameter_count": EXPECTED_CORTEX_PARAMS,
                "worlds_integrated": worlds_integrated,
                "grouped_v4_backend_preserved": grouped_v4,
                "logical_expert_hidden": 338,
                "physical_expert_hidden": 344,
                "pass": builder_pass,
            },
            "timing_measured": False,
            "optimizer_step_executed": False,
            "corpus_accessed": False,
            "production_preflight_authorized": False,
            "full_training_authorized": False,
            "scientific_claim_authorized": False,
            "note": "Integration correctness only; no whole-model speed or scientific authority.",
        }
    except Exception as exc:
        result = {
            "status": "INTEGRATED_CUDA_EQUIVALENCE_FAIL",
            "classification": "ENGINEERING_INTEGRATED_CUDA_EQUIVALENCE_ONLY",
            "source_sha": source_sha,
            "engineering_seed": ENGINEERING_SEED,
            "error": f"{type(exc).__name__}: {exc}",
            "timing_measured": False,
            "optimizer_step_executed": False,
            "corpus_accessed": False,
            "production_preflight_authorized": False,
            "full_training_authorized": False,
            "scientific_claim_authorized": False,
        }

    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    volume.commit()
    print(json.dumps({"result": result}, indent=2, sort_keys=True), flush=True)
    return json.dumps(result, sort_keys=True)


@app.local_entrypoint()
def main(phase: str = "integrated-cuda-equivalence-v1", source_sha: str = ""):
    if phase.strip().lower() != "integrated-cuda-equivalence-v1":
        raise ValueError("only phase='integrated-cuda-equivalence-v1' is supported")
    source_sha = source_sha.strip().lower()
    if source_sha != FROZEN_SOURCE_SHA:
        raise ValueError("source_sha must be the frozen PR #821 integration head")

    gate_raw = verify_zero_gpu.remote(source_sha)
    gate = json.loads(gate_raw)
    print(json.dumps({"zero_gpu": gate}, indent=2, sort_keys=True), flush=True)

    result_raw = h100_integrated_cuda_equivalence.remote(source_sha)
    result = json.loads(result_raw)
    print(json.dumps({"h100": result}, indent=2, sort_keys=True), flush=True)
    if result.get("status") != "INTEGRATED_CUDA_EQUIVALENCE_PASS":
        raise RuntimeError(
            "integrated CUDA equivalence did not pass; durable RESULT.json is authoritative"
        )
