from __future__ import annotations

import base64
import json
from pathlib import Path
import time
from typing import Any

import modal

JOB_ID = "rlt-systems-scan-scale15m-batch768-modal-20261004-ac"
ENGINEERING_SEED = 20_261_040
SEQ_LEN = 64
BATCH_SIZE = 768
CHECK_BATCH = 2
CHECK_SEQ_LEN = 16
CAUSAL_MUTATION_INDEX = 37
WARMUP_STEPS = 1
MEASURED_STEPS = 3
EXPECTED_PARAMETERS = 15_129_344

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11", "numpy>=2.0,<3")
    .add_local_python_source("tam_research", "experiments")
)
app = modal.App("tam-rlt-systems-scan-scale15m-batch768-20261004-ac")


def _encode(value: dict[str, Any]) -> str:
    return base64.urlsafe_b64encode(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).decode()


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=4.0,
    memory=65536,
    timeout=70 * 60,
    retries=0,
    max_containers=1,
)
def run_profile_remote(job_json: str) -> str:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from experiments.rlt.model import RLTConfig, RecurrentLoopedTransformer, parameter_count
    from experiments.rlt.model_scan import ScanRecurrentLoopedTransformer
    from tam_research.models import ModelConfig, ResearchLM, parameter_count as transformer_parameter_count

    started = time.time()
    job = json.loads(job_json)
    expected = {
        "schema": 1,
        "job_id": JOB_ID,
        "task": "systems_scan_scale15m_batch768_ac",
        "engineering_seed": ENGINEERING_SEED,
        "requires_block4_replication_job_id": "rlt-block4-scale15m-paired-4m-modal-20260930-b",
        "requires_optimizer_tuned_compute_match_job_id": "rlt-compute-matched-optimizer-tuned-scale15m-60s-modal-20260928-a",
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "expected_parameters_each": EXPECTED_PARAMETERS,
        "automatic_retry": False,
    }
    if job != expected:
        raise RuntimeError(f"scan profile preregistration mismatch: {job}")
    if not torch.cuda.is_available():
        raise RuntimeError("scan systems profile requires CUDA")

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    rlt_cfg = RLTConfig(
        vocab_size=50_257, d_model=256, n_heads=8, n_stages=2,
        max_seq_len=128, ff_mult=4, swa_window=32,
    )
    tr_cfg = ModelConfig(
        vocab_size=50_257, d_model=256, n_layers=3, n_heads=8,
        max_seq_len=128, ff_mult=4, ff_inner=938, architecture="transformer",
    )

    def seed_all(seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def new_reference() -> RecurrentLoopedTransformer:
        seed_all(ENGINEERING_SEED)
        m = RecurrentLoopedTransformer(rlt_cfg).to(device)
        if parameter_count(m) != EXPECTED_PARAMETERS:
            raise RuntimeError(f"reference parameter drift: {parameter_count(m)}")
        return m

    def new_scan() -> ScanRecurrentLoopedTransformer:
        seed_all(ENGINEERING_SEED)
        m = ScanRecurrentLoopedTransformer(rlt_cfg).to(device)
        if parameter_count(m) != EXPECTED_PARAMETERS:
            raise RuntimeError(f"scan parameter drift: {parameter_count(m)}")
        return m

    def new_transformer() -> ResearchLM:
        seed_all(ENGINEERING_SEED)
        m = ResearchLM(tr_cfg).to(device)
        if transformer_parameter_count(m) != EXPECTED_PARAMETERS:
            raise RuntimeError(f"Transformer parameter drift: {transformer_parameter_count(m)}")
        return m

    reference = new_reference()
    scan = new_scan()
    schema_equal = tuple(reference.state_dict().keys()) == tuple(scan.state_dict().keys())
    if not schema_equal:
        raise RuntimeError("scan model state/parameter schema differs from frozen RLT")
    del reference
    torch.cuda.empty_cache()

    # Verify the differentiable vectorized recurrence against its serial algebra.
    g = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 90)
    memory = torch.randn(
        CHECK_BATCH, CHECK_SEQ_LEN, rlt_cfg.d_model,
        generator=g, dtype=torch.float32,
    ).to(device)

    def serial_states(model: ScanRecurrentLoopedTransformer, mem: torch.Tensor) -> torch.Tensor:
        d = model.cfg.d_model
        weight = model.merge.weight
        candidate = torch.tanh(F.linear(mem, weight[:, :d]))
        gate_logits = F.linear(mem, weight[:, d:])
        gate = model.GATE_FLOOR + (
            model.GATE_CEILING - model.GATE_FLOOR
        ) * torch.sigmoid(gate_logits)
        state = model.start_state.float().view(1, -1).expand(mem.size(0), -1)
        rows = []
        for token_index in range(mem.size(1)):
            state = gate[:, token_index, :].float() * state + (
                1.0 - gate[:, token_index, :].float()
            ) * candidate[:, token_index, :].float()
            rows.append(state)
        return torch.stack(rows, dim=1)

    scan.zero_grad(set_to_none=True)
    vector_states = scan._scan_states(memory)
    vector_loss = vector_states.float().square().mean()
    vector_loss.backward()
    vector_weight_grad = scan.merge.weight.grad.detach().clone()
    vector_start_grad = scan.start_state.grad.detach().clone()

    scan.zero_grad(set_to_none=True)
    serial = serial_states(scan, memory)
    serial_loss = serial.square().mean()
    serial_loss.backward()
    serial_weight_grad = scan.merge.weight.grad.detach().clone()
    serial_start_grad = scan.start_state.grad.detach().clone()

    state_abs = float((vector_states.float() - serial).abs().max().item())
    state_loss_abs = float((vector_loss - serial_loss).abs().item())
    weight_grad_abs = float((vector_weight_grad - serial_weight_grad).abs().max().item())
    start_grad_abs = float((vector_start_grad - serial_start_grad).abs().max().item())
    recurrence_passed = (
        state_abs <= 2e-5
        and state_loss_abs <= 1e-6
        and weight_grad_abs <= 2e-5
        and start_grad_abs <= 2e-5
    )
    if not recurrence_passed:
        raise RuntimeError(
            "vectorized scan != serial recurrence within tolerance: "
            f"state={state_abs}, loss={state_loss_abs}, "
            f"weight_grad={weight_grad_abs}, start_grad={start_grad_abs}"
        )

    # Full-model future-token causality gate.
    scan = new_scan()
    g2 = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 91)
    causal_x = torch.randint(
        0, rlt_cfg.vocab_size, (CHECK_BATCH, SEQ_LEN),
        generator=g2, dtype=torch.long,
    ).to(device)
    changed_x = causal_x.clone()
    changed_x[:, CAUSAL_MUTATION_INDEX] = (
        changed_x[:, CAUSAL_MUTATION_INDEX] + 1
    ) % rlt_cfg.vocab_size

    scan.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        original_logits = scan(causal_x)
        changed_logits = scan(changed_x)
    torch.cuda.synchronize(device)
    prefix_abs = float(
        (
            original_logits[:, :CAUSAL_MUTATION_INDEX, :]
            - changed_logits[:, :CAUSAL_MUTATION_INDEX, :]
        ).abs().max().item()
    )
    changed_position_abs = float(
        (
            original_logits[:, CAUSAL_MUTATION_INDEX, :]
            - changed_logits[:, CAUSAL_MUTATION_INDEX, :]
        ).abs().max().item()
    )
    causality_passed = prefix_abs == 0.0 and changed_position_abs > 0.0
    if not causality_passed:
        raise RuntimeError(
            "scan future-token causality failed: "
            f"prefix={prefix_abs}, changed={changed_position_abs}"
        )
    del scan, original_logits, changed_logits
    torch.cuda.empty_cache()

    batch_gen = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 1)
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(WARMUP_STEPS + MEASURED_STEPS + 2):
        x = torch.randint(
            0, 50_257, (BATCH_SIZE, SEQ_LEN),
            generator=batch_gen, dtype=torch.long,
        ).to(device)
        y = torch.randint(
            0, 50_257, (BATCH_SIZE, SEQ_LEN),
            generator=batch_gen, dtype=torch.long,
        ).to(device)
        batches.append((x, y))

    def one_step(
        callable_model: Any,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        offset: int,
    ) -> float:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        x, y = batches[offset % len(batches)]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = callable_model(x)
            loss = F.cross_entropy(logits.float().reshape(-1, 50_257), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return float(loss.detach().cpu())

    def benchmark(kind: str) -> dict[str, Any]:
        if kind == "scan":
            base: nn.Module = new_scan()
        elif kind == "transformer":
            base = new_transformer()
        else:
            raise ValueError(kind)

        optimizer = torch.optim.AdamW(
            base.parameters(), lr=3e-4, betas=(0.9, 0.95),
            weight_decay=0.1, fused=True,
        )
        compiled = torch.compile(base, fullgraph=True, dynamic=False, mode="default")
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        first_loss = one_step(compiled, base, optimizer, 0)
        torch.cuda.synchronize(device)
        first_step_seconds = time.perf_counter() - t0

        for i in range(WARMUP_STEPS):
            one_step(compiled, base, optimizer, 1 + i)
        torch.cuda.synchronize(device)

        torch.cuda.reset_peak_memory_stats(device)
        t1 = time.perf_counter()
        last_loss = first_loss
        for i in range(MEASURED_STEPS):
            last_loss = one_step(
                compiled, base, optimizer, 1 + WARMUP_STEPS + i
            )
        torch.cuda.synchronize(device)
        measured_seconds = time.perf_counter() - t1
        measured_tokens = MEASURED_STEPS * BATCH_SIZE * SEQ_LEN

        out = {
            "parameters": sum(p.numel() for p in base.parameters()),
            "first_step_seconds_including_compile": first_step_seconds,
            "measured_seconds": measured_seconds,
            "measured_tokens": measured_tokens,
            "tokens_per_second": measured_tokens / max(measured_seconds, 1e-9),
            "last_loss": last_loss,
            "peak_vram_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
        }
        del compiled, optimizer, base
        torch.cuda.empty_cache()
        torch._dynamo.reset()
        return out

    scan_profile = benchmark("scan")
    transformer = benchmark("transformer")
    gap = (
        float(transformer["tokens_per_second"])
        / max(float(scan_profile["tokens_per_second"]), 1e-9)
    )
    if gap <= 1.15:
        classification = "PASS_SCAN_SCALE15M_GAP_LE_1_15X"
    elif gap <= 1.35:
        classification = "PASS_SCAN_SCALE15M_GAP_LE_1_35X"
    else:
        classification = "PASS_SCAN_SCALE15M_GAP_GT_1_35X"

    return _encode({
        "schema": 1,
        "job_id": JOB_ID,
        "status": "complete",
        "classification": classification,
        "engineering_only": True,
        "scientific_execution": False,
        "breakthrough_claim_supported": False,
        "engineering_seed": ENGINEERING_SEED,
        "started_unix": started,
        "finished_unix": time.time(),
        "runtime": {
            "torch_version": str(torch.__version__),
            "cuda_runtime": None if torch.version.cuda is None else str(torch.version.cuda),
            "gpu_name": str(torch.cuda.get_device_name(0)),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        },
        "architecture": {
            "variant": "scan_recurrent_rlt",
            "token_state_updates_per_seq64": 64,
            "serial_python_token_loop": False,
            "parameters": EXPECTED_PARAMETERS,
            "state_schema_equal_to_frozen_rlt": schema_equal,
            "parameter_neutral": True,
            "gate_floor": 0.5,
            "gate_ceiling": 0.99,
        },
        "implementation_validation": {
            "vectorized_vs_serial_state_recurrence": {
                "passed": recurrence_passed,
                "state_max_abs": state_abs,
                "loss_abs": state_loss_abs,
                "merge_weight_grad_max_abs": weight_grad_abs,
                "start_state_grad_max_abs": start_grad_abs,
                "tolerances": {
                    "state": 2e-5,
                    "loss": 1e-6,
                    "gradient": 2e-5,
                },
            },
            "future_token_causality": {
                "passed": causality_passed,
                "mutation_index": CAUSAL_MUTATION_INDEX,
                "prefix_logits_max_abs": prefix_abs,
                "mutated_position_logits_max_abs": changed_position_abs,
            },
        },
        "profile": {
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "warmup_steps": WARMUP_STEPS,
            "measured_steps": MEASURED_STEPS,
            "expected_parameters_each": EXPECTED_PARAMETERS,
        },
        "scan": scan_profile,
        "transformer": transformer,
        "derived": {
            "transformer_throughput_multiple_vs_scan": gap,
            "scan_fraction_of_transformer_throughput": 1.0 / gap,
        },
        "interpretation_ceiling": (
            "Engineering-only causality, recurrence-algebra, and throughput profile "
            "for a parameter-neutral token-state scan RLT variant. Random-token losses "
            "are not quality evidence. A favorable speed result only authorizes a fresh "
            "equal-parameter/equal-token quality experiment."
        ),
    })


@app.local_entrypoint()
def main(job_path: str) -> None:
    result = run_profile_remote.remote(Path(job_path).read_text())
    if not isinstance(result, str):
        raise TypeError("remote scan systems result must be base64 text")
    print("RLT_SYSTEMS_SCAN_SCALE15M_BATCH768_AC_RESULT_B64=" + result)
