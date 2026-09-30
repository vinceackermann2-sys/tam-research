from __future__ import annotations

import base64
import json
from pathlib import Path
import time
from typing import Any

import modal

JOB_ID = "rlt-systems-state-scan-block4-scale15m-batch768-modal-20260930-ac"
ENGINEERING_SEED = 20_261_040
SEQ_LEN = 64
BLOCK_SIZE = 4
BATCH_SIZE = 768
CHECK_BATCH = 2
CAUSAL_MUTATION_INDEX = 37
WARMUP_STEPS = 1
MEASURED_STEPS = 2
EXPECTED_PARAMETERS = 15_129_344

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11", "numpy>=2.0,<3")
    .add_local_python_source("tam_research", "experiments")
)
app = modal.App("tam-rlt-systems-state-scan-block4-scale15m-20260930-ac")


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
    from experiments.rlt.model_block import BlockRecurrentLoopedTransformer
    from experiments.rlt.model_state_scan import StateScanBlockRecurrentLoopedTransformer
    from tam_research.models import (
        ModelConfig,
        ResearchLM,
        parameter_count as transformer_parameter_count,
    )

    started = time.time()
    job = json.loads(job_json)
    expected = {
        "schema": 1,
        "job_id": JOB_ID,
        "task": "systems_state_scan_block4_scale15m_batch768_ac",
        "engineering_seed": ENGINEERING_SEED,
        "requires_block_sweep_job_id": "rlt-systems-block-sweep-scale15m-batch768-modal-20260929-ab",
        "requires_block4_aggregate_id": "rlt-block4-scale15m-4m-ab-aggregate-20260930-a",
        "block_size": BLOCK_SIZE,
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "expected_parameters_each": EXPECTED_PARAMETERS,
        "automatic_retry": False,
    }
    if job != expected:
        raise RuntimeError(f"state-scan block4 profile preregistration mismatch: {job}")
    if not torch.cuda.is_available():
        raise RuntimeError("state-scan block4 systems profile requires CUDA")

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

    def new_scan(block_size: int = BLOCK_SIZE) -> StateScanBlockRecurrentLoopedTransformer:
        seed_all(ENGINEERING_SEED)
        m = StateScanBlockRecurrentLoopedTransformer(rlt_cfg, block_size=block_size).to(device)
        if parameter_count(m) != EXPECTED_PARAMETERS:
            raise RuntimeError(f"state-scan parameter drift: {parameter_count(m)}")
        return m

    def new_plain_block4() -> BlockRecurrentLoopedTransformer:
        seed_all(ENGINEERING_SEED)
        m = BlockRecurrentLoopedTransformer(rlt_cfg, block_size=BLOCK_SIZE).to(device)
        if parameter_count(m) != EXPECTED_PARAMETERS:
            raise RuntimeError(f"plain block4 parameter drift: {parameter_count(m)}")
        return m

    def new_transformer() -> ResearchLM:
        seed_all(ENGINEERING_SEED)
        m = ResearchLM(tr_cfg).to(device)
        if transformer_parameter_count(m) != EXPECTED_PARAMETERS:
            raise RuntimeError(f"Transformer parameter drift: {transformer_parameter_count(m)}")
        return m

    # Exact anchor: state-scan block_size=1 must route through the frozen RLT graph.
    reference = new_reference()
    scan1 = new_scan(1)
    scan1.load_state_dict(reference.state_dict(), strict=True)
    gen = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 91)
    x0 = torch.randint(0, rlt_cfg.vocab_size, (CHECK_BATCH, SEQ_LEN), generator=gen, dtype=torch.long).to(device)
    y0 = torch.randint(0, rlt_cfg.vocab_size, (CHECK_BATCH, SEQ_LEN), generator=gen, dtype=torch.long).to(device)
    reference.zero_grad(set_to_none=True)
    scan1.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        a = reference(x0)
        b = scan1(x0)
        la = F.cross_entropy(a.float().reshape(-1, rlt_cfg.vocab_size), y0.reshape(-1))
        lb = F.cross_entropy(b.float().reshape(-1, rlt_cfg.vocab_size), y0.reshape(-1))
    la.backward(); lb.backward(); torch.cuda.synchronize(device)
    logits_abs = float((a-b).abs().max().item())
    loss_abs = float((la-lb).abs().item())
    grad_abs = 0.0
    ar = dict(reference.named_parameters()); br = dict(scan1.named_parameters())
    if ar.keys() != br.keys():
        raise RuntimeError("state-scan block1 parameter names differ from frozen RLT")
    for name in ar:
        xg, yg = ar[name].grad, br[name].grad
        if xg is None or yg is None:
            if xg is not yg:
                grad_abs = float("inf"); break
            continue
        grad_abs = max(grad_abs, float((xg-yg).abs().max().item()))
    exact = logits_abs == 0.0 and loss_abs == 0.0 and grad_abs == 0.0
    if not exact:
        raise RuntimeError(f"state-scan block1 exact anchor failed: {logits_abs}, {loss_abs}, {grad_abs}")
    ref_schema = tuple(reference.state_dict().keys())
    del reference, scan1, a, b
    torch.cuda.empty_cache()

    # Parameter/state schema and future-token causality.
    scan4 = new_scan()
    if tuple(scan4.state_dict().keys()) != ref_schema:
        raise RuntimeError("state-scan block4 changed state schema")
    gen = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 92)
    x = torch.randint(0, rlt_cfg.vocab_size, (CHECK_BATCH, SEQ_LEN), generator=gen, dtype=torch.long).to(device)
    x2 = x.clone()
    x2[:, CAUSAL_MUTATION_INDEX] = (x2[:, CAUSAL_MUTATION_INDEX] + 1) % rlt_cfg.vocab_size
    scan4.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y = scan4(x)
        y2 = scan4(x2)
    torch.cuda.synchronize(device)
    prefix_abs = float((y[:, :CAUSAL_MUTATION_INDEX, :] - y2[:, :CAUSAL_MUTATION_INDEX, :]).abs().max().item())
    changed_abs = float((y[:, CAUSAL_MUTATION_INDEX, :] - y2[:, CAUSAL_MUTATION_INDEX, :]).abs().max().item())
    causal = prefix_abs == 0.0 and changed_abs > 0.0
    if not causal:
        raise RuntimeError(f"state-scan block4 causality failed: prefix={prefix_abs}, changed={changed_abs}")
    del scan4, y, y2
    torch.cuda.empty_cache()

    gen = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 1)
    batches = []
    for _ in range(WARMUP_STEPS + MEASURED_STEPS + 2):
        bx = torch.randint(0, rlt_cfg.vocab_size, (BATCH_SIZE, SEQ_LEN), generator=gen, dtype=torch.long).to(device)
        by = torch.randint(0, rlt_cfg.vocab_size, (BATCH_SIZE, SEQ_LEN), generator=gen, dtype=torch.long).to(device)
        batches.append((bx, by))

    def one_step(callable_model: Any, model: nn.Module, opt: torch.optim.Optimizer, offset: int) -> float:
        model.train(); opt.zero_grad(set_to_none=True)
        bx, by = batches[offset % len(batches)]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = callable_model(bx)
            loss = F.cross_entropy(logits.float().reshape(-1, 50_257), by.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        return float(loss.detach().cpu())

    def benchmark(kind: str) -> dict[str, Any]:
        if kind == "state_scan_block4":
            base: nn.Module = new_scan()
        elif kind == "plain_block4":
            base = new_plain_block4()
        elif kind == "transformer":
            base = new_transformer()
        else:
            raise ValueError(kind)

        opt = torch.optim.AdamW(base.parameters(), lr=3e-4, betas=(0.9,0.95), weight_decay=0.1, fused=True)
        compiled = torch.compile(base, fullgraph=True, dynamic=False, mode="default")
        torch.cuda.synchronize(device)
        t0=time.perf_counter()
        first_loss=one_step(compiled,base,opt,0)
        torch.cuda.synchronize(device)
        first_seconds=time.perf_counter()-t0
        for i in range(WARMUP_STEPS):
            one_step(compiled,base,opt,1+i)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        t1=time.perf_counter()
        last_loss=first_loss
        for i in range(MEASURED_STEPS):
            last_loss=one_step(compiled,base,opt,1+WARMUP_STEPS+i)
        torch.cuda.synchronize(device)
        measured=time.perf_counter()-t1
        tokens=MEASURED_STEPS*BATCH_SIZE*SEQ_LEN
        out={
            "parameters":sum(p.numel() for p in base.parameters()),
            "first_step_seconds_including_compile":first_seconds,
            "measured_seconds":measured,
            "measured_tokens":tokens,
            "tokens_per_second":tokens/max(measured,1e-9),
            "last_loss":last_loss,
            "peak_vram_gb":torch.cuda.max_memory_allocated(device)/(1024**3),
        }
        del compiled,opt,base
        torch.cuda.empty_cache(); torch._dynamo.reset()
        return out

    state_scan=benchmark("state_scan_block4")
    plain=benchmark("plain_block4")
    transformer=benchmark("transformer")
    tr=float(transformer["tokens_per_second"])
    scan=float(state_scan["tokens_per_second"])
    plain_tps=float(plain["tokens_per_second"])

    gap=tr/max(scan,1e-9)
    if gap <= 1.5:
        classification="PASS_STATE_SCAN_BLOCK4_SCALE15M_GAP_LE_1_5X"
    elif gap <= 2.0:
        classification="PASS_STATE_SCAN_BLOCK4_SCALE15M_GAP_LE_2X"
    else:
        classification="PASS_STATE_SCAN_BLOCK4_SCALE15M_GAP_GT_2X"

    return _encode({
        "schema":1,
        "job_id":JOB_ID,
        "status":"complete",
        "classification":classification,
        "engineering_only":True,
        "scientific_execution":False,
        "breakthrough_claim_supported":False,
        "engineering_seed":ENGINEERING_SEED,
        "started_unix":started,
        "finished_unix":time.time(),
        "runtime":{
            "torch_version":str(torch.__version__),
            "cuda_runtime":None if torch.version.cuda is None else str(torch.version.cuda),
            "gpu_name":str(torch.cuda.get_device_name(0)),
            "bf16_supported":bool(torch.cuda.is_bf16_supported()),
        },
        "architecture":{
            "variant":"state_scan_block_recurrent_rlt",
            "block_size":BLOCK_SIZE,
            "expensive_decoder_blocks_per_seq64":SEQ_LEN//BLOCK_SIZE,
            "cheap_merge_state_updates_per_seq64":SEQ_LEN,
            "parameters":EXPECTED_PARAMETERS,
            "parameter_neutral":True,
            "state_schema_equal_to_frozen_rlt":True,
        },
        "implementation_validation":{
            "block_size_1_exact_reference_equivalence":{
                "passed":exact,"logits_max_abs":logits_abs,"loss_abs":loss_abs,"grad_max_abs":grad_abs,
            },
            "block_size_4_future_token_causality":{
                "passed":causal,"mutation_index":CAUSAL_MUTATION_INDEX,
                "prefix_logits_max_abs":prefix_abs,"mutated_position_logits_max_abs":changed_abs,
            },
        },
        "profile":{
            "batch_size":BATCH_SIZE,"seq_len":SEQ_LEN,"block_size":BLOCK_SIZE,
            "warmup_steps":WARMUP_STEPS,"measured_steps":MEASURED_STEPS,
            "expected_parameters_each":EXPECTED_PARAMETERS,
        },
        "state_scan_block4":state_scan,
        "plain_block4":plain,
        "transformer":transformer,
        "derived":{
            "transformer_throughput_multiple_vs_state_scan_block4":gap,
            "transformer_throughput_multiple_vs_plain_block4":tr/max(plain_tps,1e-9),
            "state_scan_throughput_fraction_vs_plain_block4":scan/max(plain_tps,1e-9),
            "state_scan_fraction_of_transformer_throughput":scan/max(tr,1e-9),
        },
        "interpretation_ceiling":(
            "Engineering-only throughput/causality profile for a parameter-neutral state-scan block4 RLT. "
            "Random-token losses are not quality evidence. Proceed to fresh equal-token training only if "
            "the state-scan variant retains a material systems advantage."
        ),
    })


@app.local_entrypoint()
def main(job_path: str) -> None:
    result=run_profile_remote.remote(Path(job_path).read_text())
    if not isinstance(result,str):
        raise TypeError("remote state-scan block4 systems result must be base64 text")
    print("RLT_SYSTEMS_STATE_SCAN_BLOCK4_SCALE15M_BATCH768_AC_RESULT_B64="+result)
