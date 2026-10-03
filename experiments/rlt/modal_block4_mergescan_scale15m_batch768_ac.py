from __future__ import annotations

import base64
import json
from pathlib import Path
import time
from typing import Any

import modal

JOB_ID = "rlt-systems-block4-mergescan-scale15m-batch768-modal-20261003-ac"
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
app = modal.App("tam-rlt-systems-block4-mergescan-scale15m-batch768-20261003-ac")


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
    from experiments.rlt.model_block_mergescan import MergeScanBlockRecurrentLoopedTransformer
    from tam_research.models import ModelConfig, ResearchLM, parameter_count as transformer_parameter_count

    started = time.time()
    job = json.loads(job_json)
    expected = {
        "schema": 1,
        "job_id": JOB_ID,
        "task": "systems_block4_mergescan_scale15m_batch768_ac",
        "engineering_seed": ENGINEERING_SEED,
        "requires_block_sweep_job_id": "rlt-systems-block-sweep-scale15m-batch768-modal-20260929-ab",
        "requires_block4_crossdata_job_id": "rlt-crossdata-block4-4m-ab-modal-20260930-a",
        "block_size": BLOCK_SIZE,
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "expected_parameters_each": EXPECTED_PARAMETERS,
        "automatic_retry": False,
    }
    if job != expected:
        raise RuntimeError(f"merge-scan profile preregistration mismatch: {job}")
    if not torch.cuda.is_available():
        raise RuntimeError("merge-scan systems profile requires CUDA")

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

    def seed_local(seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def new_reference() -> RecurrentLoopedTransformer:
        seed_local(ENGINEERING_SEED)
        m = RecurrentLoopedTransformer(rlt_cfg).to(device)
        if parameter_count(m) != EXPECTED_PARAMETERS:
            raise RuntimeError(f"reference parameter drift: {parameter_count(m)}")
        return m

    def new_plain() -> BlockRecurrentLoopedTransformer:
        seed_local(ENGINEERING_SEED)
        m = BlockRecurrentLoopedTransformer(rlt_cfg, block_size=BLOCK_SIZE).to(device)
        if parameter_count(m) != EXPECTED_PARAMETERS:
            raise RuntimeError(f"plain block4 parameter drift: {parameter_count(m)}")
        return m

    def new_scan(block_size: int = BLOCK_SIZE) -> MergeScanBlockRecurrentLoopedTransformer:
        seed_local(ENGINEERING_SEED)
        m = MergeScanBlockRecurrentLoopedTransformer(rlt_cfg, block_size=block_size).to(device)
        if parameter_count(m) != EXPECTED_PARAMETERS:
            raise RuntimeError(f"merge-scan parameter drift: {parameter_count(m)}")
        return m

    def new_transformer() -> ResearchLM:
        seed_local(ENGINEERING_SEED)
        m = ResearchLM(tr_cfg).to(device)
        if transformer_parameter_count(m) != EXPECTED_PARAMETERS:
            raise RuntimeError(f"Transformer parameter drift: {transformer_parameter_count(m)}")
        return m

    # Exact anchor: merge-scan block_size=1 must be the frozen token RLT.
    ref = new_reference()
    scan1 = new_scan(1)
    scan1.load_state_dict(ref.state_dict(), strict=True)
    g = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 90)
    x0 = torch.randint(0, rlt_cfg.vocab_size, (CHECK_BATCH, SEQ_LEN), generator=g, dtype=torch.long).to(device)
    y0 = torch.randint(0, rlt_cfg.vocab_size, (CHECK_BATCH, SEQ_LEN), generator=g, dtype=torch.long).to(device)
    ref.zero_grad(set_to_none=True); scan1.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        a = ref(x0); b = scan1(x0)
        la = F.cross_entropy(a.float().reshape(-1, rlt_cfg.vocab_size), y0.reshape(-1))
        lb = F.cross_entropy(b.float().reshape(-1, rlt_cfg.vocab_size), y0.reshape(-1))
    la.backward(); lb.backward(); torch.cuda.synchronize(device)
    logits_abs = float((a-b).abs().max().item())
    loss_abs = float((la-lb).abs().item())
    grad_abs = 0.0
    ra=dict(ref.named_parameters()); rb=dict(scan1.named_parameters())
    if ra.keys()!=rb.keys():
        raise RuntimeError("merge-scan block1 parameter names differ from frozen RLT")
    for name in ra:
        ga,gb=ra[name].grad,rb[name].grad
        if ga is None or gb is None:
            if ga is not gb: grad_abs=float("inf"); break
            continue
        grad_abs=max(grad_abs,float((ga-gb).abs().max().item()))
    exact = logits_abs==0.0 and loss_abs==0.0 and grad_abs==0.0
    if not exact:
        raise RuntimeError(f"merge-scan block1 exact anchor failed: logits={logits_abs}, loss={loss_abs}, grad={grad_abs}")
    reference_schema=tuple(ref.state_dict().keys())
    del ref,scan1,a,b
    torch.cuda.empty_cache()

    # Causality and schema/parameter neutrality at block4.
    scan_check=new_scan()
    if tuple(scan_check.state_dict().keys())!=reference_schema:
        raise RuntimeError("merge-scan state schema drift")
    g=torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED+91)
    x=torch.randint(0,rlt_cfg.vocab_size,(CHECK_BATCH,SEQ_LEN),generator=g,dtype=torch.long).to(device)
    x2=x.clone()
    x2[:,CAUSAL_MUTATION_INDEX]=(x2[:,CAUSAL_MUTATION_INDEX]+1)%rlt_cfg.vocab_size
    scan_check.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda",dtype=torch.bfloat16):
        sa=scan_check(x); sb=scan_check(x2)
    torch.cuda.synchronize(device)
    prefix_abs=float((sa[:,:CAUSAL_MUTATION_INDEX,:]-sb[:,:CAUSAL_MUTATION_INDEX,:]).abs().max().item())
    changed_abs=float((sa[:,CAUSAL_MUTATION_INDEX,:]-sb[:,CAUSAL_MUTATION_INDEX,:]).abs().max().item())
    causal=prefix_abs==0.0 and changed_abs>0.0
    if not causal:
        raise RuntimeError(f"merge-scan causality failed: prefix={prefix_abs}, changed={changed_abs}")

    # Prove the merge-scan is a real architectural change from plain block4.
    plain_check=new_plain()
    plain_check.load_state_dict(scan_check.state_dict(),strict=True)
    with torch.no_grad(), torch.autocast(device_type="cuda",dtype=torch.bfloat16):
        plain_logits=plain_check(x)
    torch.cuda.synchronize(device)
    scan_vs_plain=float((sa-plain_logits).abs().max().item())
    if scan_vs_plain==0.0:
        raise RuntimeError("merge-scan unexpectedly identical to plain block4")
    del scan_check,plain_check,sa,sb,plain_logits
    torch.cuda.empty_cache()

    g=torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED+1)
    batches=[]
    for _ in range(WARMUP_STEPS+MEASURED_STEPS+2):
        bx=torch.randint(0,rlt_cfg.vocab_size,(BATCH_SIZE,SEQ_LEN),generator=g,dtype=torch.long).to(device)
        by=torch.randint(0,rlt_cfg.vocab_size,(BATCH_SIZE,SEQ_LEN),generator=g,dtype=torch.long).to(device)
        batches.append((bx,by))

    def one_step(callable_model:Any,model:nn.Module,opt:torch.optim.Optimizer,offset:int)->float:
        model.train(); opt.zero_grad(set_to_none=True)
        bx,by=batches[offset%len(batches)]
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
            logits=callable_model(bx)
            loss=F.cross_entropy(logits.float().reshape(-1,50_257),by.reshape(-1))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        return float(loss.detach().cpu())

    def benchmark(kind:str)->dict[str,Any]:
        if kind=="merge_scan":
            base:nn.Module=new_scan()
        elif kind=="plain_block4":
            base=new_plain()
        elif kind=="transformer":
            base=new_transformer()
        else:
            raise ValueError(kind)
        opt=torch.optim.AdamW(base.parameters(),lr=3e-4,betas=(0.9,0.95),weight_decay=0.1,fused=True)
        compiled=torch.compile(base,fullgraph=True,dynamic=False,mode="default")
        torch.cuda.synchronize(device)
        t0=time.perf_counter(); first_loss=one_step(compiled,base,opt,0); torch.cuda.synchronize(device)
        compile_seconds=time.perf_counter()-t0
        for i in range(WARMUP_STEPS): one_step(compiled,base,opt,1+i)
        torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
        t1=time.perf_counter(); last_loss=first_loss
        for i in range(MEASURED_STEPS):
            last_loss=one_step(compiled,base,opt,1+WARMUP_STEPS+i)
        torch.cuda.synchronize(device); measured=time.perf_counter()-t1
        tokens=MEASURED_STEPS*BATCH_SIZE*SEQ_LEN
        out={
            "parameters":sum(p.numel() for p in base.parameters()),
            "first_step_seconds_including_compile":compile_seconds,
            "measured_seconds":measured,
            "measured_tokens":tokens,
            "tokens_per_second":tokens/max(measured,1e-9),
            "last_loss":last_loss,
            "peak_vram_gb":torch.cuda.max_memory_allocated(device)/(1024**3),
        }
        del compiled,opt,base
        torch.cuda.empty_cache(); torch._dynamo.reset()
        return out

    merge_scan=benchmark("merge_scan")
    plain=benchmark("plain_block4")
    transformer=benchmark("transformer")
    tr_tps=float(transformer["tokens_per_second"])
    ms_tps=float(merge_scan["tokens_per_second"])
    plain_tps=float(plain["tokens_per_second"])
    tr_gap=tr_tps/max(ms_tps,1e-9)
    plain_cost=plain_tps/max(ms_tps,1e-9)

    if tr_gap<=1.5:
        classification="PASS_MERGESCAN_BLOCK4_SCALE15M_GAP_LE_1_5X"
    elif tr_gap<=2.0:
        classification="PASS_MERGESCAN_BLOCK4_SCALE15M_GAP_LE_2X"
    else:
        classification="PASS_MERGESCAN_BLOCK4_SCALE15M_GAP_GT_2X"

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
            "variant":"merge_scan_block_recurrent_rlt",
            "block_size":BLOCK_SIZE,
            "expensive_decoder_recurrences_per_seq64":SEQ_LEN//BLOCK_SIZE,
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
            "merge_scan_differs_from_plain_block4_logits_max_abs":scan_vs_plain,
        },
        "profile":{
            "batch_size":BATCH_SIZE,"seq_len":SEQ_LEN,"block_size":BLOCK_SIZE,
            "warmup_steps":WARMUP_STEPS,"measured_steps":MEASURED_STEPS,
            "expected_parameters_each":EXPECTED_PARAMETERS,
        },
        "merge_scan_block4":merge_scan,
        "plain_block4":plain,
        "transformer":transformer,
        "derived":{
            "transformer_throughput_multiple_vs_merge_scan":tr_gap,
            "merge_scan_fraction_of_transformer_throughput":ms_tps/max(tr_tps,1e-9),
            "plain_block4_throughput_multiple_vs_merge_scan":plain_cost,
            "merge_scan_fraction_of_plain_block4_throughput":ms_tps/max(plain_tps,1e-9),
        },
        "interpretation_ceiling":(
            "Engineering-only throughput/causality profile of a parameter-neutral merge-scan block4 RLT. "
            "It restores a cheap recurrent Merge state update at every token while keeping expensive decoder "
            "execution block-parallel. Random-token losses are not quality evidence; a fresh matched training "
            "run is required before any quality conclusion."
        ),
    })


@app.local_entrypoint()
def main(job_path:str)->None:
    result=run_profile_remote.remote(Path(job_path).read_text())
    if not isinstance(result,str):
        raise TypeError("remote merge-scan profile result must be base64 text")
    print("RLT_SYSTEMS_BLOCK4_MERGESCAN_SCALE15M_BATCH768_AC_RESULT_B64="+result)
