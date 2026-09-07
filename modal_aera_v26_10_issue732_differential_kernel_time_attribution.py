from __future__ import annotations
"""Issue #732 diagnostic-only differential CUDA kernel-time attribution after #731."""
import gc, hashlib, json, math, statistics
from pathlib import Path
from typing import Any
import modal
import modal_aera_v26_10_issue710_memory_safe_harness as frozen710

APP_NAME="aera-v26-10-issue732-differential-kernel-time-attribution"
VOLUME_NAME=frozen710.VOLUME_NAME
RESULT_PATH="/vol/aera-v26/issue732-post731-differential-kernel-time-attribution/result.json"
PARENT_RESULT_PATH="/vol/aera-v26/issue724-v26-10-source-bound-provenance-repair1/result.json"
SOURCE_MAIN="267f6d6018a0d355436869919ed73c7723315eab"
SOURCE_TREE="1739ccd80ad48636a4439cbc9d39ba1ab34df807"
RESEARCH_ISSUE=732
PARENT_TRIGGER=731
PARENT_EVIDENCE_COMMENT=5574730812
PARENT_RUN=34154255434
PARENT_JOB=101842659688
PARENT_ATTEMPT=1
PARENT_DECISION="FAIL_V26_10_LATENT_DEPTH_SYNC_COALESCING_MICROBENCH"
ISSUE710_LAUNCHER="modal_aera_v26_10_issue710_memory_safe_harness.py"
ISSUE710_LAUNCHER_BLOB="b885250753ea169cd89dd7a978bb3647fd261fe8"
V26_10_IMPL_BLOB="d8f691c198eed1fa96bcbb78a4e76cad82d18779"
CHECKPOINT_HASHES=dict(frozen710.CHECKPOINT_HASHES)
BATCHES=(8,64)
TOKEN_SEED_BASE=frozen710.TOKEN_SEED_BASE
WARMUP_CALLS=3
BASELINE_CALLS=3
PROFILE_CALLS=1
INTEGRATED_ATOL=frozen710.INTEGRATED_ATOL
INTEGRATED_RTOL=frozen710.INTEGRATED_RTOL
SMALL_KERNEL_LIMIT_US=50.0
VERY_SMALL_KERNEL_LIMIT_US=10.0
REPEATED_MIN_CALLS=100
REPEATED_MIN_MS=1.0
REPEATED_MIN_SHARE=0.10
WHOLE_STAGE_MIN_SMALL_CALLS=500
WHOLE_STAGE_MIN_SMALL_ACTIVE_SHARE=0.20
WHOLE_STAGE_MIN_IDLE_MS=1.0
PARENT_BASELINE_MS={8:33.98348808288574,64:76.74127960205078}
PARENT_CANDIDATE_MS={8:33.72492790222168,64:76.23878479003906}
PARENT_SYNC={8:(49,30),64:(51,30)}
MAX_GPU_SECONDS=420
PREAUTH_MARKER="AERA_V26_10_ISSUE732_PREAUTH_JSON="
L4_START_MARKER="AERA_V26_10_ISSUE732_L4_START_JSON="
RESULT_MARKER="AERA_V26_10_ISSUE732_RESULT_JSON="
SUMMARY_MARKER="AERA_V26_10_ISSUE732_SUMMARY_JSON="

image=frozen710.image.add_local_file(ISSUE710_LAUNCHER,f"/root/{ISSUE710_LAUNCHER}")
app=modal.App(APP_NAME)
volume=modal.Volume.from_name(VOLUME_NAME,create_if_missing=False)

def _blob(path:Path)->str:
    data=path.read_bytes(); return hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()

def _summary(xs:list[float])->dict[str,float]:
    ys=[float(x) for x in xs]
    if not ys:return {"samples":0.0,"mean":0.0,"median":0.0,"p95":0.0,"min":0.0,"max":0.0}
    s=sorted(ys); rank=max(0,min(len(s)-1,math.ceil(.95*len(s))-1))
    return {"samples":float(len(ys)),"mean":float(statistics.fmean(ys)),"median":float(statistics.median(ys)),
            "p95":float(s[rank]),"min":float(min(ys)),"max":float(max(ys))}

def _parent()->dict[str,Any]:
    pth=Path(PARENT_RESULT_PATH)
    if not pth.exists():raise RuntimeError("issue732 parent #731 durable result missing")
    p=json.loads(pth.read_text())
    if p.get("decision")!=PARENT_DECISION or p.get("overall_pass") is not False or p.get("fresh_full_e2e_systems_gate_authorized") is not False:
        raise RuntimeError("issue732 parent decision drift")
    rows=p.get("rows")
    if not isinstance(rows,dict) or set(rows)!={"8","64"}:raise RuntimeError("issue732 parent rows drift")
    for batch in BATCHES:
        r=rows[str(batch)]
        if not math.isclose(float(r["timing"]["baseline_v26_9"]["median_ms"]),PARENT_BASELINE_MS[batch],rel_tol=0,abs_tol=1e-12):raise RuntimeError("issue732 parent baseline drift")
        if not math.isclose(float(r["timing"]["candidate_v26_10"]["median_ms"]),PARENT_CANDIDATE_MS[batch],rel_tol=0,abs_tol=1e-12):raise RuntimeError("issue732 parent candidate drift")
        bsync,csync=PARENT_SYNC[batch]
        pr=r["post_timing_profiler"]
        if int(pr["baseline"]["cudaStreamSynchronize"])!=bsync or int(pr["candidate"]["cudaStreamSynchronize"])!=csync:raise RuntimeError("issue732 parent sync drift")
    return p

def _frozen_blobs()->dict[str,str]:
    got=dict(frozen710._frozen_blobs())
    got["issue710_launcher"]=_blob(Path(f"/root/{ISSUE710_LAUNCHER}"))
    if got.get("issue710_launcher")!=ISSUE710_LAUNCHER_BLOB:raise RuntimeError("issue732 #710 launcher drift")
    if got.get("v26_10_impl")!=V26_10_IMPL_BLOB:raise RuntimeError("issue732 v26.10 implementation drift")
    return got

@app.function(image=image,cpu=4,memory=8192,timeout=180,volumes={"/vol":volume})
def preflight()->dict[str,Any]:
    import tam_research.aera_v26_5_end_to_end_systems as base
    volume.reload()
    if Path(RESULT_PATH).exists():raise RuntimeError("issue732 result already exists")
    _parent(); blobs=_frozen_blobs(); hashes=base.checkpoint_hashes(base.CHECKPOINT_RELATIVE_DIR)
    if hashes!=CHECKPOINT_HASHES:raise RuntimeError("issue732 checkpoint drift")
    return {"research_issue":732,"source_main":SOURCE_MAIN,"source_tree":SOURCE_TREE,
      "parent_trigger":PARENT_TRIGGER,"parent_evidence_comment":PARENT_EVIDENCE_COMMENT,"parent_run":PARENT_RUN,
      "parent_job":PARENT_JOB,"parent_attempt":PARENT_ATTEMPT,"parent_decision":PARENT_DECISION,
      "result_path":RESULT_PATH,"parent_result_path":PARENT_RESULT_PATH,"frozen_blobs":blobs,"checkpoint_hashes":hashes,
      "result_absent":True,"gpu_used":False,"model_constructed":False,"new_measurement_performed":False,
      "optimization_authorized":False,"full_e2e_systems_gate_authorized":False,"systems_pass_earned":False,
      "architecture_freeze_authorized":False,"s2_authorized":False,"fresh_scientific_seed_authorized":False,
      "independent_replication_credit":False,"100m_authorized":False,"breakthrough_proven":False}

def _clip_x(e:dict[str,Any],lo:float,hi:float):
    if e.get("ph")!="X":return None
    try:a=float(e["ts"]); b=a+float(e["dur"])
    except (KeyError,TypeError,ValueError):return None
    x=max(a,lo); y=min(b,hi); return (x,y) if y>x else None

def _union(xs:list[tuple[float,float]])->list[tuple[float,float]]:
    if not xs:return []
    out=[]
    for a,b in sorted(xs):
        if out and a<=out[-1][1]:out[-1]=(out[-1][0],max(out[-1][1],b))
        else:out.append((a,b))
    return out

def _union_us(xs:list[tuple[float,float]])->float:return float(sum(b-a for a,b in _union(xs)))

def _dist_stats(ds:list[float])->dict[str,float]:
    return _summary(ds)

def _bucket_name(d:float)->str:
    if d<=10:return "le_10us"
    if d<=25:return "10_25us"
    if d<=50:return "25_50us"
    if d<=100:return "50_100us"
    return "gt_100us"

def _trace(path:Path,label:str)->dict[str,Any]:
    raw=json.loads(path.read_text()); events=raw.get("traceEvents")
    if not isinstance(events,list):raise RuntimeError("issue732 traceEvents missing")
    bounds=[]
    for e in events:
        if isinstance(e,dict) and e.get("name")==label and str(e.get("cat","")).lower()=="user_annotation":
            x=_clip_x(e,-float("inf"),float("inf"))
            if x:bounds.append(x)
    if len(bounds)!=1:raise RuntimeError(f"issue732 expected one call annotation, got {len(bounds)}")
    start,end=bounds[0]; device=[]; runtime=[]
    for e in events:
        if not isinstance(e,dict):continue
        x=_clip_x(e,start,end)
        if not x:continue
        cat=str(e.get("cat","")).lower(); name=str(e.get("name","")); args=e.get("args") if isinstance(e.get("args"),dict) else {}
        row={"category":cat,"name":name,"start_us_rel":float(x[0]-start),"end_us_rel":float(x[1]-start),"duration_us":float(x[1]-x[0]),"stream":args.get("stream"),"correlation":args.get("correlation")}
        if cat in {"kernel","gpu_memcpy","gpu_memset","memcpy","memset"}:device.append(row)
        elif cat=="cuda_runtime":runtime.append(row)
    device.sort(key=lambda r:(r["start_us_rel"],r["end_us_rel"]))
    merged=_union([(r["start_us_rel"],r["end_us_rel"]) for r in device])
    gaps=[merged[i+1][0]-merged[i][1] for i in range(len(merged)-1)]
    kernels=[r for r in device if r["category"]=="kernel"]
    memcpys=[r for r in device if "memcpy" in r["category"]]
    groups={}
    for r in kernels:
        g=groups.setdefault(r["name"],{"name":r["name"],"count":0,"total_us":0.0,"durations_us":[]})
        g["count"]+=1; g["total_us"]+=r["duration_us"]; g["durations_us"].append(r["duration_us"])
    names=[]
    for g in groups.values():
        ds=g.pop("durations_us"); s=_dist_stats(ds); g["median_us"]=s["median"]; g["p95_us"]=s["p95"]; names.append(g)
    by_total=sorted(names,key=lambda r:(r["total_us"],r["count"]),reverse=True)
    by_count=sorted(names,key=lambda r:(r["count"],r["total_us"]),reverse=True)
    buckets={k:{"count":0,"sum_us":0.0,"union_us":0.0} for k in ("le_10us","10_25us","25_50us","50_100us","gt_100us")}
    for k in buckets:
        subset=[r for r in kernels if _bucket_name(r["duration_us"])==k]
        buckets[k]["count"]=len(subset); buckets[k]["sum_us"]=float(sum(r["duration_us"] for r in subset)); buckets[k]["union_us"]=_union_us([(r["start_us_rel"],r["end_us_rel"]) for r in subset])
    launch=sum(1 for r in runtime if "launch" in r["name"].lower())
    sync=sum(1 for r in runtime if "synchron" in r["name"].lower() or "wait" in r["name"].lower())
    memcpy=sum(1 for r in runtime if "memcpy" in r["name"].lower() or "memset" in r["name"].lower())
    return {"annotation_duration_us":float(end-start),"device_activity_count":len(device),"kernel_count":len(kernels),"memcpy_count":len(memcpys),
      "device_active_union_us":_union_us([(r["start_us_rel"],r["end_us_rel"]) for r in device]),
      "idle_gap_total_us":float(sum(gaps)),"idle_gap_stats_us":_dist_stats(gaps),"kernel_duration_buckets":buckets,
      "top_kernel_names_by_total":by_total[:30],"top_kernel_names_by_count":by_count[:30],"all_kernel_names":{r["name"]:r for r in names},
      "cuda_runtime_counts":{"launch":launch,"synchronize_or_wait":sync,"memcpy_or_memset":memcpy,"total":len(runtime)},
      "chronological_device_activities":device,
      "unavailable_fields":{"kernel_aten_ownership":"not inferred; raw CUDA kernel names/device durations only"}}

def _profile(call,label:str)->dict[str,Any]:
    import torch
    path=Path(f"/tmp/{label.replace('.','_')}.json")
    a=torch.cuda.Event(enable_timing=True); z=torch.cuda.Event(enable_timing=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False,with_stack=False) as prof:
        with torch.profiler.record_function(label):
            a.record(); out=call(); z.record(); torch.cuda.synchronize()
    elapsed=float(a.elapsed_time(z)); prof.export_chrome_trace(str(path)); t=_trace(path,label); path.unlink(missing_ok=True); del out,prof
    t["profiled_cuda_event_ms"]=elapsed; return t

def _normalize(trace:dict[str,Any],unprofiled_median:float)->dict[str,Any]:
    p=float(trace["profiled_cuda_event_ms"]); scale=(unprofiled_median/p) if p>0 else 0.0
    def norm_us(us:float)->float:return min(float(us)/1000.0*scale,unprofiled_median)
    names={}
    for name,r in trace["all_kernel_names"].items():
        names[name]={**r,"normalized_total_ms":norm_us(r["total_us"])}
    buckets={k:{**v,"normalized_sum_ms":norm_us(v["sum_us"]),"normalized_union_ms":norm_us(v["union_us"])} for k,v in trace["kernel_duration_buckets"].items()}
    return {"distortion_ratio":p/unprofiled_median if unprofiled_median>0 else float("inf"),"normalization_scale":scale,
      "normalized_device_active_ms":norm_us(trace["device_active_union_us"]),"normalized_idle_gap_ms":norm_us(trace["idle_gap_total_us"]),
      "kernel_names":names,"kernel_duration_buckets":buckets}

def _choose(rows:dict[str,Any])->tuple[str|None,dict[str,Any]]:
    evidence={"repeated_candidates":{},"explicit_candidates":{},"whole_stage":{}}
    common=None
    for b in ("8","64"):
        cand=rows[b]["candidate_v26_10"]; names=cand["normalized"]["kernel_names"]; base=PARENT_BASELINE_MS[int(b)]
        qualifying={n:r for n,r in names.items() if r["count"]>=REPEATED_MIN_CALLS and r["normalized_total_ms"]>=REPEATED_MIN_MS and r["normalized_total_ms"]/base>=REPEATED_MIN_SHARE}
        evidence["repeated_candidates"][b]=qualifying
        common=set(qualifying) if common is None else common & set(qualifying)
    if common:
        best=max(common,key=lambda n:sum(rows[b]["candidate_v26_10"]["normalized"]["kernel_names"][n]["normalized_total_ms"] for b in ("8","64")))
        return f"repeated_small_kernel_fusion::{best}",evidence
    largest=[]
    for b in ("8","64"):
        names=rows[b]["candidate_v26_10"]["normalized"]["kernel_names"]; base=PARENT_BASELINE_MS[int(b)]
        if names:
            n=max(names,key=lambda x:names[x]["normalized_total_ms"]); r=names[n]
            ok=r["normalized_total_ms"]>=REPEATED_MIN_MS and r["normalized_total_ms"]/base>=REPEATED_MIN_SHARE
            evidence["explicit_candidates"][b]={"name":n,"row":r,"qualifies":ok}; largest.append((n,ok))
    if len(largest)==2 and largest[0][0]==largest[1][0] and all(x[1] for x in largest):return f"explicit_device_kernel_optimization::{largest[0][0]}",evidence
    ws=[]
    for b in ("8","64"):
        c=rows[b]["candidate_v26_10"]; norm=c["normalized"]; tr=c["trace"]
        small=tr["kernel_duration_buckets"]["le_10us"]["count"]+tr["kernel_duration_buckets"]["10_25us"]["count"]+tr["kernel_duration_buckets"]["25_50us"]["count"]
        small_union=norm["kernel_duration_buckets"]["le_10us"]["normalized_union_ms"]+norm["kernel_duration_buckets"]["10_25us"]["normalized_union_ms"]+norm["kernel_duration_buckets"]["25_50us"]["normalized_union_ms"]
        active=norm["normalized_device_active_ms"]; share=small_union/active if active>0 else 0.0
        ok=small>=WHOLE_STAGE_MIN_SMALL_CALLS and share>=WHOLE_STAGE_MIN_SMALL_ACTIVE_SHARE and norm["normalized_idle_gap_ms"]>=WHOLE_STAGE_MIN_IDLE_MS
        evidence["whole_stage"][b]={"small_kernel_count":small,"small_kernel_active_share":share,"normalized_idle_gap_ms":norm["normalized_idle_gap_ms"],"qualifies":ok}; ws.append(ok)
    if all(ws):return "whole_stage_launch_coalescing",evidence
    return None,evidence

@app.function(image=image,gpu="L4",cpu=4,memory=16384,timeout=MAX_GPU_SECONDS,volumes={"/vol":volume})
def run_diagnostic()->dict[str,Any]:
    import torch
    import tam_research.aera_v25_post8471_triage as triage
    import tam_research.aera_v26_5_end_to_end_systems as base
    import tam_research.aera_v26_9_issue643_bounded_memory_end_to_end_systems as systems
    from tam_research.aera_hardware_core import HardwareAERAState
    from tam_research.aera_hardware_core_v26_10_latent_depth_sync_coalescing import install_latent_depth_sync_coalescing_v26_10
    from tam_research.aera_hardware_core_v26_9_ficem_read_identity_weight_visibility import IdentityWeightVisibilityTritonFICEMReadWriteBackend
    volume.reload()
    if Path(RESULT_PATH).exists():raise RuntimeError("issue732 result already exists")
    _parent(); blobs=_frozen_blobs()
    if not torch.cuda.is_available():raise RuntimeError("issue732 requires authorized NVIDIA L4")
    device=torch.device("cuda"); torch.set_float32_matmul_precision("high")
    before=base.checkpoint_hashes(base.CHECKPOINT_RELATIVE_DIR)
    if before!=CHECKPOINT_HASHES:raise RuntimeError("issue732 checkpoint drift")
    payload=torch.load(Path(base.CHECKPOINT_RELATIVE_DIR)/"aera.pt",map_location="cpu",weights_only=False)
    baseline=base._build_v26(payload,device); candidate=base._build_v26(payload,device)
    bb=systems._install_v26_9_candidate_backend(baseline); cb=systems._install_v26_9_candidate_backend(candidate)
    installed=install_latent_depth_sync_coalescing_v26_10(candidate); expected=IdentityWeightVisibilityTritonFICEMReadWriteBackend.name
    if not all(x==expected for x in bb+cb) or installed!=tuple(range(len(candidate.stages))):raise RuntimeError("issue732 runtime identity drift")
    schema=frozen710.frozen706._schemas_and_weights_exact(baseline,candidate)
    if not schema["pass"]:raise RuntimeError("issue732 schema drift")
    bv=base._parameter_versions(baseline); cv=base._parameter_versions(candidate); rows={}
    with torch.inference_mode():
      for batch in BATCHES:
        gen=torch.Generator(device="cpu").manual_seed(TOKEN_SEED_BASE+batch)
        tokens=torch.randint(0,triage.VOCAB_SIZE,(batch,triage.SEQ_LEN),generator=gen).to(device)
        bcall=lambda:base._model_call(baseline,tokens,update_memory=True); ccall=lambda:base._model_call(candidate,tokens,update_memory=True)
        bo,bd=frozen710.frozen706._capture_decisions(baseline,bcall); co,cd=frozen710.frozen706._capture_decisions(candidate,ccall)
        if not isinstance(bo.get("state"),HardwareAERAState) or not isinstance(co.get("state"),HardwareAERAState):raise RuntimeError("issue732 state missing")
        deq=frozen710.frozen706._decision_equivalence(bd,cd); route=frozen710.frozen706._route_exact(base,bo,co)
        leq=frozen710._chunked_logit_equivalence(bo["logits"],co["logits"]); seq=base._state_equivalence(bo["state"],co["state"])
        finite=bool(base._finite_output(bo) and base._finite_output(co)); correct=bool(deq["pass"] and route and leq["pass"] and seq["pass"] and finite)
        if not correct:raise RuntimeError(f"issue732 correctness drift batch {batch}")
        del bo,co; gc.collect(); torch.cuda.empty_cache()
        for _ in range(WARMUP_CALLS):
            x=bcall(); y=ccall(); del x,y
        samples={"baseline_v26_9":[],"candidate_v26_10":[]}
        for i in range(BASELINE_CALLS):
            order=(("baseline_v26_9",bcall),("candidate_v26_10",ccall)) if i%2==0 else (("candidate_v26_10",ccall),("baseline_v26_9",bcall))
            for n,call in order:samples[n].append(frozen710.frozen706._event_timed_call(call))
        timing={n:_summary(v) for n,v in samples.items()}
        bt=_profile(bcall,f"aera732.baseline.batch{batch}"); ct=_profile(ccall,f"aera732.candidate.batch{batch}")
        bn=_normalize(bt,timing["baseline_v26_9"]["median"]); cn=_normalize(ct,timing["candidate_v26_10"]["median"])
        common=set(bn["kernel_names"])&set(cn["kernel_names"]); differential=[]
        for name in common:
            br=bn["kernel_names"][name]; cr=cn["kernel_names"][name]
            differential.append({"name":name,"baseline_count":br["count"],"candidate_count":cr["count"],"baseline_normalized_total_ms":br["normalized_total_ms"],"candidate_normalized_total_ms":cr["normalized_total_ms"],"candidate_minus_baseline_ms":cr["normalized_total_ms"]-br["normalized_total_ms"]})
        differential=sorted(differential,key=lambda r:max(r["baseline_normalized_total_ms"],r["candidate_normalized_total_ms"]),reverse=True)[:50]
        rows[str(batch)]={"batch_size":batch,"token_seed":TOKEN_SEED_BASE+batch,"correctness":{"pass":correct,"route_exact":route,"decision_equivalence":deq,"logit_equivalence":leq,"state_equivalence":seq,"finite":finite},
          "baseline_v26_9":{"unprofiled":timing["baseline_v26_9"],"trace":bt,"normalized":bn},
          "candidate_v26_10":{"unprofiled":timing["candidate_v26_10"],"trace":ct,"normalized":cn},"differential_top_common_kernels":differential}
        del tokens; gc.collect(); torch.cuda.empty_cache()
    after=base.checkpoint_hashes(base.CHECKPOINT_RELATIVE_DIR); versions_ok=(bv==base._parameter_versions(baseline) and cv==base._parameter_versions(candidate))
    if after!=before or not versions_ok:raise RuntimeError("issue732 immutability drift")
    next_target,evidence=_choose(rows)
    result={"scope":"aera_v26_10_issue732_differential_kernel_time_attribution","research_issue":732,"source_main":SOURCE_MAIN,"source_tree":SOURCE_TREE,
      "parent_trigger":PARENT_TRIGGER,"parent_evidence_comment":PARENT_EVIDENCE_COMMENT,"parent_run":PARENT_RUN,"parent_job":PARENT_JOB,"parent_attempt":PARENT_ATTEMPT,
      "parent_decision":PARENT_DECISION,"device":torch.cuda.get_device_name(device),"checkpoint_hashes_before":before,"checkpoint_hashes_after":after,"checkpoint_hashes_unchanged":True,
      "parameter_versions_unchanged":True,"schema_check":schema,"frozen_blobs":blobs,"batches":list(BATCHES),"warmup_calls":WARMUP_CALLS,"unprofiled_calls_per_condition":BASELINE_CALLS,"profile_calls_per_condition":PROFILE_CALLS,
      "rows":rows,"next_target":next_target,"decision_evidence":evidence,"diagnostic_only":True,"optimization_authorized":False,"full_e2e_systems_gate_authorized":False,"systems_pass_earned":False,
      "architecture_freeze_authorized":False,"s2_authorized":False,"fresh_scientific_seed_authorized":False,"independent_replication_credit":False,"100m_authorized":False,"breakthrough_proven":False}
    p=Path(RESULT_PATH); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n"); volume.commit()
    summary={"research_issue":732,"device":result["device"],"parent_decision":PARENT_DECISION,"next_target":next_target,
      "batches":{b:{"baseline_unprofiled":r["baseline_v26_9"]["unprofiled"],"candidate_unprofiled":r["candidate_v26_10"]["unprofiled"],
        "baseline_profiled_ms":r["baseline_v26_9"]["trace"]["profiled_cuda_event_ms"],"candidate_profiled_ms":r["candidate_v26_10"]["trace"]["profiled_cuda_event_ms"],
        "baseline_kernel_count":r["baseline_v26_9"]["trace"]["kernel_count"],"candidate_kernel_count":r["candidate_v26_10"]["trace"]["kernel_count"],
        "baseline_normalized_idle_gap_ms":r["baseline_v26_9"]["normalized"]["normalized_idle_gap_ms"],"candidate_normalized_idle_gap_ms":r["candidate_v26_10"]["normalized"]["normalized_idle_gap_ms"],
        "baseline_buckets":r["baseline_v26_9"]["normalized"]["kernel_duration_buckets"],"candidate_buckets":r["candidate_v26_10"]["normalized"]["kernel_duration_buckets"],
        "top_common_kernels":r["differential_top_common_kernels"][:10]} for b,r in rows.items()},"decision_evidence":evidence,
      "optimization_authorized":False,"full_e2e_systems_gate_authorized":False,"systems_pass_earned":False,"architecture_freeze_authorized":False,"s2_authorized":False,"fresh_scientific_seed_authorized":False,"independent_replication_credit":False,"100m_authorized":False,"breakthrough_proven":False}
    print(RESULT_MARKER+json.dumps(summary,sort_keys=True)); return summary

@app.local_entrypoint()
def preauth_main()->None:print(PREAUTH_MARKER+json.dumps(preflight.remote(),sort_keys=True))

@app.local_entrypoint()
def l4_main()->None:
    print(PREAUTH_MARKER+json.dumps(preflight.remote(),sort_keys=True))
    print(L4_START_MARKER+json.dumps({"research_issue":732,"gpu":"L4","max_gpu_seconds":MAX_GPU_SECONDS,"result_path":RESULT_PATH,"diagnostic_only":True},sort_keys=True))
    print(SUMMARY_MARKER+json.dumps(run_diagnostic.remote(),sort_keys=True))
