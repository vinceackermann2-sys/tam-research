from __future__ import annotations
"""Issue #702 diagnostic-only CUDA timeline attribution successor after #701 governance incident."""
import gc, hashlib, json, math, statistics
from pathlib import Path
from typing import Any
import modal
import modal_aera_v26_9_issue665_frozen_throughput_component_attribution as issue665

APP_NAME="aera-v26-9-issue702-cuda-timeline-dispatch-gap-attribution"
VOLUME_NAME=issue665.VOLUME_NAME
RESULT_PATH="/vol/aera-v26/issue702-cuda-timeline-dispatch-gap-attribution/result.json"
PARENT_RESULT_PATH="/vol/aera-v26/issue697-micro-op-attribution/result.json"
RESEARCH_ISSUE=702
SOURCE_MAIN="6fad37d0e4c611fcaff480899b25dc190e9e9127"
SOURCE_TREE="87a22d981caf77263574503749757ba77290cc9a"
GOVERNANCE_PREDECESSOR_ISSUE=701
GOVERNANCE_INCIDENT_COMMENT=5571988281
EXCLUDED_INERT_REF="__noop_probe_do_not_create"
PARENT_TRIGGER=700
PARENT_EVIDENCE_COMMENT=5571118103
PARENT_RUN=34125640230
PARENT_JOB=101753558368
PARENT_ATTEMPT=1
SOURCE_DECISION="FAIL_FROZEN_E2E_SYSTEMS_GATE"
PARENT_NEXT_TARGET="dispatch_or_unattributed_gap"
ISSUE665_LAUNCHER="modal_aera_v26_9_issue665_frozen_throughput_component_attribution.py"
ISSUE687_LAUNCHER="modal_aera_v26_9_issue687_stage_internal_throughput_attribution.py"
ISSUE697_LAUNCHER="modal_aera_v26_9_issue697_micro_op_attribution.py"
EXPECTED_BLOBS={
"issue665_launcher":"72f27391ff2f0a7bff8d4532f307ddc4869cf494",
"issue687_launcher":"b9950c032686a496c6d944c30c441428499cb367",
"issue697_launcher":"c82affc01d1ecb096371e8865dbd7b18e739f5e1",
"scientific_adapter":"512572340cc09e2e7ad6729712258c12cb377ef2",
"base_systems":"c9731cae7e386f09b2a190b045532591c4fa00be",
"runtime_interface":"268644ac4edee15a4cc4e29d3fed7f61eeb3caa7",
"stage_v25_1":"1c3456d8040455b4cd1194db4c8586f77d0f3e43",
"compact_v25_1":"4e336b6e1a6238dac782fa320751d68281493ee1",
"tokenwise_v19":"98008bceb8c68af3bc346e5dfcc7a8218875661e",
"base_core":"ffe0341829fb905e40ef9ae3544f797471f1aa9c",
"nohost_v25_1":"237e5615cf32f644e8675808a6b3e9adaf04fb23",
"v26_9_backend":"b81cc209f5d95abbe1fb8bd620c78e87c067bc19",
"triage":"e5ce8cdda0777dce97816e2640f4492803a6b191",
}
CHECKPOINT_HASHES={
"aera":"f8aa92421801e8f190247e420632be5f0c20bc5ea8bf6bdeefe06686b3a31b30",
"transformer":"cdd5cab4439a709468d6607d45d82081b33e876b2e40d91d4a38ba139b219dd7",
}
SYSTEM_BATCH_SIZES=tuple(issue665.SYSTEM_BATCH_SIZES)
TOKEN_SEED_BASE=issue665.TOKEN_SEED_BASE
TOKEN_SEED_OFFSET=issue665.TOKEN_SEED_OFFSET
WARMUP_CALLS=issue665.DIAGNOSTIC_WARMUP_CALLS
BASELINE_CALLS=3
PROFILE_CALLS=1
SMALL_KERNEL_MAX_US=50.0
REPEATED_MIN_COUNT=2
MATERIAL_SHARE_MIN=0.15
MATERIAL_MS_MIN=1.0
MAX_GPU_SECONDS=300
EXPECTED_PARENT={
"8":{"gap":9.2633691290416,"distortion":2.6684698738709085},
"64":{"gap":12.370303117411762,"distortion":1.6960100325791625},
}
PREAUTH_MARKER="AERA_V26_9_ISSUE702_CUDA_TIMELINE_PREAUTH_JSON="
L4_START_MARKER="AERA_V26_9_ISSUE702_CUDA_TIMELINE_L4_START_JSON="
RESULT_MARKER="AERA_V26_9_ISSUE702_CUDA_TIMELINE_RESULT_JSON="
SUMMARY_MARKER="AERA_V26_9_ISSUE702_CUDA_TIMELINE_SUMMARY_JSON="

image=(issue665.image
 .add_local_file(ISSUE665_LAUNCHER,f"/root/{ISSUE665_LAUNCHER}")
 .add_local_file(ISSUE687_LAUNCHER,f"/root/{ISSUE687_LAUNCHER}")
 .add_local_file(ISSUE697_LAUNCHER,f"/root/{ISSUE697_LAUNCHER}"))
app=modal.App(APP_NAME)
volume=modal.Volume.from_name(VOLUME_NAME,create_if_missing=False)

def _blob(path:Path)->str:
    data=path.read_bytes(); return hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()

def _summary(xs:list[float])->dict[str,float]:
    xs=[float(x) for x in xs]
    if not xs:return {"samples":0.0,"mean":0.0,"median":0.0,"p95":0.0,"min":0.0,"max":0.0}
    ys=sorted(xs); rank=max(0,min(len(ys)-1,math.ceil(0.95*len(ys))-1))
    return {"samples":float(len(xs)),"mean":float(statistics.fmean(xs)),"median":float(statistics.median(xs)),
            "p95":float(ys[rank]),"min":float(min(xs)),"max":float(max(xs))}

def _parent()->dict[str,Any]:
    path=Path(PARENT_RESULT_PATH)
    if not path.exists():raise RuntimeError("issue702 parent #697 result missing")
    p=json.loads(path.read_text())
    if p.get("research_issue")!=697 or p.get("source_decision")!=SOURCE_DECISION:raise RuntimeError("issue702 parent authority drift")
    if p.get("next_target")!=PARENT_NEXT_TARGET:raise RuntimeError("issue702 parent target drift")
    for key in ("optimization_authorized","systems_pass_earned","architecture_freeze_authorized","s2_authorized",
                "fresh_scientific_seed_authorized","independent_replication_credit","100m_authorized","breakthrough_proven"):
        if p.get(key) is not False:raise RuntimeError(f"issue702 parent higher-stage drift: {key}")
    rows=p.get("rows")
    if not isinstance(rows,dict) or set(rows)!={"8","64"}:raise RuntimeError("issue702 parent rows drift")
    for b,e in EXPECTED_PARENT.items():
        row=rows[b]; be=row.get("baseline_equivalent")
        if not isinstance(be,dict):raise RuntimeError(f"issue702 parent baseline-equivalent missing {b}")
        gap=float(be["glue"].get(PARENT_NEXT_TARGET,0.0))+float(be["experts"].get(PARENT_NEXT_TARGET,0.0))
        distortion=float(row["profiler_distortion_ratio"])
        if not math.isclose(gap,e["gap"],rel_tol=0.0,abs_tol=1e-12):raise RuntimeError(f"issue702 parent gap drift {b}: {gap}")
        if not math.isclose(distortion,e["distortion"],rel_tol=0.0,abs_tol=1e-12):raise RuntimeError(f"issue702 parent distortion drift {b}: {distortion}")
    return p

def _frozen_blobs()->dict[str,str]:
    import tam_research.aera_hardware_core as core
    import tam_research.aera_hardware_core_v19 as v19
    import tam_research.aera_hardware_core_v25_1 as v251
    import tam_research.aera_hardware_core_v25_1_compact as compact
    import tam_research.aera_hardware_core_v25_1_nohost as nohost
    import tam_research.aera_hardware_core_v26 as runtime
    import tam_research.aera_hardware_core_v26_9_ficem_read_identity_weight_visibility as backend
    import tam_research.aera_v25_post8471_triage as triage
    import tam_research.aera_v26_5_end_to_end_systems as base
    import tam_research.aera_v26_9_issue643_bounded_memory_end_to_end_systems as systems
    got={"issue665_launcher":_blob(Path(f"/root/{ISSUE665_LAUNCHER}")),
      "issue687_launcher":_blob(Path(f"/root/{ISSUE687_LAUNCHER}")),"issue697_launcher":_blob(Path(f"/root/{ISSUE697_LAUNCHER}")),
      "scientific_adapter":_blob(Path(systems.__file__)),"base_systems":_blob(Path(base.__file__)),
      "runtime_interface":_blob(Path(runtime.__file__)),"stage_v25_1":_blob(Path(v251.__file__)),
      "compact_v25_1":_blob(Path(compact.__file__)),"tokenwise_v19":_blob(Path(v19.__file__)),
      "base_core":_blob(Path(core.__file__)),"nohost_v25_1":_blob(Path(nohost.__file__)),
      "v26_9_backend":_blob(Path(backend.__file__)),"triage":_blob(Path(triage.__file__))}
    if got!=EXPECTED_BLOBS:raise RuntimeError(f"issue702 frozen blob drift: {got}")
    return got

@app.function(image=image,cpu=4,memory=8192,timeout=180,volumes={"/vol":volume})
def preflight()->dict[str,Any]:
    import tam_research.aera_v26_5_end_to_end_systems as base
    volume.reload()
    if Path(RESULT_PATH).exists():raise RuntimeError(f"issue702 result already exists: {RESULT_PATH}")
    _parent(); blobs=_frozen_blobs(); hashes=base.checkpoint_hashes(base.CHECKPOINT_RELATIVE_DIR)
    if hashes!=CHECKPOINT_HASHES:raise RuntimeError("issue702 checkpoint drift")
    return {"research_issue":RESEARCH_ISSUE,"source_main":SOURCE_MAIN,"source_tree":SOURCE_TREE,
      "governance_predecessor_issue":GOVERNANCE_PREDECESSOR_ISSUE,"governance_incident_comment":GOVERNANCE_INCIDENT_COMMENT,
      "excluded_inert_ref":EXCLUDED_INERT_REF,"parent_trigger":PARENT_TRIGGER,"parent_evidence_comment":PARENT_EVIDENCE_COMMENT,
      "parent_run":PARENT_RUN,"parent_job":PARENT_JOB,"parent_attempt":PARENT_ATTEMPT,"source_decision":SOURCE_DECISION,
      "parent_next_target":PARENT_NEXT_TARGET,"result_path":RESULT_PATH,"parent_result_path":PARENT_RESULT_PATH,
      "frozen_blobs":blobs,"checkpoint_hashes":hashes,"result_absent":True,"gpu_used":False,"model_constructed":False,
      "new_measurement_performed":False,"optimization_authorized":False,"systems_pass_earned":False,
      "architecture_freeze_authorized":False,"s2_authorized":False,"fresh_scientific_seed_authorized":False,
      "independent_replication_credit":False,"100m_authorized":False,"breakthrough_proven":False}

def _route_rows(output:dict[str,object])->list[dict[str,Any]]:
    routes=output.get("stage_routes")
    if not isinstance(routes,list):raise RuntimeError("issue702 output missing stage_routes")
    rows=[]
    for chunk_i,chunk in enumerate(routes):
        if not isinstance(chunk,list):raise RuntimeError("issue702 malformed route chunk")
        for stage_i,row in enumerate(chunk):
            if not isinstance(row,dict):raise RuntimeError("issue702 malformed route row")
            gate=row.get("stage_route_gate"); frac=row.get("executed_fraction")
            if not hasattr(gate,"detach") or not isinstance(frac,(float,int)):raise RuntimeError("issue702 route evidence missing")
            selected_gate=gate.detach().ge(0.5).cpu(); selected=int(selected_gate.sum().item()); population=int(selected_gate.numel())
            expected=selected/population if population else 0.0
            if not math.isclose(float(frac),expected,rel_tol=0.0,abs_tol=1e-12):raise RuntimeError("issue702 executed_fraction disagrees with physical route gate")
            rows.append({"chunk_index":chunk_i,"stage_index":stage_i,"foundation":stage_i==0,"selected":selected,
                         "population":population,"executed_fraction":float(frac)})
    return rows

def _clip_x(event:dict[str,Any],start:float,end:float):
    if event.get("ph")!="X":return None
    try:a=float(event["ts"]); b=a+float(event["dur"])
    except (KeyError,TypeError,ValueError):return None
    lo=max(a,start); hi=min(b,end); return (lo,hi) if hi>lo else None

def _union(intervals:list[tuple[float,float]])->list[tuple[float,float]]:
    if not intervals:return []
    xs=sorted(intervals); out=[xs[0]]
    for a,b in xs[1:]:
        pa,pb=out[-1]
        if a<=pb:out[-1]=(pa,max(pb,b))
        else:out.append((a,b))
    return out

def _union_us(intervals:list[tuple[float,float]])->float:return float(sum(b-a for a,b in _union(intervals)))

def _runtime_family(name:str)->str:
    n=name.lower()
    if "launch" in n:return "launch"
    if "memcpy" in n or "memset" in n:return "memcpy_or_memset"
    if "synchron" in n or "wait" in n:return "synchronize_or_wait"
    return "other"

def _top_names(events:list[dict[str,Any]],limit:int=20)->list[dict[str,Any]]:
    groups={}
    for e in events:
        name=str(e["name"]); row=groups.setdefault(name,{"name":name,"count":0,"total_us":0.0,"durations":[]})
        row["count"]+=1; row["total_us"]+=float(e["duration_us"]); row["durations"].append(float(e["duration_us"]))
    out=[]
    for row in groups.values():
        ds=row.pop("durations"); row["median_us"]=float(statistics.median(ds)); out.append(row)
    return sorted(out,key=lambda r:(r["total_us"],r["count"]),reverse=True)[:limit]

def _analyze_trace(path:Path,label:str)->dict[str,Any]:
    trace=json.loads(path.read_text()); events=trace.get("traceEvents")
    if not isinstance(events,list):raise RuntimeError("issue702 traceEvents missing")
    bounds=[]
    for e in events:
        if isinstance(e,dict) and e.get("name")==label and str(e.get("cat","")).lower()=="user_annotation":
            x=_clip_x(e,-float("inf"),float("inf"))
            if x is not None:bounds.append(x)
    if len(bounds)!=1:raise RuntimeError(f"issue702 expected one call annotation, got {len(bounds)}")
    start,end=bounds[0]; cpu_ops=[]; runtime=[]; device=[]; device_categories={"kernel","gpu_memcpy","gpu_memset","memcpy","memset"}
    for e in events:
        if not isinstance(e,dict):continue
        clipped=_clip_x(e,start,end)
        if clipped is None:continue
        cat=str(e.get("cat","")).lower(); name=str(e.get("name","")); args=e.get("args") if isinstance(e.get("args"),dict) else {}; ext=args.get("External id")
        base={"category":cat,"name":name,"start_us_rel":float(clipped[0]-start),"duration_us":float(clipped[1]-clipped[0]),
              "end_us_rel":float(clipped[1]-start),"external_id":ext,"stream":args.get("stream"),"correlation":args.get("correlation")}
        if cat=="cpu_op":cpu_ops.append(base)
        elif cat=="cuda_runtime":runtime.append(base)
        elif cat in device_categories:device.append(base)
    device.sort(key=lambda e:(e["start_us_rel"],e["end_us_rel"]))
    aten_ids={e["external_id"] for e in cpu_ops if e["external_id"] is not None and e["name"].startswith("aten::")}
    for e in device:e["aten_linked"]=(e["external_id"] in aten_ids) if e["external_id"] is not None else None
    intervals=[(e["start_us_rel"],e["end_us_rel"]) for e in device]; merged=_union(intervals); gaps=[merged[i+1][0]-merged[i][1] for i in range(len(merged)-1)]
    kernels=[e for e in device if e["category"]=="kernel"]
    complete=bool(kernels) and all(e["aten_linked"] is not None for e in kernels)
    unlinked=[e for e in kernels if e["aten_linked"] is False] if complete else []
    unlinked_groups=_top_names(unlinked,limit=100000)
    small_names={r["name"] for r in unlinked_groups if r["count"]>=REPEATED_MIN_COUNT and r["median_us"]<=SMALL_KERNEL_MAX_US}
    small_events=[e for e in unlinked if e["name"] in small_names]; large_events=[e for e in unlinked if e["name"] not in small_names]
    large_by_name={name:_union_us([(e["start_us_rel"],e["end_us_rel"]) for e in large_events if e["name"]==name]) for name in sorted({e["name"] for e in large_events})}
    runtime_groups={}
    for e in runtime:
        fam=_runtime_family(e["name"]); row=runtime_groups.setdefault(fam,{"count":0,"trace_duration_us":0.0,"names":{}})
        row["count"]+=1; row["trace_duration_us"]+=e["duration_us"]; row["names"][e["name"]]=row["names"].get(e["name"],0)+1
    for row in runtime_groups.values():row["names"]=dict(sorted(row["names"].items(),key=lambda kv:(kv[1],kv[0]),reverse=True))
    all_large=[r for r in _top_names(kernels) if not (r["count"]>=REPEATED_MIN_COUNT and r["median_us"]<=SMALL_KERNEL_MAX_US)]
    return {"call_annotation":{"label":label,"duration_us":float(end-start)},"device_activity_categories_observed":sorted({e["category"] for e in device}),
      "device_activity_count":len(device),"device_active_union_us":_union_us(intervals),"idle_gaps_us":_summary(gaps),"idle_gap_total_us":float(sum(gaps)),
      "chronological_device_activities":device,"top_device_activity_names":_top_names(device),"top_kernel_names":_top_names(kernels),"top_large_kernel_names":all_large[:20],
      "cuda_runtime":{"available":bool(runtime),"events":runtime,"families":runtime_groups,"self_time_exposed":False,
                      "note":"Chrome cuda_runtime X duration/count persisted; self time is not invented."},
      "aten_external_ids_observed":len(aten_ids),"kernel_external_id_attribution_complete":complete,
      "unlinked_kernel_count":len(unlinked),"unlinked_kernel_active_union_us":_union_us([(e["start_us_rel"],e["end_us_rel"]) for e in unlinked]),
      "repeated_small_unlinked_kernel_names":sorted(small_names),"repeated_small_unlinked_active_union_us":_union_us([(e["start_us_rel"],e["end_us_rel"]) for e in small_events]),
      "unlinked_large_kernel_by_name_union_us":large_by_name,"top_unlinked_large_kernel_names":_top_names(large_events),
      "unavailable_fields":{"kernel_external_id_attribution":None if complete else "one or more observed kernel events lacked External id; all kernel mechanism classification disabled",
        "cuda_runtime_self_time":"not separately exposed by Chrome X events; trace duration/count reported without inventing self time",
        "cuda_runtime":None if runtime else "no cuda_runtime X events exposed inside call annotation",
        "gpu_memcpy":None if any(e["category"] in {"gpu_memcpy","memcpy"} for e in device) else "no GPU memcpy activity exposed inside call annotation"}}

def _candidate_row(trace:dict[str,Any],baseline_ms:float,profiled_ms:float,parent_gap:float)->dict[str,Any]:
    distortion=profiled_ms/baseline_ms if baseline_ms>0 else float("inf"); correction=1.0/distortion if math.isfinite(distortion) and distortion>0 else 0.0
    idle=float(trace["idle_gap_total_us"])/1000.0*correction; complete=bool(trace["kernel_external_id_attribution_complete"])
    small=float(trace["repeated_small_unlinked_active_union_us"])/1000.0*correction if complete else 0.0
    unlinked=float(trace["unlinked_kernel_active_union_us"])/1000.0*correction if complete else 0.0
    large={name:float(us)/1000.0*correction for name,us in trace["unlinked_large_kernel_by_name_union_us"].items()} if complete else {}
    residual=max(parent_gap-idle-unlinked,0.0)
    return {"profile_distortion_ratio":distortion,"distortion_correction":correction,"kernel_mechanism_classification_enabled":complete,
      "baseline_equivalent_ms":{"launch_or_idle_reduction":idle,"repeated_small_kernel_fusion":small,
        "explicit_device_kernel_optimization_by_name":large,"all_unlinked_kernel_activity":unlinked},
      "parent_unresolved_gap_ms":parent_gap,"unavailable_or_unattributed_residual_ms":residual,"materiality":{"ms_min":MATERIAL_MS_MIN,"share_min":MATERIAL_SHARE_MIN}}

def _decide(rows:dict[str,Any]):
    per_batch={}
    for b,row in rows.items():
        c=row["candidate"]; parent=c["parent_unresolved_gap_ms"]; vals={"launch_or_idle_reduction":c["baseline_equivalent_ms"]["launch_or_idle_reduction"]}
        if c["kernel_mechanism_classification_enabled"]:
            vals["repeated_small_kernel_fusion"]=c["baseline_equivalent_ms"]["repeated_small_kernel_fusion"]
            large=c["baseline_equivalent_ms"]["explicit_device_kernel_optimization_by_name"]
            if large:
                name=max(large,key=large.get); vals[f"explicit_device_kernel_optimization::{name}"]=large[name]
        ranked=sorted(vals,key=lambda k:vals[k],reverse=True)
        per_batch[b]={"ranked":ranked,"values_ms":vals,"shares":{k:(vals[k]/parent if parent>0 else 0.0) for k in vals}}
    common=set(per_batch["8"]["values_ms"])&set(per_batch["64"]["values_ms"]); material=[]
    for key in common:
        if all(per_batch[b]["values_ms"][key]>=MATERIAL_MS_MIN and per_batch[b]["shares"][key]>=MATERIAL_SHARE_MIN for b in ("8","64")):material.append(key)
    target=None
    if material:
        best=max(material,key=lambda k:sum(per_batch[b]["values_ms"][k] for b in ("8","64")))
        if all(per_batch[b]["ranked"] and per_batch[b]["ranked"][0]==best for b in ("8","64")):
            target="explicit_device_kernel_optimization" if best.startswith("explicit_device_kernel_optimization::") else best
    return target,{"per_batch":per_batch,"common_material_candidates":sorted(material),"unavailable_residual_cannot_authorize":True}

@app.function(image=image,gpu="L4",cpu=4,memory=16384,timeout=MAX_GPU_SECONDS,volumes={"/vol":volume})
def run_diagnostic()->dict[str,Any]:
    import torch
    import tam_research.aera_v25_post8471_triage as triage
    import tam_research.aera_v26_5_end_to_end_systems as base
    import tam_research.aera_v26_9_issue643_bounded_memory_end_to_end_systems as systems
    from tam_research.aera_hardware_core_v26_9_ficem_read_identity_weight_visibility import IdentityWeightVisibilityTritonFICEMReadWriteBackend
    volume.reload()
    if Path(RESULT_PATH).exists():raise RuntimeError(f"issue702 result already exists: {RESULT_PATH}")
    _parent(); blobs=_frozen_blobs()
    if not torch.cuda.is_available():raise RuntimeError("issue702 requires authorized NVIDIA L4")
    device=torch.device("cuda"); torch.set_float32_matmul_precision("high")
    before=base.checkpoint_hashes(base.CHECKPOINT_RELATIVE_DIR)
    if before!=CHECKPOINT_HASHES:raise RuntimeError("issue702 checkpoint drift before model load")
    reference,candidate,transformer,backend_names=systems.load_models_v26_9(run_dir=base.CHECKPOINT_RELATIVE_DIR,device=device)
    expected=IdentityWeightVisibilityTritonFICEMReadWriteBackend.name
    if tuple(backend_names)!=tuple(expected for _ in candidate.stages):raise RuntimeError("issue702 backend drift")
    del reference,transformer; gc.collect(); torch.cuda.empty_cache(); rows={}
    with torch.inference_mode():
        for batch in SYSTEM_BATCH_SIZES:
            b=str(batch); gen=torch.Generator(device="cpu").manual_seed(TOKEN_SEED_BASE+TOKEN_SEED_OFFSET+batch)
            tokens=torch.randint(0,triage.VOCAB_SIZE,(batch,triage.SEQ_LEN),generator=gen).to(device)
            for _ in range(WARMUP_CALLS):warm=base._model_call(candidate,tokens,update_memory=True); del warm
            baseline=[]; baseline_routes=[]
            for _ in range(BASELINE_CALLS):
                a=torch.cuda.Event(enable_timing=True); z=torch.cuda.Event(enable_timing=True); a.record(); out=base._model_call(candidate,tokens,update_memory=True); z.record(); torch.cuda.synchronize()
                baseline.append(float(a.elapsed_time(z))); baseline_routes.append(_route_rows(out)); del out
            if any(r!=baseline_routes[0] for r in baseline_routes[1:]):raise RuntimeError(f"issue702 baseline route drift batch {batch}")
            prof_rows=[]
            for profile_index in range(PROFILE_CALLS):
                label=f"aera702.call.batch{batch}.profile{profile_index}"; trace_path=Path(f"/tmp/aera702-b{batch}-p{profile_index}.json")
                a=torch.cuda.Event(enable_timing=True); z=torch.cuda.Event(enable_timing=True)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False,with_stack=False) as prof:
                    with torch.profiler.record_function(label):a.record(); out=base._model_call(candidate,tokens,update_memory=True); z.record(); torch.cuda.synchronize()
                profiled_ms=float(a.elapsed_time(z)); routes=_route_rows(out)
                if routes!=baseline_routes[0]:raise RuntimeError(f"issue702 profiled route differs from baseline batch {batch}")
                prof.export_chrome_trace(str(trace_path)); trace=_analyze_trace(trace_path,label); trace_path.unlink(missing_ok=True)
                prof_rows.append({"profiled_full_ms":profiled_ms,"route_rows":routes,"trace":trace}); del out,prof
            if len(prof_rows)!=1:raise RuntimeError("issue702 frozen PROFILE_CALLS must equal one")
            base_s=_summary(baseline); profiled=prof_rows[0]; candidate_row=_candidate_row(profiled["trace"],base_s["median"],profiled["profiled_full_ms"],EXPECTED_PARENT[b]["gap"])
            rows[b]={"batch_size":batch,"token_seed":TOKEN_SEED_BASE+TOKEN_SEED_OFFSET+batch,"sequence_length":int(triage.SEQ_LEN),
              "route_mode":"hard_sparse","hard":True,"update_memory":True,"cuda_bf16_autocast_via_frozen_model_call":True,
              "parent_issue697":EXPECTED_PARENT[b],"unprofiled_full":base_s,"profiled_full_ms":profiled["profiled_full_ms"],
              "route_rows":profiled["route_rows"],"route_matches_all_unprofiled_calls":True,"trace":profiled["trace"],"candidate":candidate_row}
            del tokens; torch.cuda.empty_cache()
    after=base.checkpoint_hashes(base.CHECKPOINT_RELATIVE_DIR)
    if after!=before:raise RuntimeError("issue702 checkpoint hashes changed")
    next_target,decision=_decide(rows)
    result={"scope":"aera_v26_9_issue702_cuda_timeline_dispatch_gap_attribution","research_issue":RESEARCH_ISSUE,
      "source_main":SOURCE_MAIN,"source_tree":SOURCE_TREE,"governance_predecessor_issue":GOVERNANCE_PREDECESSOR_ISSUE,
      "governance_incident_comment":GOVERNANCE_INCIDENT_COMMENT,"excluded_inert_ref":EXCLUDED_INERT_REF,
      "parent_trigger":PARENT_TRIGGER,"parent_evidence_comment":PARENT_EVIDENCE_COMMENT,"parent_run":PARENT_RUN,"parent_job":PARENT_JOB,
      "parent_attempt":PARENT_ATTEMPT,"parent_result_path":PARENT_RESULT_PATH,"source_decision":SOURCE_DECISION,"source_decision_changed":False,
      "device":torch.cuda.get_device_name(device),"checkpoint_hashes_before":before,"checkpoint_hashes_after":after,"checkpoint_hashes_unchanged":True,
      "frozen_blobs":blobs,"candidate_backend_names":list(backend_names),"batches":list(SYSTEM_BATCH_SIZES),
      "token_seed_rule":"138471 + 10000 + batch_size","warmup_calls":WARMUP_CALLS,"baseline_calls":BASELINE_CALLS,"profile_calls":PROFILE_CALLS,
      "small_kernel_max_us":SMALL_KERNEL_MAX_US,"repeated_min_count":REPEATED_MIN_COUNT,"material_share_min":MATERIAL_SHARE_MIN,"material_ms_min":MATERIAL_MS_MIN,
      "profiler":{"activities":["CPU","CUDA"],"record_shapes":False,"profile_memory":False,"with_stack":False,"chrome_trace_used":True,
                  "whole_call_user_annotation":True,"profiled_measurements_may_be_perturbed":True,"comparative_systems_evidence":False},
      "rows":rows,"next_target":next_target,"decision_evidence":decision,"comparative_gate_rerun":False,"reference_model_executed":False,
      "transformer_model_executed":False,"training_performed":False,"optimizer_created":False,"backward_performed":False,"scientific_seed_consumed":False,
      "optimization_authorized":False,"systems_pass_earned":False,"architecture_freeze_authorized":False,"s2_authorized":False,
      "fresh_scientific_seed_authorized":False,"independent_replication_credit":False,"100m_authorized":False,"breakthrough_proven":False}
    path=Path(RESULT_PATH); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n"); volume.commit()
    summary={"scope":result["scope"],"research_issue":RESEARCH_ISSUE,"device":result["device"],"source_decision":SOURCE_DECISION,
      "checkpoint_hashes_unchanged":True,"next_target":next_target,"batches":{b:{"parent_gap_ms":r["candidate"]["parent_unresolved_gap_ms"],
        "baseline_full_median_ms":r["unprofiled_full"]["median"],"profiled_full_ms":r["profiled_full_ms"],"profiler_distortion_ratio":r["candidate"]["profile_distortion_ratio"],
        "route_rows":r["route_rows"],"device_activity_count":r["trace"]["device_activity_count"],"device_active_union_us":r["trace"]["device_active_union_us"],
        "idle_gaps_us":r["trace"]["idle_gaps_us"],"cuda_runtime_families":r["trace"]["cuda_runtime"]["families"],
        "top_device_activity_names":r["trace"]["top_device_activity_names"],"top_large_kernel_names":r["trace"]["top_large_kernel_names"],
        "kernel_external_id_attribution_complete":r["trace"]["kernel_external_id_attribution_complete"],
        "repeated_small_unlinked_kernel_names":r["trace"]["repeated_small_unlinked_kernel_names"],"top_unlinked_large_kernel_names":r["trace"]["top_unlinked_large_kernel_names"],
        "baseline_equivalent_ms":r["candidate"]["baseline_equivalent_ms"],"unavailable_or_unattributed_residual_ms":r["candidate"]["unavailable_or_unattributed_residual_ms"],
        "unavailable_fields":r["trace"]["unavailable_fields"]} for b,r in rows.items()},"decision_evidence":decision,
      "optimization_authorized":False,"systems_pass_earned":False,"architecture_freeze_authorized":False,"s2_authorized":False,
      "fresh_scientific_seed_authorized":False,"independent_replication_credit":False,"100m_authorized":False,"breakthrough_proven":False}
    print(RESULT_MARKER+json.dumps(summary,sort_keys=True)); return summary

@app.local_entrypoint()
def preauth_main()->None:print(PREAUTH_MARKER+json.dumps(preflight.remote(),sort_keys=True))

@app.local_entrypoint()
def l4_main()->None:
    pre=preflight.remote(); print(PREAUTH_MARKER+json.dumps(pre,sort_keys=True))
    print(L4_START_MARKER+json.dumps({"research_issue":RESEARCH_ISSUE,"gpu":"L4","max_gpu_seconds":MAX_GPU_SECONDS,"result_path":RESULT_PATH,"diagnostic_only":True},sort_keys=True))
    print(SUMMARY_MARKER+json.dumps(run_diagnostic.remote(),sort_keys=True))
