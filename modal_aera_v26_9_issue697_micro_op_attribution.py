from __future__ import annotations
"""Issue #697 diagnostic-only micro-op attribution after #687/#696."""
from contextlib import contextmanager
import gc, hashlib, json, math, statistics
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Iterator
import modal
import modal_aera_v26_9_issue665_frozen_throughput_component_attribution as issue665

APP_NAME="aera-v26-9-issue697-micro-op-attribution"
VOLUME_NAME=issue665.VOLUME_NAME
RESULT_PATH="/vol/aera-v26/issue697-micro-op-attribution/result.json"
PARENT_RESULT_PATH="/vol/aera-v26/issue687-stage-internal-throughput-attribution/result.json"
MAX_GPU_SECONDS=300
RESEARCH_ISSUE=697
SOURCE_MAIN="2b89fd0e6d12ad3f0d2d0ca870d838669cd7fddd"
SOURCE_TREE="ad75aae06273fe42769855f5273f5e516a1dd20f"
PARENT_TRIGGER=696
PARENT_EVIDENCE_COMMENT=5569795957
PARENT_RUN=34115412471
PARENT_JOB=101720942690
PARENT_ATTEMPT=1
SOURCE_DECISION="FAIL_FROZEN_E2E_SYSTEMS_GATE"
ISSUE665_LAUNCHER="modal_aera_v26_9_issue665_frozen_throughput_component_attribution.py"
ISSUE687_LAUNCHER="modal_aera_v26_9_issue687_stage_internal_throughput_attribution.py"
EXPECTED_BLOBS={
"issue665_launcher":"72f27391ff2f0a7bff8d4532f307ddc4869cf494",
"issue687_launcher":"b9950c032686a496c6d944c30c441428499cb367",
"scientific_adapter":"512572340cc09e2e7ad6729712258c12cb377ef2",
"runtime_interface":"268644ac4edee15a4cc4e29d3fed7f61eeb3caa7",
"stage_v25_1":"1c3456d8040455b4cd1194db4c8586f77d0f3e43",
"tokenwise_v19":"98008bceb8c68af3bc346e5dfcc7a8218875661e",
"base_core":"ffe0341829fb905e40ef9ae3544f797471f1aa9c",
"nohost_v25_1":"237e5615cf32f644e8675808a6b3e9adaf04fb23",
"v26_9_backend":"b81cc209f5d95abbe1fb8bd620c78e87c067bc19",
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
PROFILE_CALLS=3
MATERIAL_SHARE_MIN=0.15
MATERIAL_MS_MIN=1.0
EXPECTED_PARENT={
"8":{"full":41.13203239440918,"stage":26.655376447364688,"glue":6.067023945972323,"experts":3.601471960544586},
"64":{"full":76.7083511352539,"stage":39.28654378838837,"glue":6.752576315775514,"experts":13.073423981666565},
}
PREAUTH_MARKER="AERA_V26_9_ISSUE697_MICRO_OP_PREAUTH_JSON="
L4_START_MARKER="AERA_V26_9_ISSUE697_MICRO_OP_L4_START_JSON="
RESULT_MARKER="AERA_V26_9_ISSUE697_MICRO_OP_RESULT_JSON="
SUMMARY_MARKER="AERA_V26_9_ISSUE697_MICRO_OP_SUMMARY_JSON="
image=(issue665.image
 .add_local_file(ISSUE665_LAUNCHER,f"/root/{ISSUE665_LAUNCHER}")
 .add_local_file(ISSUE687_LAUNCHER,f"/root/{ISSUE687_LAUNCHER}"))
app=modal.App(APP_NAME)
volume=modal.Volume.from_name(VOLUME_NAME,create_if_missing=False)

def _blob(path:Path)->str:
    data=path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()

def _summary(xs:list[float])->dict[str,float]:
    xs=[float(x) for x in xs]
    if not xs:return {"samples":0.0,"mean_ms":0.0,"median_ms":0.0,"min_ms":0.0,"max_ms":0.0}
    return {"samples":float(len(xs)),"mean_ms":float(statistics.fmean(xs)),
            "median_ms":float(statistics.median(xs)),"min_ms":float(min(xs)),"max_ms":float(max(xs))}

def _parent()->dict[str,Any]:
    path=Path(PARENT_RESULT_PATH)
    if not path.exists():raise RuntimeError("issue697 parent #687 result missing")
    p=json.loads(path.read_text())
    if p.get("research_issue")!=687 or p.get("source_decision")!=SOURCE_DECISION:
        raise RuntimeError("issue697 parent authority drift")
    if p.get("next_target") is not None or p.get("optimization_authorized") is not False or p.get("systems_pass_earned") is not False:
        raise RuntimeError("issue697 parent decision drift")
    rows=p.get("rows")
    if not isinstance(rows,dict) or set(rows)!={"8","64"}:raise RuntimeError("issue697 parent rows drift")
    for b,e in EXPECTED_PARENT.items():
        m=rows[b]["measurement"]; cats=m["exclusive_categories"]
        got={"full":m["full_call"]["median_ms"],"stage":m["stage_compute_excluding_ficem"]["median_ms"],
             "glue":cats["residual_stage_glue"]["median_ms"],"experts":cats["sparse_experts"]["median_ms"]}
        if got!=e:raise RuntimeError(f"issue697 parent medians drift {b}: {got}")
    return p

def _frozen_blobs()->dict[str,str]:
    import tam_research.aera_hardware_core as core
    import tam_research.aera_hardware_core_v19 as v19
    import tam_research.aera_hardware_core_v25_1 as v251
    import tam_research.aera_hardware_core_v25_1_nohost as nohost
    import tam_research.aera_hardware_core_v26 as runtime
    import tam_research.aera_hardware_core_v26_9_ficem_read_identity_weight_visibility as backend
    import tam_research.aera_v26_9_issue643_bounded_memory_end_to_end_systems as systems
    got={
      "issue665_launcher":_blob(Path(f"/root/{ISSUE665_LAUNCHER}")),
      "issue687_launcher":_blob(Path(f"/root/{ISSUE687_LAUNCHER}")),
      "scientific_adapter":_blob(Path(systems.__file__)),"runtime_interface":_blob(Path(runtime.__file__)),
      "stage_v25_1":_blob(Path(v251.__file__)),"tokenwise_v19":_blob(Path(v19.__file__)),
      "base_core":_blob(Path(core.__file__)),"nohost_v25_1":_blob(Path(nohost.__file__)),
      "v26_9_backend":_blob(Path(backend.__file__)),
    }
    if got!=EXPECTED_BLOBS:raise RuntimeError(f"issue697 frozen blob drift: {got}")
    return got

@app.function(image=image,cpu=4,memory=8192,timeout=180,volumes={"/vol":volume})
def preflight()->dict[str,Any]:
    import tam_research.aera_v26_5_end_to_end_systems as base
    volume.reload()
    if Path(RESULT_PATH).exists():raise RuntimeError(f"issue697 result already exists: {RESULT_PATH}")
    _parent(); blobs=_frozen_blobs(); hashes=base.checkpoint_hashes(base.CHECKPOINT_RELATIVE_DIR)
    if hashes!=CHECKPOINT_HASHES:raise RuntimeError("issue697 checkpoint drift")
    return {"research_issue":697,"source_main":SOURCE_MAIN,"source_tree":SOURCE_TREE,
      "parent_trigger":PARENT_TRIGGER,"parent_evidence_comment":PARENT_EVIDENCE_COMMENT,
      "parent_run":PARENT_RUN,"parent_job":PARENT_JOB,"parent_attempt":PARENT_ATTEMPT,
      "source_decision":SOURCE_DECISION,"result_path":RESULT_PATH,"parent_result_path":PARENT_RESULT_PATH,
      "frozen_blobs":blobs,"checkpoint_hashes":hashes,"result_absent":True,
      "gpu_used":False,"model_constructed":False,"new_measurement_performed":False,
      "optimization_authorized":False,"systems_pass_earned":False,"architecture_freeze_authorized":False,
      "s2_authorized":False,"fresh_scientific_seed_authorized":False,"independent_replication_credit":False,
      "100m_authorized":False,"breakthrough_proven":False}


def _restore(obj:Any,name:str,had:bool,previous:Any)->None:
    if had:object.__setattr__(obj,name,previous)
    elif name in getattr(obj,"__dict__",{}):object.__delattr__(obj,name)

class _Recorder:
    def __init__(self,torch_module):
        self.torch=torch_module; self.enabled=False; self.current_stage=None; self.events={}
    def reset(self):self.events={}
    def record(self,label:str,call:Callable[[],Any])->Any:
        if not self.enabled:return call()
        a=self.torch.cuda.Event(enable_timing=True); b=self.torch.cuda.Event(enable_timing=True); a.record()
        try:
            with self.torch.profiler.record_function(f"aera697.{label}"):return call()
        finally:
            b.record(); self.events.setdefault(label,[]).append((a,b))
    def elapsed(self)->dict[str,list[float]]:
        return {k:[float(a.elapsed_time(b)) for a,b in v] for k,v in self.events.items()}

@contextmanager
def _instrument(candidate,torch_module)->Iterator[_Recorder]:
    import tam_research.aera_hardware_core_v25_1 as v251
    r=_Recorder(torch_module); restores=[]
    def install(obj,name,label,stage_name):
        original=getattr(obj,name); had=name in getattr(obj,"__dict__",{}); previous=getattr(obj,"__dict__",{}).get(name)
        def wrapped(this,*args,_o=original,_l=label,_s=stage_name,**kwargs):
            return r.record(f"{_l}.{_s}",lambda:_o(*args,**kwargs))
        object.__setattr__(obj,name,MethodType(wrapped,obj))
        restores.append(lambda o=obj,n=name,h=had,p=previous:_restore(o,n,h,p))
    for i,stage in enumerate(candidate.stages):
        s="foundation" if i==0 else f"optional_{i}"
        original=stage.forward_chunk; had="forward_chunk" in stage.__dict__; previous=stage.__dict__.get("forward_chunk")
        def wf(this,events,state,*,hard,update_memory,_o=original,_s=s):
            prev=r.current_stage; r.current_stage=_s
            try:return r.record(f"stage.{_s}",lambda:_o(events,state,hard=hard,update_memory=update_memory))
            finally:r.current_stage=prev
        object.__setattr__(stage,"forward_chunk",MethodType(wf,stage))
        restores.append(lambda o=stage,h=had,p=previous:_restore(o,"forward_chunk",h,p))
        original_ctx=stage._tokenwise_context; had_ctx="_tokenwise_context" in stage.__dict__; prev_ctx=stage.__dict__.get("_tokenwise_context")
        def wc(this,h,state,start_control,_o=original_ctx,_s=s):
            return r.record(f"child.context.{_s}",lambda:_o(h,state,start_control))
        object.__setattr__(stage,"_tokenwise_context",MethodType(wc,stage))
        restores.append(lambda o=stage,h=had_ctx,p=prev_ctx:_restore(o,"_tokenwise_context",h,p))
        for attr,label in (("norm","child.norm"),("state_to_chunk","child.state_to_chunk"),
          ("attn","child.attention"),("reasoner","child.reasoner"),("reason_to_chunk","child.reason_to_chunk"),
          ("out_norm","child.out_norm"),("stream_input_norm","child.stream_input_norm"),
          ("stream_cell","child.stream_cell"),("pair_write_gate","child.pair_write_gate")):
            install(getattr(stage,attr),"forward",label,s)
        install(stage.experts,"forward","experts",s)
        install(stage.controller,"forward","child.controller",s)
        backend=stage.memory._execution_backend
        for name in ("read","update","update_from_projected"):install(backend,name,f"child.ficem_{name}",s)
    original_select=v251.select_budgeted_event_pairs
    def select(*args,**kwargs):
        s=r.current_stage or "unattributed"
        return r.record(f"child.event_pair_select.{s}",lambda:original_select(*args,**kwargs))
    v251.select_budgeted_event_pairs=select
    try:yield r
    finally:
        v251.select_budgeted_event_pairs=original_select
        for restore in reversed(restores):restore()

def _sum(t:dict[str,list[float]],label:str)->float:return float(sum(t.get(label,[])))

def _regions(t:dict[str,list[float]],stages:list[str])->dict[str,float]:
    glue=experts=stage_total=0.0
    for s in stages:
        stage=_sum(t,f"stage.{s}")
        fr=_sum(t,f"child.ficem_read.{s}"); fu=_sum(t,f"child.ficem_update.{s}"); fp=_sum(t,f"child.ficem_update_from_projected.{s}")
        ficem=fr+fu+fp; stage_ex=max(stage-ficem,0.0)
        context=max(_sum(t,f"child.context.{s}")-fr,0.0); ex=_sum(t,f"experts.{s}")
        accounted=(_sum(t,f"child.norm.{s}")+_sum(t,f"child.controller.{s}")+context+
          _sum(t,f"child.attention.{s}")+ex+_sum(t,f"child.reasoner.{s}")+
          _sum(t,f"child.reason_to_chunk.{s}")+_sum(t,f"child.out_norm.{s}")+
          _sum(t,f"child.stream_input_norm.{s}")+_sum(t,f"child.stream_cell.{s}")+
          _sum(t,f"child.pair_write_gate.{s}")+_sum(t,f"child.event_pair_select.{s}"))
        glue+=max(stage_ex-accounted,0.0); experts+=ex; stage_total+=stage_ex
    return {"glue":glue,"experts":experts,"stage":stage_total}

def _scope(event):
    p=getattr(event,"cpu_parent",None)
    while p is not None:
        n=str(getattr(p,"name",""))
        if n.startswith("aera697.experts."):return "experts",p
        if n.startswith("aera697.child."):return "child",p
        if n.startswith("aera697.stage."):return "glue",p
        p=getattr(p,"cpu_parent",None)
    return None,None

def _family(name:str)->str:
    n=name.lower()
    if any(x in n for x in ("bmm","matmul","addmm","einsum")) or n.endswith("::mm"):return "matmul_gemm"
    if any(x in n for x in ("softmax","topk","argmax","sort","kthvalue")):return "softmax_topk_sort"
    if any(x in n for x in ("gelu","relu","silu")):return "activation"
    if any(x in n for x in ("index","gather","scatter","select")):return "index_gather_scatter"
    if any(x in n for x in ("_to_copy","copy_","::to","type_as")):return "transfer_cast_copy"
    if any(x in n for x in ("view","reshape","expand","contiguous","transpose","permute","slice","squeeze","unsqueeze","narrow")):return "shape_view_layout"
    if any(x in n for x in ("sum","mean","prod","amax","amin")):return "reduction"
    if any(x in n for x in ("zeros","empty","arange","fill_","ones")):return "allocation_mask"
    if any(x in n for x in ("add","mul","div","sub","sigmoid","clamp","ge","gt","le","lt","where","maximum","minimum")):return "elementwise"
    return "other"

def _profile(prof,regions:dict[str,float])->dict[str,Any]:
    dev={"glue":{},"experts":{}}; cpu={"glue":{},"experts":{}}; groups={}
    for e in prof.events():
        name=str(getattr(e,"name",""))
        if not name.startswith("aten::"):continue
        sc,scope_event=_scope(e)
        if sc not in dev:continue
        fam=_family(name); dm=float(getattr(e,"self_device_time_total",0.0))/1000.0; cm=float(getattr(e,"self_cpu_time_total",0.0))/1000.0
        dev[sc][fam]=dev[sc].get(fam,0.0)+dm; cpu[sc][fam]=cpu[sc].get(fam,0.0)+cm
        if sc=="experts" and scope_event is not None:groups.setdefault(id(scope_event),[]).append(e)
    detail={}; detail_cpu={}
    def add(label,e):
        detail[label]=detail.get(label,0.0)+float(getattr(e,"self_device_time_total",0.0))/1000.0
        detail_cpu[label]=detail_cpu.get(label,0.0)+float(getattr(e,"self_cpu_time_total",0.0))/1000.0
    for es in groups.values():
        proj=0; output_done=False
        es=sorted(es,key=lambda e:float(getattr(getattr(e,"time_range",None),"start",0.0)))
        for e in es:
            fam=_family(str(getattr(e,"name","")))
            if output_done:add("telemetry_tail",e); continue
            if fam=="matmul_gemm":
                proj+=1; add("projection_1" if proj==1 else "projection_2" if proj==2 else "projection_other",e)
            elif fam=="activation":add("activation",e)
            elif fam=="softmax_topk_sort":add("routing_selection",e)
            elif fam=="index_gather_scatter":add("weight_gather_and_indexing",e)
            elif fam=="reduction" and proj>=2:
                add("weighted_output_reduction",e); output_done=True
            elif proj>=2:add("weighted_output_math",e)
            else:add("routing_weight_math_and_layout",e)
    for sc,key in (("glue","glue"),("experts","experts")):
        dev[sc]["dispatch_or_unattributed_gap"]=max(regions[key]-sum(dev[sc].values()),0.0)
    detail["dispatch_or_unattributed_gap"]=max(regions["experts"]-sum(detail.values()),0.0)
    return {"device":dev,"cpu":cpu,"expert_detail":detail,"expert_detail_cpu":detail_cpu}

def _maps(rows:list[dict[str,float]])->dict[str,dict[str,float]]:
    keys=sorted({k for row in rows for k in row})
    return {k:_summary([row.get(k,0.0) for row in rows]) for k in keys}

def _normalize(stats:dict[str,dict[str,float]],parent_ms:float)->dict[str,float]:
    vals={k:max(float(v["median_ms"]),0.0) for k,v in stats.items()}; total=sum(vals.values())
    return {} if total<=0 else {k:parent_ms*v/total for k,v in vals.items()}


def _decide(rows:dict[str,Any]):
    impacts={}; totals={}
    for b,row in rows.items():
        g=row["baseline_equivalent"]["glue"]; e=row["baseline_equivalent"]["experts"]
        fams=set(g)|set(e); impacts[b]={k:g.get(k,0.0)+e.get(k,0.0) for k in fams}
        totals[b]=EXPECTED_PARENT[b]["glue"]+EXPECTED_PARENT[b]["experts"]
    fams=sorted({k for x in impacts.values() for k in x})
    combined={k:sum(impacts[b].get(k,0.0) for b in impacts) for k in fams}
    ranked=sorted(fams,key=lambda k:combined[k],reverse=True)
    target=None
    if ranked:
        c=ranked[0]
        if all(impacts[b].get(c,0.0)>=MATERIAL_MS_MIN and impacts[b].get(c,0.0)/totals[b]>=MATERIAL_SHARE_MIN for b in impacts):
            target=c
    regime={b:(max(v,key=v.get) if v else "none") for b,v in impacts.items()}
    return target,regime,{"impacts_ms":impacts,"unresolved_parent_ms":totals,"combined_ms":combined,
      "ranked":ranked,"material_share_min":MATERIAL_SHARE_MIN,"material_ms_min":MATERIAL_MS_MIN}

@app.function(image=image,gpu="L4",cpu=4,memory=16384,timeout=MAX_GPU_SECONDS,volumes={"/vol":volume})
def run_diagnostic()->dict[str,Any]:
    import torch
    import tam_research.aera_v25_post8471_triage as triage
    import tam_research.aera_v26_5_end_to_end_systems as base
    import tam_research.aera_v26_9_issue643_bounded_memory_end_to_end_systems as systems
    from tam_research.aera_hardware_core_v26_9_ficem_read_identity_weight_visibility import IdentityWeightVisibilityTritonFICEMReadWriteBackend
    volume.reload()
    if Path(RESULT_PATH).exists():raise RuntimeError(f"issue697 result already exists: {RESULT_PATH}")
    _parent(); blobs=_frozen_blobs()
    if not torch.cuda.is_available():raise RuntimeError("issue697 requires NVIDIA L4")
    device=torch.device("cuda"); torch.set_float32_matmul_precision("high")
    before=base.checkpoint_hashes(base.CHECKPOINT_RELATIVE_DIR)
    if before!=CHECKPOINT_HASHES:raise RuntimeError("issue697 checkpoint drift before model load")
    reference,candidate,transformer,backend_names=systems.load_models_v26_9(run_dir=base.CHECKPOINT_RELATIVE_DIR,device=device)
    expected=IdentityWeightVisibilityTritonFICEMReadWriteBackend.name
    if tuple(backend_names)!=tuple(expected for _ in candidate.stages):raise RuntimeError("issue697 backend drift")
    del reference,transformer; gc.collect(); torch.cuda.empty_cache()
    stages=["foundation" if i==0 else f"optional_{i}" for i in range(len(candidate.stages))]
    rows={}
    with torch.inference_mode(),_instrument(candidate,torch) as recorder:
        for batch in SYSTEM_BATCH_SIZES:
            b=str(batch); gen=torch.Generator(device="cpu").manual_seed(TOKEN_SEED_BASE+TOKEN_SEED_OFFSET+batch)
            tokens=torch.randint(0,triage.VOCAB_SIZE,(batch,triage.SEQ_LEN),generator=gen).to(device)
            recorder.enabled=False
            for _ in range(WARMUP_CALLS):
                out=base._model_call(candidate,tokens,update_memory=True); del out
            baseline=[]
            for _ in range(BASELINE_CALLS):
                a=torch.cuda.Event(enable_timing=True); z=torch.cuda.Event(enable_timing=True); a.record()
                out=base._model_call(candidate,tokens,update_memory=True); z.record(); torch.cuda.synchronize()
                baseline.append(float(a.elapsed_time(z))); del out
            prof_rows=[]
            for _ in range(PROFILE_CALLS):
                recorder.reset(); recorder.enabled=True
                a=torch.cuda.Event(enable_timing=True); z=torch.cuda.Event(enable_timing=True)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True,profile_memory=False,with_stack=False) as prof:
                    a.record(); out=base._model_call(candidate,tokens,update_memory=True); z.record(); torch.cuda.synchronize()
                full=float(a.elapsed_time(z)); timings=recorder.elapsed(); recorder.enabled=False
                regions=_regions(timings,stages); ops=_profile(prof,regions)
                prof_rows.append({"full":full,"regions":regions,**ops}); del out,prof; torch.cuda.empty_cache()
            glue=_maps([x["device"]["glue"] for x in prof_rows]); experts=_maps([x["device"]["experts"] for x in prof_rows])
            detail=_maps([x["expert_detail"] for x in prof_rows]); glue_cpu=_maps([x["cpu"]["glue"] for x in prof_rows])
            experts_cpu=_maps([x["cpu"]["experts"] for x in prof_rows]); detail_cpu=_maps([x["expert_detail_cpu"] for x in prof_rows])
            base_s=_summary(baseline); prof_s=_summary([x["full"] for x in prof_rows])
            rows[b]={"batch_size":batch,"token_seed":TOKEN_SEED_BASE+TOKEN_SEED_OFFSET+batch,"sequence_length":int(triage.SEQ_LEN),
              "route_mode":"hard_sparse","hard":True,"update_memory":True,"parent_issue687":EXPECTED_PARENT[b],
              "unprofiled_full":base_s,"profiled_full":prof_s,
              "profiler_distortion_ratio":prof_s["median_ms"]/base_s["median_ms"] if base_s["median_ms"]>0 else float("inf"),
              "profiled_regions":{"glue":_summary([x["regions"]["glue"] for x in prof_rows]),
                                  "experts":_summary([x["regions"]["experts"] for x in prof_rows]),
                                  "stage":_summary([x["regions"]["stage"] for x in prof_rows])},
              "device_generic":{"glue":glue,"experts":experts},"cpu_generic":{"glue":glue_cpu,"experts":experts_cpu},
              "expert_detail_device":detail,"expert_detail_cpu":detail_cpu,
              "baseline_equivalent":{"glue":_normalize(glue,EXPECTED_PARENT[b]["glue"]),
                                     "experts":_normalize(experts,EXPECTED_PARENT[b]["experts"]),
                                     "expert_detail":_normalize(detail,EXPECTED_PARENT[b]["experts"])},
              "raw_profile_calls":prof_rows}
            del tokens; torch.cuda.empty_cache()
    after=base.checkpoint_hashes(base.CHECKPOINT_RELATIVE_DIR)
    if after!=before:raise RuntimeError("issue697 checkpoint hashes changed")
    target,regime,decision=_decide(rows)
    result={"scope":"aera_v26_9_issue697_micro_op_attribution","research_issue":697,"source_main":SOURCE_MAIN,
      "source_tree":SOURCE_TREE,"parent_trigger":PARENT_TRIGGER,"parent_evidence_comment":PARENT_EVIDENCE_COMMENT,
      "parent_run":PARENT_RUN,"parent_job":PARENT_JOB,"parent_attempt":PARENT_ATTEMPT,"parent_result_path":PARENT_RESULT_PATH,
      "source_decision":SOURCE_DECISION,"source_decision_changed":False,"device":torch.cuda.get_device_name(device),
      "checkpoint_hashes_before":before,"checkpoint_hashes_after":after,"checkpoint_hashes_unchanged":True,
      "frozen_blobs":blobs,"candidate_backend_names":list(backend_names),"batches":list(SYSTEM_BATCH_SIZES),
      "token_seed_rule":"138471 + 10000 + batch_size","warmup_calls":WARMUP_CALLS,"baseline_calls":BASELINE_CALLS,
      "profile_calls":PROFILE_CALLS,"profiler":{"activities":["CPU","CUDA"],"record_shapes":True,
      "uses_record_function_annotations":True,"uses_same_stream_cuda_events":True,
      "profiled_measurements_may_be_perturbed":True,"comparative_systems_evidence":False},
      "rows":rows,"next_target":target,"regime_specific_targets":regime,"decision_evidence":decision,
      "comparative_gate_rerun":False,"reference_model_executed":False,"transformer_model_executed":False,
      "training_performed":False,"optimizer_created":False,"backward_performed":False,"scientific_seed_consumed":False,
      "optimization_authorized":False,"systems_pass_earned":False,"architecture_freeze_authorized":False,
      "s2_authorized":False,"fresh_scientific_seed_authorized":False,"independent_replication_credit":False,
      "100m_authorized":False,"breakthrough_proven":False}
    path=Path(RESULT_PATH); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n"); volume.commit()
    summary={"scope":result["scope"],"research_issue":697,"device":result["device"],"source_decision":SOURCE_DECISION,
      "checkpoint_hashes_unchanged":True,"next_target":target,"regime_specific_targets":regime,
      "material_share_min":MATERIAL_SHARE_MIN,"material_ms_min":MATERIAL_MS_MIN,
      "batches":{b:{"parent_glue_ms":EXPECTED_PARENT[b]["glue"],"parent_experts_ms":EXPECTED_PARENT[b]["experts"],
        "baseline_full_median_ms":r["unprofiled_full"]["median_ms"],"profiled_full_median_ms":r["profiled_full"]["median_ms"],
        "profiler_distortion_ratio":r["profiler_distortion_ratio"],
        "generic_impacts_ms":{k:r["baseline_equivalent"]["glue"].get(k,0.0)+r["baseline_equivalent"]["experts"].get(k,0.0)
          for k in sorted(set(r["baseline_equivalent"]["glue"])|set(r["baseline_equivalent"]["experts"]))},
        "expert_detail_ms":r["baseline_equivalent"]["expert_detail"]} for b,r in rows.items()},
      "optimization_authorized":False,"systems_pass_earned":False,"architecture_freeze_authorized":False,
      "s2_authorized":False,"fresh_scientific_seed_authorized":False,"independent_replication_credit":False,
      "100m_authorized":False,"breakthrough_proven":False}
    print(RESULT_MARKER+json.dumps(summary,sort_keys=True)); return summary

@app.local_entrypoint()
def preauth_main()->None:
    print(PREAUTH_MARKER+json.dumps(preflight.remote(),sort_keys=True))

@app.local_entrypoint()
def l4_main()->None:
    pre=preflight.remote(); print(PREAUTH_MARKER+json.dumps(pre,sort_keys=True))
    print(L4_START_MARKER+json.dumps({"research_issue":697,"gpu":"L4","max_gpu_seconds":MAX_GPU_SECONDS,
      "result_path":RESULT_PATH,"diagnostic_only":True},sort_keys=True))
    print(SUMMARY_MARKER+json.dumps(run_diagnostic.remote(),sort_keys=True))
