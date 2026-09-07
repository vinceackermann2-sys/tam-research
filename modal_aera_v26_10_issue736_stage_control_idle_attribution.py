from __future__ import annotations
"""Issue #736 diagnostic-only stage/control-flow idle attribution after #732/#735."""
from contextlib import contextmanager
import gc, hashlib, json, math, statistics
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Iterator
import modal
import modal_aera_v26_10_issue710_memory_safe_harness as frozen710

APP_NAME="aera-v26-10-issue736-stage-control-idle-attribution"
VOLUME_NAME=frozen710.VOLUME_NAME
RESULT_PATH="/vol/aera-v26/issue736-stage-control-idle-attribution/result.json"
PARENT_RESULT_PATH="/vol/aera-v26/issue732-post731-differential-kernel-time-attribution/result.json"
SOURCE_MAIN="d15d94f39e59d0c6cc8d1514f44a33452d279a74"
SOURCE_TREE="fe5181034b3cb07076b17a43c7137829124d8cbb"
RESEARCH_ISSUE=736
PARENT_TRIGGER=735
PARENT_EVIDENCE_COMMENT=5574994790
PARENT_RUN=34156438962
PARENT_JOB=101849137998
PARENT_ATTEMPT=1
ISSUE710_LAUNCHER="modal_aera_v26_10_issue710_memory_safe_harness.py"
ISSUE710_LAUNCHER_BLOB="b885250753ea169cd89dd7a978bb3647fd261fe8"
V26_10_IMPL_BLOB="d8f691c198eed1fa96bcbb78a4e76cad82d18779"
CHECKPOINT_HASHES=dict(frozen710.CHECKPOINT_HASHES)
BATCHES=(8,64)
TOKEN_SEED_BASE=frozen710.TOKEN_SEED_BASE
WARMUP_CALLS=3
UNPROFILED_CALLS=3
PROFILE_CALLS=1
MIN_REGION_MS=1.0
MIN_REGION_SHARE=0.10
MAX_GPU_SECONDS=420
PARENT_OBS={
"8":{"candidate_median_ms":52.51891326904297,"candidate_profiled_ms":89.11689758300781,"candidate_kernel_count":1716,"candidate_idle_ms":47.15760479756141},
"64":{"candidate_median_ms":101.74742126464844,"candidate_profiled_ms":126.5898208618164,"candidate_kernel_count":1755,"candidate_idle_ms":58.12677116908063}}
ALLOWED_TARGETS=("stage_route_control","expert_dispatch_control","latent_reasoner_control","ficem_read_write_control",
"write_select_update_control","chunk_or_stage_glue_control","context_attention_control","norm_start_controller_control",
"end_controller_control","reason_to_chunk_output_norm_control","recurrent_stream_update_control")
PREAUTH_MARKER="AERA_V26_10_ISSUE736_PREAUTH_JSON="
L4_START_MARKER="AERA_V26_10_ISSUE736_L4_START_JSON="
RESULT_MARKER="AERA_V26_10_ISSUE736_RESULT_JSON="
SUMMARY_MARKER="AERA_V26_10_ISSUE736_SUMMARY_JSON="

image=frozen710.image.add_local_file(ISSUE710_LAUNCHER,f"/root/{ISSUE710_LAUNCHER}")
app=modal.App(APP_NAME)
volume=modal.Volume.from_name(VOLUME_NAME,create_if_missing=False)

def _blob(path:Path)->str:
    data=path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()

def _summary(xs:list[float])->dict[str,float]:
    ys=[float(x) for x in xs]
    if not ys:return {"samples":0.0,"mean":0.0,"median":0.0,"p95":0.0,"min":0.0,"max":0.0}
    s=sorted(ys); rank=max(0,min(len(s)-1,math.ceil(.95*len(s))-1))
    return {"samples":float(len(ys)),"mean":float(statistics.fmean(ys)),"median":float(statistics.median(ys)),
            "p95":float(s[rank]),"min":float(min(ys)),"max":float(max(ys))}

def _parent()->dict[str,Any]:
    pth=Path(PARENT_RESULT_PATH)
    if not pth.exists():raise RuntimeError("issue736 parent #732 durable result missing")
    p=json.loads(pth.read_text())
    if p.get("research_issue")!=732 or p.get("next_target") is not None:raise RuntimeError("issue736 parent identity/target drift")
    for k in ("optimization_authorized","full_e2e_systems_gate_authorized","systems_pass_earned","architecture_freeze_authorized",
              "s2_authorized","fresh_scientific_seed_authorized","independent_replication_credit","100m_authorized","breakthrough_proven"):
        if p.get(k) is not False:raise RuntimeError(f"issue736 parent authority drift: {k}")
    rows=p.get("rows")
    if not isinstance(rows,dict) or set(rows)!={"8","64"}:raise RuntimeError("issue736 parent rows drift")
    for b,e in PARENT_OBS.items():
        r=rows[b]["candidate_v26_10"]
        checks=(("median",float(r["unprofiled"]["median"]),e["candidate_median_ms"]),
                ("profiled",float(r["trace"]["profiled_cuda_event_ms"]),e["candidate_profiled_ms"]),
                ("idle",float(r["normalized"]["normalized_idle_gap_ms"]),e["candidate_idle_ms"]))
        for name,got,want in checks:
            if not math.isclose(got,want,rel_tol=0,abs_tol=1e-12):raise RuntimeError(f"issue736 parent {name} drift {b}")
        if int(r["trace"]["kernel_count"])!=e["candidate_kernel_count"]:raise RuntimeError(f"issue736 parent kernel drift {b}")
    return p

def _frozen_blobs()->dict[str,str]:
    got=dict(frozen710._frozen_blobs()); got["issue710_launcher"]=_blob(Path(f"/root/{ISSUE710_LAUNCHER}"))
    if got.get("issue710_launcher")!=ISSUE710_LAUNCHER_BLOB:raise RuntimeError("issue736 #710 launcher drift")
    if got.get("v26_10_impl")!=V26_10_IMPL_BLOB:raise RuntimeError("issue736 v26.10 implementation drift")
    return got

@app.function(image=image,cpu=4,memory=8192,timeout=180,volumes={"/vol":volume})
def preflight()->dict[str,Any]:
    import tam_research.aera_v26_5_end_to_end_systems as base
    volume.reload()
    if Path(RESULT_PATH).exists():raise RuntimeError("issue736 result already exists")
    _parent(); blobs=_frozen_blobs(); hashes=base.checkpoint_hashes(base.CHECKPOINT_RELATIVE_DIR)
    if hashes!=CHECKPOINT_HASHES:raise RuntimeError("issue736 checkpoint drift")
    return {"research_issue":736,"source_main":SOURCE_MAIN,"source_tree":SOURCE_TREE,"parent_trigger":PARENT_TRIGGER,
      "parent_evidence_comment":PARENT_EVIDENCE_COMMENT,"parent_run":PARENT_RUN,"parent_job":PARENT_JOB,"parent_attempt":PARENT_ATTEMPT,
      "parent_next_target":None,"result_path":RESULT_PATH,"parent_result_path":PARENT_RESULT_PATH,"frozen_blobs":blobs,
      "checkpoint_hashes":hashes,"result_absent":True,"gpu_used":False,"model_constructed":False,"new_measurement_performed":False,
      "optimization_authorized":False,"full_e2e_systems_gate_authorized":False,"systems_pass_earned":False,
      "architecture_freeze_authorized":False,"s2_authorized":False,"fresh_scientific_seed_authorized":False,
      "independent_replication_credit":False,"100m_authorized":False,"breakthrough_proven":False}

def _restore(obj:Any,name:str,had:bool,previous:Any)->None:
    if had:object.__setattr__(obj,name,previous)
    elif name in getattr(obj,"__dict__",{}):object.__delattr__(obj,name)

@contextmanager
def _instrument(candidate,torch_module)->Iterator[None]:
    import tam_research.aera_hardware_core_v25_1 as v25_1
    restored:list[Callable[[],None]]=[]; counts={f"stage{i}":0 for i in range(len(candidate.stages))}
    stage_names={id(s):f"stage{i}" for i,s in enumerate(candidate.stages)}
    def ann(label:str,call:Callable[[],Any])->Any:
        with torch_module.profiler.record_function(label):return call()
    original=candidate._route_one_stage; had="_route_one_stage" in candidate.__dict__; prev=candidate.__dict__.get("_route_one_stage")
    def route(this,x,stage,stage_state,router,*,route_mode,update_memory):
        s=stage_names[id(stage)]
        return ann(f"issue736.route_scope.{s}",lambda:original(x,stage,stage_state,router,route_mode=route_mode,update_memory=update_memory))
    object.__setattr__(candidate,"_route_one_stage",MethodType(route,candidate)); restored.append(lambda:_restore(candidate,"_route_one_stage",had,prev))
    for i,router in enumerate(candidate.stage_routers):
        orig=router.forward; h="forward" in router.__dict__; p=router.__dict__.get("forward")
        def rf(this,*args,_o=orig,_i=i,**kwargs):return ann(f"issue736.route_gate.stage{_i}",lambda:_o(*args,**kwargs))
        object.__setattr__(router,"forward",MethodType(rf,router)); restored.append(lambda o=router,h=h,p=p:_restore(o,"forward",h,p))
    for i,stage in enumerate(candidate.stages):
        s=f"stage{i}"; orig=stage.forward_chunk; h="forward_chunk" in stage.__dict__; p=stage.__dict__.get("forward_chunk")
        def sf(this,events,state,*,hard,update_memory,_o=orig,_s=s):
            counts[_s]=0; return ann(f"issue736.stage_scope.{_s}",lambda:_o(events,state,hard=hard,update_memory=update_memory))
        object.__setattr__(stage,"forward_chunk",MethodType(sf,stage)); restored.append(lambda o=stage,h=h,p=p:_restore(o,"forward_chunk",h,p))
        if hasattr(stage,"_tokenwise_context"):
            orig=stage._tokenwise_context; h="_tokenwise_context" in stage.__dict__; p=stage.__dict__.get("_tokenwise_context")
            def cf(this,*args,_o=orig,_s=s,**kwargs):return ann(f"issue736.context_scope.{_s}",lambda:_o(*args,**kwargs))
            object.__setattr__(stage,"_tokenwise_context",MethodType(cf,stage)); restored.append(lambda o=stage,h=h,p=p:_restore(o,"_tokenwise_context",h,p))
        controller=stage.controller; orig=controller.forward; h="forward" in controller.__dict__; p=controller.__dict__.get("forward")
        def ctrl(this,*args,_o=orig,_s=s,**kwargs):
            n=counts[_s]; counts[_s]=n+1; part="controller_start" if n==0 else "controller_end"
            return ann(f"issue736.{part}.{_s}",lambda:_o(*args,**kwargs))
        object.__setattr__(controller,"forward",MethodType(ctrl,controller)); restored.append(lambda o=controller,h=h,p=p:_restore(o,"forward",h,p))
        for attr,label in (("norm","norm"),("state_to_chunk","state_to_chunk"),("attn","attn"),
                           ("reason_to_chunk","reason_to_chunk"),("out_norm","out_norm"),
                           ("stream_input_norm","stream_input_norm"),("stream_cell","stream_cell"),
                           ("pair_write_gate","pair_write_gate")):
            mod=getattr(stage,attr); orig=mod.forward; h="forward" in mod.__dict__; p=mod.__dict__.get("forward")
            def mf(this,*args,_o=orig,_l=label,_s=s,**kwargs):return ann(f"issue736.{_l}.{_s}",lambda:_o(*args,**kwargs))
            object.__setattr__(mod,"forward",MethodType(mf,mod)); restored.append(lambda o=mod,h=h,p=p:_restore(o,"forward",h,p))
        experts=stage.experts; orig=experts.forward; h="forward" in experts.__dict__; p=experts.__dict__.get("forward")
        def ef(this,*args,_o=orig,_s=s,**kwargs):return ann(f"issue736.expert_scope.{_s}",lambda:_o(*args,**kwargs))
        object.__setattr__(experts,"forward",MethodType(ef,experts)); restored.append(lambda o=experts,h=h,p=p:_restore(o,"forward",h,p))
        seen={id(experts)}
        for path,mod in experts.named_modules():
            if not path or id(mod) in seen or any(True for _ in mod.children()):continue
            seen.add(id(mod)); orig=mod.forward; h="forward" in mod.__dict__; p=mod.__dict__.get("forward"); clean=path.replace(".","_")
            def ec(this,*args,_o=orig,_s=s,_p=clean,**kwargs):return ann(f"issue736.expert_child.{_s}.{_p}",lambda:_o(*args,**kwargs))
            object.__setattr__(mod,"forward",MethodType(ec,mod)); restored.append(lambda o=mod,h=h,p=p:_restore(o,"forward",h,p))
        reasoner=stage.reasoner; orig=reasoner.forward; h="forward" in reasoner.__dict__; p=reasoner.__dict__.get("forward")
        def rr(this,*args,_o=orig,_s=s,**kwargs):return ann(f"issue736.reasoner_scope.{_s}",lambda:_o(*args,**kwargs))
        object.__setattr__(reasoner,"forward",MethodType(rr,reasoner)); restored.append(lambda o=reasoner,h=h,p=p:_restore(o,"forward",h,p))
        cell=reasoner.cell; orig=cell.forward; h="forward" in cell.__dict__; p=cell.__dict__.get("forward")
        def rc(this,*args,_o=orig,_s=s,**kwargs):return ann(f"issue736.reasoner_cell.{_s}",lambda:_o(*args,**kwargs))
        object.__setattr__(cell,"forward",MethodType(rc,cell)); restored.append(lambda o=cell,h=h,p=p:_restore(o,"forward",h,p))
        backend=stage.memory._execution_backend
        for method in ("read","update","update_from_projected"):
            orig=getattr(backend,method); h=method in getattr(backend,"__dict__",{}); p=getattr(backend,"__dict__",{}).get(method)
            def bf(this,*args,_o=orig,_s=s,_m=method,**kwargs):return ann(f"issue736.ficem_scope.{_s}.{_m}",lambda:_o(*args,**kwargs))
            object.__setattr__(backend,method,MethodType(bf,backend)); restored.append(lambda o=backend,n=method,h=h,p=p:_restore(o,n,h,p))
    orig_select=v25_1.select_budgeted_event_pairs
    def ws(*args,**kwargs):return ann("issue736.write_select",lambda:orig_select(*args,**kwargs))
    v25_1.select_budgeted_event_pairs=ws; restored.append(lambda:setattr(v25_1,"select_budgeted_event_pairs",orig_select))
    try:yield
    finally:
        for fn in reversed(restored):fn()

def _clip(e:dict[str,Any],lo:float,hi:float):
    if e.get("ph")!="X":return None
    try:a=float(e["ts"]); b=a+float(e["dur"])
    except (KeyError,TypeError,ValueError):return None
    x=max(a,lo); y=min(b,hi); return (x,y) if y>x else None

def _union(xs:list[tuple[float,float]])->list[tuple[float,float]]:
    out=[]
    for a,b in sorted(xs):
        if out and a<=out[-1][1]:out[-1]=(out[-1][0],max(out[-1][1],b))
        else:out.append((a,b))
    return out

def _subtract(parent:tuple[float,float],children:list[tuple[float,float]])->list[tuple[float,float]]:
    segs=[parent]
    for ca,cb in _union(children):
        nxt=[]
        for a,b in segs:
            if cb<=a or ca>=b:nxt.append((a,b));continue
            if ca>a:nxt.append((a,min(ca,b)))
            if cb<b:nxt.append((max(cb,a),b))
        segs=nxt
    return [(a,b) for a,b in segs if b>a]

def _intersect(events:list[tuple[float,float]],segs:list[tuple[float,float]])->list[tuple[float,float]]:
    out=[]
    for ea,eb in events:
        for sa,sb in segs:
            a=max(ea,sa); b=min(eb,sb)
            if b>a:out.append((a,b))
    return _union(out)

def _us(xs:list[tuple[float,float]])->float:return float(sum(b-a for a,b in _union(xs)))
def _gaps(xs:list[tuple[float,float]])->list[float]:
    m=_union(xs); return [m[i+1][0]-m[i][1] for i in range(len(m)-1)]

def _family(label:str)->str|None:
    if label=="issue736.whole_call":return "chunk_or_stage_glue_control"
    if label.startswith("issue736.route_scope.") or label.startswith("issue736.route_gate."):return "chunk_or_stage_glue_control" if label.endswith(".stage0") else "stage_route_control"
    if label.startswith("issue736.stage_scope."):return "chunk_or_stage_glue_control"
    if label.startswith("issue736.norm.") or label.startswith("issue736.controller_start."):return "norm_start_controller_control"
    if label.startswith("issue736.context_scope.") or label.startswith("issue736.state_to_chunk.") or label.startswith("issue736.attn."):return "context_attention_control"
    if label.startswith("issue736.expert_scope.") or label.startswith("issue736.expert_child."):return "expert_dispatch_control"
    if label.startswith("issue736.controller_end."):return "end_controller_control"
    if label.startswith("issue736.reasoner_scope.") or label.startswith("issue736.reasoner_cell."):return "latent_reasoner_control"
    if label.startswith("issue736.reason_to_chunk.") or label.startswith("issue736.out_norm."):return "reason_to_chunk_output_norm_control"
    if label.startswith("issue736.stream_input_norm.") or label.startswith("issue736.stream_cell."):return "recurrent_stream_update_control"
    if label.startswith("issue736.pair_write_gate.") or label=="issue736.write_select":return "write_select_update_control"
    if label.startswith("issue736.ficem_scope."):return "ficem_read_write_control"
    return None

def _analyze(path:Path,whole_label:str,unprofiled_median:float,profiled_ms:float)->dict[str,Any]:
    raw=json.loads(path.read_text()); events=raw.get("traceEvents")
    if not isinstance(events,list):raise RuntimeError("issue736 traceEvents missing")
    anns=[]; dev=[]; runtime=[]
    for e in events:
        if not isinstance(e,dict):continue
        x=_clip(e,-float("inf"),float("inf"))
        if not x:continue
        cat=str(e.get("cat","")).lower(); name=str(e.get("name",""))
        if cat=="user_annotation" and name.startswith("issue736."):anns.append({"name":name,"start":x[0],"end":x[1]})
        elif cat in {"kernel","gpu_memcpy","gpu_memset","memcpy","memset"}:dev.append((x[0],x[1],cat,name))
        elif cat=="cuda_runtime":runtime.append((x[0],x[1],name))
    wholes=[a for a in anns if a["name"]==whole_label]
    if len(wholes)!=1:raise RuntimeError(f"issue736 expected one whole annotation, got {len(wholes)}")
    ws,we=wholes[0]["start"],wholes[0]["end"]
    anns=[a for a in anns if a["start"]>=ws and a["end"]<=we]; dev=[d for d in dev if d[1]>ws and d[0]<we]; runtime=[r for r in runtime if r[1]>ws and r[0]<we]
    devints=[(max(a,ws),min(b,we)) for a,b,_,_ in dev]
    inv=[]; families={}; unmapped=[]
    for idx,a in enumerate(anns):
        parent=(a["start"],a["end"])
        children=[(c["start"],c["end"]) for j,c in enumerate(anns) if j!=idx and c["start"]>=a["start"] and c["end"]<=a["end"] and (c["start"]>a["start"] or c["end"]<a["end"])]
        excl=_subtract(parent,children); full_dev=_intersect(devints,[parent]); excl_dev=_intersect(devints,excl); idle=_gaps(excl_dev)
        full_cpu=_us([parent]); excl_cpu=_us(excl); excl_dev_us=_us(excl_dev); residual=max(excl_cpu-excl_dev_us,0.0)
        rtev=[r for r in runtime if any(sa<=r[0]<sb for sa,sb in excl)]
        counts={"launch":sum("launch" in n.lower() for _,_,n in rtev),"synchronize_or_wait":sum(("synchron" in n.lower() or "wait" in n.lower()) for _,_,n in rtev),"memcpy_or_memset":sum(("memcpy" in n.lower() or "memset" in n.lower()) for _,_,n in rtev),"total":len(rtev)}
        fam=_family(a["name"]); idle_stats=_summary(idle); idle_total=float(sum(idle))
        inv.append({"name":a["name"],"family":fam,"nested":bool(children),"children_subtracted":True,"cpu_annotation_elapsed_us":full_cpu,"full_device_active_union_us":_us(full_dev),"exclusive_cpu_us":excl_cpu,"exclusive_device_active_union_us":excl_dev_us,"exclusive_residual_us":residual,"exclusive_inter_device_idle_total_us":idle_total,"exclusive_inter_device_idle_stats_us":idle_stats,"cuda_runtime_counts_exclusive":counts})
        if fam:
            g=families.setdefault(fam,{"invocation_count":0,"exclusive_cpu_us":0.0,"exclusive_device_active_union_us":0.0,"exclusive_residual_us":0.0,"exclusive_inter_device_idle_total_us":0.0,"_idle_values":[],"cuda_runtime_counts":{"launch":0,"synchronize_or_wait":0,"memcpy_or_memset":0,"total":0}})
            g["invocation_count"]+=1; g["exclusive_cpu_us"]+=excl_cpu; g["exclusive_device_active_union_us"]+=excl_dev_us; g["exclusive_residual_us"]+=residual; g["exclusive_inter_device_idle_total_us"]+=idle_total; g["_idle_values"].extend(idle)
            for k,v in counts.items():g["cuda_runtime_counts"][k]+=v
        else:unmapped.append(a["name"])
    scale=unprofiled_median/profiled_ms if profiled_ms>0 else 0.0
    for fam,g in families.items():
        g["exclusive_inter_device_idle_stats_us"]=_summary(g.pop("_idle_values")); ms=min(g["exclusive_residual_us"]/1000.0*scale,unprofiled_median)
        g["distortion_normalized_host_control_idle_ms"]=ms; g["share_of_unprofiled_median"]=ms/unprofiled_median if unprofiled_median>0 else 0.0; g["directly_observed"]=True
    wholedev=_intersect(devints,[(ws,we)]); gaps=_gaps(wholedev)
    return {"annotation_count":len(anns),"raw_invocations":inv,"exclusive_family_table":families,"unmapped_annotations":sorted(set(unmapped)),"profiled_cuda_event_ms":profiled_ms,"profiler_distortion_ratio":profiled_ms/unprofiled_median if unprofiled_median>0 else float("inf"),"normalization_scale":scale,"whole_device_active_union_us":_us(wholedev),"whole_inter_device_idle_total_us":float(sum(gaps)),"whole_inter_device_idle_stats_us":_summary(gaps),"device_activity_count":len(dev),"cuda_runtime_event_count":len(runtime),"exclusive_partition_no_double_count":True,"unavailable_fields":{"durable_state_pack_output":"not directly isolated without runtime source edit","first_vs_second_expert_subphases":"leaf module children observed, but routing ownership not inferred beyond expert parent","kernel_to_aten_ownership":"not inferred"}}

def _routes(out:dict[str,Any])->list[dict[str,Any]]:
    routes=out.get("stage_routes")
    if not isinstance(routes,list):raise RuntimeError("issue736 missing stage_routes")
    rows=[]
    for ci,chunk in enumerate(routes):
        if not isinstance(chunk,list):raise RuntimeError("issue736 malformed route chunk")
        for si,r in enumerate(chunk):
            gate=r.get("stage_route_gate"); frac=r.get("executed_fraction")
            if not hasattr(gate,"detach") or not isinstance(frac,(float,int)):raise RuntimeError("issue736 route evidence missing")
            h=gate.detach().ge(.5).cpu(); selected=int(h.sum().item()); population=int(h.numel()); expected=selected/population if population else 0.0
            if not math.isclose(float(frac),expected,rel_tol=0,abs_tol=1e-12):raise RuntimeError("issue736 executed_fraction drift")
            rows.append({"chunk_index":ci,"stage_index":si,"selected":selected,"population":population,"executed_fraction":float(frac)})
    return rows

def _choose(rows:dict[str,Any])->tuple[str|None,dict[str,Any]]:
    rankings={}; q={}
    for b in ("8","64"):
        tab=rows[b]["trace"]["exclusive_family_table"]; order=sorted(tab,key=lambda k:tab[k]["distortion_normalized_host_control_idle_ms"],reverse=True)
        rankings[b]=[{"family":k,"ms":tab[k]["distortion_normalized_host_control_idle_ms"],"share":tab[k]["share_of_unprofiled_median"]} for k in order]
        q[b]={k for k in ALLOWED_TARGETS if k in tab and tab[k]["directly_observed"] and tab[k]["distortion_normalized_host_control_idle_ms"]>=MIN_REGION_MS and tab[k]["share_of_unprofiled_median"]>=MIN_REGION_SHARE}
    common=q["8"]&q["64"]; target=None
    if common:
        top8=next((r["family"] for r in rankings["8"] if r["family"] in common),None); top64=next((r["family"] for r in rankings["64"] if r["family"] in common),None)
        if top8==top64:target=top8
    return target,{"rankings":rankings,"qualifying":{"8":sorted(q["8"]),"64":sorted(q["64"])},"common_qualifying":sorted(common),"same_largest_common_required":True,"unavailable_unattributed_cannot_authorize":True}

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
    if Path(RESULT_PATH).exists():raise RuntimeError("issue736 result already exists")
    _parent(); blobs=_frozen_blobs()
    if not torch.cuda.is_available():raise RuntimeError("issue736 requires authorized NVIDIA L4")
    device=torch.device("cuda"); torch.set_float32_matmul_precision("high")
    before=base.checkpoint_hashes(base.CHECKPOINT_RELATIVE_DIR)
    if before!=CHECKPOINT_HASHES:raise RuntimeError("issue736 checkpoint drift before model load")
    payload=torch.load(Path(base.CHECKPOINT_RELATIVE_DIR)/"aera.pt",map_location="cpu",weights_only=False)
    if payload.get("seed")!=base.SOURCE_CHECKPOINT_SEED:raise RuntimeError("issue736 seed drift")
    baseline=base._build_v26(payload,device); candidate=base._build_v26(payload,device)
    bb=systems._install_v26_9_candidate_backend(baseline); cb=systems._install_v26_9_candidate_backend(candidate); installed=install_latent_depth_sync_coalescing_v26_10(candidate)
    expected=IdentityWeightVisibilityTritonFICEMReadWriteBackend.name
    if not all(x==expected for x in bb+cb):raise RuntimeError("issue736 backend drift")
    if installed!=tuple(range(len(candidate.stages))):raise RuntimeError("issue736 v26.10 installer drift")
    schema=frozen710.frozen706._schemas_and_weights_exact(baseline,candidate)
    if not schema["pass"]:raise RuntimeError("issue736 schema drift")
    bv=base._parameter_versions(baseline); cv=base._parameter_versions(candidate); rows={}
    with torch.inference_mode():
        for batch in BATCHES:
            gen=torch.Generator(device="cpu").manual_seed(TOKEN_SEED_BASE+batch); tokens=torch.randint(0,triage.VOCAB_SIZE,(batch,triage.SEQ_LEN),generator=gen).to(device)
            bcall=lambda:base._model_call(baseline,tokens,update_memory=True); ccall=lambda:base._model_call(candidate,tokens,update_memory=True)
            bout,bdec=frozen710.frozen706._capture_decisions(baseline,bcall); cout,cdec=frozen710.frozen706._capture_decisions(candidate,ccall)
            if not isinstance(bout.get("state"),HardwareAERAState) or not isinstance(cout.get("state"),HardwareAERAState):raise RuntimeError("issue736 state missing")
            blog=bout.get("logits"); clog=cout.get("logits")
            if not isinstance(blog,torch.Tensor) or not isinstance(clog,torch.Tensor):raise RuntimeError("issue736 logits missing")
            deq=frozen710.frozen706._decision_equivalence(bdec,cdec); route=frozen710.frozen706._route_exact(base,bout,cout); logeq=frozen710._chunked_logit_equivalence(blog,clog); stateeq=base._state_equivalence(bout["state"],cout["state"]); finite=bool(base._finite_output(bout) and base._finite_output(cout))
            if not (deq["pass"] and route and logeq["pass"] and stateeq["pass"] and finite):raise RuntimeError(f"issue736 correctness failed batch {batch}")
            del bout,cout,blog,clog; gc.collect(); torch.cuda.empty_cache()
            for _ in range(WARMUP_CALLS):o=ccall(); del o
            samples=[frozen710.frozen706._event_timed_call(ccall) for _ in range(UNPROFILED_CALLS)]; unprofiled=frozen710.frozen706._summary(samples)
            label="issue736.whole_call"; path=Path(f"/tmp/issue736-b{batch}.json"); a=torch.cuda.Event(enable_timing=True); z=torch.cuda.Event(enable_timing=True)
            with _instrument(candidate,torch):
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False,with_stack=False) as prof:
                    with torch.profiler.record_function(label):a.record(); pout=ccall(); z.record(); torch.cuda.synchronize()
            profiled=float(a.elapsed_time(z)); route_rows=_routes(pout); prof.export_chrome_trace(str(path)); trace=_analyze(path,label,float(unprofiled["median_ms"]),profiled); path.unlink(missing_ok=True); del pout,prof
            rows[str(batch)]={"batch_size":batch,"token_seed":TOKEN_SEED_BASE+batch,"route_mode":"hard_sparse","hard":True,"update_memory":True,"cuda_bf16_autocast_via_frozen_model_call":True,"correctness":{"pass":True,"route_exact":route,"decision_equivalence":deq,"logit_equivalence":logeq,"state_equivalence":stateeq,"finite":finite},"route_rows":route_rows,"unprofiled_whole_call":unprofiled,"trace":trace}
            del tokens; gc.collect(); torch.cuda.empty_cache()
    after=base.checkpoint_hashes(base.CHECKPOINT_RELATIVE_DIR); versions=(bv==base._parameter_versions(baseline) and cv==base._parameter_versions(candidate))
    if after!=before or not versions:raise RuntimeError("issue736 immutability drift")
    target,evidence=_choose(rows)
    result={"scope":"aera_v26_10_issue736_stage_control_idle_attribution","research_issue":736,"source_main":SOURCE_MAIN,"source_tree":SOURCE_TREE,"parent_trigger":PARENT_TRIGGER,"parent_evidence_comment":PARENT_EVIDENCE_COMMENT,"parent_run":PARENT_RUN,"parent_job":PARENT_JOB,"parent_attempt":PARENT_ATTEMPT,"parent_next_target":None,"device":torch.cuda.get_device_name(device),"checkpoint_hashes_before":before,"checkpoint_hashes_after":after,"checkpoint_hashes_unchanged":True,"parameter_versions_unchanged":True,"schema_check":schema,"frozen_blobs":blobs,"batches":list(BATCHES),"warmup_calls":WARMUP_CALLS,"unprofiled_calls":UNPROFILED_CALLS,"profile_calls":PROFILE_CALLS,"min_region_ms":MIN_REGION_MS,"min_region_share":MIN_REGION_SHARE,"allowed_targets":list(ALLOWED_TARGETS),"rows":rows,"next_target_region":target,"decision_evidence":evidence,"diagnostic_only":True,"optimization_authorized":False,"full_e2e_systems_gate_authorized":False,"systems_pass_earned":False,"architecture_freeze_authorized":False,"s2_authorized":False,"fresh_scientific_seed_authorized":False,"independent_replication_credit":False,"100m_authorized":False,"breakthrough_proven":False}
    p=Path(RESULT_PATH); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n"); volume.commit()
    summary={"research_issue":736,"device":result["device"],"next_target_region":target,"batches":{b:{"unprofiled_whole_call":r["unprofiled_whole_call"],"profiled_cuda_event_ms":r["trace"]["profiled_cuda_event_ms"],"profiler_distortion_ratio":r["trace"]["profiler_distortion_ratio"],"exclusive_family_table":r["trace"]["exclusive_family_table"],"route_rows":r["route_rows"],"unavailable_fields":r["trace"]["unavailable_fields"]} for b,r in rows.items()},"decision_evidence":evidence,"optimization_authorized":False,"full_e2e_systems_gate_authorized":False,"systems_pass_earned":False,"architecture_freeze_authorized":False,"s2_authorized":False,"fresh_scientific_seed_authorized":False,"independent_replication_credit":False,"100m_authorized":False,"breakthrough_proven":False}
    print(RESULT_MARKER+json.dumps(summary,sort_keys=True)); return summary

@app.local_entrypoint()
def preauth_main()->None:print(PREAUTH_MARKER+json.dumps(preflight.remote(),sort_keys=True))
@app.local_entrypoint()
def l4_main()->None:
    pre=preflight.remote(); print(PREAUTH_MARKER+json.dumps(pre,sort_keys=True)); print(L4_START_MARKER+json.dumps({"research_issue":736,"gpu":"L4","max_gpu_seconds":MAX_GPU_SECONDS,"result_path":RESULT_PATH,"diagnostic_only":True},sort_keys=True)); print(SUMMARY_MARKER+json.dumps(run_diagnostic.remote(),sort_keys=True))
