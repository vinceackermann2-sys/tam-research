from __future__ import annotations

"""Frozen seed-1 capability harness implementation for research issue #748.

This module is executable on CPU for smoke/invariant testing. Scientific model
seeds are guarded and cannot be consumed unless a later, separately authorized
runner explicitly opts in.
"""

from dataclasses import dataclass
import math
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F

from tam_research import aera_memory_capability_gate_v1 as gate
from tam_research.aera import AERAState, FastMemoryState
from tam_research.aera_hardware_core import HardwareAERAConfig, HardwareAERAState
from tam_research.aera_hardware_core_v18 import HardwareAwareAERATextLMV18, PretrainableDeltaFastMemory

RESEARCH_ISSUE=748
PARENT_RESEARCH_ISSUE=746
SOURCE_MAIN="9c17fc802e10383bb6abf0f3eb5098516970bb00"
SOURCE_TREE="92662708edc844f9d04067631ae2639f0a69ddfb"
PARENT_FIXTURE_BLOB="981602432b684989f7a5011ff953e6965368c5e5"
PARENT_PROTOCOL_BLOB="d2e184c9dc1469b172bbda8e96ed65c129931af6"
PARENT_CPU_TEST_BLOB="2e9072176ee79d2442fe12151b1856ec7582c873"
BRANCH="research/aera-issue748-seed1-capability-harness-v1"
SEED1_RESULT_PATH="/vol/aera-capability/issue748-seed1/result.json"
SEED1_CHECKPOINT_DIR="/vol/aera-capability/issue748-seed1/checkpoints"
SEED1_PREAUTH_PREFIX="[aera-issue748-seed1-preauth]"
SEED1_GPU_PREFIX="[aera-issue748-seed1-l4]"
SCIENTIFIC_MODEL_SEEDS=tuple(gate.SCIENTIFIC_MODEL_SEEDS)
SEED1_MODEL_SEED=SCIENTIFIC_MODEL_SEEDS[0]
CPU_TEST_SEED=gate.CPU_TEST_SEED
MODEL_CONFIG={"vocab_size":gate.VOCAB_SIZE,"d_model":128,"n_stages":4,"n_heads":4,"chunk_size":gate.CHUNK_SIZE,"n_experts":8,"max_active_experts":2,"expert_mult":4,"memory_dim":32,"max_reason_steps":4,"block_size":4,"fast_memory_lr":0.2,"fast_memory_decay":0.999}
TRAIN_SEQUENCE_CHUNKS=16
MICROBATCH_SIZE=2
GRAD_ACCUM_STEPS=2
TOKENS_PER_SEQUENCE=TRAIN_SEQUENCE_CHUNKS*gate.CHUNK_SIZE
TOKENS_PER_OPTIMIZER_STEP=MICROBATCH_SIZE*GRAD_ACCUM_STEPS*TOKENS_PER_SEQUENCE
OPTIMIZER_STEPS=gate.TOKEN_BUDGET_PER_TRAINED_VARIANT//TOKENS_PER_OPTIMIZER_STEP
CHECKPOINT_EVERY_STEPS=gate.CHECKPOINT_INTERVAL_TOKENS//TOKENS_PER_OPTIMIZER_STEP
ROUTER_CALIBRATION_EVERY=8
ADAMW_LR=3e-4
ADAMW_BETAS=(0.9,0.95)
ADAMW_EPS=1e-8
ADAMW_WEIGHT_DECAY=0.1
GRAD_CLIP_NORM=1.0
LR_WARMUP_STEPS=64
LR_MIN=3e-5
ROUTING_RANK_WEIGHT=0.10
ROUTING_BUDGET_WEIGHT=0.05
ROUTING_POLARIZATION_WEIGHT=0.01
EVAL_NONRESET_CASES_PER_DISTANCE=108
EVAL_RESET_CASES_PER_LONG_DISTANCE=54
EVAL_BATCH_SIZE=8
LATENCY_BATCH_SIZE=8
LATENCY_WARMUP_CALLS=3
LATENCY_TIMED_CALLS=10
GPU_AUTHORIZED=False
SCIENTIFIC_TRAINING_AUTHORIZED=False
FRESH_SCIENTIFIC_SEED_AUTHORIZED=False
SEEDS_2_3_AUTHORIZED=False
SYSTEMS_OPTIMIZATION_AUTHORIZED=False
ARCHITECTURE_FREEZE_AUTHORIZED=False
SCALING_AUTHORIZED=False
BREAKTHROUGH_PROVEN=False
TRAINED_VARIANTS=tuple(gate.TRAINED_VARIANTS)
REFERENCE_VARIANTS=("E_simple_retrieval","F_backbone_plus_recurrent_state")
if TOKENS_PER_OPTIMIZER_STEP*OPTIMIZER_STEPS!=gate.TOKEN_BUDGET_PER_TRAINED_VARIANT:raise RuntimeError("issue748 token budget geometry is not exact")
if CHECKPOINT_EVERY_STEPS*TOKENS_PER_OPTIMIZER_STEP!=gate.CHECKPOINT_INTERVAL_TOKENS:raise RuntimeError("issue748 checkpoint geometry is not exact")

@dataclass(frozen=True)
class MaterializedBatch:
    tokens:torch.Tensor
    targets:torch.Tensor
    prediction_positions:torch.Tensor
    reset_after_chunks:torch.Tensor
    cases:tuple[gate.MemoryCase,...]

@dataclass(frozen=True)
class CheckpointRecord:
    step:int
    tokens:int
    cumulative_gpu_seconds:float
    path:str|None

def scientific_config()->HardwareAERAConfig:return HardwareAERAConfig(**MODEL_CONFIG)
def _variant_spec(name:str)->gate.VariantSpec:
    for spec in gate.VARIANTS:
        if spec.name==name:return spec
    raise ValueError(f"unknown capability variant: {name}")
def _assert_model_seed_allowed(seed:int,*,scientific_seed_authorized:bool)->None:
    if seed in SCIENTIFIC_MODEL_SEEDS and not scientific_seed_authorized:raise PermissionError("scientific model seed requires separate explicit authorization")
def _restore_state_dtype(base:AERAState,update:AERAState)->AERAState:return AERAState(stream=update.stream.to(dtype=base.stream.dtype),memory=FastMemoryState(update.memory.matrix.to(dtype=base.memory.matrix.dtype)))
def _select_state(state:AERAState,idx:torch.Tensor)->AERAState:return AERAState(stream=state.stream.index_select(0,idx),memory=FastMemoryState(state.memory.matrix.index_select(0,idx)))
def _merge_state(base:AERAState,update:AERAState,idx:torch.Tensor)->AERAState:
    update=_restore_state_dtype(_select_state(base,idx),update)
    return AERAState(stream=base.stream.index_copy(0,idx,update.stream),memory=FastMemoryState(base.memory.matrix.index_copy(0,idx,update.memory.matrix)))
def _blend_state(base:AERAState,update:AERAState,gate_value:torch.Tensor)->AERAState:
    g1=gate_value.to(base.stream.dtype);g2=gate_value[:,:,None].to(base.memory.matrix.dtype);update=_restore_state_dtype(base,update)
    return AERAState(stream=base.stream+g1*(update.stream-base.stream),memory=FastMemoryState(base.memory.matrix+g2*(update.memory.matrix-base.memory.matrix)))
def _reset_rows(state:HardwareAERAState,mask:torch.Tensor)->HardwareAERAState:
    if mask.ndim!=1:raise ValueError("reset mask must be [batch]")
    if not bool(mask.any()):return state
    stages=[]
    for s in state.stages:
        stages.append(AERAState(stream=torch.where(mask[:,None],torch.zeros_like(s.stream),s.stream),memory=FastMemoryState(torch.where(mask[:,None,None],torch.zeros_like(s.memory.matrix),s.memory.matrix))))
    return HardwareAERAState(stages)

class CapabilityAblationLM(HardwareAwareAERATextLMV18):
    def __init__(self,variant_name:str,cfg:HardwareAERAConfig|None=None)->None:
        spec=_variant_spec(variant_name)
        if variant_name not in TRAINED_VARIANTS and variant_name!="F_backbone_plus_recurrent_state":raise ValueError("learned capability scaffold supports A/B/C/D and F only")
        super().__init__(scientific_config() if cfg is None else cfg)
        self.variant_spec=spec;self.variant_name=variant_name
        self.memory_enabled=bool(spec.persistent_learned_memory);self.adaptive_routing_enabled=bool(spec.adaptive_routing);self.recurrent_state_enabled=bool(spec.recurrent_state)
        self.set_router_task_gradient_isolation(True)
        for stage in self.stages:
            for p in stage.reasoner.parameters():p.requires_grad_(False)
            for p in stage.reason_to_chunk.parameters():p.requires_grad_(False)
            if not self.recurrent_state_enabled:
                for p in stage.stream_cell.parameters():p.requires_grad_(False)
                for p in stage.state_to_chunk.parameters():p.requires_grad_(False)
            if not self.memory_enabled:
                for p in stage.memory.parameters():p.requires_grad_(False)
        if not self.adaptive_routing_enabled:
            for router in self.stage_routers:
                for p in router.parameters():p.requires_grad_(False)
        else:self.set_optional_stage_routers_trainable(True)
        self.set_memory_pretraining_mode(False)
        for module in (self.next_event,self.block_draft,self.stream_forecast_projectors):
            for p in module.parameters():p.requires_grad_(False)
    def _stage_forward_without_latent_reasoning(self,stage,events:torch.Tensor,state:AERAState,*,intrinsic_hard:bool,update_memory:bool):
        h=stage.norm(events);start_control=stage.controller(h[:,0],state.stream)
        memory_read=stage.memory.read(h[:,:1],state.memory).squeeze(1) if self.memory_enabled else torch.zeros_like(state.stream)
        carried=stage.state_to_chunk(state.stream) if self.recurrent_state_enabled else torch.zeros_like(state.stream)
        h=h+(start_control["state_read"]*carried+start_control["memory_read"]*memory_read)[:,None,:]
        h=h+stage.attn(h);h=h+stage.experts(h,start_control["expert_logits"],start_control["expert_count_logits"],hard=intrinsic_hard)
        end_summary=h[:,-1];end_control=stage.controller(end_summary,state.stream);h=stage.out_norm(h)
        final_stream=stage.stream_cell(end_summary,state.stream) if self.recurrent_state_enabled else torch.zeros_like(state.stream)
        if self.memory_enabled:
            memory_state=state.memory
            if update_memory:
                write=(end_control["novelty"]*end_control["memory_write"]).clamp(0.0,1.0)
                memory_state=stage.memory.local_update(end_summary[:,None,:],write[:,None,:],state.memory)
        else:memory_state=FastMemoryState(torch.zeros_like(state.memory.matrix))
        return h,AERAState(final_stream,memory_state),{"start":start_control,"end":end_control}
    def _run_stage(self,x,stage,stage_state,router,stage_index:int,*,route_mode:str,intrinsic_hard:bool,update_memory:bool):
        fixed=(not self.adaptive_routing_enabled) or stage_index==self.FOUNDATION_STAGE
        if fixed:
            processed,new_state,controls=self._stage_forward_without_latent_reasoning(stage,x,stage_state,intrinsic_hard=intrinsic_hard,update_memory=update_memory)
            one=torch.ones(x.size(0),1,device=x.device,dtype=x.dtype)
            return processed,new_state,{"stage_route_probability":one,"stage_route_gate":one,"executed_fraction":1.0,"start":controls["start"],"end":controls["end"],"fixed_depth":True}
        if route_mode not in {"straight_through","hard_sparse"}:raise ValueError("adaptive capability routing requires straight_through or hard_sparse")
        gate_value,logits=router(x[:,0],stage_state.stream,mode=route_mode);probability=torch.sigmoid(logits)
        if route_mode=="straight_through":
            processed,processed_state,controls=self._stage_forward_without_latent_reasoning(stage,x,stage_state,intrinsic_hard=intrinsic_hard,update_memory=update_memory)
            task_gate=gate_value.detach();processed=processed.to(dtype=x.dtype);processed_state=_restore_state_dtype(stage_state,processed_state)
            y=x+task_gate[:,None,:].to(dtype=x.dtype)*(processed-x);new_state=_blend_state(stage_state,processed_state,task_gate)
            return y,new_state,{"stage_route_probability":probability,"stage_route_gate":task_gate,"executed_fraction":1.0,"start":controls["start"],"end":controls["end"],"task_router_gradient_isolated":True}
        run_idx=(gate_value[:,0]>=0.5).nonzero(as_tuple=False).squeeze(-1)
        if run_idx.numel()==0:return x,stage_state,{"stage_route_probability":probability,"stage_route_gate":gate_value,"executed_fraction":0.0,"start":None,"end":None}
        selected_x=x.index_select(0,run_idx);selected_state=_select_state(stage_state,run_idx)
        selected_y,selected_new_state,controls=self._stage_forward_without_latent_reasoning(stage,selected_x,selected_state,intrinsic_hard=intrinsic_hard,update_memory=update_memory)
        y=x.index_copy(0,run_idx,selected_y.to(dtype=x.dtype));new_state=_merge_state(stage_state,selected_new_state,run_idx)
        return y,new_state,{"stage_route_probability":probability,"stage_route_gate":gate_value,"executed_fraction":float(run_idx.numel()/x.size(0)),"start":controls["start"],"end":controls["end"]}
    def forward_answer_tokens(self,tokens:torch.Tensor,prediction_positions:torch.Tensor,reset_after_chunks:torch.Tensor,*,route_mode:str,intrinsic_hard:bool,update_memory:bool,state:HardwareAERAState|None=None):
        if tokens.ndim!=2 or tokens.size(1)%self.cfg.chunk_size:raise ValueError("tokens must be [batch, whole_chunks]")
        batch,total_tokens=tokens.shape;chunks=total_tokens//self.cfg.chunk_size
        if prediction_positions.shape!=(batch,):raise ValueError("prediction_positions must be [batch]")
        if reset_after_chunks.shape!=(batch,chunks):raise ValueError("reset_after_chunks shape mismatch")
        if state is None:state=self.empty_state(tokens)
        outputs=[];route_history=[];execution=[0.0 for _ in self.stages];current_state=state
        for chunk_index in range(chunks):
            start=chunk_index*self.cfg.chunk_size;chunk=tokens[:,start:start+self.cfg.chunk_size];pos=torch.arange(chunk.size(1),device=tokens.device);x=self.token_emb(chunk)+self.local_pos(pos)[None]
            new_states=[];stage_routes=[]
            for stage_index,(stage,stage_state,router) in enumerate(zip(self.stages,current_state.stages,self.stage_routers)):
                x,new_state,info=self._run_stage(x,stage,stage_state,router,stage_index,route_mode=route_mode,intrinsic_hard=intrinsic_hard,update_memory=update_memory)
                new_states.append(new_state);stage_routes.append(info);execution[stage_index]+=float(info["executed_fraction"])
            outputs.append(x);route_history.append(stage_routes);current_state=HardwareAERAState(new_states);current_state=_reset_rows(current_state,reset_after_chunks[:,chunk_index])
        all_hidden=torch.cat(outputs,dim=1)
        if bool((prediction_positions<0).any()) or bool((prediction_positions>=total_tokens).any()):raise ValueError("prediction position outside sequence")
        rows=torch.arange(batch,device=tokens.device);answer_hidden=self.norm(all_hidden[rows,prediction_positions]);answer_logits=self.lm_head(answer_hidden)
        self.last_stage_execution=[{"stage":float(i),"mean_executed_fraction":execution[i]/chunks} for i in range(len(execution))]
        return {"answer_logits":answer_logits,"state":current_state,"stage_routes":route_history,"routing_mode":route_mode,"prediction_positions":prediction_positions}

def build_model(variant_name:str,model_seed:int,*,scientific_seed_authorized:bool=False,device:torch.device|str="cpu")->CapabilityAblationLM:
    _assert_model_seed_allowed(model_seed,scientific_seed_authorized=scientific_seed_authorized);torch.manual_seed(int(model_seed));return CapabilityAblationLM(variant_name,scientific_config()).to(device)
def model_accounting(model:CapabilityAblationLM)->dict[str,int|float|str]:
    return {"variant":model.variant_name,"total_parameters":int(sum(p.numel() for p in model.parameters())),"trainable_parameters":int(sum(p.numel() for p in model.parameters() if p.requires_grad)),"memory_parameters":int(sum(p.numel() for stage in model.stages for p in stage.memory.parameters())),"stage_router_parameters":int(sum(p.numel() for p in model.stage_routers.parameters())),"latent_reasoner_parameters_registered_but_inactive":int(sum(p.numel() for stage in model.stages for p in stage.reasoner.parameters())),"recurrent_parameters_registered":int(sum(p.numel() for stage in model.stages for p in stage.stream_cell.parameters()))}
def _padding_chunk(sample_index:int,chunk_index:int)->tuple[int,...]:
    base=(gate.TRAIN_DATA_SEED*97409+sample_index*65537+chunk_index*257)%gate.NOISE_COUNT
    return tuple(gate.NOISE_BASE+((base+i*131)%gate.NOISE_COUNT) for i in range(gate.CHUNK_SIZE))
def materialize_cases(cases:Sequence[gate.MemoryCase],*,total_chunks:int|None=None,device:torch.device|str="cpu")->MaterializedBatch:
    if not cases:raise ValueError("cases must be non-empty")
    required=max(len(case.chunks) for case in cases);total=required if total_chunks is None else int(total_chunks)
    if total<required:raise ValueError("total_chunks shorter than case")
    token_rows=[];targets=[];positions=[];reset_rows=[]
    for case in cases:
        offset=total-len(case.chunks);chunks=[_padding_chunk(case.sample_index,i) for i in range(offset)]+list(case.chunks);token_rows.append([t for c in chunks for t in c])
        target_pos=offset*gate.CHUNK_SIZE+case.answer_chunk_index*gate.CHUNK_SIZE+case.answer_token_index;positions.append(target_pos-1);targets.append(case.expected_answer)
        reset_mask=[False]*total
        for source_chunk_index,source_chunk in enumerate(case.chunks):
            if gate.RESET in source_chunk:reset_mask[offset+source_chunk_index]=True
        reset_rows.append(reset_mask)
    return MaterializedBatch(torch.tensor(token_rows,dtype=torch.long,device=device),torch.tensor(targets,dtype=torch.long,device=device),torch.tensor(positions,dtype=torch.long,device=device),torch.tensor(reset_rows,dtype=torch.bool,device=device),tuple(cases))
def training_case(sample_index:int)->gate.MemoryCase:
    distance=gate.TRAIN_RETENTION_DISTANCES[sample_index%len(gate.TRAIN_RETENTION_DISTANCES)];concurrent=gate.TRAIN_CONCURRENT_FACTS[(sample_index//4)%len(gate.TRAIN_CONCURRENT_FACTS)];distractors=gate.TRAIN_DISTRACTOR_RECORDS[(sample_index//12)%len(gate.TRAIN_DISTRACTOR_RECORDS)];correction=bool((sample_index//36)%2);reset=bool(((sample_index//72)%8)==7 and distance>=2)
    return gate.generate_case(split="train",seed=gate.TRAIN_DATA_SEED,sample_index=sample_index,retention_distance_chunks=distance,concurrent_facts=concurrent,distractor_records=distractors,correction=correction,reset_before_query=reset)
def evaluation_cases()->tuple[gate.MemoryCase,...]:
    cases=[];combos=[(c,d,u) for c in gate.EVAL_CONCURRENT_FACTS for d in gate.EVAL_DISTRACTOR_RECORDS for u in (False,True)];reps=EVAL_NONRESET_CASES_PER_DISTANCE//len(combos)
    if reps*len(combos)!=EVAL_NONRESET_CASES_PER_DISTANCE:raise RuntimeError("non-reset evaluator count does not balance frozen grid")
    for distance in gate.EVAL_RETENTION_DISTANCES:
        for combo_index,(concurrent,distractors,correction) in enumerate(combos):
            for rep in range(reps):
                sample_index=1_000_000+distance*10_000+combo_index*reps+rep;cases.append(gate.generate_case(split="eval",seed=gate.HELDOUT_DATA_SEED,sample_index=sample_index,retention_distance_chunks=distance,concurrent_facts=concurrent,distractor_records=distractors,correction=correction,reset_before_query=False))
    reset_reps=EVAL_RESET_CASES_PER_LONG_DISTANCE//len(combos)
    if reset_reps*len(combos)!=EVAL_RESET_CASES_PER_LONG_DISTANCE:raise RuntimeError("reset evaluator count does not balance frozen grid")
    for distance in (8,32):
        for combo_index,(concurrent,distractors,correction) in enumerate(combos):
            for rep in range(reset_reps):
                sample_index=2_000_000+distance*10_000+combo_index*reset_reps+rep;cases.append(gate.generate_case(split="eval",seed=gate.HELDOUT_DATA_SEED,sample_index=sample_index,retention_distance_chunks=distance,concurrent_facts=concurrent,distractor_records=distractors,correction=correction,reset_before_query=True))
    return tuple(cases)
def simple_retrieval_prediction(case:gate.MemoryCase)->int:
    tokens=case.flat_tokens;prediction_position=case.answer_chunk_index*gate.CHUNK_SIZE+case.answer_token_index-1;cache={};i=0
    while i<=prediction_position:
        token=int(tokens[i])
        if token in {gate.WRITE,gate.UPDATE} and i+3<=prediction_position:cache[int(tokens[i+1])]=int(tokens[i+2]);i+=4;continue
        if token==gate.RESET:cache.clear();i+=1;continue
        if token==gate.QUERY and i+2<=prediction_position:
            key=int(tokens[i+1])
            if int(tokens[i+2])!=gate.ANSWER:raise RuntimeError("malformed query record")
            return cache.get(key,gate.UNKNOWN)
        i+=1
    raise RuntimeError("query not found before answer prediction position")
def _learning_rate(step:int)->float:
    if step<LR_WARMUP_STEPS:return ADAMW_LR*float(step+1)/float(LR_WARMUP_STEPS)
    progress=(step-LR_WARMUP_STEPS)/max(1,OPTIMIZER_STEPS-LR_WARMUP_STEPS-1);cosine=0.5*(1.0+math.cos(math.pi*min(max(progress,0.0),1.0)));return LR_MIN+(ADAMW_LR-LR_MIN)*cosine
def _answer_loss_and_routing(model:CapabilityAblationLM,batch:MaterializedBatch,*,optimizer_step:int):
    calibration=optimizer_step%ROUTER_CALIBRATION_EVERY==0;route_mode="straight_through" if model.adaptive_routing_enabled and calibration else "hard_sparse" if model.adaptive_routing_enabled else "fixed"
    output=model.forward_answer_tokens(batch.tokens,batch.prediction_positions,batch.reset_after_chunks,route_mode=route_mode,intrinsic_hard=not calibration,update_memory=model.memory_enabled);logits=output["answer_logits"]
    if not isinstance(logits,torch.Tensor):raise RuntimeError("missing answer logits")
    per_example=F.cross_entropy(logits.float(),batch.targets,reduction="none");loss=per_example.mean();terms={"answer_ce":float(loss.detach())}
    if model.adaptive_routing_enabled and calibration:
        routes=output["stage_routes"]
        if not isinstance(routes,list):raise RuntimeError("missing stage routes")
        chunk_losses=torch.zeros(batch.tokens.size(0),len(routes),device=batch.tokens.device,dtype=torch.float32);final_chunk=batch.prediction_positions//model.cfg.chunk_size;chunk_losses.scatter_(1,final_chunk[:,None],per_example.detach()[:,None]);routing=model.routing_supervision(output,chunk_losses);rank=routing["stage_difficulty_rank"];budget=routing["stage_budget"];polarization=routing["stage_polarization"];loss=loss+ROUTING_RANK_WEIGHT*rank+ROUTING_BUDGET_WEIGHT*budget+ROUTING_POLARIZATION_WEIGHT*polarization;terms.update(routing_rank=float(rank.detach()),routing_budget=float(budget.detach()),routing_polarization=float(polarization.detach()))
    terms["total"]=float(loss.detach());return loss,terms
def train_variant(variant_name:str,model_seed:int,*,device:torch.device|str,scientific_seed_authorized:bool=False,max_steps:int|None=None,checkpoint_dir:str|Path|None=None):
    if variant_name not in TRAINED_VARIANTS:raise ValueError("only A/B/C/D are trained by the frozen gate")
    _assert_model_seed_allowed(model_seed,scientific_seed_authorized=scientific_seed_authorized);device=torch.device(device);model=build_model(variant_name,model_seed,scientific_seed_authorized=scientific_seed_authorized,device=device);model.train();model.set_memory_pretraining_mode(model.memory_enabled)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=ADAMW_LR,betas=ADAMW_BETAS,eps=ADAMW_EPS,weight_decay=ADAMW_WEIGHT_DECAY);steps=OPTIMIZER_STEPS if max_steps is None else int(max_steps)
    if not 1<=steps<=OPTIMIZER_STEPS:raise ValueError("max_steps outside frozen range")
    cumulative_gpu_seconds=0.0;records=[];last_terms={};peak_memory=0;completed_steps=0;time_budget_exceeded=False
    if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)
    for step in range(steps):
        for group in optimizer.param_groups:group["lr"]=_learning_rate(step)
        optimizer.zero_grad(set_to_none=True)
        if device.type=="cuda":start_event=torch.cuda.Event(enable_timing=True);end_event=torch.cuda.Event(enable_timing=True);start_event.record();cpu_start=None
        else:start_event=end_event=None;cpu_start=time.perf_counter()
        for ai in range(GRAD_ACCUM_STEPS):
            base_index=step*GRAD_ACCUM_STEPS*MICROBATCH_SIZE+ai*MICROBATCH_SIZE;cases=tuple(training_case(base_index+row) for row in range(MICROBATCH_SIZE));batch=materialize_cases(cases,total_chunks=TRAIN_SEQUENCE_CHUNKS,device=device);loss,last_terms=_answer_loss_and_routing(model,batch,optimizer_step=step);(loss/GRAD_ACCUM_STEPS).backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],GRAD_CLIP_NORM);optimizer.step()
        if device.type=="cuda":end_event.record();end_event.synchronize();cumulative_gpu_seconds+=float(start_event.elapsed_time(end_event))/1000.0;peak_memory=max(peak_memory,int(torch.cuda.max_memory_allocated(device)))
        else:cumulative_gpu_seconds+=time.perf_counter()-cpu_start
        completed_step=step+1;completed_steps=completed_step;completed_tokens=completed_step*TOKENS_PER_OPTIMIZER_STEP
        if completed_step%CHECKPOINT_EVERY_STEPS==0:
            path=None
            if checkpoint_dir is not None:
                root=Path(checkpoint_dir);root.mkdir(parents=True,exist_ok=True);out=root/f"{variant_name}-seed{model_seed}-step{completed_step}.pt";torch.save({"research_issue":RESEARCH_ISSUE,"variant":variant_name,"model_seed":model_seed,"step":completed_step,"tokens":completed_tokens,"cumulative_gpu_seconds":cumulative_gpu_seconds,"model":model.state_dict(),"optimizer":optimizer.state_dict()},out);path=str(out)
            records.append(CheckpointRecord(completed_step,completed_tokens,cumulative_gpu_seconds,path))
        if cumulative_gpu_seconds>gate.MAX_GPU_SECONDS_PER_VARIANT_SEED and steps==OPTIMIZER_STEPS:time_budget_exceeded=True;break
    return model,{"variant":variant_name,"model_seed":model_seed,"steps":completed_steps,"tokens":completed_steps*TOKENS_PER_OPTIMIZER_STEP,"complete_frozen_token_budget":bool(completed_steps==OPTIMIZER_STEPS and not time_budget_exceeded),"time_budget_exceeded":time_budget_exceeded,"cumulative_gpu_seconds":cumulative_gpu_seconds,"checkpoint_records":[r.__dict__ for r in records],"parameter_accounting":model_accounting(model),"peak_memory_bytes":peak_memory,"last_terms":last_terms}
def select_equal_gpu_time_checkpoints(per_variant_records:dict[str,Sequence[dict[str,Any]]])->dict[str,dict[str,Any]]:
    if set(per_variant_records)!=set(TRAINED_VARIANTS):raise ValueError("equal-GPU-time selection requires A/B/C/D")
    if any(not records for records in per_variant_records.values()):raise ValueError("checkpoint records cannot be empty")
    target=min(float(records[-1]["cumulative_gpu_seconds"]) for records in per_variant_records.values());selected={}
    for variant,records in per_variant_records.items():
        eligible=[r for r in records if float(r["cumulative_gpu_seconds"])<=target]
        if not eligible:raise RuntimeError(f"no equal-time checkpoint available for {variant}")
        selected[variant]=dict(eligible[-1])
    selected["_target"]={"cumulative_gpu_seconds":target};return selected
def load_model_checkpoint(variant_name:str,model_seed:int,checkpoint_path:str|Path,*,device:torch.device|str,scientific_seed_authorized:bool=False)->CapabilityAblationLM:
    model=build_model(variant_name,model_seed,scientific_seed_authorized=scientific_seed_authorized,device=device);payload=torch.load(checkpoint_path,map_location=device,weights_only=False)
    if int(payload.get("research_issue",-1))!=RESEARCH_ISSUE:raise RuntimeError("checkpoint research issue drift")
    if payload.get("variant")!=variant_name or int(payload.get("model_seed",-1))!=model_seed:raise RuntimeError("checkpoint variant/seed drift")
    model.load_state_dict(payload["model"],strict=True);return model
def latency_cases()->tuple[gate.MemoryCase,...]:return tuple(gate.generate_case(split="eval",seed=gate.HELDOUT_DATA_SEED,sample_index=3_000_000+row,retention_distance_chunks=32,concurrent_facts=8,distractor_records=32,correction=True,reset_before_query=False) for row in range(LATENCY_BATCH_SIZE))
def measure_model_inference_latency_ms(model:CapabilityAblationLM,*,device:torch.device|str)->float:
    device=torch.device(device);model.eval();model.set_memory_pretraining_mode(False);batch=materialize_cases(latency_cases(),device=device)
    def call():
        out=model.forward_answer_tokens(batch.tokens,batch.prediction_positions,batch.reset_after_chunks,route_mode="hard_sparse" if model.adaptive_routing_enabled else "fixed",intrinsic_hard=True,update_memory=model.memory_enabled);del out
    values=[]
    with torch.inference_mode():
        for _ in range(LATENCY_WARMUP_CALLS):call()
        if device.type=="cuda":torch.cuda.synchronize(device)
        for _ in range(LATENCY_TIMED_CALLS):
            if device.type=="cuda":start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record();call();end.record();end.synchronize();values.append(float(start.elapsed_time(end)))
            else:begin=time.perf_counter();call();values.append((time.perf_counter()-begin)*1000.0)
    values.sort();return float(values[len(values)//2])
def measure_simple_retrieval_latency_ms()->float:
    cases=latency_cases();values=[]
    for _ in range(LATENCY_WARMUP_CALLS):[simple_retrieval_prediction(c) for c in cases]
    for _ in range(LATENCY_TIMED_CALLS):begin=time.perf_counter();[simple_retrieval_prediction(c) for c in cases];values.append((time.perf_counter()-begin)*1000.0)
    values.sort();return float(values[len(values)//2])
def _batch_iter(items:Sequence[Any],size:int)->Iterable[Sequence[Any]]:
    for start in range(0,len(items),size):yield items[start:start+size]
def evaluate_model(model:CapabilityAblationLM,cases:Sequence[gate.MemoryCase],*,device:torch.device|str,batch_size:int=EVAL_BATCH_SIZE)->dict[str,Any]:
    device=torch.device(device);model.eval();model.set_memory_pretraining_mode(False);predictions=[];gold_lp=[];ordered=[]
    with torch.inference_mode():
        for group in _batch_iter(list(cases),batch_size):
            batch=materialize_cases(group,device=device);output=model.forward_answer_tokens(batch.tokens,batch.prediction_positions,batch.reset_after_chunks,route_mode="hard_sparse" if model.adaptive_routing_enabled else "fixed",intrinsic_hard=True,update_memory=model.memory_enabled);logits=output["answer_logits"];log_probs=F.log_softmax(logits.float(),dim=-1);pred=logits.argmax(dim=-1);rows=torch.arange(batch.targets.size(0),device=device);gold=log_probs[rows,batch.targets];predictions.extend(int(x) for x in pred.detach().cpu().tolist());gold_lp.extend(float(x) for x in gold.detach().cpu().tolist());ordered.extend(group)
    metrics=_aggregate_metrics(tuple(ordered),predictions,gold_lp);metrics["parameter_accounting"]=model_accounting(model);metrics["stage_execution_last_batch"]=list(model.last_stage_execution);return metrics
def evaluate_simple_retrieval(cases:Sequence[gate.MemoryCase])->dict[str,Any]:
    start=time.perf_counter();predictions=[simple_retrieval_prediction(c) for c in cases];metrics=_aggregate_metrics(tuple(cases),predictions,None);metrics["reference_wall_seconds"]=time.perf_counter()-start;return metrics
def _aggregate_metrics(cases:tuple[gate.MemoryCase,...],predictions:Sequence[int],gold_log_probabilities:Sequence[float]|None)->dict[str,Any]:
    if len(cases)!=len(predictions):raise ValueError("case/prediction length mismatch")
    correct=[int(p==c.expected_answer) for p,c in zip(predictions,cases)];nonreset=[i for i,c in enumerate(cases) if not c.reset_before_query];reset=[i for i,c in enumerate(cases) if c.reset_before_query];by_distance={}
    for d in gate.EVAL_RETENTION_DISTANCES:
        idx=[i for i in nonreset if cases[i].retention_distance_chunks==d];by_distance[str(d)]=sum(correct[i] for i in idx)/len(idx) if idx else float("nan")
    long_accuracy=.5*(by_distance["8"]+by_distance["32"]);correction_idx=[i for i in nonreset if cases[i].correction];correction_accuracy=sum(correct[i] for i in correction_idx)/len(correction_idx);stale_error=gate.stale_value_error_rate([predictions[i] for i in correction_idx],[cases[i].stale_value for i in correction_idx]);distractor_idx=[i for i in nonreset if cases[i].distractor_records>0];interference_error=1.0-sum(correct[i] for i in distractor_idx)/len(distractor_idx);reset_leak=gate.session_reset_leakage_rate([predictions[i] for i in reset],[cases[i].latest_value for i in reset]) if reset else 0.0;concurrency={};distractor_curve={}
    for v in gate.EVAL_CONCURRENT_FACTS:
        idx=[i for i in nonreset if cases[i].concurrent_facts==v];concurrency[str(v)]=sum(correct[i] for i in idx)/len(idx)
    for v in gate.EVAL_DISTRACTOR_RECORDS:
        idx=[i for i in nonreset if cases[i].distractor_records==v];distractor_curve[str(v)]=sum(correct[i] for i in idx)/len(idx)
    result={"examples":len(cases),"predictions":list(predictions),"correct":correct,"case_ids":[c.sample_index for c in cases],"long_distance_accuracy":long_accuracy,"retention_accuracy":by_distance,"correction_accuracy":correction_accuracy,"stale_value_error_rate":stale_error,"distractor_interference_error_rate":interference_error,"session_reset_leakage_rate":reset_leak,"concurrent_fact_curve":concurrency,"distractor_curve":distractor_curve}
    if gold_log_probabilities is not None:result["answer_token_nll"]=gate.mean_answer_nll(gold_log_probabilities);result["gold_log_probabilities"]=list(gold_log_probabilities)
    else:result["answer_token_nll"]=None
    return result
def paired_primary_delta(a:dict[str,Any],b:dict[str,Any])->dict[str,float]:
    if a["case_ids"]!=b["case_ids"]:raise ValueError("paired primary comparison requires identical case order")
    metadata={c.sample_index:c for c in evaluation_cases()};keep=[i for i,sid in enumerate(a["case_ids"]) if not metadata[int(sid)].reset_before_query and metadata[int(sid)].retention_distance_chunks in {8,32}];delta,lo,hi=gate.paired_bootstrap_accuracy_delta([int(a["correct"][i]) for i in keep],[int(b["correct"][i]) for i in keep]);return {"delta":delta,"ci95_lower":lo,"ci95_upper":hi}
REQUIRED_EVALUATION_METRIC_KEYS=("examples","predictions","correct","case_ids","long_distance_accuracy","retention_accuracy","correction_accuracy","stale_value_error_rate","distractor_interference_error_rate","session_reset_leakage_rate","concurrent_fact_curve","distractor_curve","answer_token_nll")
def validate_evaluation_metrics(metrics:dict[str,Any])->None:
    missing=[k for k in REQUIRED_EVALUATION_METRIC_KEYS if k not in metrics]
    if missing:raise ValueError(f"evaluation metrics missing frozen keys: {missing}")
    if set(metrics["retention_accuracy"])!={"2","8","32"}:raise ValueError("retention curve schema drift")
    if set(metrics["concurrent_fact_curve"])!={"1","4","8"}:raise ValueError("concurrent-fact curve schema drift")
    if set(metrics["distractor_curve"])!={"0","16","32"}:raise ValueError("distractor curve schema drift")
def validate_seed1_result_schema(result:dict[str,Any])->None:
    required={"research_issue","source_main","source_tree","model_seed","equal_token","equal_gpu_time","training","inference_latency_ms","simple_retrieval","equal_gpu_time_checkpoint_selection","decision","authority"};missing=sorted(required-set(result))
    if missing:raise ValueError(f"seed1 result missing frozen keys: {missing}")
    if int(result["research_issue"])!=RESEARCH_ISSUE:raise ValueError("seed1 result research issue drift")
    if int(result["model_seed"])!=SEED1_MODEL_SEED:raise ValueError("seed1 result seed drift")
    for surface_name in ("equal_token","equal_gpu_time"):
        surface=result[surface_name]
        if set(surface)!=set(TRAINED_VARIANTS):raise ValueError(f"{surface_name} variant schema drift")
        for metrics in surface.values():validate_evaluation_metrics(metrics)
    validate_evaluation_metrics(result["simple_retrieval"])
    if set(result["training"])!=set(TRAINED_VARIANTS):raise ValueError("training variant schema drift")
    if set(result["inference_latency_ms"])!=set(TRAINED_VARIANTS)|{"E_simple_retrieval"}:raise ValueError("latency schema drift")
    for key in ("seeds_2_3_authorized","systems_optimization_authorized","architecture_freeze_authorized","scaling_authorized","breakthrough_proven"):
        if result["authority"].get(key) is not False:raise ValueError(f"seed1 result authority drift: {key}")
def seed1_decision(*,equal_token:dict[str,dict[str,Any]],equal_gpu_time:dict[str,dict[str,Any]],training:dict[str,dict[str,Any]],inference_latency_ms:dict[str,float],simple_retrieval:dict[str,Any])->dict[str,Any]:
    for surface in (equal_token,equal_gpu_time):
        if set(surface)!=set(TRAINED_VARIANTS):raise ValueError("decision surfaces require A/B/C/D")
    checks={}
    for name,surface in (("equal_token",equal_token),("equal_gpu_time",equal_gpu_time)):
        a,b,c=surface["A_backbone"],surface["B_backbone_plus_memory"],surface["C_backbone_plus_routing"];primary=paired_primary_delta(a,b);ba=float(b["long_distance_accuracy"])-float(a["long_distance_accuracy"]);correction=float(b["correction_accuracy"])-float(a["correction_accuracy"]);stale_reduction=float(a["stale_value_error_rate"])-float(b["stale_value_error_rate"]);bc=float(b["long_distance_accuracy"])-float(c["long_distance_accuracy"]);reset_ok=float(b["session_reset_leakage_rate"])<=gate.SESSION_RESET_LEAKAGE_MAX
        checks[name]={"B_minus_A_long_accuracy":ba,"paired_primary":primary,"B_minus_A_correction_accuracy":correction,"A_minus_B_stale_error_reduction":stale_reduction,"B_minus_C_long_accuracy":bc,"reset_ok":reset_ok,"pass":bool(ba>=gate.LONG_DISTANCE_ACCURACY_GAIN_MIN and primary["ci95_lower"]>gate.PAIRED_CI_LOWER_ACCURACY_GAIN_MIN and correction>=gate.CORRECTION_ACCURACY_GAIN_MIN and stale_reduction>=gate.STALE_VALUE_ERROR_REDUCTION_MIN and bc>=gate.ROUTING_CONTROL_ACCURACY_MARGIN_MIN and reset_ok)}
    full_b=equal_token["B_backbone_plus_memory"];retrieval_dominates=bool(float(simple_retrieval["long_distance_accuracy"])-float(full_b["long_distance_accuracy"])>=gate.SIMPLE_RETRIEVAL_DOMINANCE_MARGIN and float(simple_retrieval["correction_accuracy"])-float(full_b["correction_accuracy"])>=gate.SIMPLE_RETRIEVAL_DOMINANCE_MARGIN and float(inference_latency_ms["E_simple_retrieval"])<=float(inference_latency_ms["B_backbone_plus_memory"]));b_train=float(training["B_backbone_plus_memory"]["cumulative_gpu_seconds"]);a_train=float(training["A_backbone"]["cumulative_gpu_seconds"]);train_multiplier=b_train/a_train;latency_multiplier=float(inference_latency_ms["B_backbone_plus_memory"])/float(inference_latency_ms["A_backbone"]);capability_pass=checks["equal_token"]["pass"] and checks["equal_gpu_time"]["pass"];cost_ok=bool(train_multiplier<=gate.MEMORY_TRAIN_TIME_MULTIPLIER_MAX and latency_multiplier<=gate.MEMORY_INFERENCE_LATENCY_MULTIPLIER_MAX)
    recommendation="STOP_MEMORY_CAPABILITY_SCREEN_FAIL" if not capability_pass else "STOP_LEARNED_MEMORY_SIMPLER_RETRIEVAL_DOMINATES" if retrieval_dominates else "CAPABILITY_POSITIVE_BUT_COSTLY_DO_NOT_PRESERVE_FOR_SYSTEMS" if not cost_ok else "CONFIRM_SEEDS_2_3_RECOMMENDED_REQUIRES_SEPARATE_AUTHORIZATION"
    return {"research_issue":RESEARCH_ISSUE,"checks":checks,"retrieval_dominates":retrieval_dominates,"memory_train_time_multiplier_B_over_A":train_multiplier,"memory_inference_latency_multiplier_B_over_A":latency_multiplier,"cost_ok":cost_ok,"capability_screen_pass":capability_pass,"recommendation":recommendation,"seeds_2_3_authorized":False,"systems_optimization_authorized":False,"architecture_freeze_authorized":False,"scaling_authorized":False,"breakthrough_proven":False}
def harness_protocol_snapshot()->dict[str,Any]:
    return {"research_issue":RESEARCH_ISSUE,"parent_research_issue":PARENT_RESEARCH_ISSUE,"source_main":SOURCE_MAIN,"source_tree":SOURCE_TREE,"parent_blobs":{"fixture":PARENT_FIXTURE_BLOB,"protocol":PARENT_PROTOCOL_BLOB,"cpu_test":PARENT_CPU_TEST_BLOB},"branch":BRANCH,"future_execution":{"result_path":SEED1_RESULT_PATH,"checkpoint_dir":SEED1_CHECKPOINT_DIR,"preauth_prefix":SEED1_PREAUTH_PREFIX,"gpu_prefix":SEED1_GPU_PREFIX,"seed1_model_seed":SEED1_MODEL_SEED},"model_config":dict(MODEL_CONFIG),"variants":{"trained":list(TRAINED_VARIANTS),"reference":list(REFERENCE_VARIANTS),"common_stored_modules":True,"latent_reasoning_executed":False,"A_B_C_D_recurrent_carry_executed":False,"B_D_v18_pretrainable_delta_memory":True,"C_D_v17_optional_stage_supervision":True,"intrinsic_expert_schedule_identical_across_A_B_C_D":True,"F_scientific_training_in_seed1":False},"training":{"sequence_chunks":TRAIN_SEQUENCE_CHUNKS,"chunk_size":gate.CHUNK_SIZE,"microbatch_size":MICROBATCH_SIZE,"grad_accum_steps":GRAD_ACCUM_STEPS,"tokens_per_optimizer_step":TOKENS_PER_OPTIMIZER_STEP,"optimizer_steps":OPTIMIZER_STEPS,"tokens_per_variant_seed":gate.TOKEN_BUDGET_PER_TRAINED_VARIANT,"checkpoint_every_steps":CHECKPOINT_EVERY_STEPS,"checkpoint_interval_tokens":gate.CHECKPOINT_INTERVAL_TOKENS,"router_calibration_every":ROUTER_CALIBRATION_EVERY,"optimizer":"AdamW","lr":ADAMW_LR,"betas":list(ADAMW_BETAS),"eps":ADAMW_EPS,"weight_decay":ADAMW_WEIGHT_DECAY,"grad_clip_norm":GRAD_CLIP_NORM,"lr_warmup_steps":LR_WARMUP_STEPS,"lr_min":LR_MIN,"schedule":"linear-warmup-cosine-decay","routing_rank_weight":ROUTING_RANK_WEIGHT,"routing_budget_weight":ROUTING_BUDGET_WEIGHT,"routing_polarization_weight":ROUTING_POLARIZATION_WEIGHT,"max_gpu_seconds_per_variant_seed":gate.MAX_GPU_SECONDS_PER_VARIANT_SEED},"evaluation":{"nonreset_cases_per_distance":EVAL_NONRESET_CASES_PER_DISTANCE,"reset_cases_per_long_distance":EVAL_RESET_CASES_PER_LONG_DISTANCE,"batch_size":EVAL_BATCH_SIZE,"latency_batch_size":LATENCY_BATCH_SIZE,"latency_warmup_calls":LATENCY_WARMUP_CALLS,"latency_timed_calls":LATENCY_TIMED_CALLS,"primary_distances":[8,32],"bootstrap_resamples":gate.BOOTSTRAP_RESAMPLES},"decision":{"must_pass_equal_token_and_equal_gpu_time":True,"long_distance_B_minus_A_min":gate.LONG_DISTANCE_ACCURACY_GAIN_MIN,"paired_ci95_lower_min_strict":gate.PAIRED_CI_LOWER_ACCURACY_GAIN_MIN,"correction_B_minus_A_min":gate.CORRECTION_ACCURACY_GAIN_MIN,"stale_error_A_minus_B_min":gate.STALE_VALUE_ERROR_REDUCTION_MIN,"B_minus_C_long_min":gate.ROUTING_CONTROL_ACCURACY_MARGIN_MIN,"session_reset_leakage_max":gate.SESSION_RESET_LEAKAGE_MAX,"simple_retrieval_dominance_margin":gate.SIMPLE_RETRIEVAL_DOMINANCE_MARGIN,"memory_inference_latency_multiplier_max":gate.MEMORY_INFERENCE_LATENCY_MULTIPLIER_MAX,"memory_train_time_multiplier_max":gate.MEMORY_TRAIN_TIME_MULTIPLIER_MAX},"authority":{"gpu_authorized":GPU_AUTHORIZED,"scientific_training_authorized":SCIENTIFIC_TRAINING_AUTHORIZED,"fresh_scientific_seed_authorized":FRESH_SCIENTIFIC_SEED_AUTHORIZED,"seeds_2_3_authorized":SEEDS_2_3_AUTHORIZED,"systems_optimization_authorized":SYSTEMS_OPTIMIZATION_AUTHORIZED,"architecture_freeze_authorized":ARCHITECTURE_FREEZE_AUTHORIZED,"scaling_authorized":SCALING_AUTHORIZED,"breakthrough_proven":BREAKTHROUGH_PROVEN}}
