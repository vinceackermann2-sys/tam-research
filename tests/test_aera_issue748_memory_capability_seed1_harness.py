from __future__ import annotations
import json
from pathlib import Path
import pytest
import torch
from tam_research import aera_issue748_memory_capability_seed1_harness as h
from tam_research import aera_memory_capability_gate_v1 as gate

def test_issue748_budget_geometry_and_authority():
    p=h.harness_protocol_snapshot()
    assert h.MICROBATCH_SIZE==2 and h.GRAD_ACCUM_STEPS==2
    assert h.TOKENS_PER_OPTIMIZER_STEP==16384 and h.OPTIMIZER_STEPS==1024
    assert h.CHECKPOINT_EVERY_STEPS==128
    assert h.OPTIMIZER_STEPS*h.TOKENS_PER_OPTIMIZER_STEP==gate.TOKEN_BUDGET_PER_TRAINED_VARIANT
    assert p["decision"]["must_pass_equal_token_and_equal_gpu_time"] is True
    assert all(v is False for v in p["authority"].values())

def test_issue748_scientific_seed_guard_is_closed():
    with pytest.raises(PermissionError):h.build_model("A_backbone",h.SEED1_MODEL_SEED)

def test_issue748_scaffold_parameter_capacity_is_identical():
    totals={};trainables={}
    for variant in h.TRAINED_VARIANTS:
        model=h.build_model(variant,h.CPU_TEST_SEED);a=h.model_accounting(model);totals[variant]=a["total_parameters"];trainables[variant]=a["trainable_parameters"]
        assert model.lm_head.weight is model.token_emb.weight
    assert len(set(totals.values()))==1
    assert trainables["B_backbone_plus_memory"]>trainables["A_backbone"]
    assert trainables["C_backbone_plus_routing"]>trainables["A_backbone"]
    assert trainables["D_combined"]>=max(trainables["B_backbone_plus_memory"],trainables["C_backbone_plus_routing"])

def test_issue748_training_materialization_is_deterministic_and_query_is_final_chunk():
    cases=tuple(h.training_case(i) for i in range(4));a=h.materialize_cases(cases,total_chunks=h.TRAIN_SEQUENCE_CHUNKS);b=h.materialize_cases(cases,total_chunks=h.TRAIN_SEQUENCE_CHUNKS)
    assert torch.equal(a.tokens,b.tokens) and torch.equal(a.targets,b.targets)
    assert a.tokens.shape==(4,4096)
    assert torch.all(a.prediction_positions//gate.CHUNK_SIZE==15)

def test_issue748_reset_mask_is_applied_before_query_chunk():
    case=gate.generate_case(split="train",seed=gate.TRAIN_DATA_SEED,sample_index=98765,retention_distance_chunks=2,concurrent_facts=1,distractor_records=0,correction=False,reset_before_query=True)
    batch=h.materialize_cases((case,),total_chunks=4)
    assert int(batch.reset_after_chunks.sum())==1
    assert int(batch.reset_after_chunks[0].nonzero(as_tuple=False)[0])==2
    assert int(batch.prediction_positions[0]//gate.CHUNK_SIZE)==3

def test_issue748_simple_retrieval_obeys_update_and_reset():
    corrected=gate.generate_case(split="eval",seed=gate.HELDOUT_DATA_SEED,sample_index=81001,retention_distance_chunks=8,concurrent_facts=4,distractor_records=16,correction=True,reset_before_query=False)
    reset=gate.generate_case(split="eval",seed=gate.HELDOUT_DATA_SEED,sample_index=81002,retention_distance_chunks=8,concurrent_facts=4,distractor_records=16,correction=True,reset_before_query=True)
    assert h.simple_retrieval_prediction(corrected)==corrected.latest_value
    assert h.simple_retrieval_prediction(reset)==gate.UNKNOWN

def _forbid(*args,**kwargs):raise AssertionError("latent reasoner/recurrent carry executed")

def test_issue748_B_memory_is_differentiable_without_reasoner_or_recurrent_carry():
    model=h.build_model("B_backbone_plus_memory",h.CPU_TEST_SEED);model.train();model.set_memory_pretraining_mode(True)
    for stage in model.stages:stage.reasoner.forward=_forbid;stage.stream_cell.forward=_forbid
    cases=tuple(gate.generate_case(split="train",seed=gate.TRAIN_DATA_SEED,sample_index=41000+i,retention_distance_chunks=1,concurrent_facts=1,distractor_records=0,correction=False,reset_before_query=False) for i in range(2))
    batch=h.materialize_cases(cases,total_chunks=2);loss,terms=h._answer_loss_and_routing(model,batch,optimizer_step=1);assert torch.isfinite(loss) and terms["answer_ce"]>0;loss.backward()
    grads=[p.grad for stage in model.stages for p in stage.memory.parameters() if p.requires_grad]
    assert grads and any(g is not None and bool(torch.isfinite(g).all()) and float(g.abs().sum())>0 for g in grads)

def test_issue748_A_memory_is_registered_but_frozen():
    model=h.build_model("A_backbone",h.CPU_TEST_SEED);params=[p for stage in model.stages for p in stage.memory.parameters()]
    assert params and all(not p.requires_grad for p in params)

def test_issue748_C_calibration_produces_optional_router_gradients():
    model=h.build_model("C_backbone_plus_routing",h.CPU_TEST_SEED);model.train()
    cases=tuple(gate.generate_case(split="train",seed=gate.TRAIN_DATA_SEED,sample_index=42000+i,retention_distance_chunks=2,concurrent_facts=1,distractor_records=i*4,correction=bool(i),reset_before_query=False) for i in range(2))
    batch=h.materialize_cases(cases,total_chunks=3);loss,terms=h._answer_loss_and_routing(model,batch,optimizer_step=0);assert "routing_rank" in terms;loss.backward()
    grads=[p.grad for si,r in enumerate(model.stage_routers) if si!=model.FOUNDATION_STAGE for p in r.parameters() if p.requires_grad]
    assert grads and any(g is not None and bool(torch.isfinite(g).all()) for g in grads)
    assert all(not p.requires_grad for p in model.stage_routers[model.FOUNDATION_STAGE].parameters())

def test_issue748_eval_grid_is_frozen_balanced_and_heldout():
    cases=h.evaluation_cases();non=[c for c in cases if not c.reset_before_query];reset=[c for c in cases if c.reset_before_query]
    assert len(non)==324 and len(reset)==108 and all(c.split=="eval" for c in cases)
    for d in gate.EVAL_RETENTION_DISTANCES:assert sum(c.retention_distance_chunks==d for c in non)==108

def _surface(correct,long,correction,stale,reset):
    cases=h.evaluation_cases();return {"case_ids":[c.sample_index for c in cases],"correct":[correct for _ in cases],"long_distance_accuracy":long,"correction_accuracy":correction,"stale_value_error_rate":stale,"session_reset_leakage_rate":reset}

def test_issue748_decision_requires_equal_token_and_equal_gpu_time_pass():
    a=_surface(0,.50,.50,.50,0);b=_surface(1,.75,.75,.20,0);c=_surface(0,.60,.60,.40,0);d=_surface(1,.80,.80,.15,0);good={"A_backbone":a,"B_backbone_plus_memory":b,"C_backbone_plus_routing":c,"D_combined":d}
    training={"A_backbone":{"cumulative_gpu_seconds":100.},"B_backbone_plus_memory":{"cumulative_gpu_seconds":110.},"C_backbone_plus_routing":{"cumulative_gpu_seconds":105.},"D_combined":{"cumulative_gpu_seconds":115.}}
    latency={"A_backbone":10.,"B_backbone_plus_memory":15.,"C_backbone_plus_routing":11.,"D_combined":16.,"E_simple_retrieval":1.};retrieval={"long_distance_accuracy":.79,"correction_accuracy":.79}
    decision=h.seed1_decision(equal_token=good,equal_gpu_time=good,training=training,inference_latency_ms=latency,simple_retrieval=retrieval);assert decision["capability_screen_pass"] is True and decision["recommendation"].startswith("CONFIRM_SEEDS_2_3")
    bad=dict(good);bad["B_backbone_plus_memory"]=_surface(0,.52,.52,.49,0);failed=h.seed1_decision(equal_token=good,equal_gpu_time=bad,training=training,inference_latency_ms=latency,simple_retrieval=retrieval);assert failed["capability_screen_pass"] is False and failed["recommendation"]=="STOP_MEMORY_CAPABILITY_SCREEN_FAIL"

def test_issue748_protocol_json_matches_python_contract():
    root=Path(__file__).resolve().parents[1];expected=json.loads((root/"docs"/"aera_issue748_seed1_harness_protocol.json").read_text());assert expected==h.harness_protocol_snapshot()
