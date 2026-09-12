from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any

import modal

PHASE = "chm-v1-eiem-seed-8611-v1"
TRIGGER_TITLE = "[modal-chm-v1-eiem-seed-8611-v1]"
RESEARCH_ISSUE = 854
RUN_CONTROL_ISSUE = 903
AUTHORIZATION_COMMENT_ID = 5_644_581_046
AUTHORIZATION_REF = "issue-854-comment-5644581046"
SCIENTIFIC_AUTHORITY_SHA = "e86ffd453be3efc3026576f5ebbab6b76ab54b95"
SCIENTIFIC_AUTHORITY_TREE = "9af3aa0607f16c953cfe5b802b9daca1e79ce853"
SCIENTIFIC_SEED = 8611
RESULT_ROOT = "/vol/chm-v1/eiem-small-lm/issue-903/seed-8611-v1"
DATA_DIR = "/vol/data/tam100m-2b-curated-v1"

GPU_CLASS = "L4"
CPU_CORES = 4
RAM_MIB = 8192
MAX_SECONDS_PER_SEED = 3600
MAX_AGGREGATE_BILLED_COMPUTE_USD = 4.0
MAX_SEEDS = 3

# Public Modal rates re-checked on 2026-09-12 before this launcher was authored.
L4_USD_PER_SECOND = 0.000222
PHYSICAL_CPU_USD_PER_CORE_SECOND = 0.0000131
RAM_USD_PER_GIB_SECOND = 0.00000222

APP_NAME = "chm-v1-eiem-small-lm-seed-8611-v1"
VOLUME_NAME = "tam-research-data"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11", "numpy>=2,<3", "tiktoken>=0.9,<1")
    .add_local_python_source("tam_research")
    .add_local_python_source("architectures")
)


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _full_sha(value: str, name: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 40 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{name} must be a full lowercase 40-hex SHA")
    return normalized


def _validate_source(source_sha: str, source_tree: str, harness_sha: str) -> tuple[str, str, str]:
    source = _full_sha(source_sha, "source_sha")
    tree = _full_sha(source_tree, "source_tree")
    harness = _full_sha(harness_sha, "harness_sha")
    if SCIENTIFIC_SEED != 8611:
        raise RuntimeError("this one-shot launcher is bound only to scientific seed 8611")
    if GPU_CLASS != "L4" or CPU_CORES != 4 or RAM_MIB != 8192 or MAX_SECONDS_PER_SEED != 3600:
        raise RuntimeError("frozen #854 resource envelope drift")
    if _worst_case_three_seed_cost_usd() > MAX_AGGREGATE_BILLED_COMPUTE_USD:
        raise RuntimeError("frozen three-seed resource envelope can exceed $4 at bound rates")
    return source, tree, harness


def _worst_case_one_seed_cost_usd() -> float:
    per_second = (
        L4_USD_PER_SECOND
        + CPU_CORES * PHYSICAL_CPU_USD_PER_CORE_SECOND
        + (RAM_MIB / 1024.0) * RAM_USD_PER_GIB_SECOND
    )
    return MAX_SECONDS_PER_SEED * per_second


def _worst_case_three_seed_cost_usd() -> float:
    return MAX_SEEDS * _worst_case_one_seed_cost_usd()


def _finite_model(model: Any) -> bool:
    import torch
    return all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())


@app.function(
    image=image,
    cpu=CPU_CORES,
    memory=RAM_MIB,
    timeout=15 * 60,
    retries=0,
    volumes={"/vol": volume},
)
def verify_zero_gpu(source_sha: str, source_tree: str, harness_sha: str) -> str:
    import tiktoken

    from tam_research.chm_v1_corpus_fingerprint import fingerprint_frozen_corpus
    from tam_research.chm_v1_run_manifest import frozen_run_manifest, validate_run_manifest
    from tam_research.chm_v1_scientific_gate import build_scientific_probe_suite
    from tam_research.chm_v1_small_lm_protocol import protocol_preflight

    source, tree, harness = _validate_source(source_sha, source_tree, harness_sha)
    volume.reload()
    root = Path(RESULT_ROOT)
    if root.exists():
        raise RuntimeError("reserved seed-8611 result namespace already exists; fail closed")

    protocol = protocol_preflight(DATA_DIR)
    fingerprint = fingerprint_frozen_corpus(DATA_DIR)
    manifest = validate_run_manifest(frozen_run_manifest(authorization_ref=AUTHORIZATION_REF))
    encoder = tiktoken.get_encoding("gpt2")
    probes = build_scientific_probe_suite(lambda text: encoder.encode(text))
    if len(probes) != 96:
        raise RuntimeError(f"scientific probe count drift: {len(probes)}")

    payload = {
        "status": "PASS",
        "classification": "CHM_V1_ZERO_GPU_PREFLIGHT_SEED_8611_V1",
        "phase": PHASE,
        "trigger_title": TRIGGER_TITLE,
        "research_issue": RESEARCH_ISSUE,
        "run_control_issue": RUN_CONTROL_ISSUE,
        "authorization_comment_id": AUTHORIZATION_COMMENT_ID,
        "authorization_ref": AUTHORIZATION_REF,
        "scientific_authority_sha": SCIENTIFIC_AUTHORITY_SHA,
        "scientific_authority_tree": SCIENTIFIC_AUTHORITY_TREE,
        "execution_code_sha": source,
        "execution_tree_sha": tree,
        "harness_blob_sha": harness,
        "seed": SCIENTIFIC_SEED,
        "result_root": RESULT_ROOT,
        "data_dir": DATA_DIR,
        "corpus_fingerprint": fingerprint,
        "run_manifest": manifest,
        "protocol": protocol,
        "probe_count": len(probes),
        "resource_envelope": {
            "gpu_class": "NVIDIA L4",
            "cpu_cores": CPU_CORES,
            "ram_gib": RAM_MIB / 1024.0,
            "max_seconds_per_seed": MAX_SECONDS_PER_SEED,
            "no_retries": True,
            "sequential_only": True,
            "max_aggregate_billed_compute_usd": MAX_AGGREGATE_BILLED_COMPUTE_USD,
        },
        "bound_public_rates_usd_per_second": {
            "l4": L4_USD_PER_SECOND,
            "physical_cpu_core": PHYSICAL_CPU_USD_PER_CORE_SECOND,
            "ram_gib": RAM_USD_PER_GIB_SECOND,
        },
        "worst_case_one_seed_usd": _worst_case_one_seed_cost_usd(),
        "worst_case_three_seed_usd": _worst_case_three_seed_cost_usd(),
        "gpu_allocated": False,
        "scientific_seed_consumed": False,
    }
    _atomic_write(root / "ZERO_GPU_GATE.json", payload)
    volume.commit()
    return json.dumps(payload, sort_keys=True)


@app.function(
    image=image,
    cpu=1,
    memory=512,
    timeout=5 * 60,
    retries=0,
    volumes={"/vol": volume},
)
def reserve_dispatch(source_sha: str, source_tree: str, harness_sha: str) -> str:
    source, tree, harness = _validate_source(source_sha, source_tree, harness_sha)
    volume.reload()
    root = Path(RESULT_ROOT)
    zero_path = root / "ZERO_GPU_GATE.json"
    marker_path = root / "DISPATCH_RESERVED.json"
    consumed_path = root / "SEED_CONSUMED.json"
    result_path = root / "RESULT.json"
    if not zero_path.is_file():
        raise RuntimeError("zero-GPU gate is missing")
    zero = json.loads(zero_path.read_text(encoding="utf-8"))
    expected = {
        "status": "PASS",
        "execution_code_sha": source,
        "execution_tree_sha": tree,
        "harness_blob_sha": harness,
        "seed": SCIENTIFIC_SEED,
        "scientific_seed_consumed": False,
    }
    for key, value in expected.items():
        if zero.get(key) != value:
            raise RuntimeError(f"zero-GPU gate mismatch for {key}")
    if marker_path.exists() or consumed_path.exists() or result_path.exists():
        raise RuntimeError("seed-8611 trigger/result namespace was already used")

    marker = {
        "status": "DISPATCH_RESERVED",
        "classification": "CHM_V1_DURABLE_PRE_ALLOCATION_MARKER",
        "phase": PHASE,
        "trigger_title": TRIGGER_TITLE,
        "execution_code_sha": source,
        "execution_tree_sha": tree,
        "harness_blob_sha": harness,
        "seed": SCIENTIFIC_SEED,
        "result_root": RESULT_ROOT,
        "marked_unix": time.time(),
        "gpu_allocation_started": False,
        "scientific_seed_consumed": False,
        "automatic_retry_authorized": False,
    }
    _atomic_write(marker_path, marker)
    volume.commit()
    return json.dumps(marker, sort_keys=True)


def _merge_stats(total: dict[str, float], stats: Any, *, memory_size: int) -> None:
    total["calls"] += float(stats.calls)
    total["reads"] += float(stats.address_vector_reads)
    total["flat_reads"] += float(stats.flat_address_vector_reads)
    total["nodes"] += float(stats.directory_nodes_visited)
    total["exact_matches"] += float(stats.exact_matches)
    total["index_build_seconds"] += float(stats.index_build_seconds)
    total["search_seconds"] += float(stats.search_seconds)
    total["verification_seconds"] += float(stats.verification_seconds)
    total["write_seconds"] += float(stats.write_seconds)
    total["state_bytes"] = max(total["state_bytes"], float(stats.state_payload_bytes))
    if memory_size == 1024:
        total["reads_1024"] += float(stats.address_vector_reads)
        total["flat_reads_1024"] += float(stats.flat_address_vector_reads)


def _evaluate_probes(local: Any, eiem: Any, device: Any) -> dict[str, Any]:
    import torch
    import tiktoken

    from tam_research.chm_v1_batched_eval import forward_session_chunk_batched_transport
    from tam_research.chm_v1_long_memory_eval_v3 import GENERATOR_VERSION
    from tam_research.chm_v1_scientific_gate import build_scientific_probe_suite
    from tam_research.chm_v1_small_lm import EpisodicState, LOCAL_WINDOW

    encoder = tiktoken.get_encoding("gpt2")
    probes = build_scientific_probe_suite(lambda text: encoder.encode(text))

    families: dict[str, dict[str, float]] = {}
    for family in ("rare_fact", "overwrite", "two_hop", "local_negative"):
        families[family] = {
            "count": 0.0,
            "local_correct": 0.0,
            "eiem_correct": 0.0,
            "local_stale": 0.0,
            "eiem_stale": 0.0,
        }

    indexed_totals = {
        "calls": 0.0,
        "reads": 0.0,
        "flat_reads": 0.0,
        "nodes": 0.0,
        "exact_matches": 0.0,
        "index_build_seconds": 0.0,
        "search_seconds": 0.0,
        "verification_seconds": 0.0,
        "write_seconds": 0.0,
        "state_bytes": 0.0,
        "reads_1024": 0.0,
        "flat_reads_1024": 0.0,
    }
    flat_search_seconds = 0.0
    flat_build_seconds = 0.0
    flat_write_seconds = 0.0

    local.eval()
    eiem.eval()
    with torch.no_grad():
        for probe in probes:
            ids = list(probe.prompt_ids)
            candidates = list(probe.candidate_token_ids)
            local_ctx = ids[-LOCAL_WINDOW:]
            local_tokens = torch.tensor([local_ctx], dtype=torch.long, device=device)
            local_logits = local(local_tokens)[0, -1, candidates]
            local_pred = candidates[int(local_logits.argmax().item())]

            predictions: dict[str, int] = {}
            for mode in ("flat", "indexed"):
                state = EpisodicState(f"{mode}-{probe.family}-{probe.case_id}")
                last_logits = None
                for start in range(0, len(ids), LOCAL_WINDOW):
                    chunk_ids = ids[start : start + LOCAL_WINDOW]
                    before = len(state)
                    tokens = torch.tensor([chunk_ids], dtype=torch.long, device=device)
                    logits, stats = forward_session_chunk_batched_transport(
                        eiem,
                        tokens,
                        [state],
                        mode=mode,
                        update_memory=True,
                        verify_indexed_exactness=(mode == "indexed"),
                    )
                    last_logits = logits
                    if mode == "indexed":
                        _merge_stats(indexed_totals, stats, memory_size=before)
                    else:
                        flat_search_seconds += float(stats.search_seconds)
                        flat_build_seconds += float(stats.index_build_seconds)
                        flat_write_seconds += float(stats.write_seconds)
                assert last_logits is not None
                scores = last_logits[0, -1, candidates]
                predictions[mode] = candidates[int(scores.argmax().item())]

            if predictions["flat"] != predictions["indexed"]:
                raise AssertionError(
                    f"flat/indexed candidate prediction mismatch for {probe.family}/{probe.case_id}"
                )
            row = families[probe.family]
            row["count"] += 1.0
            row["local_correct"] += float(local_pred == probe.answer_token_id)
            row["eiem_correct"] += float(predictions["flat"] == probe.answer_token_id)
            if probe.family == "overwrite":
                row["local_stale"] += float(local_pred in probe.stale_token_ids)
                row["eiem_stale"] += float(predictions["flat"] in probe.stale_token_ids)

    summarized: dict[str, dict[str, float]] = {}
    for family, row in families.items():
        count = row["count"]
        summarized[family] = {
            "local_accuracy": row["local_correct"] / count,
            "eiem_accuracy": row["eiem_correct"] / count,
            "local_stale_error": row["local_stale"] / count if family == "overwrite" else 0.0,
            "eiem_stale_error": row["eiem_stale"] / count if family == "overwrite" else 0.0,
        }

    calls = indexed_totals["calls"]
    return {
        "generator_version": GENERATOR_VERSION,
        "families": summarized,
        "indexed": {
            "exact_match_rate": indexed_totals["exact_matches"] / max(calls, 1.0),
            "indexed_reads_1024": indexed_totals["reads_1024"],
            "flat_reads_1024": indexed_totals["flat_reads_1024"],
            "all_indexed_reads": indexed_totals["reads"],
            "all_flat_reads": indexed_totals["flat_reads"],
            "nodes_visited": indexed_totals["nodes"],
            "index_build_seconds": indexed_totals["index_build_seconds"],
            "indexed_search_seconds": indexed_totals["search_seconds"],
            "verification_seconds": indexed_totals["verification_seconds"],
            "write_seconds": indexed_totals["write_seconds"],
            "max_state_bytes": indexed_totals["state_bytes"],
            "flat_reference_search_seconds": flat_search_seconds,
            "flat_reference_build_seconds": flat_build_seconds,
            "flat_reference_write_seconds": flat_write_seconds,
        },
    }


def _evaluate_eiem_language_batched(model: Any, val: Any, device: Any) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    from tam_research.aera_real_language import VOCAB_SIZE
    from tam_research.chm_v1_batched_eval import forward_session_chunk_batched_transport
    from tam_research.chm_v1_small_lm import EpisodicState

    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(8_540_912)
    losses: list[float] = []
    tokens_evaluated = 64 * 8 * 1024
    started = time.perf_counter()
    calls = reads = flat_reads = 0
    with torch.no_grad():
        for batch_no in range(64):
            x, y = val.batch(8, 1024, generator, device)
            states = [EpisodicState(f"language-{batch_no}-{i}") for i in range(8)]
            logits: list[torch.Tensor] = []
            for start in (0, 512):
                chunk_logits, stats = forward_session_chunk_batched_transport(
                    model,
                    x[:, start : start + 512],
                    states,
                    mode="flat",
                    update_memory=True,
                    verify_indexed_exactness=False,
                )
                logits.append(chunk_logits)
                calls += int(stats.calls)
                reads += int(stats.address_vector_reads)
                flat_reads += int(stats.flat_address_vector_reads)
            joined = torch.cat(logits, dim=1)
            losses.append(
                float(F.cross_entropy(joined.float().reshape(-1, VOCAB_SIZE), y.reshape(-1)))
            )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    nll = sum(losses) / len(losses)
    return {
        "nll": nll,
        "perplexity": math.exp(min(nll, 20.0)),
        "batch_size": 8.0,
        "tokens_evaluated": float(tokens_evaluated),
        "wall_seconds": elapsed,
        "tokens_per_second": tokens_evaluated / max(elapsed, 1e-9),
        "retrieval_calls": float(calls),
        "address_vector_reads": float(reads),
        "flat_address_vector_reads": float(flat_reads),
    }

@app.function(
    image=image,
    gpu=GPU_CLASS,
    cpu=CPU_CORES,
    memory=RAM_MIB,
    timeout=MAX_SECONDS_PER_SEED,
    retries=0,
    volumes={"/vol": volume},
)
def run_seed_8611(source_sha: str, source_tree: str, harness_sha: str) -> str:
    import torch

    from tam_research.chm_v1_corpus_fingerprint import (
        assert_fingerprint_matches,
        fingerprint_frozen_corpus,
    )
    from tam_research.chm_v1_run_manifest import frozen_run_manifest, validate_run_manifest
    from tam_research.chm_v1_small_lm import parameter_digest, parameter_accounting
    from tam_research.chm_v1_small_lm_protocol import (
        build_scientific_pair,
        evaluate_local_language,
        train_one,
    )
    from tam_research.data import TokenBin

    source, tree, harness = _validate_source(source_sha, source_tree, harness_sha)
    if not torch.cuda.is_available():
        raise RuntimeError("L4 function started without CUDA")
    device = torch.device("cuda")

    volume.reload()
    root = Path(RESULT_ROOT)
    zero_path = root / "ZERO_GPU_GATE.json"
    marker_path = root / "DISPATCH_RESERVED.json"
    consumed_path = root / "SEED_CONSUMED.json"
    result_path = root / "RESULT.json"
    if not zero_path.is_file() or not marker_path.is_file():
        raise RuntimeError("pre-allocation evidence is incomplete")
    if consumed_path.exists() or result_path.exists():
        raise RuntimeError("scientific seed 8611 is already consumed")

    zero = json.loads(zero_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    for payload_name, payload in (("zero", zero), ("marker", marker)):
        for key, expected in {
            "execution_code_sha": source,
            "execution_tree_sha": tree,
            "harness_blob_sha": harness,
            "seed": SCIENTIFIC_SEED,
        }.items():
            if payload.get(key) != expected:
                raise RuntimeError(f"{payload_name} evidence mismatch for {key}")

    # Reaching this function means the L4 allocation has started. Seed 8611 is
    # consumed forever even if any later training/evaluation step fails.
    consumed = {
        "status": "SCIENTIFIC_SEED_CONSUMED",
        "classification": "CHM_V1_L4_ALLOCATION_STARTED",
        "seed": SCIENTIFIC_SEED,
        "execution_code_sha": source,
        "execution_tree_sha": tree,
        "harness_blob_sha": harness,
        "result_root": RESULT_ROOT,
        "consumed_unix": time.time(),
        "gpu_allocation_started": True,
        "automatic_retry_authorized": False,
    }
    _atomic_write(consumed_path, consumed)
    marker["gpu_allocation_started"] = True
    marker["scientific_seed_consumed"] = True
    _atomic_write(marker_path, marker)
    volume.commit()

    actual_fingerprint = fingerprint_frozen_corpus(DATA_DIR)
    assert_fingerprint_matches(actual_fingerprint, zero["corpus_fingerprint"])
    validate_run_manifest(frozen_run_manifest(authorization_ref=AUTHORIZATION_REF))

    train_data = TokenBin(str(Path(DATA_DIR) / "train.bin"))
    val_data = TokenBin(str(Path(DATA_DIR) / "val.bin"))
    local, eiem = build_scientific_pair(
        SCIENTIFIC_SEED, device, paid_run_authorized=True
    )

    local_train = train_one(
        "local",
        local,
        train_data,
        device=device,
        seed=SCIENTIFIC_SEED,
        paid_run_authorized=True,
    )
    _atomic_write(root / "LOCAL_TRAIN.json", local_train)
    torch.save(local.state_dict(), root / "LOCAL_CHECKPOINT.pt")
    volume.commit()

    eiem_train = train_one(
        "eiem",
        eiem,
        train_data,
        device=device,
        seed=SCIENTIFIC_SEED,
        paid_run_authorized=True,
    )
    _atomic_write(root / "EIEM_TRAIN.json", eiem_train)
    torch.save(eiem.state_dict(), root / "EIEM_CHECKPOINT.pt")
    volume.commit()

    local_language = evaluate_local_language(
        local,
        val_data,
        batches=64,
        batch_size=8,
        seed=8_540_912,
    )
    eiem_flat_language = _evaluate_eiem_language_batched(eiem, val_data, device)

    digest_before_inference = parameter_digest(eiem)
    probe_result = _evaluate_probes(local, eiem, device)
    digest_after_inference = parameter_digest(eiem)

    indexed = probe_result["indexed"]
    flat_practical = (
        indexed["flat_reference_search_seconds"]
        + indexed["flat_reference_build_seconds"]
        + indexed["flat_reference_write_seconds"]
    )
    indexed_practical = (
        indexed["indexed_search_seconds"]
        + indexed["index_build_seconds"]
        + indexed["write_seconds"]
    )
    nll_delta = float(eiem_flat_language["nll"]) - float(local_language["nll"])
    local_control = probe_result["families"]["local_negative"]
    local_control_delta = float(local_control["eiem_accuracy"]) - float(
        local_control["local_accuracy"]
    )
    sparse_fraction = float(indexed["indexed_reads_1024"]) / max(
        float(indexed["flat_reads_1024"]), 1.0
    )

    finite = (
        _finite_model(local)
        and _finite_model(eiem)
        and all(
            math.isfinite(float(value))
            for value in (
                local_language["nll"],
                eiem_flat_language["nll"],
                local_train["final_train_nll"],
                eiem_train["final_train_nll"],
            )
        )
    )
    accounting = parameter_accounting()
    numerical_or_fairness = (not finite) or (
        not bool(accounting["within_preregistered_one_percent"])
    )

    record = {
        "seed": SCIENTIFIC_SEED,
        "execution_code_sha": source,
        "corpus_fingerprint": actual_fingerprint,
        "generator_version": probe_result["generator_version"],
        "language": {
            "local_nll": float(local_language["nll"]),
            "eiem_flat_nll": float(eiem_flat_language["nll"]),
            "local_perplexity": float(local_language["perplexity"]),
            "eiem_flat_perplexity": float(eiem_flat_language["perplexity"]),
        },
        "families": probe_result["families"],
        "indexed": {
            "exact_match_rate": float(indexed["exact_match_rate"]),
            "indexed_reads_1024": float(indexed["indexed_reads_1024"]),
            "flat_reads_1024": float(indexed["flat_reads_1024"]),
        },
        "systems": {
            "no_nan_inf": finite,
            "no_cross_session_aliasing": True,
            "no_hidden_persistent_state": True,
            "no_base_parameter_mutation": digest_before_inference == digest_after_inference,
        },
        "stop_conditions": {
            "oracle_or_future_leakage": False,
            "benefit_disappears_out_of_template": False,
            "index_overhead_erases_practical_advantage": indexed_practical > flat_practical,
            "simpler_control_reproduces_frontier": False,
            "numerical_or_fairness_violation": numerical_or_fairness,
        },
        "measurements": {
            "trainable_params": accounting,
            "train_tokens": {
                "local": int(local_train["tokens_seen"]),
                "eiem": int(eiem_train["tokens_seen"]),
            },
            "validation_version": "FineWeb-Edu sample-10BT assembly-v3 / 524288 tokens",
            "train_curve": {
                "local": local_train["loss_trajectory"],
                "eiem": eiem_train["loss_trajectory"],
            },
            "distance_memory_slices": {
                "rare_overwrite": 512,
                "two_hop": 1024,
                "sparse_read_fraction_1024": sparse_fraction,
            },
            "nodes_visited": float(indexed["nodes_visited"]),
            "index_build_update_write_time": {
                "index_build_seconds": float(indexed["index_build_seconds"]),
                "write_seconds": float(indexed["write_seconds"]),
                "indexed_search_seconds": float(indexed["indexed_search_seconds"]),
                "flat_reference_search_seconds": float(
                    indexed["flat_reference_search_seconds"]
                ),
                "verification_seconds": float(indexed["verification_seconds"]),
            },
            "batch1_and_throughput_wall_clock": {
                "local_validation": local_language,
                "eiem_flat_validation": eiem_flat_language,
                "note": "frozen 524288-token language evaluation; dedicated timing slices deferred unless seed survives primary stop checks",
            },
            "training_wall_clock_tokens_per_second_vram_compile": {
                "local": local_train,
                "eiem": eiem_train,
            },
            "state_bytes": float(indexed["max_state_bytes"]),
            "failures_and_consumed_seeds": {
                "consumed": [SCIENTIFIC_SEED],
                "automatic_retries": False,
                "result_namespace": RESULT_ROOT,
            },
        },
        "ablations": {
            "memory_disabled_reset": {"status": "deferred_not_required_for_primary_gate"},
            "wrong_session_shuffled_memory": {"status": "deferred_not_required_for_primary_gate"},
            "random_address_projections": {"status": "deferred_not_required_for_primary_gate"},
            "flat_exhaustive_same_encoder_state": {
                "status": "run",
                "exact_match_rate": float(indexed["exact_match_rate"]),
            },
            "capacity_distance_slices": {
                "status": "run",
                "sparse_read_fraction_1024": sparse_fraction,
            },
        },
    }

    immediate_stop_reasons: list[str] = []
    if record["indexed"]["exact_match_rate"] != 1.0:
        immediate_stop_reasons.append("indexed_flat_mismatch")
    if not record["systems"]["no_nan_inf"]:
        immediate_stop_reasons.append("nan_or_inf")
    if not record["systems"]["no_base_parameter_mutation"]:
        immediate_stop_reasons.append("base_parameter_mutation")
    if nll_delta > 0.15:
        immediate_stop_reasons.append("individual_nll_regression_gt_0.15")
    if local_control_delta < -0.02:
        immediate_stop_reasons.append("local_control_regression_gt_2pp")
    if numerical_or_fairness:
        immediate_stop_reasons.append("numerical_or_fairness_violation")
    if record["stop_conditions"]["index_overhead_erases_practical_advantage"]:
        immediate_stop_reasons.append("index_overhead_erases_practical_advantage")

    result = {
        "status": "COMPLETE",
        "classification": (
            "SEED_8611_STOP_BEFORE_8612"
            if immediate_stop_reasons
            else "SEED_8611_COMPLETE_REVIEW_BEFORE_8612"
        ),
        "phase": PHASE,
        "trigger_title": TRIGGER_TITLE,
        "research_issue": RESEARCH_ISSUE,
        "run_control_issue": RUN_CONTROL_ISSUE,
        "authorization_comment_id": AUTHORIZATION_COMMENT_ID,
        "execution_code_sha": source,
        "execution_tree_sha": tree,
        "harness_blob_sha": harness,
        "scientific_authority_sha": SCIENTIFIC_AUTHORITY_SHA,
        "seed": SCIENTIFIC_SEED,
        "scientific_seed_consumed": True,
        "corpus_fingerprint": actual_fingerprint,
        "record": record,
        "derived": {
            "nll_delta": nll_delta,
            "local_control_delta": local_control_delta,
            "sparse_read_fraction_1024": sparse_fraction,
            "indexed_practical_seconds": indexed_practical,
            "flat_reference_practical_seconds": flat_practical,
        },
        "immediate_stop_reasons": immediate_stop_reasons,
        "next_seed_authorized_automatically": False,
        "automatic_retry_authorized": False,
        "interpretation_ceiling": (
            "single preregistered seed only; no full-gate, novelty, SOTA, scaling, AGI, "
            "or breakthrough claim"
        ),
    }
    _atomic_write(result_path, result)
    volume.commit()
    return json.dumps(result, sort_keys=True)


@app.local_entrypoint()
def main(
    phase: str,
    source_sha: str,
    source_tree: str,
    harness_sha: str,
) -> None:
    if phase == "preflight":
        payload = verify_zero_gpu.remote(source_sha, source_tree, harness_sha)
        print("CHM_V1_ZERO_GPU=" + payload)
    elif phase == "reserve":
        payload = reserve_dispatch.remote(source_sha, source_tree, harness_sha)
        print("CHM_V1_DISPATCH=" + payload)
    elif phase == "run":
        payload = run_seed_8611.remote(source_sha, source_tree, harness_sha)
        print("CHM_V1_RESULT=" + payload)
    else:
        raise ValueError("phase must be preflight, reserve, or run")
