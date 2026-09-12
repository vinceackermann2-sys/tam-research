from __future__ import annotations

from contextlib import nullcontext
import json
import math
from pathlib import Path
import time
from typing import Any

import modal

PHASE = "chm-v1-eiem-seed-8611-v2"
TRIGGER_TITLE = "[modal-chm-v1-eiem-seed-8611-v2]"
RESEARCH_ISSUE = 854
RUN_CONTROL_ISSUE = 903
AUTHORIZATION_COMMENT_ID = 5_644_581_046
AUTHORIZATION_REF = "issue-854-comment-5644581046"
SCIENTIFIC_AUTHORITY_SHA = "e86ffd453be3efc3026576f5ebbab6b76ab54b95"
SCIENTIFIC_AUTHORITY_TREE = "9af3aa0607f16c953cfe5b802b9daca1e79ce853"
SCIENTIFIC_SEED = 8611
RESULT_ROOT = "/vol/chm-v1/eiem-small-lm/issue-903/seed-8611-v2"
DATA_DIR = "/vol/data/tam100m-2b-curated-v1"

GPU_CLASS = "L4"
CPU_CORES = 4
RAM_MIB = 8192
MAX_SECONDS_PER_SEED = 3600
MAX_AGGREGATE_BILLED_COMPUTE_USD = 4.0
MAX_SEEDS = 3

# Reference hourly rates observed from `modal billing rates --json` on the
# pre-allocation V1 attempt. The workflow re-reads live rates and blocks if the
# actual three-seed bound exceeds $4; these constants are a second fail-closed
# launcher-side sanity check, not a claim that prices are immutable.
L4_USD_PER_HOUR = 0.80000
PHYSICAL_CPU_USD_PER_CORE_HOUR = 0.04730
RAM_USD_PER_GIB_HOUR = 0.00800

# Frozen evaluator-only identities from issue #854 / RUN_MANIFEST.md.
LONG_MEMORY_EVALUATOR_SEED = 8_540_911
VALIDATION_SAMPLING_SEED = 8_540_912
SYSTEMS_TIMING_SEED = 8_540_913
VALIDATION_BATCHES = 64
VALIDATION_BATCH_SIZE = 8
SESSION_TOKENS = 1024
TIMING_BATCH1_SESSIONS = 8
TIMING_THROUGHPUT_BATCHES = 8
TIMING_THROUGHPUT_BATCH_SIZE = 8

APP_NAME = "chm-v1-eiem-small-lm-seed-8611-v2"
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


def _worst_case_one_seed_cost_usd() -> float:
    hours = MAX_SECONDS_PER_SEED / 3600.0
    return hours * (
        L4_USD_PER_HOUR
        + CPU_CORES * PHYSICAL_CPU_USD_PER_CORE_HOUR
        + (RAM_MIB / 1024.0) * RAM_USD_PER_GIB_HOUR
    )


def _worst_case_three_seed_cost_usd() -> float:
    return MAX_SEEDS * _worst_case_one_seed_cost_usd()


def _validate_source(source_sha: str, source_tree: str, harness_sha: str) -> tuple[str, str, str]:
    source = _full_sha(source_sha, "source_sha")
    tree = _full_sha(source_tree, "source_tree")
    harness = _full_sha(harness_sha, "harness_sha")
    if SCIENTIFIC_SEED != 8611:
        raise RuntimeError("this one-shot launcher is bound only to scientific seed 8611")
    if GPU_CLASS != "L4" or CPU_CORES != 4 or RAM_MIB != 8192 or MAX_SECONDS_PER_SEED != 3600:
        raise RuntimeError("frozen #854 resource envelope drift")
    if (
        VALIDATION_BATCHES != 64
        or VALIDATION_BATCH_SIZE != 8
        or SESSION_TOKENS != 1024
        or TIMING_BATCH1_SESSIONS != 8
        or TIMING_THROUGHPUT_BATCHES != 8
        or TIMING_THROUGHPUT_BATCH_SIZE != 8
    ):
        raise RuntimeError("frozen #854 evaluation envelope drift")
    if _worst_case_three_seed_cost_usd() > MAX_AGGREGATE_BILLED_COMPUTE_USD:
        raise RuntimeError("three-seed resource envelope can exceed $4 at reference rates")
    return source, tree, harness


def _finite_model(model: Any) -> bool:
    import torch

    return all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())


def _autocast(device: Any):
    import torch

    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )


def _empty_stats() -> dict[str, float]:
    return {
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


def _merge_stats(total: dict[str, float], stats: Any, *, memory_size: int | None = None) -> None:
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
        raise RuntimeError("reserved seed-8611 V2 result namespace already exists; fail closed")

    protocol = protocol_preflight(DATA_DIR)
    fingerprint = fingerprint_frozen_corpus(DATA_DIR)
    manifest = validate_run_manifest(frozen_run_manifest(authorization_ref=AUTHORIZATION_REF))
    encoder = tiktoken.get_encoding("gpt2")
    probes = build_scientific_probe_suite(lambda text: encoder.encode(text))
    if len(probes) != 96:
        raise RuntimeError(f"scientific probe count drift: {len(probes)}")

    payload = {
        "status": "PASS",
        "classification": "CHM_V1_ZERO_GPU_PREFLIGHT_SEED_8611_V2",
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
        "evaluation_envelope": {
            "long_memory_evaluator_seed": LONG_MEMORY_EVALUATOR_SEED,
            "validation_sampling_seed": VALIDATION_SAMPLING_SEED,
            "systems_timing_seed": SYSTEMS_TIMING_SEED,
            "validation_tokens_per_model": VALIDATION_BATCHES * VALIDATION_BATCH_SIZE * SESSION_TOKENS,
            "ordinary_indexed_exactness_batch_tokens": VALIDATION_BATCH_SIZE * SESSION_TOKENS,
            "timing_batch1_sessions": TIMING_BATCH1_SESSIONS,
            "timing_throughput_batches": TIMING_THROUGHPUT_BATCHES,
            "timing_throughput_batch_size": TIMING_THROUGHPUT_BATCH_SIZE,
        },
        "reference_rates_usd_per_hour": {
            "l4": L4_USD_PER_HOUR,
            "physical_cpu_core": PHYSICAL_CPU_USD_PER_CORE_HOUR,
            "ram_gib": RAM_USD_PER_GIB_HOUR,
        },
        "worst_case_one_seed_usd": _worst_case_one_seed_cost_usd(),
        "worst_case_three_seed_usd": _worst_case_three_seed_cost_usd(),
        "v1_preallocation_failure_run": 34686481894,
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
    failure_path = root / "ATTEMPT_FAILURE.json"
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
    if marker_path.exists() or consumed_path.exists() or result_path.exists() or failure_path.exists():
        raise RuntimeError("seed-8611 V2 trigger/result namespace was already used")

    marker = {
        "status": "DISPATCH_RESERVED",
        "classification": "CHM_V1_DURABLE_PRE_ALLOCATION_MARKER_V2",
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
            "minimum_evidence_distance": float("inf"),
            "maximum_evidence_distance": 0.0,
            "memory_items_at_query": -1.0,
        }

    indexed_totals = _empty_stats()
    flat_totals = _empty_stats()

    local.eval()
    eiem.eval()
    with torch.no_grad():
        for probe in probes:
            ids = list(probe.prompt_ids)
            candidates = list(probe.candidate_token_ids)
            local_ctx = ids[-LOCAL_WINDOW:]
            local_tokens = torch.tensor([local_ctx], dtype=torch.long, device=device)
            with _autocast(device):
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
                    with _autocast(device):
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
                        _merge_stats(flat_totals, stats, memory_size=before)
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
            row["minimum_evidence_distance"] = min(
                row["minimum_evidence_distance"], float(probe.evidence_distance)
            )
            row["maximum_evidence_distance"] = max(
                row["maximum_evidence_distance"], float(probe.evidence_distance)
            )
            memory_items = float((probe.query_token // LOCAL_WINDOW) * LOCAL_WINDOW)
            if row["memory_items_at_query"] < 0:
                row["memory_items_at_query"] = memory_items
            elif row["memory_items_at_query"] != memory_items:
                raise RuntimeError(f"memory-size slice drift inside family {probe.family}")
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
            "minimum_evidence_distance": row["minimum_evidence_distance"],
            "maximum_evidence_distance": row["maximum_evidence_distance"],
            "memory_items_at_query": row["memory_items_at_query"],
        }

    calls = indexed_totals["calls"]
    return {
        "generator_version": GENERATOR_VERSION,
        "families": summarized,
        "indexed": {
            "calls": calls,
            "exact_matches": indexed_totals["exact_matches"],
            "exact_match_rate": indexed_totals["exact_matches"] / max(calls, 1.0),
            "indexed_reads_1024": indexed_totals["reads_1024"],
            "flat_reads_1024": indexed_totals["flat_reads_1024"],
            "all_indexed_reads": indexed_totals["reads"],
            "all_flat_reads": indexed_totals["flat_reads"],
            "mean_indexed_reads_per_query": indexed_totals["reads"] / max(calls, 1.0),
            "mean_flat_reads_per_query": indexed_totals["flat_reads"] / max(calls, 1.0),
            "nodes_visited": indexed_totals["nodes"],
            "mean_nodes_visited_per_query": indexed_totals["nodes"] / max(calls, 1.0),
            "index_build_seconds": indexed_totals["index_build_seconds"],
            "indexed_search_seconds": indexed_totals["search_seconds"],
            "verification_seconds": indexed_totals["verification_seconds"],
            "write_seconds": indexed_totals["write_seconds"],
            "max_state_bytes": indexed_totals["state_bytes"],
            "flat_reference_search_seconds": flat_totals["search_seconds"],
            "flat_reference_build_seconds": flat_totals["index_build_seconds"],
            "flat_reference_write_seconds": flat_totals["write_seconds"],
        },
    }


def _run_eiem_batch(
    model: Any,
    x: Any,
    device: Any,
    *,
    mode: str,
    verify_indexed_exactness: bool,
    session_prefix: str,
) -> tuple[Any, dict[str, float]]:
    import torch

    from tam_research.chm_v1_batched_eval import forward_session_chunk_batched_transport
    from tam_research.chm_v1_small_lm import EpisodicState

    states = [EpisodicState(f"{session_prefix}-{i}") for i in range(x.shape[0])]
    logits: list[torch.Tensor] = []
    total = _empty_stats()
    for start in (0, 512):
        with _autocast(device):
            chunk_logits, stats = forward_session_chunk_batched_transport(
                model,
                x[:, start : start + 512],
                states,
                mode=mode,
                update_memory=True,
                verify_indexed_exactness=verify_indexed_exactness,
            )
        logits.append(chunk_logits)
        _merge_stats(total, stats, memory_size=start)
    return torch.cat(logits, dim=1), total


def _evaluate_eiem_language_batched(model: Any, val: Any, device: Any) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    from tam_research.aera_real_language import VOCAB_SIZE

    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(VALIDATION_SAMPLING_SEED)
    losses: list[float] = []
    tokens_evaluated = VALIDATION_BATCHES * VALIDATION_BATCH_SIZE * SESSION_TOKENS
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    calls = reads = flat_reads = 0
    with torch.no_grad():
        for batch_no in range(VALIDATION_BATCHES):
            x, y = val.batch(VALIDATION_BATCH_SIZE, SESSION_TOKENS, generator, device)
            joined, stats = _run_eiem_batch(
                model,
                x,
                device,
                mode="flat",
                verify_indexed_exactness=False,
                session_prefix=f"language-{batch_no}",
            )
            calls += int(stats["calls"])
            reads += int(stats["reads"])
            flat_reads += int(stats["flat_reads"])
            losses.append(
                float(F.cross_entropy(joined.float().reshape(-1, VOCAB_SIZE), y.reshape(-1)))
            )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    nll = sum(losses) / len(losses)
    return {
        "nll": nll,
        "perplexity": math.exp(min(nll, 20.0)),
        "batch_size": float(VALIDATION_BATCH_SIZE),
        "tokens_evaluated": float(tokens_evaluated),
        "wall_seconds": elapsed,
        "tokens_per_second": tokens_evaluated / max(elapsed, 1e-9),
        "retrieval_calls": float(calls),
        "address_vector_reads": float(reads),
        "flat_address_vector_reads": float(flat_reads),
        "bf16_autocast": True,
    }


def _ordinary_indexed_exactness_audit(model: Any, val: Any, device: Any) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    from tam_research.aera_real_language import VOCAB_SIZE

    generator = torch.Generator(device="cpu").manual_seed(VALIDATION_SAMPLING_SEED)
    x, y = val.batch(VALIDATION_BATCH_SIZE, SESSION_TOKENS, generator, device)
    model.eval()
    with torch.no_grad():
        flat_logits, flat_stats = _run_eiem_batch(
            model,
            x,
            device,
            mode="flat",
            verify_indexed_exactness=False,
            session_prefix="ordinary-exactness-flat",
        )
        indexed_logits, indexed_stats = _run_eiem_batch(
            model,
            x,
            device,
            mode="indexed",
            verify_indexed_exactness=True,
            session_prefix="ordinary-exactness-indexed",
        )
    torch.cuda.synchronize(device)
    diff = (flat_logits.float() - indexed_logits.float()).abs()
    max_abs_logit_delta = float(diff.max())
    logits_within_tolerance = bool(
        torch.allclose(flat_logits.float(), indexed_logits.float(), rtol=0.0, atol=1e-5)
    )
    flat_nll = float(F.cross_entropy(flat_logits.float().reshape(-1, VOCAB_SIZE), y.reshape(-1)))
    indexed_nll = float(
        F.cross_entropy(indexed_logits.float().reshape(-1, VOCAB_SIZE), y.reshape(-1))
    )
    calls = float(indexed_stats["calls"])
    exact_matches = float(indexed_stats["exact_matches"])
    return {
        "seed": VALIDATION_SAMPLING_SEED,
        "batch_size": VALIDATION_BATCH_SIZE,
        "session_tokens": SESSION_TOKENS,
        "tokens_evaluated": VALIDATION_BATCH_SIZE * SESSION_TOKENS,
        "retrieval_calls": calls,
        "exact_matches": exact_matches,
        "exact_match_rate": exact_matches / max(calls, 1.0),
        "max_abs_logit_delta": max_abs_logit_delta,
        "logits_within_tolerance": logits_within_tolerance,
        "flat_nll": flat_nll,
        "indexed_nll": indexed_nll,
        "nll_delta": indexed_nll - flat_nll,
        "indexed_verification_seconds": float(indexed_stats["verification_seconds"]),
        "indexed_reads": float(indexed_stats["reads"]),
        "flat_reads": float(indexed_stats["flat_reads"]),
        "bf16_autocast": True,
    }


def _time_local_inputs(model: Any, inputs: list[Any], device: Any) -> dict[str, float]:
    import torch

    from tam_research.chm_v1_small_lm_protocol import local_session_logits

    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for x in inputs:
            with _autocast(device):
                local_session_logits(model, x)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    tokens = sum(int(x.numel()) for x in inputs)
    return {
        "wall_seconds": elapsed,
        "tokens": float(tokens),
        "tokens_per_second": tokens / max(elapsed, 1e-9),
        "peak_vram_bytes": float(torch.cuda.max_memory_allocated(device)),
        "bf16_autocast": True,
    }


def _time_eiem_inputs(
    model: Any,
    inputs: list[Any],
    device: Any,
    *,
    mode: str,
) -> dict[str, float]:
    import torch

    model.eval()
    total = _empty_stats()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for batch_no, x in enumerate(inputs):
            _, stats = _run_eiem_batch(
                model,
                x,
                device,
                mode=mode,
                verify_indexed_exactness=False,
                session_prefix=f"timing-{mode}-{batch_no}",
            )
            for key in total:
                if key == "state_bytes":
                    total[key] = max(total[key], stats[key])
                else:
                    total[key] += stats[key]
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    tokens = sum(int(x.numel()) for x in inputs)
    return {
        "wall_seconds": elapsed,
        "tokens": float(tokens),
        "tokens_per_second": tokens / max(elapsed, 1e-9),
        "peak_vram_bytes": float(torch.cuda.max_memory_allocated(device)),
        "retrieval_calls": total["calls"],
        "address_vector_reads": total["reads"],
        "flat_address_vector_reads": total["flat_reads"],
        "directory_nodes_visited": total["nodes"],
        "index_build_seconds": total["index_build_seconds"],
        "search_seconds": total["search_seconds"],
        "write_seconds": total["write_seconds"],
        "verification_seconds": total["verification_seconds"],
        "max_state_bytes": total["state_bytes"],
        "verification_disabled_after_exactness_audit": mode == "indexed",
        "bf16_autocast": True,
    }


def _systems_timing(local: Any, eiem: Any, val: Any, device: Any) -> dict[str, Any]:
    import torch

    generator = torch.Generator(device="cpu").manual_seed(SYSTEMS_TIMING_SEED)
    batch1_inputs = [
        val.batch(1, SESSION_TOKENS, generator, device)[0]
        for _ in range(TIMING_BATCH1_SESSIONS)
    ]
    throughput_inputs = [
        val.batch(TIMING_THROUGHPUT_BATCH_SIZE, SESSION_TOKENS, generator, device)[0]
        for _ in range(TIMING_THROUGHPUT_BATCHES)
    ]
    expected_batch1_tokens = TIMING_BATCH1_SESSIONS * SESSION_TOKENS
    expected_throughput_tokens = (
        TIMING_THROUGHPUT_BATCHES * TIMING_THROUGHPUT_BATCH_SIZE * SESSION_TOKENS
    )
    if sum(int(x.numel()) for x in batch1_inputs) != expected_batch1_tokens:
        raise RuntimeError("batch-1 timing token envelope drift")
    if sum(int(x.numel()) for x in throughput_inputs) != expected_throughput_tokens:
        raise RuntimeError("throughput timing token envelope drift")

    return {
        "seed": SYSTEMS_TIMING_SEED,
        "batch1": {
            "sessions": TIMING_BATCH1_SESSIONS,
            "batch_size": 1,
            "session_tokens": SESSION_TOKENS,
            "local": _time_local_inputs(local, batch1_inputs, device),
            "eiem_flat": _time_eiem_inputs(eiem, batch1_inputs, device, mode="flat"),
            "eiem_indexed": _time_eiem_inputs(eiem, batch1_inputs, device, mode="indexed"),
        },
        "throughput": {
            "batches": TIMING_THROUGHPUT_BATCHES,
            "batch_size": TIMING_THROUGHPUT_BATCH_SIZE,
            "session_tokens": SESSION_TOKENS,
            "local": _time_local_inputs(local, throughput_inputs, device),
            "eiem_flat": _time_eiem_inputs(eiem, throughput_inputs, device, mode="flat"),
            "eiem_indexed": _time_eiem_inputs(eiem, throughput_inputs, device, mode="indexed"),
        },
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
    failure_path = root / "ATTEMPT_FAILURE.json"
    if not zero_path.is_file() or not marker_path.is_file():
        raise RuntimeError("pre-allocation evidence is incomplete")
    if consumed_path.exists() or result_path.exists() or failure_path.exists():
        raise RuntimeError("scientific seed 8611 V2 is already consumed")

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
        "classification": "CHM_V1_L4_ALLOCATION_STARTED_V2",
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

    try:
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
            batches=VALIDATION_BATCHES,
            batch_size=VALIDATION_BATCH_SIZE,
            seed=VALIDATION_SAMPLING_SEED,
        )
        eiem_flat_language = _evaluate_eiem_language_batched(eiem, val_data, device)
        if int(local_language["tokens_evaluated"]) != 524_288:
            raise RuntimeError("LOCAL validation token count drift")
        if int(eiem_flat_language["tokens_evaluated"]) != 524_288:
            raise RuntimeError("EIEM-FLAT validation token count drift")

        digest_before_inference = parameter_digest(eiem)
        ordinary_exactness = _ordinary_indexed_exactness_audit(eiem, val_data, device)
        probe_result = _evaluate_probes(local, eiem, device)
        indexed = probe_result["indexed"]

        combined_exact_calls = float(indexed["calls"]) + float(
            ordinary_exactness["retrieval_calls"]
        )
        combined_exact_matches = float(indexed["exact_matches"]) + float(
            ordinary_exactness["exact_matches"]
        )
        combined_exact_rate = combined_exact_matches / max(combined_exact_calls, 1.0)
        exactness_audit_passed = (
            combined_exact_rate == 1.0
            and bool(ordinary_exactness["logits_within_tolerance"])
        )
        if not exactness_audit_passed:
            systems_timing: dict[str, Any] = {
                "status": "SKIPPED_INDEXED_TIMING_EXACTNESS_AUDIT_FAILED",
                "seed": SYSTEMS_TIMING_SEED,
            }
        else:
            systems_timing = _systems_timing(local, eiem, val_data, device)

        digest_after_inference = parameter_digest(eiem)

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
                    ordinary_exactness["flat_nll"],
                    ordinary_exactness["indexed_nll"],
                )
            )
        )
        accounting = parameter_accounting()
        numerical_or_fairness = (not finite) or (
            not bool(accounting["within_preregistered_one_percent"])
        )

        if exactness_audit_passed:
            flat_practical = float(
                systems_timing["batch1"]["eiem_flat"]["wall_seconds"]
            ) + float(systems_timing["throughput"]["eiem_flat"]["wall_seconds"])
            indexed_practical = float(
                systems_timing["batch1"]["eiem_indexed"]["wall_seconds"]
            ) + float(
                systems_timing["throughput"]["eiem_indexed"]["wall_seconds"]
            )
            index_overhead_erases = indexed_practical >= flat_practical
        else:
            flat_practical = float("nan")
            indexed_practical = float("nan")
            index_overhead_erases = False

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
                "exact_match_rate": combined_exact_rate,
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
                "index_overhead_erases_practical_advantage": index_overhead_erases,
                "simpler_control_reproduces_frontier": False,
                "numerical_or_fairness_violation": numerical_or_fairness,
            },
            "measurements": {
                "trainable_params": accounting,
                "train_tokens": {
                    "local": int(local_train["tokens_seen"]),
                    "eiem": int(eiem_train["tokens_seen"]),
                },
                "validation_version": "FineWeb-Edu sample-10BT assembly-v3 / 524288 tokens/model",
                "train_curve": {
                    "local": local_train["loss_trajectory"],
                    "eiem": eiem_train["loss_trajectory"],
                },
                "distance_memory_slices": {
                    "families": probe_result["families"],
                    "sparse_read_fraction_1024": sparse_fraction,
                },
                "nodes_visited": {
                    "total": float(indexed["nodes_visited"]),
                    "mean_per_query": float(indexed["mean_nodes_visited_per_query"]),
                    "mean_indexed_address_reads_per_query": float(
                        indexed["mean_indexed_reads_per_query"]
                    ),
                    "mean_flat_address_reads_per_query": float(
                        indexed["mean_flat_reads_per_query"]
                    ),
                },
                "index_build_update_write_time": {
                    "index_build_seconds": float(indexed["index_build_seconds"]),
                    "write_seconds": float(indexed["write_seconds"]),
                    "indexed_search_seconds": float(indexed["indexed_search_seconds"]),
                    "flat_reference_search_seconds": float(
                        indexed["flat_reference_search_seconds"]
                    ),
                    "verification_seconds_probe_suite": float(
                        indexed["verification_seconds"]
                    ),
                    "verification_seconds_ordinary_batch": float(
                        ordinary_exactness["indexed_verification_seconds"]
                    ),
                },
                "ordinary_indexed_exactness_audit": ordinary_exactness,
                "batch1_and_throughput_wall_clock": systems_timing,
                "training_wall_clock_tokens_per_second_vram_compile": {
                    "local": local_train,
                    "eiem": eiem_train,
                },
                "state_bytes": float(indexed["max_state_bytes"]),
                "failures_and_consumed_seeds": {
                    "consumed": [SCIENTIFIC_SEED],
                    "automatic_retries": False,
                    "result_namespace": RESULT_ROOT,
                    "abandoned_preallocation_trigger_issue": 919,
                },
                "systems_evidence": {
                    "bf16_gpu_evaluation": True,
                    "fresh_ephemeral_state_per_session": True,
                    "current_chunk_written_only_after_logits": True,
                    "inference_parameter_digest_before": digest_before_inference,
                    "inference_parameter_digest_after": digest_after_inference,
                },
            },
            "ablations": {
                "memory_disabled_reset": {
                    "status": "deferred_mandatory_before_scaling_not_used_for_small_gate_pass_interpretation"
                },
                "wrong_session_shuffled_memory": {
                    "status": "deferred_mandatory_before_scaling_not_used_for_small_gate_pass_interpretation"
                },
                "random_address_projections": {
                    "status": "deferred_mandatory_before_scaling_not_used_for_small_gate_pass_interpretation"
                },
                "flat_exhaustive_same_encoder_state": {
                    "status": "run",
                    "combined_exact_match_rate": combined_exact_rate,
                    "ordinary_max_abs_logit_delta": float(
                        ordinary_exactness["max_abs_logit_delta"]
                    ),
                },
                "capacity_distance_slices": {
                    "status": "run",
                    "sparse_read_fraction_1024": sparse_fraction,
                    "families": probe_result["families"],
                },
            },
        }

        immediate_stop_reasons: list[str] = []
        if record["indexed"]["exact_match_rate"] != 1.0:
            immediate_stop_reasons.append("indexed_flat_mismatch")
        if not bool(ordinary_exactness["logits_within_tolerance"]):
            immediate_stop_reasons.append("indexed_flat_logit_or_nll_mismatch")
        if sparse_fraction > 0.25:
            immediate_stop_reasons.append("sparse_read_fraction_gt_0.25")
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
                "combined_exact_match_rate": combined_exact_rate,
                "ordinary_max_abs_logit_delta": float(
                    ordinary_exactness["max_abs_logit_delta"]
                ),
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
    except Exception as exc:
        failure = {
            "status": "ATTEMPT_FAILED_AFTER_L4_ALLOCATION",
            "classification": "UNCLASSIFIED_EXECUTION_FAILURE_NOT_SCIENTIFIC_EVIDENCE",
            "seed": SCIENTIFIC_SEED,
            "execution_code_sha": source,
            "execution_tree_sha": tree,
            "harness_blob_sha": harness,
            "scientific_seed_consumed": True,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "failed_unix": time.time(),
            "automatic_retry_authorized": False,
            "next_seed_authorized_automatically": False,
        }
        _atomic_write(failure_path, failure)
        volume.commit()
        raise


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