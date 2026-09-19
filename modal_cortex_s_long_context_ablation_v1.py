from __future__ import annotations

import gc
import json
from pathlib import Path
import time
from typing import Any

import modal


APP_NAME = "cortex-s-long-context-ablation-v1"
VOLUME_NAME = "tam-research-data"
TRIGGER_TITLE = "[modal-cortex-s-long-context-ablation-v1]"
SYNTHETIC_SEED = 2_026_091_501
RESULT_ROOT = "/vol/cortex-s-v0/systems/long-context-ablation-v1"
RESULT_PATH = f"{RESULT_ROOT}/RESULT.json"
DISPATCH_PATH = f"{RESULT_ROOT}/DISPATCH_CONSUMED.json"
CONTEXT_BATCHES = (
    (512, 32),
    (1_024, 16),
    (2_048, 8),
    (4_096, 4),
    (8_192, 2),
    (16_384, 1),
)
TOKENS_PER_BATCH = 16_384
WARMUP_ITERS = 2
MEASURE_ITERS = 4
HIGH_CONTEXTS = (8_192, 16_384)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
github_secret = modal.Secret.from_name("github-secret")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11", "PyGithub>=2.3,<3")
    .add_local_python_source("architectures")
)


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _comment(repo_full_name: str, issue_number: int, body: str) -> None:
    if not repo_full_name or not issue_number:
        print(f"[status] {body}", flush=True)
        return
    import os

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print(f"[status] {body}", flush=True)
        return
    try:
        import github

        client = github.Github(auth=github.Auth.Token(token))
        client.get_repo(repo_full_name).get_issue(number=issue_number).create_comment(body)
    except Exception as exc:  # status reporting is deliberately non-authoritative.
        print(
            f"[status-report-nonfatal] {type(exc).__name__}: {exc}; body={body}",
            flush=True,
        )


def _validate_sha(value: str, label: str) -> str:
    value = value.strip().lower()
    if len(value) != 40 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{label} must be a full lowercase SHA")
    return value


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=10 * 60,
    volumes={"/vol": volume},
    secrets=[github_secret],
)
def reserve_dispatch(
    source_sha: str,
    source_tree: str,
    harness_sha: str,
    repo_full_name: str = "",
    issue_number: int = 0,
) -> str:
    source = _validate_sha(source_sha, "source_sha")
    tree = _validate_sha(source_tree, "source_tree")
    harness = _validate_sha(harness_sha, "harness_sha")
    volume.reload()
    root = Path(RESULT_ROOT)
    if root.exists():
        raise RuntimeError("long-context systems namespace already exists; fail closed")
    marker = {
        "status": "SYSTEMS_DISPATCH_CONSUMED",
        "classification": "ENGINEERING_LONG_CONTEXT_ABLATION_SINGLE_ATTEMPT",
        "trigger_title": TRIGGER_TITLE,
        "source_sha": source,
        "source_tree": tree,
        "harness_sha": harness,
        "synthetic_seed": SYNTHETIC_SEED,
        "result_root": RESULT_ROOT,
        "gpu_allocation_started": False,
        "training_authorized": False,
        "scientific_claim_authorized": False,
        "created_unix": time.time(),
    }
    _atomic_write(Path(DISPATCH_PATH), marker)
    volume.commit()
    _comment(
        repo_full_name,
        issue_number,
        "🔒 **CORTEX-S long-context systems dispatch consumed** — one H100 systems-only attempt; synthetic tokens only; no optimizer, weight update, checkpoint mutation, or training.",
    )
    return json.dumps(marker, sort_keys=True)


def _synthetic_batch(length: int, batch: int, device: Any):
    import torch

    generator = torch.Generator(device="cpu").manual_seed(SYNTHETIC_SEED + length)
    x_cpu = torch.randint(
        0,
        50_257,
        (batch, length),
        generator=generator,
        dtype=torch.long,
    )
    y_cpu = torch.randint(
        0,
        50_257,
        (batch, length),
        generator=generator,
        dtype=torch.long,
    )
    return x_cpu.to(device=device), y_cpu.to(device=device)


def _forward_measurement(model: Any, x: Any) -> dict[str, Any]:
    import torch

    model.eval()
    try:
        for _ in range(WARMUP_ITERS):
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                logits = model(x)
            del logits
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for _ in range(MEASURE_ITERS):
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                logits = model(x)
            del logits
        torch.cuda.synchronize()
        elapsed = max(time.perf_counter() - started, 1e-9)
        return {
            "status": "COMPLETE",
            "iterations": MEASURE_ITERS,
            "elapsed_seconds": elapsed,
            "tokens_per_second": (x.numel() * MEASURE_ITERS) / elapsed,
            "peak_vram_gib": torch.cuda.max_memory_allocated() / (1024**3),
        }
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        return {
            "status": "OOM",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _train_measurement(model: Any, x: Any, y: Any) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    model.train()

    def one_iteration() -> float:
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
        loss = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )
        loss.backward()
        value = float(loss.detach())
        del loss, logits
        return value

    try:
        warm_loss = float("nan")
        for _ in range(WARMUP_ITERS):
            warm_loss = one_iteration()
        torch.cuda.synchronize()
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        measured_loss = float("nan")
        for _ in range(MEASURE_ITERS):
            measured_loss = one_iteration()
        torch.cuda.synchronize()
        elapsed = max(time.perf_counter() - started, 1e-9)
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        model.zero_grad(set_to_none=True)
        return {
            "status": "COMPLETE",
            "iterations": MEASURE_ITERS,
            "elapsed_seconds": elapsed,
            "tokens_per_second": (x.numel() * MEASURE_ITERS) / elapsed,
            "peak_vram_gib": peak,
            "last_loss": measured_loss,
            "warmup_last_loss": warm_loss,
        }
    except torch.cuda.OutOfMemoryError as exc:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return {
            "status": "OOM",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _condition_for(variant_result: dict[str, Any], length: int) -> dict[str, Any] | None:
    for row in variant_result.get("contexts", []):
        if int(row.get("context_length", -1)) == length:
            return row
    return None


def _complete_train(row: dict[str, Any] | None) -> bool:
    return bool(row and (row.get("train") or {}).get("status") == "COMPLETE")


def _classify(results: dict[str, Any]) -> dict[str, Any]:
    transformer = results["transformer"]
    cortex = results["full_cortex"]
    t_512 = _condition_for(transformer, 512)
    c_512 = _condition_for(cortex, 512)
    c_8192 = _condition_for(cortex, 8_192)
    t_8192 = _condition_for(transformer, 8_192)

    long_pass = False
    long_reason = ""
    high_length: int | None = None
    high_tps_ratio: float | None = None
    high_mem_ratio: float | None = None
    ratio_512: float | None = None
    trend_delta: float | None = None

    if not (_complete_train(c_8192) and _complete_train(t_8192)):
        long_reason = "both Transformer and full CORTEX must complete through 8192"
    else:
        # A Transformer OOM at 16K while CORTEX completes is a direct preregistered pass.
        t_16 = _condition_for(transformer, 16_384)
        c_16 = _condition_for(cortex, 16_384)
        if (
            c_16
            and (c_16.get("train") or {}).get("status") == "COMPLETE"
            and t_16
            and (t_16.get("train") or {}).get("status") == "OOM"
        ):
            high_length = 16_384
            long_pass = True
            long_reason = "Transformer OOM at 16K while full CORTEX completed"
        else:
            for length in (16_384, 8_192):
                t_row = _condition_for(transformer, length)
                c_row = _condition_for(cortex, length)
                if _complete_train(t_row) and _complete_train(c_row):
                    high_length = length
                    t_train = t_row["train"]
                    c_train = c_row["train"]
                    high_tps_ratio = float(c_train["tokens_per_second"]) / float(
                        t_train["tokens_per_second"]
                    )
                    high_mem_ratio = float(c_train["peak_vram_gib"]) / float(
                        t_train["peak_vram_gib"]
                    )
                    break
            if high_length is None:
                long_reason = "no common completed 8K/16K train condition"
            elif not (_complete_train(t_512) and _complete_train(c_512)):
                long_reason = "512 baseline condition incomplete"
            else:
                ratio_512 = float(c_512["train"]["tokens_per_second"]) / float(
                    t_512["train"]["tokens_per_second"]
                )
                trend_delta = high_tps_ratio - ratio_512
                advantage = (
                    high_tps_ratio >= 1.10 and high_mem_ratio <= 1.10
                ) or (
                    high_mem_ratio <= 0.80 and high_tps_ratio >= 0.90
                )
                trend = trend_delta >= 0.10
                long_pass = bool(advantage and trend)
                long_reason = (
                    f"high-context advantage={advantage}; throughput-ratio trend={trend}"
                )

    # Ablation dominance is evaluated at the highest high context where full CORTEX
    # completes. OOM in a simpler ablation cannot make it a dominating alternative.
    full_high_length = None
    for length in (16_384, 8_192):
        row = _condition_for(cortex, length)
        if _complete_train(row):
            full_high_length = length
            break

    dominated_by: list[str] = []
    dominance_details: dict[str, Any] = {}
    if full_high_length is not None:
        full_row = _condition_for(cortex, full_high_length)
        assert full_row is not None
        full_train = full_row["train"]
        for name in ("world_only", "reduced_attention_only", "moe_only"):
            row = _condition_for(results[name], full_high_length)
            if not _complete_train(row):
                dominance_details[name] = {"status": "NOT_COMPLETE_AT_FULL_HIGH_CONTEXT"}
                continue
            train = row["train"]
            tps_ratio = float(train["tokens_per_second"]) / float(
                full_train["tokens_per_second"]
            )
            mem_ratio = float(train["peak_vram_gib"]) / float(
                full_train["peak_vram_gib"]
            )
            dominates = tps_ratio >= 1.10 and mem_ratio <= 0.90
            dominance_details[name] = {
                "status": "COMPLETE",
                "throughput_ratio_vs_full_cortex": tps_ratio,
                "memory_ratio_vs_full_cortex": mem_ratio,
                "materially_dominates": dominates,
            }
            if dominates:
                dominated_by.append(name)

    ablation_pass = bool(full_high_length is not None and not dominated_by)
    progression = bool(long_pass and ablation_pass)
    return {
        "long_context": {
            "classification": (
                "LONG_CONTEXT_SYSTEMS_PASS"
                if long_pass
                else "LONG_CONTEXT_SYSTEMS_FAIL_OR_NO_CLEAR_ADVANTAGE"
            ),
            "pass": long_pass,
            "reason": long_reason,
            "highest_common_high_context": high_length,
            "throughput_ratio_high": high_tps_ratio,
            "memory_ratio_high": high_mem_ratio,
            "throughput_ratio_512": ratio_512,
            "throughput_ratio_trend_delta": trend_delta,
        },
        "ablation": {
            "classification": (
                "ABLATION_SYSTEMS_PASS"
                if ablation_pass
                else "ABLATION_SYSTEMS_FAIL_FULL_CORTEX_DOMINATED_OR_INCOMPLETE"
            ),
            "pass": ablation_pass,
            "full_cortex_high_context": full_high_length,
            "dominated_by": dominated_by,
            "details": dominance_details,
        },
        "progression": {
            "classification": (
                "PREPARE_250M_5B_PREREG_ALLOWED"
                if progression
                else "DO_NOT_SCALE_THIS_EXACT_DESIGN_YET"
            ),
            "allowed": progression,
            "250m_training_authorized": False,
        },
    }


def _run_systems_benchmark() -> dict[str, Any]:
    import torch

    from architectures.cortex_s.long_context_ablation_v1 import (
        VARIANT_ORDER,
        architecture_contract,
        build_variant,
        expected_parameter_count,
        parameter_count,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("long-context systems benchmark requires CUDA")
    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    if capability[0] < 9:
        raise RuntimeError(
            f"preregistered systems benchmark requires H100/SM90-class CUDA, got {capability}"
        )
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(SYNTHETIC_SEED)

    contract = architecture_contract()
    results: dict[str, Any] = {}
    for variant_index, name in enumerate(VARIANT_ORDER):
        # Deterministic initialization per variant is not scientific evidence, but
        # makes repeated source inspection reproducible. The dispatch itself remains
        # single-use and must never be rerun for performance evidence.
        torch.manual_seed(SYNTHETIC_SEED + variant_index)
        model = build_variant(name)
        actual_parameters = parameter_count(model)
        expected_parameters = expected_parameter_count(name)
        if actual_parameters != expected_parameters:
            raise RuntimeError(
                f"{name} actual parameter count {actual_parameters} != expected {expected_parameters}"
            )
        model = model.to(device)
        variant_rows: list[dict[str, Any]] = []
        for length, batch in CONTEXT_BATCHES:
            if length * batch != TOKENS_PER_BATCH:
                raise RuntimeError("fixed-token geometry drift")
            x, y = _synthetic_batch(length, batch, device)
            forward = _forward_measurement(model, x)
            if forward.get("status") == "OOM":
                train = {"status": "SKIPPED_AFTER_FORWARD_OOM"}
            else:
                train = _train_measurement(model, x, y)
            variant_rows.append(
                {
                    "context_length": length,
                    "batch_size": batch,
                    "tokens_per_batch": x.numel(),
                    "forward": forward,
                    "train": train,
                }
            )
            del x, y
            gc.collect()
            torch.cuda.empty_cache()
        results[name] = {
            "actual_parameters": actual_parameters,
            "contexts": variant_rows,
        }
        del model
        gc.collect()
        torch.cuda.empty_cache()

    classification = _classify(results)
    return {
        "status": "SYSTEMS_BENCHMARK_COMPLETE",
        "classification": "ENGINEERING_CORTEX_S_LONG_CONTEXT_ABLATION_FALSIFICATION_V1",
        "device": torch.cuda.get_device_name(device),
        "device_capability": list(capability),
        "synthetic_seed": SYNTHETIC_SEED,
        "context_batches": [
            {"context_length": length, "batch_size": batch}
            for length, batch in CONTEXT_BATCHES
        ],
        "tokens_per_batch": TOKENS_PER_BATCH,
        "warmup_iterations": WARMUP_ITERS,
        "measure_iterations": MEASURE_ITERS,
        "execution": "eager_bf16_autocast_identical_explicit_fp32_ce",
        "architecture_contract": contract,
        "results": results,
        "decision": classification,
        "training_performed": False,
        "optimizer_constructed": False,
        "weights_updated": False,
        "trained_checkpoint_used": False,
        "scientific_claim_authorized": False,
        "breakthrough_claim_allowed": False,
    }


@app.function(
    image=image,
    gpu="H100!",
    cpu=8,
    memory=32768,
    timeout=90 * 60,
    volumes={"/vol": volume},
    secrets=[github_secret],
)
def run_benchmark(
    source_sha: str,
    source_tree: str,
    harness_sha: str,
    repo_full_name: str = "",
    issue_number: int = 0,
) -> dict[str, Any]:
    source = _validate_sha(source_sha, "source_sha")
    tree = _validate_sha(source_tree, "source_tree")
    harness = _validate_sha(harness_sha, "harness_sha")
    volume.reload()
    marker_path = Path(DISPATCH_PATH)
    if not marker_path.exists():
        raise RuntimeError("durable systems dispatch marker is missing")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    for key, expected in {
        "status": "SYSTEMS_DISPATCH_CONSUMED",
        "source_sha": source,
        "source_tree": tree,
        "harness_sha": harness,
        "synthetic_seed": SYNTHETIC_SEED,
    }.items():
        if marker.get(key) != expected:
            raise RuntimeError(f"systems dispatch marker mismatch for {key}")
    if Path(RESULT_PATH).exists():
        raise RuntimeError("systems RESULT.json already exists; no retry allowed")

    marker["gpu_allocation_started"] = True
    marker["gpu_allocation_unix"] = time.time()
    _atomic_write(marker_path, marker)
    volume.commit()
    _comment(
        repo_full_name,
        issue_number,
        "🧪 **CORTEX-S long-context + ablation systems benchmark started on one H100** — contexts 512→16K, fixed 16,384 tokens/step, five parameter-matched variants.",
    )

    try:
        payload = _run_systems_benchmark()
        payload.update(
            {
                "source_sha": source,
                "source_tree": tree,
                "harness_sha": harness,
                "trigger_title": TRIGGER_TITLE,
                "result_path": RESULT_PATH,
            }
        )
    except Exception as exc:
        payload = {
            "status": "SYSTEMS_BENCHMARK_ERROR",
            "classification": "ENGINEERING_INFRASTRUCTURE_OR_HARNESS_ERROR_NO_RETRY",
            "source_sha": source,
            "source_tree": tree,
            "harness_sha": harness,
            "trigger_title": TRIGGER_TITLE,
            "synthetic_seed": SYNTHETIC_SEED,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "training_performed": False,
            "scientific_claim_authorized": False,
            "breakthrough_claim_allowed": False,
        }

    _atomic_write(Path(RESULT_PATH), payload)
    volume.commit()

    decision = payload.get("decision") or {}
    long_result = decision.get("long_context") or {}
    ablation_result = decision.get("ablation") or {}
    progression = decision.get("progression") or {}
    if payload.get("status") == "SYSTEMS_BENCHMARK_COMPLETE":
        _comment(
            repo_full_name,
            issue_number,
            "✅ **CORTEX-S long-context systems benchmark complete.**\n\n"
            f"- long-context: `{long_result.get('classification')}`\n"
            f"- ablation: `{ablation_result.get('classification')}`\n"
            f"- progression: `{progression.get('classification')}`\n"
            f"- durable JSON: `{RESULT_PATH}`",
        )
    else:
        _comment(
            repo_full_name,
            issue_number,
            "🟥 **CORTEX-S long-context systems attempt ended in infrastructure/harness error.** "
            "The one-shot dispatch is consumed; do not reinterpret this as architecture evidence or retry it.",
        )
    print(json.dumps(payload, sort_keys=True), flush=True)
    return payload


@app.local_entrypoint()
def main(
    source_sha: str = "",
    source_tree: str = "",
    harness_sha: str = "",
    repo_full_name: str = "",
    issue_number: int = 0,
):
    print(
        reserve_dispatch.remote(
            source_sha,
            source_tree,
            harness_sha,
            repo_full_name,
            issue_number,
        ),
        flush=True,
    )
    result = run_benchmark.remote(
        source_sha,
        source_tree,
        harness_sha,
        repo_full_name,
        issue_number,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
