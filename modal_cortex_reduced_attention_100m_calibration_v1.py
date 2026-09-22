from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable

import modal


APP_NAME = "cortex-reduced-attention-100m-calibration-v1"
VOLUME_NAME = "tam-research-data"
TRIGGER_TITLE = "[modal-cortex-reduced-attention-100m-calibration-v1]"
ENGINEERING_SEED = 2_026_092_001
RESULT_ROOT = "/vol/cortex-s-v0/100m-2b/reduced-attention-calibration-v1"
ZERO_GPU_PATH = f"{RESULT_ROOT}/ZERO_GPU_GATE.json"
DISPATCH_PATH = f"{RESULT_ROOT}/H100_DISPATCH_CONSUMED.json"
RESULT_PATH = f"{RESULT_ROOT}/RESULT.json"
DATA_DIR = "/vol/data/tam100m-2b-curated-v1"

TRAIN_SHA256 = "93e9cb0b7076a4ddd855fc696f657ea62592b8a03a05220be99c402f9043265b"
VAL_SHA256 = "ae0bc5adf36d0aa8e55e5e3903401d3f114b93037f43221944b4e88c5d1a5760"
META_SHA256 = "14bbbcf0ab0b8cba374074ef8ccb80a04ecead06c74780f35a1becd5aef1b8f3"

EXPECTED_TRANSFORMER_PARAMS = 101_803_520
EXPECTED_REDUCED_ATTENTION_PARAMS = 101_799_424

SEQ_LEN = 512
MICRO_BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 2
TOKENS_PER_STEP = SEQ_LEN * MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS
TRAIN_TOKENS = 2_000_000_000
TOTAL_OPTIMIZER_STEPS = 30_518
WARMUP_RATIO = 0.02
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
ADAMW_BETAS = (0.9, 0.95)
GRAD_CLIP = 1.0

CALIBRATION_WARMUP_STEPS = 5
MEASURED_STEPS = 40
MEASURED_TOKENS = MEASURED_STEPS * TOKENS_PER_STEP
COMPILE_MODE = "max-autotune-no-cudagraphs"
PROJECTION_OVERHEAD_MULTIPLIER = 1.10
HARD_FULL_TIMEOUT_SECONDS = 10_000.0
MAX_PREFLIGHT_VRAM_GIB = 70.0

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
github_secret = modal.Secret.from_name("github-secret")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.10,<2.11",
        "numpy>=2.0,<3",
        "PyGithub>=2.3,<3",
    )
    .add_local_python_source("tam_research")
    .add_local_python_source("architectures")
)


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha(value: str, label: str) -> str:
    value = value.strip().lower()
    if len(value) != 40 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{label} must be a full lowercase SHA")
    return value


def _comment(repo_full_name: str, issue_number: int, body: str) -> None:
    if not repo_full_name or not issue_number:
        print(f"[status] {body}", flush=True)
        return
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print(f"[status] {body}", flush=True)
        return
    try:
        import github

        client = github.Github(auth=github.Auth.Token(token))
        client.get_repo(repo_full_name).get_issue(number=issue_number).create_comment(body)
    except Exception as exc:
        print(
            f"[status-report-nonfatal] {type(exc).__name__}: {exc}; body={body}",
            flush=True,
        )


def _verify_corpus() -> dict[str, Any]:
    root = Path(DATA_DIR)
    train_path = root / "train.bin"
    val_path = root / "val.bin"
    meta_path = root / "meta.json"
    missing = [str(p) for p in (train_path, val_path, meta_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"frozen corpus missing: {missing}")

    hashes = {
        "train_sha256": _sha256(train_path),
        "val_sha256": _sha256(val_path),
        "meta_sha256": _sha256(meta_path),
    }
    expected = {
        "train_sha256": TRAIN_SHA256,
        "val_sha256": VAL_SHA256,
        "meta_sha256": META_SHA256,
    }
    for key, value in expected.items():
        if hashes[key] != value:
            raise RuntimeError(f"frozen corpus fingerprint mismatch for {key}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for key, value in {
        "assembly_version": 3,
        "train_tokens": 2_000_000_000,
        "val_tokens": 5_000_000,
        "seed": 8100,
        "tokenizer": "gpt2",
        "dtype": "uint16",
    }.items():
        if meta.get(key) != value:
            raise RuntimeError(f"frozen corpus metadata mismatch for {key}")
    if train_path.stat().st_size != 4_000_000_000:
        raise RuntimeError("frozen train corpus byte-size drift")
    if val_path.stat().st_size != 10_000_000:
        raise RuntimeError("frozen validation corpus byte-size drift")

    return {**hashes, "metadata": meta}


def _build_transformer():
    from tam_research.models import ModelConfig, ResearchLM

    cfg = ModelConfig(
        vocab_size=50_257,
        d_model=512,
        n_layers=24,
        n_heads=16,
        max_seq_len=1_024,
        ff_mult=4,
        architecture="transformer",
    )
    return ResearchLM(cfg)


def _build_reduced_attention():
    from architectures.cortex_s.reduced_attention_100m_v1 import ReducedAttentionDenseLM

    return ReducedAttentionDenseLM()


def _count_parameters(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _verify_architecture_contract() -> dict[str, Any]:
    import torch
    from architectures.cortex_s.reduced_attention_100m_v1 import architecture_contract
    from architectures.cortex_s.reduced_attention_100m_v1_protocol import validate_protocol

    protocol = validate_protocol()
    if protocol.get("engineering_calibration_seed_reserved_not_consumed") != ENGINEERING_SEED:
        raise RuntimeError("engineering calibration seed contract drift")
    if protocol.get("training_authorized") is not False:
        raise RuntimeError("quality-training authority unexpectedly enabled")
    if protocol.get("gpu_authorized") is not False:
        raise RuntimeError("candidate protocol unexpectedly authorizes GPU")
    if protocol.get("scientific_seeds_consumed") is not False:
        raise RuntimeError("scientific seed contract drift")

    candidate_contract = architecture_contract()
    if candidate_contract.get("attention_layer_numbers_one_based") != [6, 12, 18, 24]:
        raise RuntimeError("reduced-attention schedule drift")
    if candidate_contract.get("world_state") is not False or candidate_contract.get("moe") is not False:
        raise RuntimeError("candidate component contract drift")

    with torch.device("meta"):
        transformer = _build_transformer()
        reduced = _build_reduced_attention()
    transformer_params = _count_parameters(transformer)
    reduced_params = _count_parameters(reduced)
    if transformer_params != EXPECTED_TRANSFORMER_PARAMS:
        raise RuntimeError(
            f"Transformer parameter drift: {transformer_params:,} != "
            f"{EXPECTED_TRANSFORMER_PARAMS:,}"
        )
    if reduced_params != EXPECTED_REDUCED_ATTENTION_PARAMS:
        raise RuntimeError(
            f"reduced-attention parameter drift: {reduced_params:,} != "
            f"{EXPECTED_REDUCED_ATTENTION_PARAMS:,}"
        )

    return {
        "transformer_parameters": transformer_params,
        "reduced_attention_parameters": reduced_params,
        "absolute_parameter_gap": abs(transformer_params - reduced_params),
        "candidate_contract": candidate_contract,
        "protocol_experiment_id": protocol.get("experiment_id"),
        "scientific_pair_seeds_reserved_not_consumed": protocol.get(
            "scientific_pair_seeds_reserved_not_consumed"
        ),
    }


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=30 * 60,
    volumes={"/vol": volume},
    secrets=[github_secret],
)
def verify_zero_gpu(
    source_sha: str,
    source_tree: str,
    harness_sha: str,
    repo_full_name: str = "",
    issue_number: int = 0,
) -> str:
    source = _validate_sha(source_sha, "source_sha")
    tree = _validate_sha(source_tree, "source_tree")
    harness = _validate_sha(harness_sha, "harness_sha")

    if TOKENS_PER_STEP != 65_536 or MEASURED_TOKENS != 2_621_440:
        raise RuntimeError("calibration geometry drift")
    if ENGINEERING_SEED in {8_100, 48_131, 48_132, 48_133, 58_231, 58_232, 58_233}:
        raise RuntimeError("engineering seed collides with forbidden/scientific seed")

    volume.reload()
    root = Path(RESULT_ROOT)
    if root.exists():
        raise RuntimeError("calibration result namespace already exists; fail closed")

    architecture = _verify_architecture_contract()
    corpus = _verify_corpus()
    payload = {
        "status": "PASS",
        "classification": "ZERO_GPU_REDUCED_ATTENTION_CALIBRATION_V1",
        "trigger_title": TRIGGER_TITLE,
        "source_sha": source,
        "source_tree": tree,
        "harness_sha": harness,
        "engineering_seed_reserved": ENGINEERING_SEED,
        "result_root": RESULT_ROOT,
        "gpu_allocated": False,
        "architecture": architecture,
        "corpus": corpus,
        "calibration_contract": {
            "sequence_length": SEQ_LEN,
            "micro_batch_size": MICRO_BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "tokens_per_optimizer_step": TOKENS_PER_STEP,
            "warmup_steps": CALIBRATION_WARMUP_STEPS,
            "measured_steps": MEASURED_STEPS,
            "measured_tokens_per_architecture": MEASURED_TOKENS,
            "compile_mode": COMPILE_MODE,
            "bf16_autocast": True,
            "explicit_fp32_cross_entropy": True,
            "fused_adamw": True,
            "projection_overhead_multiplier": PROJECTION_OVERHEAD_MULTIPLIER,
            "hard_full_timeout_seconds": HARD_FULL_TIMEOUT_SECONDS,
            "max_vram_gib": MAX_PREFLIGHT_VRAM_GIB,
        },
        "scientific_training_authorized": False,
        "scientific_seeds_consumed": False,
        "250m_5b_authorized": False,
    }
    _atomic_write(Path(ZERO_GPU_PATH), payload)
    volume.commit()
    _comment(
        repo_full_name,
        issue_number,
        "🟨 Reduced-attention calibration zero-GPU gate passed. No H100 allocation yet.",
    )
    return json.dumps(payload, sort_keys=True)


@app.function(
    image=image,
    cpu=1,
    memory=1024,
    timeout=5 * 60,
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

    zero_path = Path(ZERO_GPU_PATH)
    marker_path = Path(DISPATCH_PATH)
    result_path = Path(RESULT_PATH)
    if not zero_path.exists():
        raise RuntimeError("zero-GPU gate missing")
    zero = json.loads(zero_path.read_text(encoding="utf-8"))
    for key, expected in {
        "status": "PASS",
        "source_sha": source,
        "source_tree": tree,
        "harness_sha": harness,
        "engineering_seed_reserved": ENGINEERING_SEED,
    }.items():
        if zero.get(key) != expected:
            raise RuntimeError(f"zero-GPU gate mismatch for {key}")
    if marker_path.exists() or result_path.exists():
        raise RuntimeError("calibration attempt already consumed")

    marker = {
        "status": "H100_DISPATCH_CONSUMED",
        "classification": "ENGINEERING_REDUCED_ATTENTION_CALIBRATION_SINGLE_ATTEMPT",
        "trigger_title": TRIGGER_TITLE,
        "source_sha": source,
        "source_tree": tree,
        "harness_sha": harness,
        "engineering_seed": ENGINEERING_SEED,
        "result_root": RESULT_ROOT,
        "marked_unix": time.time(),
        "h100_allocation_started": False,
        "scientific_training_authorized": False,
        "scientific_seeds_consumed": False,
        "250m_5b_authorized": False,
    }
    _atomic_write(marker_path, marker)
    volume.commit()
    _comment(
        repo_full_name,
        issue_number,
        "🔒 Reduced-attention calibration dispatch consumed. Exactly one H100 calibration attempt; no retry.",
    )
    return json.dumps(marker, sort_keys=True)


def _one_optimizer_step(
    *,
    model: Any,
    forward_model: Any,
    optimizer: Any,
    train_data: Any,
    batch_generator: Any,
    device: Any,
    step_index: int,
) -> float:
    import torch
    import torch.nn.functional as F
    from tam_research.train import cosine_lr

    model.train()
    forward_model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = torch.zeros((), device=device, dtype=torch.float32)

    for _ in range(GRAD_ACCUM_STEPS):
        x, y = train_data.batch(
            MICRO_BATCH_SIZE,
            SEQ_LEN,
            batch_generator,
            device,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = forward_model(x)
            loss = (
                F.cross_entropy(
                    logits.float().reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                )
                / GRAD_ACCUM_STEPS
            )
        loss.backward()
        running_loss = running_loss + loss.detach().float() * GRAD_ACCUM_STEPS

    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    all_finite = torch.isfinite(running_loss)
    for parameter in model.parameters():
        if parameter.grad is not None:
            all_finite = all_finite & torch.isfinite(parameter.grad).all()

    host_report = torch.stack(
        (running_loss, all_finite.to(dtype=running_loss.dtype))
    ).to(device="cpu")
    running_value, finite_value = host_report.tolist()
    if finite_value != 1.0 or not math.isfinite(running_value):
        optimizer.zero_grad(set_to_none=True)
        raise RuntimeError("non-finite calibration step; optimizer update suppressed")

    warmup_steps = max(1, int(TOTAL_OPTIMIZER_STEPS * WARMUP_RATIO))
    lr = cosine_lr(
        step_index,
        TOTAL_OPTIMIZER_STEPS,
        warmup_steps,
        LEARNING_RATE,
    )
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.step()
    return float(running_value)


def _calibrate_one(
    *,
    name: str,
    builder: Callable[[], Any],
    train_data: Any,
    device: Any,
) -> dict[str, Any]:
    import torch
    from tam_research.train import seed_all

    seed_all(ENGINEERING_SEED)
    torch._dynamo.reset()
    model = builder().to(device)
    parameter_count = _count_parameters(model)
    expected = {
        "transformer": EXPECTED_TRANSFORMER_PARAMS,
        "reduced_attention": EXPECTED_REDUCED_ATTENTION_PARAMS,
    }[name]
    if parameter_count != expected:
        raise RuntimeError(
            f"{name} parameter drift: {parameter_count:,} != {expected:,}"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=ADAMW_BETAS,
        weight_decay=WEIGHT_DECAY,
        fused=True,
    )
    forward_model = torch.compile(
        model,
        mode=COMPILE_MODE,
        fullgraph=False,
    )

    compile_generator = torch.Generator(device="cpu").manual_seed(
        ENGINEERING_SEED + 99_999
    )
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    compile_started = time.perf_counter()
    compile_loss = _one_optimizer_step(
        model=model,
        forward_model=forward_model,
        optimizer=optimizer,
        train_data=train_data,
        batch_generator=compile_generator,
        device=device,
        step_index=0,
    )
    torch.cuda.synchronize(device)
    compile_seconds = max(time.perf_counter() - compile_started, 0.0)
    compile_peak_vram_gib = torch.cuda.max_memory_allocated(device) / (1024**3)

    warm_generator = torch.Generator(device="cpu").manual_seed(
        ENGINEERING_SEED + 90_000
    )
    warmup_losses: list[float] = []
    for step in range(CALIBRATION_WARMUP_STEPS):
        warmup_losses.append(
            _one_optimizer_step(
                model=model,
                forward_model=forward_model,
                optimizer=optimizer,
                train_data=train_data,
                batch_generator=warm_generator,
                device=device,
                step_index=step + 1,
            )
        )

    measured_generator = torch.Generator(device="cpu").manual_seed(
        ENGINEERING_SEED + 10_000
    )
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    measured_started = time.perf_counter()
    last_loss = float("nan")
    for step in range(MEASURED_STEPS):
        last_loss = _one_optimizer_step(
            model=model,
            forward_model=forward_model,
            optimizer=optimizer,
            train_data=train_data,
            batch_generator=measured_generator,
            device=device,
            step_index=step + CALIBRATION_WARMUP_STEPS + 1,
        )
    torch.cuda.synchronize(device)
    measured_seconds = max(time.perf_counter() - measured_started, 1e-9)
    peak_vram_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
    tokens_per_second = MEASURED_TOKENS / measured_seconds
    projected_training_seconds = TRAIN_TOKENS / tokens_per_second
    projected_full_seconds = (
        projected_training_seconds * PROJECTION_OVERHEAD_MULTIPLIER
        + compile_seconds
    )

    result = {
        "status": "COMPLETE",
        "architecture": name,
        "parameter_count": parameter_count,
        "compile_mode": COMPILE_MODE,
        "compile_seconds": compile_seconds,
        "compile_peak_vram_gib": compile_peak_vram_gib,
        "compile_step_loss": compile_loss,
        "warmup_steps": CALIBRATION_WARMUP_STEPS,
        "warmup_last_loss": warmup_losses[-1],
        "measured_steps": MEASURED_STEPS,
        "measured_tokens": MEASURED_TOKENS,
        "measured_seconds": measured_seconds,
        "training_tokens_per_second": tokens_per_second,
        "peak_vram_gib": peak_vram_gib,
        "last_train_loss": last_loss,
        "projected_training_seconds": projected_training_seconds,
        "projection_overhead_multiplier": PROJECTION_OVERHEAD_MULTIPLIER,
        "projected_full_seconds": projected_full_seconds,
        "hard_timeout_seconds": HARD_FULL_TIMEOUT_SECONDS,
        "within_hard_timeout": projected_full_seconds <= HARD_FULL_TIMEOUT_SECONDS,
        "within_vram_gate": peak_vram_gib <= MAX_PREFLIGHT_VRAM_GIB,
        "engineering_seed": ENGINEERING_SEED,
        "scientific_seed_used": False,
    }

    del forward_model, optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    torch._dynamo.reset()
    return result


@app.function(
    image=image,
    gpu="H100!",
    cpu=8,
    memory=32768,
    timeout=45 * 60,
    volumes={"/vol": volume},
    secrets=[github_secret],
)
def h100_calibration(
    source_sha: str,
    source_tree: str,
    harness_sha: str,
    repo_full_name: str = "",
    issue_number: int = 0,
) -> str:
    import torch
    from tam_research.data import TokenBin

    source = _validate_sha(source_sha, "source_sha")
    tree = _validate_sha(source_tree, "source_tree")
    harness = _validate_sha(harness_sha, "harness_sha")
    volume.reload()

    marker_path = Path(DISPATCH_PATH)
    result_path = Path(RESULT_PATH)
    if not marker_path.exists():
        raise RuntimeError("durable H100 dispatch marker missing")
    if result_path.exists():
        raise RuntimeError("calibration RESULT.json already exists; no retry")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    for key, expected in {
        "status": "H100_DISPATCH_CONSUMED",
        "source_sha": source,
        "source_tree": tree,
        "harness_sha": harness,
        "engineering_seed": ENGINEERING_SEED,
        "h100_allocation_started": False,
    }.items():
        if marker.get(key) != expected:
            raise RuntimeError(f"dispatch marker mismatch for {key}")

    if not torch.cuda.is_available():
        raise RuntimeError("calibration requires CUDA")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device)[0] < 9:
        raise RuntimeError("calibration requires H100/SM90-class CUDA")
    torch.set_float32_matmul_precision("high")

    marker["h100_allocation_started"] = True
    marker["h100_function_started_unix"] = time.time()
    _atomic_write(marker_path, marker)
    volume.commit()
    _comment(
        repo_full_name,
        issue_number,
        "🧪 Reduced-attention 100M calibration started on one H100: Transformer then candidate; no scientific seed or checkpoint.",
    )

    partial: dict[str, Any] = {}
    try:
        _verify_architecture_contract()
        train_data = TokenBin(str(Path(DATA_DIR) / "train.bin"))

        transformer = _calibrate_one(
            name="transformer",
            builder=_build_transformer,
            train_data=train_data,
            device=device,
        )
        partial["transformer"] = transformer

        reduced = _calibrate_one(
            name="reduced_attention",
            builder=_build_reduced_attention,
            train_data=train_data,
            device=device,
        )
        partial["reduced_attention"] = reduced

        calibration_pass = (
            transformer["status"] == "COMPLETE"
            and reduced["status"] == "COMPLETE"
            and math.isfinite(float(transformer["last_train_loss"]))
            and math.isfinite(float(reduced["last_train_loss"]))
        )
        pair1_preparation_allowed = (
            calibration_pass
            and transformer["within_hard_timeout"]
            and reduced["within_hard_timeout"]
            and transformer["within_vram_gate"]
            and reduced["within_vram_gate"]
        )
        throughput_ratio = (
            reduced["training_tokens_per_second"]
            / transformer["training_tokens_per_second"]
        )
        vram_ratio = reduced["peak_vram_gib"] / transformer["peak_vram_gib"]

        payload = {
            "status": (
                "CALIBRATION_PASS" if calibration_pass else "CALIBRATION_FAIL"
            ),
            "classification": "ENGINEERING_REDUCED_ATTENTION_100M_CALIBRATION_V1",
            "source_sha": source,
            "source_tree": tree,
            "harness_sha": harness,
            "trigger_title": TRIGGER_TITLE,
            "engineering_seed": ENGINEERING_SEED,
            "device": torch.cuda.get_device_name(device),
            "device_capability": list(torch.cuda.get_device_capability(device)),
            "order": ["transformer", "reduced_attention"],
            "results": partial,
            "comparison": {
                "candidate_over_transformer_throughput_ratio": throughput_ratio,
                "candidate_over_transformer_vram_ratio": vram_ratio,
                "candidate_minus_transformer_projected_full_seconds": (
                    reduced["projected_full_seconds"]
                    - transformer["projected_full_seconds"]
                ),
            },
            "decision": {
                "calibration_pass": calibration_pass,
                "pair1_preparation_allowed": pair1_preparation_allowed,
                "classification": (
                    "PAIR1_PREPARATION_ALLOWED"
                    if pair1_preparation_allowed
                    else "PAIR1_PREPARATION_BLOCKED"
                ),
            },
            "training_performed": False,
            "checkpoint_written": False,
            "scientific_training_authorized": False,
            "scientific_seeds_consumed": False,
            "250m_5b_authorized": False,
            "breakthrough_claim_allowed": False,
            "result_path": RESULT_PATH,
        }
    except Exception as exc:
        payload = {
            "status": "CALIBRATION_ERROR",
            "classification": "ENGINEERING_REDUCED_ATTENTION_CALIBRATION_ERROR_NO_RETRY",
            "source_sha": source,
            "source_tree": tree,
            "harness_sha": harness,
            "trigger_title": TRIGGER_TITLE,
            "engineering_seed": ENGINEERING_SEED,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "partial_results": partial,
            "training_performed": False,
            "checkpoint_written": False,
            "scientific_training_authorized": False,
            "scientific_seeds_consumed": False,
            "250m_5b_authorized": False,
            "breakthrough_claim_allowed": False,
            "result_path": RESULT_PATH,
        }

    _atomic_write(result_path, payload)
    volume.commit()

    decision = payload.get("decision") or {}
    if payload.get("status") == "CALIBRATION_PASS":
        _comment(
            repo_full_name,
            issue_number,
            "✅ Reduced-attention calibration complete. "
            f"Pair-1 preparation allowed: {decision.get('pair1_preparation_allowed')}. "
            f"Durable result: {RESULT_PATH}",
        )
    else:
        _comment(
            repo_full_name,
            issue_number,
            "🟥 Reduced-attention calibration attempt ended without CALIBRATION_PASS. "
            "The one-shot attempt is consumed; do not retry or interpret an error as architecture evidence.",
        )

    print(json.dumps(payload, sort_keys=True), flush=True)
    return json.dumps(payload, sort_keys=True)


@app.local_entrypoint()
def main(
    source_sha: str = "",
    source_tree: str = "",
    harness_sha: str = "",
    repo_full_name: str = "",
    issue_number: int = 0,
) -> None:
    source = _validate_sha(source_sha, "source_sha")
    tree = _validate_sha(source_tree, "source_tree")
    harness = _validate_sha(harness_sha, "harness_sha")

    zero = json.loads(
        verify_zero_gpu.remote(
            source,
            tree,
            harness,
            repo_full_name,
            issue_number,
        )
    )
    print(json.dumps({"zero_gpu": zero}, sort_keys=True), flush=True)
    if zero.get("status") != "PASS":
        raise RuntimeError("zero-GPU calibration gate did not pass")

    reservation = json.loads(
        reserve_dispatch.remote(
            source,
            tree,
            harness,
            repo_full_name,
            issue_number,
        )
    )
    print(json.dumps({"dispatch": reservation}, sort_keys=True), flush=True)
    if reservation.get("status") != "H100_DISPATCH_CONSUMED":
        raise RuntimeError("H100 calibration reservation was not durably committed")

    result = json.loads(
        h100_calibration.remote(
            source,
            tree,
            harness,
            repo_full_name,
            issue_number,
        )
    )
    print(json.dumps({"h100": result}, sort_keys=True), flush=True)
