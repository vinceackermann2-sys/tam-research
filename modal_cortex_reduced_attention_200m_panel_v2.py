from __future__ import annotations

import gc
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable

import modal


PHASE = "reduced-attention-v2-100m-200m-engineering-panel"
TRIGGER_TITLE = "[modal-cortex-reduced-attention-200m-engineering-panel-v2]"
ENGINEERING_SEED = 2_026_092_901
RESULT_ROOT = "/vol/cortex-s-v0/100m-200m/reduced-attention-v2-panel-v1"
DATA_DIR = "/vol/data/tam100m-2b-curated-v1"

TRAIN_SHA256 = "93e9cb0b7076a4ddd855fc696f657ea62592b8a03a05220be99c402f9043265b"
VAL_SHA256 = "ae0bc5adf36d0aa8e55e5e3903401d3f114b93037f43221944b4e88c5d1a5760"
META_SHA256 = "14bbbcf0ab0b8cba374074ef8ccb80a04ecead06c74780f35a1becd5aef1b8f3"

MODEL_ORDER = (
    "fresh_transformer",
    "v1_replicate",
    "early_4",
    "attention_8",
)
EXPECTED_PARAMETERS = {
    "fresh_transformer": 101_803_520,
    "v1_replicate": 101_799_424,
    "early_4": 101_799_424,
    "attention_8": 101_795_328,
}

SEQ_LEN = 512
MICRO_BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 2
TOKENS_PER_STEP = 65_536
TOTAL_OPTIMIZER_STEPS = 3_052
FULL_BATCH_TOKEN_EXPOSURES = 200_015_872
LEARNING_RATE = 3e-4
ADAMW_BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.02
GRAD_CLIP = 1.0
EVAL_EXPOSURES_TOKENS = (
    50_003_968,
    100_007_936,
    150_011_904,
    200_015_872,
)
PERIODIC_EVAL_BATCHES = 20
FINAL_EVAL_BATCHES = 50
EVAL_BATCH_SIZE = 32
COMPILE_MODE = "max-autotune-no-cudagraphs"
HARD_MODEL_TIMEOUT_SECONDS = 2_400

MAX_200M_NLL_DELTA = 0.025
MIN_TRAINING_TPS_RATIO = 1.05
PAIR1_V1_200M_NLL_DELTA_REFERENCE = 0.1224118590354917

APP_NAME = "cortex-reduced-attention-200m-engineering-panel-v2"
VOLUME_NAME = "tam-research-data"
REQUIRED_MODAL_ACCOUNT = "secondary"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11", "numpy>=2.0,<3")
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


def _full_sha(value: str, label: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 40 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{label} must be a full lowercase SHA")
    return normalized


def _validate_payload(
    source_sha: str,
    source_tree: str,
    harness_sha: str,
) -> tuple[str, str, str]:
    source = _full_sha(source_sha, "source_sha")
    tree = _full_sha(source_tree, "source_tree")
    harness = _full_sha(harness_sha, "harness_sha")
    if TOKENS_PER_STEP != SEQ_LEN * MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS:
        raise RuntimeError("tokens-per-step contract drift")
    if TOTAL_OPTIMIZER_STEPS * TOKENS_PER_STEP != FULL_BATCH_TOKEN_EXPOSURES:
        raise RuntimeError("full-batch exposure contract drift")
    if FULL_BATCH_TOKEN_EXPOSURES != 200_015_872:
        raise RuntimeError("200M engineering exposure drift")
    return source, tree, harness


def _count_parameters(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _builder_for(name: str) -> Callable[[], Any]:
    from architectures.cortex_s.reduced_attention_200m_panel_v2 import (
        build_fresh_transformer,
        build_panel_variant,
    )

    if name == "fresh_transformer":
        return build_fresh_transformer
    if name in {"v1_replicate", "early_4", "attention_8"}:
        return lambda: build_panel_variant(name)
    raise ValueError(f"unknown engineering-panel model: {name}")


def _verify_model_and_protocol_contract() -> dict[str, Any]:
    import torch
    from architectures.cortex_s.reduced_attention_200m_panel_v2 import (
        architecture_contract,
    )
    from architectures.cortex_s.reduced_attention_200m_panel_v2_protocol import (
        ENGINEERING_SEED as PROTOCOL_ENGINEERING_SEED,
        validate_protocol,
    )

    protocol = validate_protocol()
    if PROTOCOL_ENGINEERING_SEED != ENGINEERING_SEED:
        raise RuntimeError("engineering seed drift")
    if protocol.get("engineering_seed_reserved_not_consumed") != ENGINEERING_SEED:
        raise RuntimeError("engineering seed reservation drift")
    for key in (
        "engineering_panel_training_authorized",
        "gpu_authorized",
        "scientific_training_authorized",
        "scientific_seeds_authorized",
        "replication_seeds_authorized",
        "250m_5b_authorized",
        "breakthrough_claim_allowed",
    ):
        if protocol.get(key) is not False:
            raise RuntimeError(f"protocol unexpectedly authorizes {key}")

    contract = architecture_contract()
    if contract.get("world_state") is not False:
        raise RuntimeError("world-state drift")
    if contract.get("moe") is not False or contract.get("router") is not False:
        raise RuntimeError("sparse/router drift")

    counts: dict[str, int] = {}
    with torch.device("meta"):
        for name in MODEL_ORDER:
            model = _builder_for(name)()
            count = _count_parameters(model)
            if count != EXPECTED_PARAMETERS[name]:
                raise RuntimeError(
                    f"{name} parameter drift: {count:,} != {EXPECTED_PARAMETERS[name]:,}"
                )
            counts[name] = count

    return {
        "protocol": protocol,
        "architecture": contract,
        "parameter_counts": counts,
    }


def _verify_corpus() -> dict[str, Any]:
    root = Path(DATA_DIR)
    train = root / "train.bin"
    val = root / "val.bin"
    meta_path = root / "meta.json"
    missing = [str(path) for path in (train, val, meta_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"frozen corpus missing: {missing}")

    hashes = {
        "train_sha256": _sha256(train),
        "val_sha256": _sha256(val),
        "meta_sha256": _sha256(meta_path),
    }
    for key, expected in {
        "train_sha256": TRAIN_SHA256,
        "val_sha256": VAL_SHA256,
        "meta_sha256": META_SHA256,
    }.items():
        if hashes[key] != expected:
            raise RuntimeError(f"frozen corpus fingerprint mismatch for {key}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for key, expected in {
        "assembly_version": 3,
        "train_tokens": 2_000_000_000,
        "val_tokens": 5_000_000,
        "seed": 8100,
        "tokenizer": "gpt2",
        "dtype": "uint16",
    }.items():
        if meta.get(key) != expected:
            raise RuntimeError(f"frozen corpus metadata mismatch for {key}")
    if train.stat().st_size != 4_000_000_000:
        raise RuntimeError("frozen train corpus byte-size drift")
    if val.stat().st_size != 10_000_000:
        raise RuntimeError("frozen validation corpus byte-size drift")
    return {**hashes, "metadata": meta}


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=20 * 60,
    volumes={"/vol": volume},
)
def verify_zero_gpu(
    source_sha: str,
    source_tree: str,
    harness_sha: str,
    selected_account: str,
) -> str:
    source, tree, harness = _validate_payload(source_sha, source_tree, harness_sha)
    account = selected_account.strip().lower()
    if account != REQUIRED_MODAL_ACCOUNT:
        raise RuntimeError(
            f"engineering panel requires Modal account {REQUIRED_MODAL_ACCOUNT!r}"
        )

    volume.reload()
    root = Path(RESULT_ROOT)
    if root.exists():
        raise RuntimeError("engineering-panel result namespace already exists; fail closed")

    model_contract = _verify_model_and_protocol_contract()
    corpus = _verify_corpus()

    payload = {
        "status": "PASS",
        "classification": "ZERO_GPU_REDUCED_ATTENTION_V2_200M_ENGINEERING_PANEL",
        "phase": PHASE,
        "trigger_title": TRIGGER_TITLE,
        "source_sha": source,
        "source_tree": tree,
        "harness_sha": harness,
        "selected_modal_account": account,
        "engineering_seed_reserved_not_consumed": ENGINEERING_SEED,
        "result_root": RESULT_ROOT,
        "model_order": list(MODEL_ORDER),
        "model_contract": model_contract,
        "corpus": corpus,
        "training_contract": {
            "full_batch_token_exposures": FULL_BATCH_TOKEN_EXPOSURES,
            "optimizer_steps": TOTAL_OPTIMIZER_STEPS,
            "seq_len": SEQ_LEN,
            "micro_batch_size": MICRO_BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "tokens_per_step": TOKENS_PER_STEP,
            "learning_rate": LEARNING_RATE,
            "adamw_betas": list(ADAMW_BETAS),
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "grad_clip": GRAD_CLIP,
            "eval_exposures_tokens": list(EVAL_EXPOSURES_TOKENS),
            "periodic_eval_batches": PERIODIC_EVAL_BATCHES,
            "final_eval_batches": FINAL_EVAL_BATCHES,
            "compile_mode": COMPILE_MODE,
        },
        "progression_gate": {
            "candidate_minus_fresh_transformer_final_nll_max": MAX_200M_NLL_DELTA,
            "candidate_over_fresh_transformer_training_tps_min": MIN_TRAINING_TPS_RATIO,
            "v1_replicate_seed_sensitivity_guard": True,
        },
        "gpu_allocated": False,
        "engineering_seed_consumed": False,
        "scientific_training_authorized": False,
        "scientific_seed_consumed": False,
        "replication_authorized": False,
        "250m_5b_authorized": False,
        "breakthrough_claim_allowed": False,
    }
    _atomic_write(root / "ZERO_GPU_GATE.json", payload)
    volume.commit()
    return json.dumps(payload, sort_keys=True)


@app.function(
    image=image,
    cpu=1,
    memory=1024,
    timeout=10 * 60,
    volumes={"/vol": volume},
)
def reserve_panel_dispatch(
    source_sha: str,
    source_tree: str,
    harness_sha: str,
    selected_account: str,
) -> str:
    source, tree, harness = _validate_payload(source_sha, source_tree, harness_sha)
    account = selected_account.strip().lower()
    if account != REQUIRED_MODAL_ACCOUNT:
        raise RuntimeError(
            f"engineering panel requires Modal account {REQUIRED_MODAL_ACCOUNT!r}"
        )
    volume.reload()
    root = Path(RESULT_ROOT)
    gate_path = root / "ZERO_GPU_GATE.json"
    marker_path = root / "ENGINEERING_PANEL_DISPATCH_RESERVED.json"
    result_path = root / "RESULT.json"
    if not gate_path.exists():
        raise RuntimeError("engineering-panel zero-GPU gate is missing")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    for key, expected in {
        "status": "PASS",
        "source_sha": source,
        "source_tree": tree,
        "harness_sha": harness,
        "selected_modal_account": account,
        "engineering_seed_reserved_not_consumed": ENGINEERING_SEED,
    }.items():
        if gate.get(key) != expected:
            raise RuntimeError(f"engineering-panel zero-GPU gate mismatch for {key}")
    if marker_path.exists() or result_path.exists():
        raise RuntimeError("engineering-panel dispatch/result already consumed")

    marker = {
        "status": "ENGINEERING_PANEL_DISPATCH_RESERVED",
        "classification": "ENGINEERING_REDUCED_ATTENTION_V2_200M_SINGLE_ATTEMPT",
        "phase": PHASE,
        "trigger_title": TRIGGER_TITLE,
        "source_sha": source,
        "source_tree": tree,
        "harness_sha": harness,
        "selected_modal_account": account,
        "engineering_seed": ENGINEERING_SEED,
        "engineering_seed_consumed": True,
        "model_order": list(MODEL_ORDER),
        "result_root": RESULT_ROOT,
        "h100_allocation_started": False,
        "automatic_retry_authorized": False,
        "resume_authorized": False,
        "scientific_training_authorized": False,
        "scientific_seed_consumed": False,
        "replication_authorized": False,
        "250m_5b_authorized": False,
        "breakthrough_claim_allowed": False,
        "marked_unix": time.time(),
    }
    _atomic_write(marker_path, marker)
    volume.commit()
    return json.dumps(marker, sort_keys=True)


def _one_optimizer_step(
    *,
    model: Any,
    forward_model: Any,
    optimizer: Any,
    train_data: Any,
    generator: Any,
    device: Any,
    step_index: int,
) -> float:
    import torch
    import torch.nn.functional as F
    from tam_research.train import cosine_lr

    model.train()
    forward_model.train()
    optimizer.zero_grad(set_to_none=True)
    running = torch.zeros((), device=device, dtype=torch.float32)

    for _ in range(GRAD_ACCUM_STEPS):
        x, y = train_data.batch(
            MICRO_BATCH_SIZE,
            SEQ_LEN,
            generator,
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
        running = running + loss.detach().float() * GRAD_ACCUM_STEPS

    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    finite = torch.isfinite(running)
    for parameter in model.parameters():
        if parameter.grad is not None:
            finite = finite & torch.isfinite(parameter.grad).all()
    report = torch.stack((running, finite.to(dtype=running.dtype))).to(device="cpu")
    loss_value, finite_value = report.tolist()
    if finite_value != 1.0 or not math.isfinite(loss_value):
        optimizer.zero_grad(set_to_none=True)
        raise RuntimeError("non-finite engineering-panel training step")

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
    return float(loss_value)


def _evaluate(
    *,
    model: Any,
    forward_model: Any,
    val_data: Any,
    device: Any,
    batches: int,
    seed: int,
) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    model.eval()
    forward_model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses: list[float] = []
    with torch.no_grad():
        for _ in range(batches):
            x, y = val_data.batch(EVAL_BATCH_SIZE, SEQ_LEN, generator, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = forward_model(x)
                loss = F.cross_entropy(
                    logits.float().reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                )
            losses.append(float(loss))
    nll = sum(losses) / len(losses)
    return {"nll": nll, "perplexity": math.exp(min(nll, 20.0))}


def _train_model(
    *,
    name: str,
    source_sha: str,
) -> dict[str, Any]:
    import torch
    from tam_research.data import TokenBin
    from tam_research.train import seed_all

    if not torch.cuda.is_available():
        raise RuntimeError("engineering panel requires CUDA")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device)[0] < 9:
        raise RuntimeError("engineering panel requires H100/SM90-class CUDA")
    torch.set_float32_matmul_precision("high")

    builder = _builder_for(name)
    train_data = TokenBin(str(Path(DATA_DIR) / "train.bin"))
    val_data = TokenBin(str(Path(DATA_DIR) / "val.bin"))

    seed_all(ENGINEERING_SEED)
    torch._dynamo.reset()
    warm_model = builder().to(device)
    if _count_parameters(warm_model) != EXPECTED_PARAMETERS[name]:
        raise RuntimeError(f"{name} parameter drift before compile warmup")
    warm_optimizer = torch.optim.AdamW(
        warm_model.parameters(),
        lr=LEARNING_RATE,
        betas=ADAMW_BETAS,
        weight_decay=WEIGHT_DECAY,
        fused=True,
    )
    warm_runner = torch.compile(
        warm_model,
        mode=COMPILE_MODE,
        fullgraph=False,
    )
    warm_generator = torch.Generator(device="cpu").manual_seed(
        ENGINEERING_SEED + 99_999
    )
    torch.cuda.synchronize(device)
    compile_started = time.perf_counter()
    _one_optimizer_step(
        model=warm_model,
        forward_model=warm_runner,
        optimizer=warm_optimizer,
        train_data=train_data,
        generator=warm_generator,
        device=device,
        step_index=0,
    )
    torch.cuda.synchronize(device)
    compile_warmup_seconds = max(time.perf_counter() - compile_started, 0.0)
    del warm_runner, warm_optimizer, warm_model
    gc.collect()
    torch.cuda.empty_cache()
    torch._dynamo.reset()

    seed_all(ENGINEERING_SEED)
    model = builder().to(device)
    if _count_parameters(model) != EXPECTED_PARAMETERS[name]:
        raise RuntimeError(f"{name} parameter drift before engineering training")
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
    generator = torch.Generator(device="cpu").manual_seed(
        ENGINEERING_SEED + 10_000
    )

    curve: list[dict[str, Any]] = []
    next_eval_index = 0
    training_seconds = 0.0
    eval_seconds = 0.0
    last_train_loss = float("nan")
    tokens_seen = 0

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    wall_started = time.perf_counter()

    for step in range(TOTAL_OPTIMIZER_STEPS):
        torch.cuda.synchronize(device)
        train_started = time.perf_counter()
        last_train_loss = _one_optimizer_step(
            model=model,
            forward_model=forward_model,
            optimizer=optimizer,
            train_data=train_data,
            generator=generator,
            device=device,
            step_index=step,
        )
        torch.cuda.synchronize(device)
        training_seconds += max(time.perf_counter() - train_started, 0.0)
        tokens_seen = (step + 1) * TOKENS_PER_STEP

        if (
            next_eval_index < len(EVAL_EXPOSURES_TOKENS)
            and tokens_seen >= EVAL_EXPOSURES_TOKENS[next_eval_index]
        ):
            eval_started = time.perf_counter()
            evaluation = _evaluate(
                model=model,
                forward_model=forward_model,
                val_data=val_data,
                device=device,
                batches=PERIODIC_EVAL_BATCHES,
                seed=ENGINEERING_SEED + 20_000,
            )
            torch.cuda.synchronize(device)
            eval_seconds += max(time.perf_counter() - eval_started, 0.0)
            curve.append(
                {
                    "step": step + 1,
                    "tokens_seen": tokens_seen,
                    "nll": evaluation["nll"],
                    "perplexity": evaluation["perplexity"],
                    "last_train_loss": last_train_loss,
                    "training_seconds": training_seconds,
                    "wall_seconds": time.perf_counter() - wall_started,
                }
            )
            next_eval_index += 1

    if tokens_seen != FULL_BATCH_TOKEN_EXPOSURES:
        raise RuntimeError(f"{name} token exposure drift: {tokens_seen}")
    if next_eval_index != len(EVAL_EXPOSURES_TOKENS):
        raise RuntimeError(f"{name} evaluation schedule incomplete")

    final_eval_started = time.perf_counter()
    final_eval = _evaluate(
        model=model,
        forward_model=forward_model,
        val_data=val_data,
        device=device,
        batches=FINAL_EVAL_BATCHES,
        seed=ENGINEERING_SEED + 30_000,
    )
    torch.cuda.synchronize(device)
    eval_seconds += max(time.perf_counter() - final_eval_started, 0.0)

    wall_seconds = max(time.perf_counter() - wall_started, 1e-9)
    peak_vram_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
    training_tps = FULL_BATCH_TOKEN_EXPOSURES / max(training_seconds, 1e-9)

    if not math.isfinite(last_train_loss) or not math.isfinite(
        float(final_eval["nll"])
    ):
        raise RuntimeError(f"{name} non-finite terminal metrics")

    return {
        "status": "COMPLETE",
        "classification": "ENGINEERING_REDUCED_ATTENTION_V2_200M_MODEL_COMPLETE",
        "source_sha": source_sha,
        "model": name,
        "parameters": _count_parameters(model),
        "engineering_seed": ENGINEERING_SEED,
        "full_batch_token_exposures": FULL_BATCH_TOKEN_EXPOSURES,
        "optimizer_steps": TOTAL_OPTIMIZER_STEPS,
        "seq_len": SEQ_LEN,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "compile_mode": COMPILE_MODE,
        "compile_warmup_seconds": compile_warmup_seconds,
        "training_seconds": training_seconds,
        "eval_seconds": eval_seconds,
        "wall_seconds": wall_seconds,
        "training_tokens_per_second": training_tps,
        "peak_vram_gib": peak_vram_gib,
        "last_train_loss": last_train_loss,
        "final_eval": final_eval,
        "curve": curve,
        "gpu_name": torch.cuda.get_device_name(device),
        "automatic_retry_authorized": False,
        "resume_authorized": False,
        "scientific_training_authorized": False,
        "scientific_seed_consumed": False,
        "250m_5b_authorized": False,
        "breakthrough_claim_allowed": False,
    }


@app.function(
    image=image,
    gpu="H100!",
    cpu=8,
    memory=32768,
    timeout=HARD_MODEL_TIMEOUT_SECONDS,
    retries=0,
    volumes={"/vol": volume},
)
def h100_train_one(
    name: str,
    source_sha: str,
    source_tree: str,
    harness_sha: str,
) -> str:
    source, tree, harness = _validate_payload(source_sha, source_tree, harness_sha)
    if name not in MODEL_ORDER:
        raise ValueError("unknown engineering-panel model")

    volume.reload()
    root = Path(RESULT_ROOT)
    reservation_path = root / "ENGINEERING_PANEL_DISPATCH_RESERVED.json"
    if not reservation_path.exists():
        raise RuntimeError("durable engineering-panel reservation is missing")
    reservation = json.loads(reservation_path.read_text(encoding="utf-8"))
    for key, expected in {
        "status": "ENGINEERING_PANEL_DISPATCH_RESERVED",
        "source_sha": source,
        "source_tree": tree,
        "harness_sha": harness,
        "engineering_seed": ENGINEERING_SEED,
        "engineering_seed_consumed": True,
        "selected_modal_account": REQUIRED_MODAL_ACCOUNT,
    }.items():
        if reservation.get(key) != expected:
            raise RuntimeError(f"engineering-panel reservation mismatch for {key}")

    run_dir = root / name
    attempt_path = run_dir / "ATTEMPT_STARTED.json"
    result_path = run_dir / "RESULT.json"
    if attempt_path.exists() or result_path.exists():
        raise RuntimeError(f"{name} engineering-panel attempt already consumed")

    attempt = {
        "status": "ATTEMPT_STARTED",
        "classification": "ENGINEERING_REDUCED_ATTENTION_V2_200M_H100_FUNCTION_BEGAN",
        "source_sha": source,
        "source_tree": tree,
        "harness_sha": harness,
        "model": name,
        "engineering_seed": ENGINEERING_SEED,
        "engineering_seed_consumed": True,
        "h100_allocation_started": True,
        "started_unix": time.time(),
        "automatic_retry_authorized": False,
        "resume_authorized": False,
        "scientific_training_authorized": False,
        "scientific_seed_consumed": False,
    }
    _atomic_write(attempt_path, attempt)
    reservation["h100_allocation_started"] = True
    _atomic_write(reservation_path, reservation)
    volume.commit()

    try:
        training = _train_model(name=name, source_sha=source)
        payload = {
            "status": "COMPLETE",
            "classification": "ENGINEERING_REDUCED_ATTENTION_V2_200M_MODEL_RESULT",
            "source_sha": source,
            "source_tree": tree,
            "harness_sha": harness,
            "model": name,
            "engineering_seed": ENGINEERING_SEED,
            "training": training,
            "automatic_retry_authorized": False,
            "resume_authorized": False,
            "scientific_training_authorized": False,
            "scientific_seed_consumed": False,
        }
    except BaseException as exc:
        payload = {
            "status": "ERROR",
            "classification": "ENGINEERING_REDUCED_ATTENTION_V2_200M_MODEL_ERROR_NO_RETRY",
            "source_sha": source,
            "source_tree": tree,
            "harness_sha": harness,
            "model": name,
            "engineering_seed": ENGINEERING_SEED,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "automatic_retry_authorized": False,
            "resume_authorized": False,
            "scientific_training_authorized": False,
            "scientific_seed_consumed": False,
        }

    _atomic_write(result_path, payload)
    volume.commit()
    print(json.dumps(payload, sort_keys=True), flush=True)
    return json.dumps(payload, sort_keys=True)


@app.function(
    image=image,
    cpu=2,
    memory=2048,
    timeout=10 * 60,
    volumes={"/vol": volume},
)
def finalize_panel(
    source_sha: str,
    source_tree: str,
    harness_sha: str,
) -> str:
    source, tree, harness = _validate_payload(source_sha, source_tree, harness_sha)
    volume.reload()
    root = Path(RESULT_ROOT)
    result_path = root / "RESULT.json"
    if result_path.exists():
        raise RuntimeError("engineering-panel terminal RESULT.json already exists")

    model_results: dict[str, Any] = {}
    for name in MODEL_ORDER:
        path = root / name / "RESULT.json"
        model_results[name] = (
            json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        )

    all_complete = all(
        result is not None and result.get("status") == "COMPLETE"
        for result in model_results.values()
    )
    comparisons: dict[str, Any] = {}
    progression_candidates: list[str] = []
    v1_seed_sensitivity_guard = False

    if all_complete:
        transformer_training = model_results["fresh_transformer"]["training"]
        transformer_nll = float(transformer_training["final_eval"]["nll"])
        transformer_tps = float(
            transformer_training["training_tokens_per_second"]
        )
        for name in MODEL_ORDER[1:]:
            training = model_results[name]["training"]
            nll = float(training["final_eval"]["nll"])
            tps = float(training["training_tokens_per_second"])
            delta = nll - transformer_nll
            ratio = tps / transformer_tps
            finite = all(math.isfinite(value) for value in (nll, tps, delta, ratio))
            quality_pass = finite and delta <= MAX_200M_NLL_DELTA
            throughput_pass = finite and ratio >= MIN_TRAINING_TPS_RATIO
            comparisons[name] = {
                "final_nll": nll,
                "fresh_transformer_final_nll": transformer_nll,
                "nll_delta": delta,
                "training_tokens_per_second": tps,
                "fresh_transformer_training_tokens_per_second": transformer_tps,
                "throughput_ratio": ratio,
                "finite": finite,
                "quality_gate_pass": quality_pass,
                "throughput_gate_pass": throughput_pass,
                "combined_gate_pass": quality_pass and throughput_pass,
            }

        v1_seed_sensitivity_guard = bool(
            comparisons["v1_replicate"]["quality_gate_pass"]
        )
        if not v1_seed_sensitivity_guard:
            progression_candidates = [
                name
                for name in ("early_4", "attention_8")
                if comparisons[name]["combined_gate_pass"]
            ]

    if not all_complete:
        classification = "ENGINEERING_PANEL_INCOMPLETE_NO_RETRY"
    elif v1_seed_sensitivity_guard:
        classification = "ENGINEERING_PANEL_SEED_SENSITIVITY_DIAGNOSTIC_ONLY"
    elif progression_candidates:
        classification = "ENGINEERING_PANEL_CANDIDATE_SCREEN_PASS"
    else:
        classification = "ENGINEERING_PANEL_NO_CANDIDATE_PASSES"

    payload = {
        "status": (
            "ENGINEERING_PANEL_COMPLETE"
            if all_complete
            else "ENGINEERING_PANEL_INCOMPLETE"
        ),
        "classification": classification,
        "source_sha": source,
        "source_tree": tree,
        "harness_sha": harness,
        "engineering_seed": ENGINEERING_SEED,
        "engineering_seed_consumed": True,
        "models": model_results,
        "comparisons": comparisons,
        "progression_candidates": progression_candidates,
        "gates": {
            "max_200m_nll_delta": MAX_200M_NLL_DELTA,
            "min_training_tps_ratio": MIN_TRAINING_TPS_RATIO,
            "v1_replicate_seed_sensitivity_guard": v1_seed_sensitivity_guard,
            "pair1_v1_200m_nll_delta_reference": (
                PAIR1_V1_200M_NLL_DELTA_REFERENCE
            ),
        },
        "scientific_training_authorized": False,
        "scientific_seed_consumed": False,
        "replication_authorized": False,
        "250m_5b_authorized": False,
        "breakthrough_claim_allowed": False,
        "automatic_retry_authorized": False,
        "resume_authorized": False,
    }
    _atomic_write(result_path, payload)
    volume.commit()
    print(json.dumps(payload, sort_keys=True), flush=True)
    return json.dumps(payload, sort_keys=True)


@app.local_entrypoint()
def main(
    source_sha: str = "",
    source_tree: str = "",
    harness_sha: str = "",
    selected_account: str = "",
) -> None:
    source, tree, harness = _validate_payload(source_sha, source_tree, harness_sha)
    account = selected_account.strip().lower()
    if account != REQUIRED_MODAL_ACCOUNT:
        raise RuntimeError(
            f"engineering panel requires Modal account {REQUIRED_MODAL_ACCOUNT!r}"
        )

    zero = json.loads(verify_zero_gpu.remote(source, tree, harness, account))
    print(json.dumps({"zero_gpu": zero}, sort_keys=True), flush=True)
    if zero.get("status") != "PASS":
        raise RuntimeError("engineering-panel zero-GPU gate did not pass")

    reservation = json.loads(
        reserve_panel_dispatch.remote(source, tree, harness, account)
    )
    print(json.dumps({"dispatch_reservation": reservation}, sort_keys=True), flush=True)
    if reservation.get("status") != "ENGINEERING_PANEL_DISPATCH_RESERVED":
        raise RuntimeError("engineering-panel reservation was not durably committed")

    prior_complete = True
    for name in MODEL_ORDER:
        if not prior_complete:
            print(
                json.dumps(
                    {
                        "model_skipped": name,
                        "reason": (
                            "prior model did not complete; fail closed to avoid "
                            "additional engineering spend"
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            continue

        result: dict[str, Any] | None = None
        try:
            result = json.loads(
                h100_train_one.remote(name, source, tree, harness)
            )
            print(json.dumps({name: result}, sort_keys=True), flush=True)
        except BaseException as exc:
            print(
                json.dumps(
                    {
                        f"{name}_remote_error": f"{type(exc).__name__}: {exc}",
                        "automatic_retry_authorized": False,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        prior_complete = bool(result and result.get("status") == "COMPLETE")

    final = json.loads(finalize_panel.remote(source, tree, harness))
    print(json.dumps({"engineering_panel": final}, sort_keys=True), flush=True)
