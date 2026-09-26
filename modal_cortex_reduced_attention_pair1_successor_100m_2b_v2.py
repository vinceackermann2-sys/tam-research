from __future__ import annotations

import gc
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable

import modal


PHASE = "reduced-attention-pair1-successor-100m-2b-v2"
TRIGGER_TITLE = "[modal-cortex-reduced-attention-pair1-successor-100m-2b-v2]"
PAIR_SEED = 59_231
RESULT_ROOT = "/vol/cortex-s-v0/100m-2b/reduced-attention-pair1-successor-v2"
PREDECESSOR_ROOT = "/vol/cortex-s-v0/100m-2b/reduced-attention-pair1-v1"
CALIBRATION_ROOT = "/vol/cortex-s-v0/100m-2b/reduced-attention-calibration-v1"
DATA_DIR = "/vol/data/tam100m-2b-curated-v1"

TRAIN_SHA256 = "93e9cb0b7076a4ddd855fc696f657ea62592b8a03a05220be99c402f9043265b"
VAL_SHA256 = "ae0bc5adf36d0aa8e55e5e3903401d3f114b93037f43221944b4e88c5d1a5760"
META_SHA256 = "14bbbcf0ab0b8cba374074ef8ccb80a04ecead06c74780f35a1becd5aef1b8f3"

EXPECTED_TRANSFORMER_PARAMS = 101_803_520
EXPECTED_REDUCED_ATTENTION_PARAMS = 101_799_424
TRAIN_TOKENS = 2_000_000_000
VAL_TOKENS = 5_000_000
SEQ_LEN = 512
MICRO_BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 2
TOKENS_PER_STEP = SEQ_LEN * MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS
TOTAL_OPTIMIZER_STEPS = 30_518
FULL_BATCH_TOKEN_EXPOSURES = TOTAL_OPTIMIZER_STEPS * TOKENS_PER_STEP
LEARNING_RATE = 3e-4
ADAMW_BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.02
GRAD_CLIP = 1.0
EVAL_EVERY_TOKENS = 200_000_000
CHECKPOINT_EVERY_TOKENS = 200_000_000
PERIODIC_EVAL_BATCHES = 20
FINAL_EVAL_BATCHES = 50
EVAL_BATCH_SIZE = 32
COMPILE_MODE = "max-autotune-no-cudagraphs"
HARD_MODEL_TIMEOUT_SECONDS = 10_000
PAIR1_SCREEN_MAX_NLL_DELTA = 0.015

APP_NAME = "cortex-reduced-attention-pair1-successor-100m-2b-v2"
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


def _validate_payload(source_sha: str, source_tree: str, harness_sha: str) -> tuple[str, str, str]:
    source = _full_sha(source_sha, "source_sha")
    tree = _full_sha(source_tree, "source_tree")
    harness = _full_sha(harness_sha, "harness_sha")
    if TOKENS_PER_STEP != 65_536:
        raise RuntimeError("tokens-per-step contract drift")
    if TOTAL_OPTIMIZER_STEPS != math.ceil(TRAIN_TOKENS / TOKENS_PER_STEP):
        raise RuntimeError("optimizer-step contract drift")
    if FULL_BATCH_TOKEN_EXPOSURES != 2_000_027_648:
        raise RuntimeError("full-batch exposure contract drift")
    return source, tree, harness


def _count_parameters(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _build_transformer() -> Any:
    from tam_research.models import ModelConfig, ResearchLM

    return ResearchLM(
        ModelConfig(
            vocab_size=50_257,
            d_model=512,
            n_layers=24,
            n_heads=16,
            max_seq_len=1_024,
            ff_mult=4,
            architecture="transformer",
        )
    )


def _build_reduced_attention() -> Any:
    from architectures.cortex_s.reduced_attention_100m_v1 import ReducedAttentionDenseLM

    return ReducedAttentionDenseLM()


def _verify_model_contract() -> dict[str, Any]:
    import torch
    from architectures.cortex_s.reduced_attention_100m_v1 import architecture_contract
    from architectures.cortex_s.reduced_attention_100m_v1_protocol import validate_protocol

    protocol = validate_protocol()
    base_reserved = protocol.get("scientific_pair_seeds_reserved_not_consumed")
    if base_reserved != [58_231, 58_232, 58_233]:
        raise RuntimeError("base prereg seed contract drift")
    forbidden = {8_100, 48_131, 48_132, 48_133, 58_231, 58_232, 58_233}
    if PAIR_SEED in forbidden or PAIR_SEED in base_reserved:
        raise RuntimeError("successor seed collides with prior scientific identity")
    if protocol.get("training_authorized") is not False:
        raise RuntimeError("base prereg protocol unexpectedly authorizes training")
    if protocol.get("scientific_seeds_consumed") is not False:
        raise RuntimeError("static base prereg protocol drift")
    if float(protocol.get("pair1_screen_max_nll_delta")) != PAIR1_SCREEN_MAX_NLL_DELTA:
        raise RuntimeError("Pair-1 NLL screen drift")

    candidate = architecture_contract()
    if candidate.get("attention_layer_numbers_one_based") != [6, 12, 18, 24]:
        raise RuntimeError("reduced-attention schedule drift")
    if candidate.get("world_state") is not False or candidate.get("moe") is not False:
        raise RuntimeError("candidate component drift")

    with torch.device("meta"):
        transformer = _build_transformer()
        reduced = _build_reduced_attention()
    t_params = _count_parameters(transformer)
    r_params = _count_parameters(reduced)
    if t_params != EXPECTED_TRANSFORMER_PARAMS:
        raise RuntimeError(f"Transformer parameter drift: {t_params}")
    if r_params != EXPECTED_REDUCED_ATTENTION_PARAMS:
        raise RuntimeError(f"reduced-attention parameter drift: {r_params}")
    return {
        "transformer_parameters": t_params,
        "reduced_attention_parameters": r_params,
        "absolute_parameter_gap": abs(t_params - r_params),
        "candidate_contract": candidate,
        "protocol_experiment_id": protocol.get("experiment_id"),
        "pair_seed_reserved": PAIR_SEED,
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
    expected = {
        "train_sha256": TRAIN_SHA256,
        "val_sha256": VAL_SHA256,
        "meta_sha256": META_SHA256,
    }
    if hashes != expected:
        raise RuntimeError(f"frozen corpus fingerprint drift: {hashes}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for key, value in {
        "assembly_version": 3,
        "train_tokens": TRAIN_TOKENS,
        "val_tokens": VAL_TOKENS,
        "seed": 8100,
        "tokenizer": "gpt2",
        "dtype": "uint16",
    }.items():
        if meta.get(key) != value:
            raise RuntimeError(f"frozen corpus metadata mismatch for {key}")
    if train.stat().st_size != TRAIN_TOKENS * 2:
        raise RuntimeError("train.bin byte-size drift")
    if val.stat().st_size != VAL_TOKENS * 2:
        raise RuntimeError("val.bin byte-size drift")
    return {**hashes, "metadata": meta}


def _verify_calibration() -> dict[str, Any]:
    path = Path(CALIBRATION_ROOT) / "RESULT.json"
    if not path.exists():
        raise FileNotFoundError("authoritative reduced-attention calibration RESULT.json is missing")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "CALIBRATION_PASS":
        raise RuntimeError("authoritative calibration is not CALIBRATION_PASS")
    decision = result.get("decision") or {}
    if decision.get("classification") != "PAIR1_PREPARATION_ALLOWED":
        raise RuntimeError("authoritative calibration does not allow Pair-1 preparation")
    if decision.get("pair1_preparation_allowed") is not True:
        raise RuntimeError("authoritative calibration Pair-1 gate is false")
    if result.get("engineering_seed") != 2_026_092_001:
        raise RuntimeError("calibration engineering seed drift")
    if result.get("scientific_seeds_consumed") is not False:
        raise RuntimeError("calibration claims scientific seed consumption")
    if result.get("scientific_training_authorized") is not False:
        raise RuntimeError("calibration unexpectedly authorized scientific training")
    return {
        "status": result["status"],
        "classification": result["classification"],
        "decision": decision,
        "source_sha": result["source_sha"],
        "source_tree": result["source_tree"],
        "harness_sha": result["harness_sha"],
        "comparison": result["comparison"],
        "results": {
            name: {
                "projected_full_seconds": payload["projected_full_seconds"],
                "peak_vram_gib": payload["peak_vram_gib"],
                "training_tokens_per_second": payload["training_tokens_per_second"],
            }
            for name, payload in result["results"].items()
        },
    }


def _verify_predecessor() -> dict[str, Any]:
    root = Path(PREDECESSOR_ROOT)
    reservation_path = root / "PAIR1_DISPATCH_RESERVED.json"
    transformer_path = root / "transformer" / "RESULT.json"
    reduced_path = root / "reduced_attention" / "RESULT.json"
    missing = [str(path) for path in (reservation_path, transformer_path, reduced_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"predecessor Pair-1 evidence missing: {missing}")

    reservation = json.loads(reservation_path.read_text(encoding="utf-8"))
    transformer = json.loads(transformer_path.read_text(encoding="utf-8"))
    reduced = json.loads(reduced_path.read_text(encoding="utf-8"))

    if reservation.get("status") != "PAIR1_DISPATCH_RESERVED":
        raise RuntimeError("predecessor reservation status drift")
    if reservation.get("pair_seed") != 58_231 or reservation.get("pair_seed_consumed") is not True:
        raise RuntimeError("predecessor seed-consumption evidence mismatch")
    if transformer.get("status") != "COMPLETE":
        raise RuntimeError("predecessor Transformer is not COMPLETE")
    if transformer.get("pair_seed") != 58_231:
        raise RuntimeError("predecessor Transformer seed mismatch")
    if reduced.get("status") != "ERROR":
        raise RuntimeError("predecessor reduced-attention result is not ERROR")
    if reduced.get("classification") != "SCIENTIFIC_REDUCED_ATTENTION_PAIR1_SUCCESSOR_ARCHITECTURE_ERROR_NO_RETRY":
        raise RuntimeError("predecessor reduced-attention classification drift")
    if reduced.get("pair_seed") != 58_231:
        raise RuntimeError("predecessor reduced-attention seed mismatch")
    if reduced.get("automatic_retry_authorized") is not False or reduced.get("resume_authorized") is not False:
        raise RuntimeError("predecessor no-retry/no-resume evidence drift")

    return {
        "result_root": PREDECESSOR_ROOT,
        "pair_seed": 58_231,
        "pair_seed_consumed": True,
        "transformer_status": transformer["status"],
        "transformer_final_nll": transformer["training"]["final_eval"]["nll"],
        "reduced_attention_status": reduced["status"],
        "reduced_attention_classification": reduced["classification"],
        "successor_is_retry": False,
    }


@app.function(image=image, cpu=4, memory=8192, timeout=30 * 60, volumes={"/vol": volume})
def verify_zero_gpu(
    source_sha: str,
    source_tree: str,
    harness_sha: str,
    selected_account: str,
) -> str:
    source, tree, harness = _validate_payload(source_sha, source_tree, harness_sha)
    account = selected_account.strip().lower()
    if account != REQUIRED_MODAL_ACCOUNT:
        raise RuntimeError(f"successor requires Modal account {REQUIRED_MODAL_ACCOUNT!r}")
    volume.reload()
    root = Path(RESULT_ROOT)
    if root.exists():
        raise RuntimeError("Pair-1 result namespace already exists; fail closed")

    model_contract = _verify_model_contract()
    corpus = _verify_corpus()
    calibration = _verify_calibration()
    predecessor = _verify_predecessor()
    payload = {
        "status": "PASS",
        "classification": "ZERO_GPU_REDUCED_ATTENTION_PAIR1_SUCCESSOR_100M_2B_V2",
        "phase": PHASE,
        "trigger_title": TRIGGER_TITLE,
        "source_sha": source,
        "source_tree": tree,
        "harness_sha": harness,
        "selected_modal_account": account,
        "pair_seed_reserved_not_consumed": PAIR_SEED,
        "result_root": RESULT_ROOT,
        "gpu_allocated": False,
        "model_contract": model_contract,
        "corpus": corpus,
        "calibration": calibration,
        "predecessor": predecessor,
        "training_contract": {
            "train_tokens": TRAIN_TOKENS,
            "val_tokens": VAL_TOKENS,
            "seq_len": SEQ_LEN,
            "micro_batch_size": MICRO_BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "tokens_per_step": TOKENS_PER_STEP,
            "total_optimizer_steps": TOTAL_OPTIMIZER_STEPS,
            "full_batch_token_exposures": FULL_BATCH_TOKEN_EXPOSURES,
            "learning_rate": LEARNING_RATE,
            "adamw_betas": list(ADAMW_BETAS),
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "grad_clip": GRAD_CLIP,
            "eval_every_tokens": EVAL_EVERY_TOKENS,
            "checkpoint_every_tokens": CHECKPOINT_EVERY_TOKENS,
            "periodic_eval_batches": PERIODIC_EVAL_BATCHES,
            "final_eval_batches": FINAL_EVAL_BATCHES,
            "compile_mode": COMPILE_MODE,
            "hard_model_timeout_seconds": HARD_MODEL_TIMEOUT_SECONDS,
        },
        "scientific_training_started": False,
        "pair_seed_consumed": False,
        "replication_seeds_authorized": False,
        "250m_5b_authorized": False,
        "architecture_superiority_claim_allowed": False,
    }
    _atomic_write(root / "ZERO_GPU_GATE.json", payload)
    volume.commit()
    return json.dumps(payload, sort_keys=True)


@app.function(image=image, cpu=2, memory=2048, timeout=10 * 60, volumes={"/vol": volume})
def reserve_pair_dispatch(
    source_sha: str,
    source_tree: str,
    harness_sha: str,
    selected_account: str,
) -> str:
    source, tree, harness = _validate_payload(source_sha, source_tree, harness_sha)
    account = selected_account.strip().lower()
    if account != REQUIRED_MODAL_ACCOUNT:
        raise RuntimeError(f"successor requires Modal account {REQUIRED_MODAL_ACCOUNT!r}")
    volume.reload()
    root = Path(RESULT_ROOT)
    gate_path = root / "ZERO_GPU_GATE.json"
    marker_path = root / "PAIR1_DISPATCH_RESERVED.json"
    pair_result = root / "RESULT.json"
    if not gate_path.exists():
        raise RuntimeError("Pair-1 zero-GPU gate is missing")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    for key, expected in {
        "status": "PASS",
        "source_sha": source,
        "source_tree": tree,
        "harness_sha": harness,
        "pair_seed_reserved_not_consumed": PAIR_SEED,
        "selected_modal_account": account,
    }.items():
        if gate.get(key) != expected:
            raise RuntimeError(f"Pair-1 zero-GPU gate mismatch for {key}")
    if marker_path.exists() or pair_result.exists():
        raise RuntimeError("Pair-1 dispatch/result already consumed")

    marker = {
        "status": "PAIR1_DISPATCH_RESERVED",
        "classification": "SCIENTIFIC_REDUCED_ATTENTION_PAIR1_SUCCESSOR_SINGLE_ATTEMPT",
        "phase": PHASE,
        "trigger_title": TRIGGER_TITLE,
        "source_sha": source,
        "source_tree": tree,
        "harness_sha": harness,
        "pair_seed": PAIR_SEED,
        "pair_seed_consumed": True,
        "selected_modal_account": account,
        "predecessor_pair_seed_consumed": 58_231,
        "result_root": RESULT_ROOT,
        "architecture_order": ["transformer", "reduced_attention"],
        "h100_allocation_started": False,
        "automatic_retry_authorized": False,
        "resume_authorized": False,
        "replication_seeds_authorized": False,
        "250m_5b_authorized": False,
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
        raise RuntimeError("non-finite scientific training step; optimizer update suppressed")

    warmup_steps = max(1, int(TOTAL_OPTIMIZER_STEPS * WARMUP_RATIO))
    lr = cosine_lr(step_index, TOTAL_OPTIMIZER_STEPS, warmup_steps, LEARNING_RATE)
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
    mean = sum(losses) / len(losses)
    return {"nll": mean, "perplexity": math.exp(min(mean, 20.0))}


def _save_checkpoint(
    *,
    path: Path,
    architecture: str,
    model: Any,
    optimizer: Any,
    step: int,
    tokens_seen: int,
    generator: Any,
    source_sha: str,
) -> None:
    import torch

    payload = {
        "experiment": PHASE,
        "classification": "SCIENTIFIC_PAIR1_SUCCESSOR_CHECKPOINT_EVIDENCE_ONLY",
        "source_sha": source_sha,
        "architecture": architecture,
        "seed": PAIR_SEED,
        "step": step,
        "tokens_seen": tokens_seen,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "batch_generator_state": generator.get_state(),
        "resume_authorized": False,
        "automatic_retry_authorized": False,
    }
    tmp = path.with_suffix(".tmp.pt")
    torch.save(payload, tmp)
    tmp.replace(path)


def _train_architecture(
    *,
    architecture: str,
    builder: Callable[[], Any],
    expected_parameters: int,
    source_sha: str,
) -> dict[str, Any]:
    import torch
    from tam_research.data import TokenBin
    from tam_research.train import seed_all

    if not torch.cuda.is_available():
        raise RuntimeError("Pair-1 scientific training requires CUDA")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device)[0] < 9:
        raise RuntimeError("Pair-1 scientific training requires H100/SM90-class CUDA")
    torch.set_float32_matmul_precision("high")

    run_dir = Path(RESULT_ROOT) / architecture
    latest_path = run_dir / "latest.pt"
    progress_path = run_dir / "PROGRESS.json"
    train_data = TokenBin(str(Path(DATA_DIR) / "train.bin"))
    val_data = TokenBin(str(Path(DATA_DIR) / "val.bin"))

    seed_all(PAIR_SEED)
    torch._dynamo.reset()
    warm_model = builder().to(device)
    if _count_parameters(warm_model) != expected_parameters:
        raise RuntimeError(f"{architecture} parameter drift before compile warmup")
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
    warm_generator = torch.Generator(device="cpu").manual_seed(PAIR_SEED + 99_999)
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
    first_compile_seconds = max(time.perf_counter() - compile_started, 0.0)
    del warm_runner, warm_optimizer, warm_model
    gc.collect()
    torch.cuda.empty_cache()

    seed_all(PAIR_SEED)
    model = builder().to(device)
    if _count_parameters(model) != expected_parameters:
        raise RuntimeError(f"{architecture} parameter drift before scientific training")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=ADAMW_BETAS,
        weight_decay=WEIGHT_DECAY,
        fused=True,
    )
    torch.cuda.synchronize(device)
    scientific_compile_started = time.perf_counter()
    forward_model = torch.compile(
        model,
        mode=COMPILE_MODE,
        fullgraph=False,
    )
    compile_generator = torch.Generator(device="cpu").manual_seed(PAIR_SEED + 199_999)
    _one_optimizer_step(
        model=model,
        forward_model=forward_model,
        optimizer=optimizer,
        train_data=train_data,
        generator=compile_generator,
        device=device,
        step_index=0,
    )
    torch.cuda.synchronize(device)
    second_compile_seconds = max(time.perf_counter() - scientific_compile_started, 0.0)
    del forward_model, optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    torch._dynamo.reset()

    seed_all(PAIR_SEED)
    model = builder().to(device)
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
    generator = torch.Generator(device="cpu").manual_seed(PAIR_SEED + 10_000)

    next_eval = EVAL_EVERY_TOKENS
    next_checkpoint = CHECKPOINT_EVERY_TOKENS
    curve: list[dict[str, Any]] = []
    training_seconds = 0.0
    eval_seconds = 0.0
    checkpoint_seconds = 0.0
    tokens_seen = 0
    last_train_loss = float("nan")

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
        tokens_seen = min(TRAIN_TOKENS, (step + 1) * TOKENS_PER_STEP)

        evaluated = False
        if tokens_seen >= next_eval or tokens_seen >= TRAIN_TOKENS:
            eval_started = time.perf_counter()
            evaluation = _evaluate(
                model=model,
                forward_model=forward_model,
                val_data=val_data,
                device=device,
                batches=PERIODIC_EVAL_BATCHES,
                seed=PAIR_SEED + 20_000,
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
            while next_eval <= tokens_seen:
                next_eval += EVAL_EVERY_TOKENS
            evaluated = True

        checkpointed = False
        if tokens_seen >= next_checkpoint or tokens_seen >= TRAIN_TOKENS:
            checkpoint_started = time.perf_counter()
            _save_checkpoint(
                path=latest_path,
                architecture=architecture,
                model=model,
                optimizer=optimizer,
                step=step + 1,
                tokens_seen=tokens_seen,
                generator=generator,
                source_sha=source_sha,
            )
            torch.cuda.synchronize(device)
            checkpoint_seconds += max(time.perf_counter() - checkpoint_started, 0.0)
            while next_checkpoint <= tokens_seen:
                next_checkpoint += CHECKPOINT_EVERY_TOKENS
            checkpointed = True

        if evaluated or checkpointed:
            _atomic_write(
                progress_path,
                {
                    "status": "IN_PROGRESS",
                    "source_sha": source_sha,
                    "architecture": architecture,
                    "seed": PAIR_SEED,
                    "step": step + 1,
                    "tokens_seen": tokens_seen,
                    "last_train_loss": last_train_loss,
                    "curve": curve,
                    "resume_authorized": False,
                    "automatic_retry_authorized": False,
                },
            )
            volume.commit()

    final_eval_started = time.perf_counter()
    final_eval = _evaluate(
        model=model,
        forward_model=forward_model,
        val_data=val_data,
        device=device,
        batches=FINAL_EVAL_BATCHES,
        seed=PAIR_SEED + 30_000,
    )
    torch.cuda.synchronize(device)
    eval_seconds += max(time.perf_counter() - final_eval_started, 0.0)
    wall_seconds = max(time.perf_counter() - wall_started, 1e-9)
    peak_vram_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
    training_tps = TRAIN_TOKENS / max(training_seconds, 1e-9)

    if tokens_seen != TRAIN_TOKENS:
        raise RuntimeError(f"{architecture} token budget drift: {tokens_seen}")
    if not math.isfinite(last_train_loss) or not math.isfinite(float(final_eval["nll"])):
        raise RuntimeError(f"{architecture} non-finite terminal metrics")

    return {
        "status": "COMPLETE",
        "classification": "SCIENTIFIC_REDUCED_ATTENTION_PAIR1_SUCCESSOR_ARCHITECTURE_COMPLETE",
        "source_sha": source_sha,
        "architecture": architecture,
        "parameters": _count_parameters(model),
        "seed": PAIR_SEED,
        "tokens_seen": TRAIN_TOKENS,
        "full_batch_token_exposures": FULL_BATCH_TOKEN_EXPOSURES,
        "optimizer_steps": TOTAL_OPTIMIZER_STEPS,
        "seq_len": SEQ_LEN,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "compile_mode": COMPILE_MODE,
        "compile_warmup_seconds": first_compile_seconds,
        "scientific_compile_warmup_seconds": second_compile_seconds,
        "training_seconds": training_seconds,
        "eval_seconds": eval_seconds,
        "checkpoint_seconds": checkpoint_seconds,
        "wall_seconds": wall_seconds,
        "training_tokens_per_second": training_tps,
        "peak_vram_gib": peak_vram_gib,
        "last_train_loss": last_train_loss,
        "final_eval": final_eval,
        "curve": curve,
        "checkpoint": str(latest_path),
        "gpu_name": torch.cuda.get_device_name(device),
        "resume_authorized": False,
        "automatic_retry_authorized": False,
        "replication_seeds_authorized": False,
        "250m_5b_authorized": False,
        "architecture_superiority_claim_allowed": False,
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
    architecture: str,
    source_sha: str,
    source_tree: str,
    harness_sha: str,
) -> str:
    source, tree, harness = _validate_payload(source_sha, source_tree, harness_sha)
    if architecture not in {"transformer", "reduced_attention"}:
        raise ValueError("unknown Pair-1 architecture")

    volume.reload()
    root = Path(RESULT_ROOT)
    reservation_path = root / "PAIR1_DISPATCH_RESERVED.json"
    if not reservation_path.exists():
        raise RuntimeError("durable Pair-1 reservation is missing")
    reservation = json.loads(reservation_path.read_text(encoding="utf-8"))
    for key, expected in {
        "status": "PAIR1_DISPATCH_RESERVED",
        "source_sha": source,
        "source_tree": tree,
        "harness_sha": harness,
        "pair_seed": PAIR_SEED,
        "pair_seed_consumed": True,
        "selected_modal_account": REQUIRED_MODAL_ACCOUNT,
    }.items():
        if reservation.get(key) != expected:
            raise RuntimeError(f"Pair-1 reservation mismatch for {key}")

    run_dir = root / architecture
    attempt_path = run_dir / "ATTEMPT_STARTED.json"
    result_path = run_dir / "RESULT.json"
    if attempt_path.exists() or result_path.exists():
        raise RuntimeError(f"{architecture} Pair-1 attempt already consumed")

    attempt = {
        "status": "ATTEMPT_STARTED",
        "classification": "SCIENTIFIC_REDUCED_ATTENTION_PAIR1_SUCCESSOR_H100_FUNCTION_BEGAN",
        "source_sha": source,
        "source_tree": tree,
        "harness_sha": harness,
        "architecture": architecture,
        "pair_seed": PAIR_SEED,
        "pair_seed_consumed": True,
        "h100_allocation_started": True,
        "started_unix": time.time(),
        "resume_authorized": False,
        "automatic_retry_authorized": False,
    }
    _atomic_write(attempt_path, attempt)
    reservation["h100_allocation_started"] = True
    _atomic_write(reservation_path, reservation)
    volume.commit()

    builder = _build_transformer if architecture == "transformer" else _build_reduced_attention
    expected_parameters = (
        EXPECTED_TRANSFORMER_PARAMS
        if architecture == "transformer"
        else EXPECTED_REDUCED_ATTENTION_PARAMS
    )

    try:
        training = _train_architecture(
            architecture=architecture,
            builder=builder,
            expected_parameters=expected_parameters,
            source_sha=source,
        )
        payload = {
            "status": "COMPLETE",
            "classification": "SCIENTIFIC_REDUCED_ATTENTION_PAIR1_SUCCESSOR_ARCHITECTURE_RESULT",
            "source_sha": source,
            "source_tree": tree,
            "harness_sha": harness,
            "architecture": architecture,
            "pair_seed": PAIR_SEED,
            "training": training,
            "resume_authorized": False,
            "automatic_retry_authorized": False,
        }
    except BaseException as exc:
        payload = {
            "status": "ERROR",
            "classification": "SCIENTIFIC_REDUCED_ATTENTION_PAIR1_SUCCESSOR_ARCHITECTURE_ERROR_NO_RETRY",
            "source_sha": source,
            "source_tree": tree,
            "harness_sha": harness,
            "architecture": architecture,
            "pair_seed": PAIR_SEED,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "resume_authorized": False,
            "automatic_retry_authorized": False,
            "posthoc_architecture_interpretation_authorized": False,
        }

    _atomic_write(result_path, payload)
    volume.commit()
    print(json.dumps(payload, sort_keys=True), flush=True)
    return json.dumps(payload, sort_keys=True)


@app.function(image=image, cpu=2, memory=2048, timeout=10 * 60, volumes={"/vol": volume})
def finalize_pair(source_sha: str, source_tree: str, harness_sha: str) -> str:
    source, tree, harness = _validate_payload(source_sha, source_tree, harness_sha)
    volume.reload()
    root = Path(RESULT_ROOT)
    pair_result_path = root / "RESULT.json"
    if pair_result_path.exists():
        raise RuntimeError("Pair-1 terminal RESULT.json already exists")

    architecture_results: dict[str, Any] = {}
    for architecture in ("transformer", "reduced_attention"):
        path = root / architecture / "RESULT.json"
        architecture_results[architecture] = (
            json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        )

    transformer = architecture_results["transformer"]
    reduced = architecture_results["reduced_attention"]
    both_complete = bool(
        transformer
        and reduced
        and transformer.get("status") == "COMPLETE"
        and reduced.get("status") == "COMPLETE"
    )
    exact_geometry = False
    finite = False
    delta = None
    if both_complete:
        t = transformer["training"]
        r = reduced["training"]
        frozen_keys = (
            "seed",
            "tokens_seen",
            "full_batch_token_exposures",
            "optimizer_steps",
            "seq_len",
            "micro_batch_size",
            "grad_accum_steps",
            "compile_mode",
        )
        exact_geometry = all(t.get(key) == r.get(key) for key in frozen_keys)
        exact_geometry = exact_geometry and all(
            t.get(key) == expected
            for key, expected in {
                "seed": PAIR_SEED,
                "tokens_seen": TRAIN_TOKENS,
                "full_batch_token_exposures": FULL_BATCH_TOKEN_EXPOSURES,
                "optimizer_steps": TOTAL_OPTIMIZER_STEPS,
                "seq_len": SEQ_LEN,
                "micro_batch_size": MICRO_BATCH_SIZE,
                "grad_accum_steps": GRAD_ACCUM_STEPS,
                "compile_mode": COMPILE_MODE,
            }.items()
        )
        t_nll = float(t["final_eval"]["nll"])
        r_nll = float(r["final_eval"]["nll"])
        finite = math.isfinite(t_nll) and math.isfinite(r_nll)
        delta = r_nll - t_nll

    screen_pass = bool(
        both_complete
        and exact_geometry
        and finite
        and delta is not None
        and delta <= PAIR1_SCREEN_MAX_NLL_DELTA
    )
    if not both_complete:
        classification = "PAIR1_SUCCESSOR_INCOMPLETE_NO_RETRY"
    elif not exact_geometry or not finite:
        classification = "PAIR1_SUCCESSOR_INVALID_OR_NONFINITE_NO_RETRY"
    elif screen_pass:
        classification = "PAIR1_SUCCESSOR_SCREEN_PASS"
    else:
        classification = "PAIR1_SUCCESSOR_SCREEN_FAIL"

    payload = {
        "status": "PAIR1_SUCCESSOR_COMPLETE" if both_complete else "PAIR1_SUCCESSOR_INCOMPLETE",
        "classification": classification,
        "source_sha": source,
        "source_tree": tree,
        "harness_sha": harness,
        "pair_seed": PAIR_SEED,
        "pair_seed_consumed": True,
        "architectures": architecture_results,
        "comparison": {
            "reduced_attention_minus_transformer_final_nll": delta,
            "screen_threshold": PAIR1_SCREEN_MAX_NLL_DELTA,
            "both_complete": both_complete,
            "exact_matched_geometry": exact_geometry,
            "finite_terminal_nll": finite,
            "pair1_screen_pass": screen_pass,
        },
        "replication_seeds_authorized": False,
        "250m_5b_authorized": False,
        "architecture_superiority_claim_allowed": False,
        "automatic_retry_authorized": False,
        "resume_authorized": False,
    }
    _atomic_write(pair_result_path, payload)
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
        raise RuntimeError(f"successor requires Modal account {REQUIRED_MODAL_ACCOUNT!r}")

    zero = json.loads(verify_zero_gpu.remote(source, tree, harness, account))
    print(json.dumps({"zero_gpu": zero}, sort_keys=True), flush=True)
    if zero.get("status") != "PASS":
        raise RuntimeError("Pair-1 zero-GPU gate did not pass")

    reservation = json.loads(reserve_pair_dispatch.remote(source, tree, harness, account))
    print(json.dumps({"dispatch_reservation": reservation}, sort_keys=True), flush=True)
    if reservation.get("status") != "PAIR1_DISPATCH_RESERVED":
        raise RuntimeError("Pair-1 reservation was not durably committed")

    transformer: dict[str, Any] | None = None
    try:
        transformer = json.loads(
            h100_train_one.remote("transformer", source, tree, harness)
        )
        print(json.dumps({"transformer": transformer}, sort_keys=True), flush=True)
    except BaseException as exc:
        print(
            json.dumps(
                {
                    "transformer_remote_error": f"{type(exc).__name__}: {exc}",
                    "automatic_retry_authorized": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if transformer is not None and transformer.get("status") == "COMPLETE":
        try:
            reduced = json.loads(
                h100_train_one.remote("reduced_attention", source, tree, harness)
            )
            print(json.dumps({"reduced_attention": reduced}, sort_keys=True), flush=True)
        except BaseException as exc:
            print(
                json.dumps(
                    {
                        "reduced_attention_remote_error": f"{type(exc).__name__}: {exc}",
                        "automatic_retry_authorized": False,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    else:
        print(
            json.dumps(
                {
                    "reduced_attention_skipped": True,
                    "reason": "Transformer did not complete; fail closed to avoid additional spend",
                },
                sort_keys=True,
            ),
            flush=True,
        )

    final = json.loads(finalize_pair.remote(source, tree, harness))
    print(json.dumps({"pair1": final}, sort_keys=True), flush=True)
