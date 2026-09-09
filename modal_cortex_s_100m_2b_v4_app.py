from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import modal

# Import-light copies used before the Modal image imports the research package.
DATA_DIR = "/vol/data/tam100m-2b-curated-v1"
TRAIN_TOKENS = 2_000_000_000
VAL_TOKENS = 5_000_000
TRAIN_SHA256 = "93e9cb0b7076a4ddd855fc696f657ea62592b8a03a05220be99c402f9043265b"
VAL_SHA256 = "ae0bc5adf36d0aa8e55e5e3903401d3f114b93037f43221944b4e88c5d1a5760"
META_SHA256 = "14bbbcf0ab0b8cba374074ef8ccb80a04ecead06c74780f35a1becd5aef1b8f3"
MAX_PROJECTED_FULL_SECONDS = 8_500
HARD_FULL_TIMEOUT_SECONDS = 10_000
CALIBRATION_SEED = 2_026_090_906
PRODUCTION_MOE_BACKEND = "physical_padded_grouped_bf16"
PRODUCTION_SYSTEMS_VARIANT = "memory_lean_cast_before_gather_pad_v4"

# v1/v2/v3 and all systems-repair namespaces are consumed. v4 gets unique roots.
APP_NAME = "cortex-s-v0-100m-2b-v4-memory-lean"
VOLUME_NAME = "tam-research-data"
PREFLIGHT_ROOT = "/vol/cortex-s-v0/100m-2b/preflight-v4-memory-lean"
RUN_ROOT = "/vol/cortex-s-v0/100m-2b/paired-seed8100-v4-memory-lean"

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
    except Exception as exc:
        print(f"[status-report-nonfatal] {type(exc).__name__}: {exc}; body={body}", flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(16 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _verify_frozen_data() -> dict:
    root = Path(DATA_DIR)
    train_path = root / "train.bin"
    val_path = root / "val.bin"
    meta_path = root / "meta.json"
    missing = [str(path) for path in (train_path, val_path, meta_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"frozen 100M/2B corpus missing: {missing}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected_meta = {
        "assembly_version": 3,
        "train_tokens": TRAIN_TOKENS,
        "val_tokens": VAL_TOKENS,
        "seed": 8100,
        "tokenizer": "gpt2",
        "dtype": "uint16",
    }
    for key, expected in expected_meta.items():
        if meta.get(key) != expected:
            raise RuntimeError(f"data metadata mismatch for {key}: {meta.get(key)!r} != {expected!r}")
    if train_path.stat().st_size != TRAIN_TOKENS * 2:
        raise RuntimeError("train.bin is not exactly 2B uint16 tokens")
    if val_path.stat().st_size != VAL_TOKENS * 2:
        raise RuntimeError("val.bin is not exactly 5M uint16 tokens")
    train_hash = _sha256(train_path)
    val_hash = _sha256(val_path)
    meta_hash = _sha256(meta_path)
    if train_hash != TRAIN_SHA256:
        raise RuntimeError(f"train corpus fingerprint mismatch: {train_hash}")
    if val_hash != VAL_SHA256:
        raise RuntimeError(f"validation corpus fingerprint mismatch: {val_hash}")
    if meta_hash != META_SHA256:
        raise RuntimeError(f"metadata fingerprint mismatch: {meta_hash}")
    return {
        "meta": meta,
        "train_sha256": train_hash,
        "val_sha256": val_hash,
        "meta_sha256": meta_hash,
        "train_bytes": train_path.stat().st_size,
        "val_bytes": val_path.stat().st_size,
        "fingerprints_established_by": "issue-767-cpu-fingerprint-v2",
    }


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=30 * 60,
    volumes={"/vol": volume},
    secrets=[github_secret],
)
def verify_data_zero_gpu(
    source_sha: str,
    repo_full_name: str = "",
    issue_number: int = 0,
) -> dict:
    """Re-hash corpus and instantiate the exact v4 candidate with no GPU."""

    from architectures.cortex_s.grouped_moe import padded_hidden
    from architectures.cortex_s.grouped_moe_v4 import (
        MemoryLeanPhysicalPaddedGroupedSparseMoE,
        SYSTEMS_VARIANT,
        build_memory_lean_grouped_cortex_100m,
    )
    from architectures.cortex_s.language_model import parameter_count
    from architectures.cortex_s.experiments.scale100m_2b.protocol import (
        EXPECTED_CORTEX_PARAMS,
        protocol_snapshot,
    )

    volume.reload()
    root = Path(PREFLIGHT_ROOT)
    target = root / "data.json"
    if target.exists() or (root / "H100_PREFLIGHT_DISPATCHED.json").exists() or (root / "h100.json").exists():
        raise RuntimeError("v4 memory-lean preflight namespace is already consumed")

    data = _verify_frozen_data()
    model = build_memory_lean_grouped_cortex_100m()
    actual = parameter_count(model)
    grouped_layers = sum(
        isinstance(block.moe, MemoryLeanPhysicalPaddedGroupedSparseMoE)
        for block in model.blocks
    )
    if actual != EXPECTED_CORTEX_PARAMS:
        raise RuntimeError("v4 memory-lean CPU model parameter count mismatch")
    if grouped_layers != len(model.blocks):
        raise RuntimeError("not every v4 MoE block uses the memory-lean grouped backend")
    if padded_hidden(model.cfg.expert_hidden) != 344:
        raise RuntimeError("v4 memory-lean physical hidden width drift")
    if SYSTEMS_VARIANT != PRODUCTION_SYSTEMS_VARIANT:
        raise RuntimeError("v4 systems variant contract mismatch")

    result = {
        "status": "PASS",
        "scientific_status": "ZERO_GPU_DATA_AND_MEMORY_LEAN_GROUPED_GATE_V4",
        "source_sha": source_sha,
        "data": data,
        "production_model": {
            "parameter_count": actual,
            "grouped_moe_layers": grouped_layers,
            "logical_expert_hidden": model.cfg.expert_hidden,
            "physical_expert_hidden": 344,
            "backend": PRODUCTION_MOE_BACKEND,
            "systems_variant": PRODUCTION_SYSTEMS_VARIANT,
        },
        "protocol": protocol_snapshot(),
        "prior_v1_preflight_consumed": True,
        "prior_v2_preflight_consumed": True,
        "prior_v3_preflight_consumed": True,
        "v3_full_training_launched": False,
    }
    root.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    volume.commit()
    _comment(
        repo_full_name,
        issue_number,
        "🟩 **CORTEX-S v4 zero-GPU gate passed** — frozen corpus and exact 101,778,112-parameter memory-lean grouped model validated. No GPU allocated.",
    )
    return result


@app.function(
    image=image,
    gpu="H100!",
    cpu=8,
    memory=32768,
    timeout=30 * 60,
    volumes={"/vol": volume},
    secrets=[github_secret],
)
def h100_preflight(
    source_sha: str,
    repo_full_name: str = "",
    issue_number: int = 0,
) -> dict:
    """Single-use v4 engineering calibration. It can never start the 2B run."""

    from architectures.cortex_s.grouped_moe_v4 import (
        run_h100_memory_lean_grouped_calibration,
    )

    volume.reload()
    data_path = Path(PREFLIGHT_ROOT) / "data.json"
    if not data_path.exists():
        raise RuntimeError("v4 memory-lean zero-GPU gate is missing")
    data_gate = json.loads(data_path.read_text(encoding="utf-8"))
    if data_gate.get("status") != "PASS" or data_gate.get("source_sha") != source_sha:
        raise RuntimeError("v4 zero-GPU gate is not valid for this source SHA")
    production_model = data_gate.get("production_model", {})
    if production_model.get("backend") != PRODUCTION_MOE_BACKEND:
        raise RuntimeError("v4 zero-GPU backend contract mismatch")
    if production_model.get("systems_variant") != PRODUCTION_SYSTEMS_VARIANT:
        raise RuntimeError("v4 zero-GPU systems variant mismatch")

    dispatch_marker = Path(PREFLIGHT_ROOT) / "H100_PREFLIGHT_DISPATCHED.json"
    result_path = Path(PREFLIGHT_ROOT) / "h100.json"
    if result_path.exists() or dispatch_marker.exists():
        raise RuntimeError("v4 memory-lean H100 preflight namespace already consumed")
    dispatch_marker.write_text(
        json.dumps(
            {
                "source_sha": source_sha,
                "calibration_seed": CALIBRATION_SEED,
                "production_moe_backend": PRODUCTION_MOE_BACKEND,
                "production_systems_variant": PRODUCTION_SYSTEMS_VARIANT,
                "dispatched_unix": time.time(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    volume.commit()

    _comment(
        repo_full_name,
        issue_number,
        "🟨 **CORTEX-S memory-lean H100 preflight v4 started** — engineering seed 2026090906 only. Paired seed 8100 remains locked and this function cannot launch full training.",
    )
    try:
        result = run_h100_memory_lean_grouped_calibration(
            data_dir=DATA_DIR,
            output_path=str(result_path),
            source_sha=source_sha,
        )
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        volume.commit()
    except Exception:
        volume.commit()
        raise

    if result.get("production_moe_backend") != PRODUCTION_MOE_BACKEND:
        raise RuntimeError("v4 calibration returned wrong production backend")
    if result.get("production_systems_variant") != PRODUCTION_SYSTEMS_VARIANT:
        raise RuntimeError("v4 calibration returned wrong systems variant")

    cost = result["projected_cost"]["total_usd_conservative"]
    if result["status"] == "PASS":
        _comment(
            repo_full_name,
            issue_number,
            "🟩 **CORTEX-S memory-lean H100 preflight v4 PASS** — "
            f"{result['training_tokens_per_second']:.0f} tok/s, peak VRAM={result['peak_vram_gb']:.2f} GiB, "
            f"projected full envelope={result['projected_full_seconds']:.0f}s, conservative projected compute=${cost:.2f}. "
            "Only a separately opened exact-source full-v4 issue may then become eligible; no full training was launched here.",
        )
    else:
        _comment(
            repo_full_name,
            issue_number,
            "🟥 **CORTEX-S memory-lean H100 preflight v4 stopped paid progression** — "
            f"projection={result['projected_full_seconds']:.0f}s vs {MAX_PROJECTED_FULL_SECONDS}s gate, "
            f"peak VRAM={result['peak_vram_gb']:.2f} GiB. No paired seed-8100 training was launched.",
        )
    return result


@app.function(
    image=image,
    gpu="H100!",
    cpu=8,
    memory=32768,
    timeout=HARD_FULL_TIMEOUT_SECONDS,
    volumes={"/vol": volume},
    secrets=[github_secret],
)
def full_2b(
    source_sha: str,
    repo_full_name: str = "",
    issue_number: int = 0,
) -> dict:
    """Exactly one paired v4 attempt, only after exact-source preflight PASS."""

    import architectures.cortex_s.experiments.scale100m_2b.train as training_module
    from architectures.cortex_s.grouped_moe_v4 import train_full_memory_lean_grouped_2b

    volume.reload()
    preflight_path = Path(PREFLIGHT_ROOT) / "h100.json"
    if not preflight_path.exists():
        raise RuntimeError("v4 memory-lean H100 preflight result is missing")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS" or not preflight.get("full_run_authorized"):
        raise RuntimeError("v4 memory-lean full run is not authorized by preflight")
    if preflight.get("source_sha") != source_sha:
        raise RuntimeError("source changed after v4 memory-lean preflight")
    if preflight.get("production_moe_backend") != PRODUCTION_MOE_BACKEND:
        raise RuntimeError("v4 preflight backend mismatch")
    if preflight.get("production_systems_variant") != PRODUCTION_SYSTEMS_VARIANT:
        raise RuntimeError("v4 preflight systems variant mismatch")
    if float(preflight.get("projected_full_seconds", float("inf"))) > MAX_PROJECTED_FULL_SECONDS:
        raise RuntimeError("v4 full run projection exceeds frozen budget gate")

    dispatch_marker = Path(RUN_ROOT) / "FULL_DISPATCH_CONSUMED.json"
    if dispatch_marker.exists() or (Path(RUN_ROOT) / "SUCCESS.json").exists():
        raise RuntimeError("v4 memory-lean paired namespace already consumed")
    Path(RUN_ROOT).mkdir(parents=True, exist_ok=True)
    dispatch_marker.write_text(
        json.dumps(
            {
                "source_sha": source_sha,
                "seed": 8100,
                "production_moe_backend": PRODUCTION_MOE_BACKEND,
                "production_systems_variant": PRODUCTION_SYSTEMS_VARIANT,
                "dispatched_unix": time.time(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    volume.commit()

    _comment(
        repo_full_name,
        issue_number,
        "🔥 **CORTEX-S 100M memory-lean grouped paired run v4 started** — seed=8100, nominal 2B budget, 30,518 optimizer steps / 2,000,027,648 literal exposures. Hard timeout and checkpoint durability remain enforced.",
    )

    original_save = training_module._save_checkpoint

    def durable_save(**kwargs):
        original_save(**kwargs)
        manifest = Path(RUN_ROOT) / "DURABLE_CHECKPOINT.json"
        manifest.write_text(
            json.dumps(
                {
                    "source_sha": source_sha,
                    "step": int(kwargs["step"]),
                    "tokens_seen_reported": int(kwargs["tokens_seen"]),
                    "checkpoint": str(kwargs["path"]),
                    "production_moe_backend": PRODUCTION_MOE_BACKEND,
                    "production_systems_variant": PRODUCTION_SYSTEMS_VARIANT,
                    "committed_unix": time.time(),
                    "automatic_resume_authorized": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        volume.commit()

    training_module._save_checkpoint = durable_save
    try:
        result = train_full_memory_lean_grouped_2b(
            data_dir=DATA_DIR,
            output_dir=RUN_ROOT,
            preflight_path=str(preflight_path),
            source_sha=source_sha,
        )
        grouped_result = Path(RUN_ROOT) / "V4_SUCCESS.json"
        grouped_result.write_text(json.dumps(result, indent=2), encoding="utf-8")
        volume.commit()
    except Exception:
        volume.commit()
        raise
    finally:
        training_module._save_checkpoint = original_save

    comparison = result["comparison"]
    _comment(
        repo_full_name,
        issue_number,
        "✅ **CORTEX-S 100M memory-lean grouped paired run v4 complete** — "
        f"NLL={result['final_eval']['nll']:.6f} vs Transformer {result['historical_transformer']['final_nll']:.6f}; "
        f"PPL={result['final_eval']['perplexity']:.3f}; throughput={result['training_tokens_per_second']:.0f} tok/s; "
        f"peak VRAM={result['peak_vram_gb']:.2f} GiB; matched-step NLL win={comparison['equal_token_nll_cortex_better']}. "
        "Classification remains paired historical-control evidence, not a breakthrough claim.",
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main(
    phase: str = "preflight-v4-memory-lean",
    source_sha: str = "",
    repo_full_name: str = "",
    issue_number: int = 0,
):
    if not source_sha:
        raise ValueError("source_sha is required and must be the exact checked-out commit")
    phase = phase.strip().lower()
    if phase == "preflight-v4-memory-lean":
        data = verify_data_zero_gpu.remote(source_sha, repo_full_name, issue_number)
        print(json.dumps({"zero_gpu": data}, indent=2), flush=True)
        result = h100_preflight.remote(source_sha, repo_full_name, issue_number)
        print(json.dumps({"h100_preflight": result}, indent=2), flush=True)
        return
    if phase == "full-v4-memory-lean":
        result = full_2b.remote(source_sha, repo_full_name, issue_number)
        print(json.dumps({"full": result}, indent=2), flush=True)
        return
    raise ValueError("phase must be 'preflight-v4-memory-lean' or 'full-v4-memory-lean'")
