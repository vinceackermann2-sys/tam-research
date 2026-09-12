from __future__ import annotations

import json
import math
import re
from pathlib import Path
import time
from typing import Any

import modal

APP_NAME = "cortex-s-v11-vs-transformer-pretrain-benchmark-v1"
VOLUME_NAME = "tam-research-data"
TRIGGER_TITLE = "[modal-cortex-s-v11-vs-transformer-pretrain-benchmark-v1]"
EVAL_SEED = 2_026_091_201
RESULT_ROOT = "/vol/cortex-s-v0/evals/cortex-s-v11-vs-transformer-pretrain-benchmark-v1"
RESULT_PATH = f"{RESULT_ROOT}/RESULT.json"
MARKER_PATH = f"{RESULT_ROOT}/EVAL_DISPATCH_CONSUMED.json"

CORTEX_CHECKPOINT = "/vol/cortex-s-v0/100m-2b/full-v11-compiled-explicit-fp32-ce/latest.pt"
TRANSFORMER_CHECKPOINT = "/vol/full100m-runs/transformer-pretrain/100m/ctx512-mb64-ga2/transformer-25m-compiled-seed8100/latest.pt"
CORTEX_TRAINING_SEED = 2_026_091_014
TRANSFORMER_TRAINING_SEED = 8_100
EXPECTED_CORTEX_PARAMS = 101_778_112
EXPECTED_TRANSFORMER_PARAMS = 101_803_520
EXPECTED_TOKENS = 2_000_000_000
EXPECTED_STEPS = 30_518
CORTEX_FINAL_NLL = 2.709058737754822
CORTEX_FINAL_PPL = 15.015135680950744
TRANSFORMER_FINAL_NLL = 2.7115590302149455
TRANSFORMER_FINAL_PPL = 15.053722884190283

EVAL_MAX_CONTEXT = 1024
MCQ_LIMIT = 200
GSM8K_LIMIT = 100

PARQUET_SPECS = (
    {
        "name": "arc_easy",
        "url": "https://huggingface.co/datasets/allenai/ai2_arc/resolve/main/ARC-Easy/validation-00000-of-00001.parquet",
        "split": "validation",
        "parser": "arc",
        "min_rows": 570,
    },
    {
        "name": "arc_challenge",
        "url": "https://huggingface.co/datasets/allenai/ai2_arc/resolve/main/ARC-Challenge/validation-00000-of-00001.parquet",
        "split": "validation",
        "parser": "arc",
        "min_rows": 299,
    },
    {
        "name": "piqa",
        "url": "https://huggingface.co/datasets/ybisk/piqa/resolve/078a131412f46a38025a762322c174a8bae2610c/plain_text/piqa-validation.parquet",
        "split": "validation",
        "parser": "piqa",
        "min_rows": 1838,
    },
    {
        "name": "hellaswag",
        "url": "https://huggingface.co/datasets/Rowan/hellaswag/resolve/main/data/validation-00000-of-00001.parquet",
        "split": "validation",
        "parser": "hellaswag",
        "min_rows": 10042,
    },
    {
        "name": "openbookqa",
        "url": "https://huggingface.co/datasets/allenai/openbookqa/resolve/main/main/validation-00000-of-00001.parquet",
        "split": "validation",
        "parser": "openbook",
        "min_rows": 500,
    },
)
GSM8K_URL = "https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
github_secret = modal.Secret.from_name("github-secret")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.10,<2.11",
        "transformers>=4.55,<5",
        "tokenizers>=0.21,<1",
        "datasets>=4.0,<5",
        "numpy>=2.0,<3",
        "huggingface-hub>=0.34,<1",
        "PyGithub>=2.3,<3",
    )
    .add_local_python_source("tam_research")
    .add_local_python_source("architectures")
)


def _full_sha(value: str, name: str) -> str:
    value = value.strip().lower()
    if len(value) != 40 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name} must be a full lowercase SHA")
    return value


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
    except Exception as exc:
        print(f"[status-report-nonfatal] {type(exc).__name__}: {exc}; body={body}", flush=True)


def _option(text: str) -> str:
    text = str(text).strip()
    return f" {text}" if text else " "


def _arc_item(row: dict) -> dict:
    labels = [str(x) for x in row["choices"]["label"]]
    answer_key = str(row["answerKey"])
    return {
        "prompt": f"Question: {row['question']}\nAnswer:",
        "options": [_option(x) for x in row["choices"]["text"]],
        "answer": labels.index(answer_key),
    }


def _openbook_item(row: dict) -> dict:
    labels = [str(x) for x in row["choices"]["label"]]
    answer_key = str(row["answerKey"])
    return {
        "prompt": f"Question: {row['question_stem']}\nAnswer:",
        "options": [_option(x) for x in row["choices"]["text"]],
        "answer": labels.index(answer_key),
    }


def _piqa_item(row: dict) -> dict:
    return {
        "prompt": f"Goal: {row['goal']}\nBest solution:",
        "options": [_option(row["sol1"]), _option(row["sol2"])],
        "answer": int(row["label"]),
    }


def _hellaswag_item(row: dict) -> dict:
    return {
        "prompt": str(row["ctx"]),
        "options": [_option(x) for x in row["endings"]],
        "answer": int(row["label"]),
    }


PARSERS = {
    "arc": _arc_item,
    "openbook": _openbook_item,
    "piqa": _piqa_item,
    "hellaswag": _hellaswag_item,
}


def _normalize_number(text: str) -> str | None:
    matches = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def _gsm_answer(answer: str) -> str | None:
    return _normalize_number(str(answer).split("####")[-1])


@app.function(image=image, cpu=2, memory=2048, timeout=600, volumes={"/vol": volume})
def reserve_evaluation_dispatch(source_sha: str, source_tree: str, harness_sha: str) -> str:
    source = _full_sha(source_sha, "source_sha")
    tree = _full_sha(source_tree, "source_tree")
    harness = _full_sha(harness_sha, "harness_sha")
    volume.reload()
    root = Path(RESULT_ROOT)
    marker = Path(MARKER_PATH)
    result = Path(RESULT_PATH)
    if root.exists() or marker.exists() or result.exists():
        raise RuntimeError("evaluation namespace already exists; refusing duplicate/retry")
    payload = {
        "status": "EVAL_DISPATCH_CONSUMED",
        "classification": "EVALUATION_ONLY_CORTEX_S_V11_VS_TRANSFORMER_PRETRAIN_V1",
        "trigger_title": TRIGGER_TITLE,
        "source_sha": source,
        "source_tree": tree,
        "harness_sha": harness,
        "evaluation_sample_seed": EVAL_SEED,
        "gpu_allocation_started": False,
        "training_authorized": False,
        "weights_updated": False,
        "reserved_scientific_training_seeds_used": False,
        "created_unix": time.time(),
    }
    _atomic_write(marker, payload)
    volume.commit()
    return json.dumps(payload, sort_keys=True)


def _load_models(device):
    import torch
    from architectures.cortex_s.language_model import parameter_count as cortex_parameter_count
    from architectures.cortex_s.production_scan_integration_v1 import (
        build_memory_lean_grouped_triton_scan_cortex_100m,
    )
    from tam_research.models import parameter_count as transformer_parameter_count
    from tam_research.posttrain import load_checkpoint_model

    cortex_path = Path(CORTEX_CHECKPOINT)
    transformer_path = Path(TRANSFORMER_CHECKPOINT)
    if not cortex_path.exists():
        raise FileNotFoundError(f"missing CORTEX checkpoint: {cortex_path}")
    if not transformer_path.exists():
        raise FileNotFoundError(f"missing Transformer checkpoint: {transformer_path}")

    cortex_ckpt = torch.load(cortex_path, map_location="cpu", weights_only=False)
    if cortex_ckpt.get("architecture") != "cortex_s":
        raise RuntimeError(f"unexpected CORTEX architecture metadata: {cortex_ckpt.get('architecture')!r}")
    if int(cortex_ckpt.get("seed", -1)) != CORTEX_TRAINING_SEED:
        raise RuntimeError("CORTEX checkpoint training-seed provenance mismatch")
    if int(cortex_ckpt.get("step", -1)) != EXPECTED_STEPS or int(cortex_ckpt.get("tokens_seen", -1)) != EXPECTED_TOKENS:
        raise RuntimeError("CORTEX checkpoint is not the completed 2B/30518-step checkpoint")
    cortex = build_memory_lean_grouped_triton_scan_cortex_100m()
    cortex.load_state_dict(cortex_ckpt["model"], strict=True)
    if cortex_parameter_count(cortex) != EXPECTED_CORTEX_PARAMS:
        raise RuntimeError("CORTEX parameter count drift")
    cortex = cortex.to(device).eval()

    transformer_ckpt = torch.load(transformer_path, map_location="cpu", weights_only=False)
    if transformer_ckpt.get("architecture") != "transformer":
        raise RuntimeError(f"unexpected Transformer architecture metadata: {transformer_ckpt.get('architecture')!r}")
    if int(transformer_ckpt.get("seed", -1)) != TRANSFORMER_TRAINING_SEED:
        raise RuntimeError("Transformer checkpoint training-seed provenance mismatch")
    if int(transformer_ckpt.get("step", -1)) != EXPECTED_STEPS or int(transformer_ckpt.get("tokens_seen", -1)) != EXPECTED_TOKENS:
        raise RuntimeError("Transformer checkpoint is not the completed 2B/30518-step checkpoint")
    transformer = load_checkpoint_model(str(transformer_path), device).eval()
    if transformer.cfg.architecture != "transformer":
        raise RuntimeError("loaded control is not Transformer")
    if transformer_parameter_count(transformer) != EXPECTED_TRANSFORMER_PARAMS:
        raise RuntimeError("Transformer parameter count drift")

    return cortex, transformer


def _run_benchmark_impl(repo_full_name: str, issue_number: int, source_sha: str, source_tree: str, harness_sha: str) -> dict:
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from transformers import AutoTokenizer

    source = _full_sha(source_sha, "source_sha")
    tree = _full_sha(source_tree, "source_tree")
    harness = _full_sha(harness_sha, "harness_sha")
    volume.reload()
    marker_path = Path(MARKER_PATH)
    result_path = Path(RESULT_PATH)
    if not marker_path.exists():
        raise RuntimeError("durable evaluation dispatch marker is missing")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    for key, expected in {
        "status": "EVAL_DISPATCH_CONSUMED",
        "source_sha": source,
        "source_tree": tree,
        "harness_sha": harness,
        "evaluation_sample_seed": EVAL_SEED,
    }.items():
        if marker.get(key) != expected:
            raise RuntimeError(f"evaluation marker mismatch for {key}")
    if result_path.exists():
        raise RuntimeError("evaluation RESULT already exists; refusing duplicate/retry")

    marker["gpu_allocation_started"] = True
    marker["gpu_started_unix"] = time.time()
    _atomic_write(marker_path, marker)
    volume.commit()

    if not torch.cuda.is_available():
        raise RuntimeError("benchmark requires CUDA")
    device = torch.device("cuda")
    torch.manual_seed(EVAL_SEED)

    _comment(
        repo_full_name,
        issue_number,
        "🧪 **CORTEX-S v11 vs Transformer 100M/2B pretrain benchmark started** — "
        "same sampled examples and scoring, pretraining checkpoints only, evaluation-only.",
    )

    benchmark_items: dict[str, list[dict]] = {}
    dataset_manifest: dict[str, dict] = {}
    for spec in PARQUET_SPECS:
        split = str(spec["split"])
        ds = load_dataset("parquet", data_files={split: str(spec["url"])}, split=split)
        if len(ds) < int(spec["min_rows"]):
            raise RuntimeError(f"{spec['name']} parquet too small: {len(ds)} < {spec['min_rows']}")
        parser = PARSERS[str(spec["parser"])]
        sample = ds.shuffle(seed=EVAL_SEED).select(range(min(MCQ_LIMIT, len(ds))))
        benchmark_items[str(spec["name"])] = [parser(dict(row)) for row in sample]
        dataset_manifest[str(spec["name"])] = {
            "url": str(spec["url"]),
            "rows": len(ds),
            "sampled": len(sample),
        }

    gsm = load_dataset("parquet", data_files={"test": GSM8K_URL}, split="test")
    if len(gsm) < 1319:
        raise RuntimeError(f"gsm8k parquet too small: {len(gsm)} < 1319")
    gsm_sample = gsm.shuffle(seed=EVAL_SEED).select(range(min(GSM8K_LIMIT, len(gsm))))
    gsm_items = [
        {"question": str(row["question"]), "answer": _gsm_answer(str(row["answer"]))}
        for row in gsm_sample
    ]
    dataset_manifest["gsm8k"] = {
        "url": GSM8K_URL,
        "rows": len(gsm),
        "sampled": len(gsm_sample),
    }

    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    cortex, transformer = _load_models(device)

    def trim_ids(prompt_ids: list[int], cont_ids: list[int], max_context: int) -> tuple[list[int], list[int]]:
        if not cont_ids:
            return prompt_ids[-max_context:], cont_ids
        max_prompt = max(1, max_context - len(cont_ids) + 1)
        return prompt_ids[-max_prompt:], cont_ids[: max_context - 1]

    def score(model, prompt: str, continuation: str) -> float:
        # Pretraining checkpoints are evaluated with the raw benchmark prompt; no
        # SFT/DPO User/Assistant wrapper is added to either architecture.
        prompt_ids = tokenizer.encode(prompt.strip(), add_special_tokens=False)
        cont_ids = tokenizer.encode(continuation, add_special_tokens=False)
        prompt_ids, cont_ids = trim_ids(
            prompt_ids,
            cont_ids,
            min(EVAL_MAX_CONTEXT, int(model.cfg.max_seq_len)),
        )
        if not prompt_ids or not cont_ids:
            return float("-inf")
        ids = prompt_ids + cont_ids
        x = torch.tensor(ids[:-1], device=device, dtype=torch.long).unsqueeze(0)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        start = len(prompt_ids) - 1
        total = sum(float(log_probs[0, start + j, token_id]) for j, token_id in enumerate(cont_ids))
        return total / len(cont_ids)

    def generate(model, question: str, max_new_tokens: int = 64) -> str:
        prefix = f"Solve the problem. Give a concise answer and end with '#### <number>'.\n\n{question.strip()}\nAnswer:\n"
        ids = tokenizer.encode(prefix, add_special_tokens=False)
        max_seq = min(EVAL_MAX_CONTEXT, int(model.cfg.max_seq_len))
        ids = ids[-max(1, max_seq - max_new_tokens):]
        tokens = torch.tensor(ids, device=device, dtype=torch.long).unsqueeze(0)
        generated: list[int] = []
        eos_id = tokenizer.eos_token_id
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(tokens[:, -max_seq:])[:, -1, :]
                next_id = int(torch.argmax(logits, dim=-1).item())
                if next_id == eos_id:
                    break
                generated.append(next_id)
                tokens = torch.cat(
                    [tokens, torch.tensor([[next_id]], device=device, dtype=torch.long)], dim=1
                )
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    def evaluate_one(name: str, model) -> dict[str, Any]:
        started = time.perf_counter()
        tasks: dict[str, dict[str, Any]] = {}
        for task_name, rows in benchmark_items.items():
            correct = 0
            for row in rows:
                scores = [score(model, row["prompt"], option) for option in row["options"]]
                pred = max(range(len(scores)), key=lambda i: scores[i])
                correct += int(pred == row["answer"])
            accuracy = correct / max(len(rows), 1)
            tasks[task_name] = {
                "correct": correct,
                "total": len(rows),
                "accuracy": accuracy,
                "stderr": math.sqrt(max(accuracy * (1 - accuracy), 0.0) / max(len(rows), 1)),
            }

        gsm_correct = 0
        gsm_rows: list[dict[str, Any]] = []
        for row in gsm_items:
            output = generate(model, row["question"])
            prediction = _normalize_number(output)
            is_correct = prediction is not None and prediction == row["answer"]
            gsm_correct += int(is_correct)
            gsm_rows.append(
                {
                    "question": row["question"],
                    "reference": row["answer"],
                    "prediction": prediction,
                    "output": output,
                    "correct": is_correct,
                }
            )
        gsm_accuracy = gsm_correct / max(len(gsm_items), 1)
        tasks["gsm8k"] = {
            "correct": gsm_correct,
            "total": len(gsm_items),
            "accuracy": gsm_accuracy,
            "stderr": math.sqrt(max(gsm_accuracy * (1 - gsm_accuracy), 0.0) / max(len(gsm_items), 1)),
        }
        macro = sum(float(item["accuracy"]) for item in tasks.values()) / len(tasks)
        return {
            "name": name,
            "tasks": tasks,
            "macro_accuracy": macro,
            "gsm8k_rows": gsm_rows,
            "elapsed_seconds": time.perf_counter() - started,
        }

    cortex_result = evaluate_one("cortex_s_v11_pretrain", cortex)
    transformer_result = evaluate_one("transformer_100m_pretrain", transformer)

    task_order = [str(x["name"]) for x in PARQUET_SPECS] + ["gsm8k"]
    comparison: dict[str, Any] = {}
    summary_rows: list[str] = []
    for task_name in task_order:
        c = float(cortex_result["tasks"][task_name]["accuracy"])
        t = float(transformer_result["tasks"][task_name]["accuracy"])
        comparison[task_name] = {
            "cortex_accuracy": c,
            "transformer_accuracy": t,
            "cortex_minus_transformer_pp": 100.0 * (c - t),
            "winner": "cortex_s" if c > t else "transformer" if t > c else "tie",
        }
        summary_rows.append(
            f"{task_name}: CORTEX={100*c:.1f}% vs Transformer={100*t:.1f}% "
            f"(CORTEX-Transformer={100*(c-t):+.1f} pp)"
        )

    macro_c = float(cortex_result["macro_accuracy"])
    macro_t = float(transformer_result["macro_accuracy"])
    result = {
        "status": "EVALUATION_COMPLETE",
        "classification": "DESCRIPTIVE_PRETRAIN_CHECKPOINT_BENCHMARK_ONLY",
        "source_sha": source,
        "source_tree": tree,
        "harness_sha": harness,
        "evaluation_sample_seed": EVAL_SEED,
        "device": torch.cuda.get_device_name(device),
        "dataset_manifest": dataset_manifest,
        "checkpoints": {
            "cortex_s": CORTEX_CHECKPOINT,
            "transformer": TRANSFORMER_CHECKPOINT,
        },
        "model_provenance": {
            "cortex_s": {
                "parameters": EXPECTED_CORTEX_PARAMS,
                "training_tokens": EXPECTED_TOKENS,
                "training_seed": CORTEX_TRAINING_SEED,
                "frozen_final_validation_nll": CORTEX_FINAL_NLL,
                "frozen_final_validation_perplexity": CORTEX_FINAL_PPL,
            },
            "transformer": {
                "parameters": EXPECTED_TRANSFORMER_PARAMS,
                "training_tokens": EXPECTED_TOKENS,
                "training_seed": TRANSFORMER_TRAINING_SEED,
                "frozen_final_validation_nll": TRANSFORMER_FINAL_NLL,
                "frozen_final_validation_perplexity": TRANSFORMER_FINAL_PPL,
            },
            "training_seed_matched": False,
        },
        "cortex_s": cortex_result,
        "transformer": transformer_result,
        "comparison": {
            "tasks": comparison,
            "macro_accuracy": {
                "cortex": macro_c,
                "transformer": macro_t,
                "cortex_minus_transformer_pp": 100.0 * (macro_c - macro_t),
                "winner": "cortex_s" if macro_c > macro_t else "transformer" if macro_t > macro_c else "tie",
            },
            "historical_equal_token_validation_nll": {
                "cortex": CORTEX_FINAL_NLL,
                "transformer": TRANSFORMER_FINAL_NLL,
                "cortex_minus_transformer": CORTEX_FINAL_NLL - TRANSFORMER_FINAL_NLL,
                "interpretation": "descriptive only; training seeds differ",
            },
        },
        "protocol": {
            "mcq_sample_cap_per_task": MCQ_LIMIT,
            "gsm8k_sample_cap": GSM8K_LIMIT,
            "mcq_scoring": "mean continuation log-likelihood on raw pretraining prompt",
            "generation": "greedy",
            "max_context": EVAL_MAX_CONTEXT,
            "same_sample_indices": True,
            "same_evaluation_seed": True,
            "same_tokenizer": "gpt2",
            "weights_updated": False,
            "optimizer_used": False,
            "backward_used": False,
            "scientific_paired_training_claim_authorized": False,
        },
    }

    _atomic_write(result_path, result)
    volume.commit()
    _comment(
        repo_full_name,
        issue_number,
        "✅ **CORTEX-S v11 vs Transformer pretrain benchmark complete.**\n\n"
        + "\n".join(f"- {row}" for row in summary_rows)
        + f"\n- macro: CORTEX={100*macro_c:.1f}% vs Transformer={100*macro_t:.1f}% "
        + f"(CORTEX-Transformer={100*(macro_c-macro_t):+.1f} pp)"
        + f"\n\nFull JSON: `{RESULT_PATH}`",
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=32768,
    timeout=7200,
    volumes={"/vol": volume},
    secrets=[github_secret],
)
def run_benchmark(repo_full_name: str, issue_number: int, source_sha: str, source_tree: str, harness_sha: str) -> str:
    try:
        result = _run_benchmark_impl(repo_full_name, issue_number, source_sha, source_tree, harness_sha)
        return json.dumps(result, sort_keys=True)
    except Exception as exc:
        volume.reload()
        marker_path = Path(MARKER_PATH)
        error = {
            "status": "EVALUATION_ERROR",
            "classification": "EVALUATION_ONLY_ERROR_NO_RETRY",
            "source_sha": source_sha,
            "source_tree": source_tree,
            "harness_sha": harness_sha,
            "evaluation_sample_seed": EVAL_SEED,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "weights_updated": False,
            "training_authorized": False,
            "retry_authorized": False,
        }
        if marker_path.exists() and not Path(RESULT_PATH).exists():
            _atomic_write(Path(RESULT_PATH), error)
            volume.commit()
        _comment(repo_full_name, issue_number, f"❌ **Evaluation failed and is consumed; no retry.** {type(exc).__name__}: {exc}")
        raise


@app.local_entrypoint()
def main(
    repo_full_name: str = "",
    issue_number: int = 0,
    source_sha: str = "",
    source_tree: str = "",
    harness_sha: str = "",
):
    reserve = reserve_evaluation_dispatch.remote(source_sha, source_tree, harness_sha)
    print(reserve, flush=True)
    result = run_benchmark.remote(repo_full_name, issue_number, source_sha, source_tree, harness_sha)
    print(result, flush=True)
