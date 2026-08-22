from __future__ import annotations

import json
import math
import re
from pathlib import Path

import modal

APP_NAME = "tam-research-transformer100m-benchmark"
VOLUME_NAME = "tam-research-data"
SEED = 8100
CHECKPOINT = "/vol/full100m-runs/Transformer-100M-2B-seed8100/final_dpo.pt"
TAM_RESULT_PATH = "/vol/full100m-runs/TAM-v3-100M-2B-seed8100/external_benchmark_v2.json"
RESULT_PATH = "/vol/full100m-runs/Transformer-100M-2B-seed8100/external_benchmark_v2.json"
EVAL_MAX_CONTEXT = 1024
MCQ_LIMIT = 200
GSM8K_LIMIT = 100

# Identical dataset files, split choices, seed and sample caps to
# modal_tam100m_benchmark_v2_app.py.
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
        "torch>=2.7,<2.11",
        "transformers>=4.55,<5",
        "tokenizers>=0.21,<1",
        "datasets>=4.0,<5",
        "numpy>=2.0,<3",
        "huggingface-hub>=0.34,<1",
        "PyGithub>=2.3,<3",
    )
    .add_local_python_source("tam_research")
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


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=32768,
    timeout=3600,
    volumes={"/vol": volume},
    secrets=[github_secret],
)
def run_benchmark(repo_full_name: str = "", issue_number: int = 0) -> dict:
    import time

    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from tam_research.posttrain import load_checkpoint_model

    volume.reload()
    checkpoint = Path(CHECKPOINT)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"matched Transformer final DPO checkpoint missing: {checkpoint}; refusing base/SFT fallback"
        )
    tam_result_file = Path(TAM_RESULT_PATH)
    if not tam_result_file.exists():
        raise FileNotFoundError(
            f"historical TAM benchmark result missing: {tam_result_file}; refusing unmatched comparison"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark requires CUDA")

    device = torch.device("cuda")
    torch.manual_seed(SEED)

    _comment(
        repo_full_name,
        issue_number,
        "🧪 **Matched Transformer-100M instruct benchmark started on one L4** — "
        "same v2 datasets/sample seed/scoring as TAM; final DPO checkpoint only; evaluation only.",
    )

    benchmark_items: dict[str, list[dict]] = {}
    dataset_manifest: dict[str, dict] = {}
    for spec in PARQUET_SPECS:
        split = str(spec["split"])
        ds = load_dataset(
            "parquet",
            data_files={split: str(spec["url"])},
            split=split,
        )
        if len(ds) < int(spec["min_rows"]):
            raise RuntimeError(
                f"{spec['name']} parquet too small: {len(ds)} < {spec['min_rows']}"
            )
        parser = PARSERS[str(spec["parser"])]
        sample = ds.shuffle(seed=SEED).select(range(min(MCQ_LIMIT, len(ds))))
        benchmark_items[str(spec["name"])] = [parser(dict(row)) for row in sample]
        dataset_manifest[str(spec["name"])] = {
            "url": str(spec["url"]),
            "rows": len(ds),
            "sampled": len(sample),
        }

    gsm = load_dataset(
        "parquet",
        data_files={"test": GSM8K_URL},
        split="test",
    )
    if len(gsm) < 1319:
        raise RuntimeError(f"gsm8k parquet too small: {len(gsm)} < 1319")
    gsm_sample = gsm.shuffle(seed=SEED).select(range(min(GSM8K_LIMIT, len(gsm))))
    gsm_items = [
        {"question": str(row["question"]), "answer": _gsm_answer(str(row["answer"]))}
        for row in gsm_sample
    ]
    dataset_manifest["gsm8k"] = {
        "url": GSM8K_URL,
        "rows": len(gsm),
        "sampled": len(gsm_sample),
    }

    _comment(
        repo_full_name,
        issue_number,
        "✅ **Matched benchmark data preflight passed inside Modal** — all six identical sampled datasets loaded; starting Transformer scoring.",
    )

    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    model = load_checkpoint_model(str(checkpoint), device)
    model.eval()
    if model.cfg.architecture != "transformer":
        raise RuntimeError(f"expected transformer checkpoint, found {model.cfg.architecture!r}")

    def trim_ids(prompt_ids: list[int], cont_ids: list[int], max_context: int) -> tuple[list[int], list[int]]:
        if not cont_ids:
            return prompt_ids[-max_context:], cont_ids
        max_prompt = max(1, max_context - len(cont_ids) + 1)
        return prompt_ids[-max_prompt:], cont_ids[: max_context - 1]

    def score(prompt: str, continuation: str) -> float:
        # Intentionally identical to score_tam() in benchmark v2: both matched
        # models use the same GPT-2 tokenizer and User/Assistant post-training format.
        prefix = f"User:\n{prompt.strip()}\nAssistant:\n"
        prompt_ids = tokenizer.encode(prefix, add_special_tokens=False)
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
        total = sum(
            float(log_probs[0, start + j, token_id])
            for j, token_id in enumerate(cont_ids)
        )
        return total / len(cont_ids)

    def generate(question: str, max_new_tokens: int = 64) -> str:
        prefix = (
            "User:\nSolve the problem. Give a concise answer and end with '#### <number>'.\n\n"
            f"{question.strip()}\nAssistant:\n"
        )
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
                    [tokens, torch.tensor([[next_id]], device=device, dtype=torch.long)],
                    dim=1,
                )
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    started = time.perf_counter()
    tasks: dict[str, dict] = {}
    for task_name, rows in benchmark_items.items():
        correct = 0
        for row in rows:
            scores = [score(row["prompt"], option) for option in row["options"]]
            pred = max(range(len(scores)), key=lambda i: scores[i])
            correct += int(pred == row["answer"])
        accuracy = correct / max(len(rows), 1)
        stderr = math.sqrt(max(accuracy * (1 - accuracy), 0.0) / max(len(rows), 1))
        tasks[task_name] = {
            "correct": correct,
            "total": len(rows),
            "accuracy": accuracy,
            "stderr": stderr,
        }
        _comment(
            repo_full_name,
            issue_number,
            f"📊 **Transformer-DPO {task_name}** — {correct}/{len(rows)} = {100*accuracy:.1f}%.",
        )

    gsm_correct = 0
    gsm_rows = []
    for row in gsm_items:
        output = generate(row["question"])
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
    _comment(
        repo_full_name,
        issue_number,
        f"📊 **Transformer-DPO gsm8k** — {gsm_correct}/{len(gsm_items)} = {100*gsm_accuracy:.1f}%.",
    )

    tam_historical = json.loads(tam_result_file.read_text())
    tam_tasks = tam_historical["tam"]["tasks"]
    comparison = {}
    rows = []
    for task_name in [str(x["name"]) for x in PARQUET_SPECS] + ["gsm8k"]:
        transformer_accuracy = float(tasks[task_name]["accuracy"])
        tam_accuracy = float(tam_tasks[task_name]["accuracy"])
        comparison[task_name] = {
            "tam_accuracy": tam_accuracy,
            "transformer_accuracy": transformer_accuracy,
            "tam_minus_transformer_pp": 100.0 * (tam_accuracy - transformer_accuracy),
        }
        rows.append(
            f"{task_name}: TAM={100*tam_accuracy:.1f}% vs Transformer={100*transformer_accuracy:.1f}% "
            f"(TAM-Transformer={100*(tam_accuracy-transformer_accuracy):+.1f} pp)"
        )

    result = {
        "benchmark": "matched TAM-v3-100M vs Transformer-100M instruct benchmark v2",
        "seed": SEED,
        "device": torch.cuda.get_device_name(device),
        "checkpoint": CHECKPOINT,
        "parameters": 101_803_520,
        "pretraining_tokens": 2_000_000_000,
        "dataset_manifest": dataset_manifest,
        "transformer": {
            "tasks": tasks,
            "gsm8k_rows": gsm_rows,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "tam_historical_result": TAM_RESULT_PATH,
        "comparison": comparison,
        "protocol": {
            "version": 2,
            "mcq_sample_cap_per_task": MCQ_LIMIT,
            "gsm8k_sample_cap": GSM8K_LIMIT,
            "mcq_scoring": "mean continuation log-likelihood; identical User/Assistant wrapper for matched TAM and Transformer",
            "generation": "greedy",
            "max_context": EVAL_MAX_CONTEXT,
            "same_sample_indices": True,
            "same_seed": True,
            "dataset_loading": "same explicit script-free Parquet URLs as TAM benchmark v2",
            "weights_updated": False,
        },
    }

    Path(RESULT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT_PATH).write_text(json.dumps(result, indent=2))
    volume.commit()

    _comment(
        repo_full_name,
        issue_number,
        "✅ **Matched TAM vs Transformer instruct benchmark complete.**\n\n"
        + "\n".join(f"- {row}" for row in rows)
        + f"\n\nFull JSON: `{RESULT_PATH}`",
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main(repo_full_name: str = "", issue_number: int = 0):
    result = run_benchmark.remote(repo_full_name, issue_number)
    print(json.dumps(result, indent=2))
