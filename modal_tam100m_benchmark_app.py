from __future__ import annotations

import json
import math
import re
from pathlib import Path

import modal

APP_NAME = "tam-research-100m-benchmark"
VOLUME_NAME = "tam-research-data"
SEED = 8100
TAM_CHECKPOINT = "/vol/full100m-runs/TAM-v3-100M-2B-seed8100/final_dpo.pt"
RESULT_PATH = "/vol/full100m-runs/TAM-v3-100M-2B-seed8100/external_benchmark_v1.json"
COMPARATOR = "HuggingFaceTB/SmolLM2-135M-Instruct"
EVAL_MAX_CONTEXT = 1024
MCQ_LIMIT = 200
GSM8K_LIMIT = 100

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
        print(
            f"[status-report-nonfatal] {type(exc).__name__}: {exc}; body={body}",
            flush=True,
        )


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


BENCHMARK_SPECS = (
    ("arc_easy", "allenai/ai2_arc", "ARC-Easy", "validation", _arc_item),
    ("arc_challenge", "allenai/ai2_arc", "ARC-Challenge", "validation", _arc_item),
    ("piqa", "ybisk/piqa", None, "validation", _piqa_item),
    ("hellaswag", "Rowan/hellaswag", None, "validation", _hellaswag_item),
    ("openbookqa", "allenai/openbookqa", "main", "validation", _openbook_item),
)


def _normalize_number(text: str) -> str | None:
    matches = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def _gsm_answer(answer: str) -> str | None:
    marker = str(answer).split("####")[-1]
    return _normalize_number(marker)


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
    import os
    import time

    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tam_research.posttrain import load_checkpoint_model

    volume.reload()
    checkpoint = Path(TAM_CHECKPOINT)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"final DPO instruct checkpoint missing: {checkpoint}; "
            "benchmark refuses to fall back to base or SFT"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark requires CUDA")

    os.environ["HF_HOME"] = "/vol/hf-cache"
    device = torch.device("cuda")
    torch.manual_seed(SEED)

    _comment(
        repo_full_name,
        issue_number,
        "🧪 **100M instruct benchmark started on one L4** — final TAM DPO checkpoint vs "
        f"`{COMPARATOR}` using the same sampled examples and scoring code. "
        "No training is performed.",
    )

    benchmark_items: dict[str, list[dict]] = {}
    for name, path, config, split, parser in BENCHMARK_SPECS:
        ds = load_dataset(path, config, split=split) if config else load_dataset(path, split=split)
        sample = ds.shuffle(seed=SEED).select(range(min(MCQ_LIMIT, len(ds))))
        benchmark_items[name] = [parser(dict(row)) for row in sample]

    gsm = load_dataset("openai/gsm8k", "main", split="test")
    gsm_sample = gsm.shuffle(seed=SEED).select(range(min(GSM8K_LIMIT, len(gsm))))
    gsm_items = [
        {"question": str(row["question"]), "answer": _gsm_answer(str(row["answer"]))}
        for row in gsm_sample
    ]

    def trim_ids(
        prompt_ids: list[int], cont_ids: list[int], max_context: int
    ) -> tuple[list[int], list[int]]:
        if not cont_ids:
            return prompt_ids[-max_context:], cont_ids
        max_prompt = max(1, max_context - len(cont_ids) + 1)
        return prompt_ids[-max_prompt:], cont_ids[: max_context - 1]

    def score_tam(model, tokenizer, prompt: str, continuation: str) -> float:
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
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            logits = model(x)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        start = len(prompt_ids) - 1
        score = sum(
            float(log_probs[0, start + j, token_id])
            for j, token_id in enumerate(cont_ids)
        )
        return score / len(cont_ids)

    def score_hf(model, tokenizer, prompt: str, continuation: str) -> float:
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt.strip()}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer.encode(prefix, add_special_tokens=False)
        cont_ids = tokenizer.encode(continuation, add_special_tokens=False)
        prompt_ids, cont_ids = trim_ids(prompt_ids, cont_ids, EVAL_MAX_CONTEXT)
        if not prompt_ids or not cont_ids:
            return float("-inf")
        ids = prompt_ids + cont_ids
        x = torch.tensor(ids[:-1], device=device, dtype=torch.long).unsqueeze(0)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            logits = model(input_ids=x).logits
        log_probs = F.log_softmax(logits.float(), dim=-1)
        start = len(prompt_ids) - 1
        score = sum(
            float(log_probs[0, start + j, token_id])
            for j, token_id in enumerate(cont_ids)
        )
        return score / len(cont_ids)

    def generate_tam(model, tokenizer, question: str, max_new_tokens: int = 64) -> str:
        prefix = (
            "User:\nSolve the problem. Give a concise answer and end with "
            f"'#### <number>'.\n\n{question.strip()}\nAssistant:\n"
        )
        ids = tokenizer.encode(prefix, add_special_tokens=False)
        max_seq = min(EVAL_MAX_CONTEXT, int(model.cfg.max_seq_len))
        ids = ids[-max(1, max_seq - max_new_tokens) :]
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
                    [
                        tokens,
                        torch.tensor([[next_id]], device=device, dtype=torch.long),
                    ],
                    dim=1,
                )
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    def generate_hf(model, tokenizer, question: str, max_new_tokens: int = 64) -> str:
        content = (
            "Solve the problem. Give a concise answer and end with '#### <number>'.\n\n"
            + question.strip()
        )
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tokenizer.encode(prefix, add_special_tokens=False)[
            -max(1, EVAL_MAX_CONTEXT - max_new_tokens) :
        ]
        x = torch.tensor(ids, device=device, dtype=torch.long).unsqueeze(0)
        with torch.inference_mode():
            out = model.generate(
                x,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0, len(ids) :], skip_special_tokens=True).strip()

    def evaluate_model(name: str, model, tokenizer, score_fn, generate_fn) -> dict:
        started = time.perf_counter()
        task_results: dict[str, dict] = {}
        for task_name, rows in benchmark_items.items():
            correct = 0
            for row in rows:
                scores = [
                    score_fn(model, tokenizer, row["prompt"], option)
                    for option in row["options"]
                ]
                pred = max(range(len(scores)), key=lambda i: scores[i])
                correct += int(pred == row["answer"])
            accuracy = correct / max(len(rows), 1)
            stderr = math.sqrt(
                max(accuracy * (1 - accuracy), 0.0) / max(len(rows), 1)
            )
            task_results[task_name] = {
                "correct": correct,
                "total": len(rows),
                "accuracy": accuracy,
                "stderr": stderr,
            }
            _comment(
                repo_full_name,
                issue_number,
                f"📊 **{name} {task_name}** — {correct}/{len(rows)} = "
                f"{100*accuracy:.1f}% (sample cap={MCQ_LIMIT}).",
            )

        gsm_correct = 0
        gsm_rows = []
        for row in gsm_items:
            output = generate_fn(model, tokenizer, row["question"])
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
        task_results["gsm8k"] = {
            "correct": gsm_correct,
            "total": len(gsm_items),
            "accuracy": gsm_accuracy,
            "stderr": math.sqrt(
                max(gsm_accuracy * (1 - gsm_accuracy), 0.0)
                / max(len(gsm_items), 1)
            ),
        }
        _comment(
            repo_full_name,
            issue_number,
            f"📊 **{name} gsm8k** — {gsm_correct}/{len(gsm_items)} = "
            f"{100*gsm_accuracy:.1f}% (greedy, sample cap={GSM8K_LIMIT}).",
        )
        return {
            "tasks": task_results,
            "gsm8k_rows": gsm_rows,
            "elapsed_seconds": time.perf_counter() - started,
        }

    overall_started = time.perf_counter()

    tam_tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    tam_model = load_checkpoint_model(str(checkpoint), device)
    tam_model.eval()
    tam_result = evaluate_model(
        "TAM-DPO", tam_model, tam_tokenizer, score_tam, generate_tam
    )
    del tam_model
    torch.cuda.empty_cache()

    hf_tokenizer = AutoTokenizer.from_pretrained(COMPARATOR, use_fast=True)
    if hf_tokenizer.pad_token_id is None:
        hf_tokenizer.pad_token_id = hf_tokenizer.eos_token_id
    hf_model = AutoModelForCausalLM.from_pretrained(
        COMPARATOR,
        torch_dtype=torch.bfloat16,
    ).to(device)
    hf_model.eval()
    comparator_result = evaluate_model(
        "SmolLM2-135M-Instruct",
        hf_model,
        hf_tokenizer,
        score_hf,
        generate_hf,
    )
    del hf_model
    torch.cuda.empty_cache()

    result = {
        "benchmark": "TAM-v3-100M instruct external benchmark v1",
        "seed": SEED,
        "device": torch.cuda.get_device_name(device),
        "tam": {
            "checkpoint": TAM_CHECKPOINT,
            "parameters": 101_806_616,
            "pretraining_tokens": 2_000_000_000,
            **tam_result,
        },
        "comparator": {
            "model": COMPARATOR,
            "parameters_approx": 135_000_000,
            "reported_pretraining_tokens": 2_000_000_000_000,
            "note": "external reference only; not a matched-data architecture control",
            **comparator_result,
        },
        "protocol": {
            "mcq_sample_cap_per_task": MCQ_LIMIT,
            "gsm8k_sample_cap": GSM8K_LIMIT,
            "mcq_scoring": (
                "mean continuation log-likelihood; model-specific instruct prompt wrapper"
            ),
            "generation": "greedy",
            "max_context": EVAL_MAX_CONTEXT,
            "same_sample_indices": True,
        },
        "total_elapsed_seconds": time.perf_counter() - overall_started,
    }

    Path(RESULT_PATH).write_text(json.dumps(result, indent=2))
    volume.commit()

    rows = []
    for task_name in [x[0] for x in BENCHMARK_SPECS] + ["gsm8k"]:
        ta = result["tam"]["tasks"][task_name]["accuracy"]
        ca = result["comparator"]["tasks"][task_name]["accuracy"]
        rows.append(f"{task_name}: TAM={100*ta:.1f}% vs SmolLM2={100*ca:.1f}%")
    _comment(
        repo_full_name,
        issue_number,
        "✅ **100M instruct benchmark complete.**\n\n"
        + "\n".join(f"- {row}" for row in rows)
        + f"\n\nFull JSON: `{RESULT_PATH}`",
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main(repo_full_name: str = "", issue_number: int = 0):
    result = run_benchmark.remote(repo_full_name, issue_number)
    print(json.dumps(result, indent=2))
