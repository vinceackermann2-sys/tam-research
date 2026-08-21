from __future__ import annotations

import json
import math
import modal

APP_NAME = "tam-research-behavior-eval"
VOLUME_NAME = "tam-research-data"
SEED = 7400

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
github_secret = modal.Secret.from_name("github-secret")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.7,<2.11",
        "transformers>=4.55,<5",
        "tokenizers>=0.21,<1",
        "numpy>=2.0,<3",
        "PyGithub>=2.3,<3",
    )
    .add_local_python_source("tam_research")
)


def _comment(repo_full_name: str, issue_number: int, body: str) -> None:
    if not repo_full_name or not issue_number:
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


MCQ_TASKS = [
    {
        "id": "arithmetic_17x6",
        "prompt": "Question: What is 17 * 6?\nAnswer:",
        "options": [" 92", " 102", " 112", " 116"],
        "answer": 1,
        "category": "arithmetic",
    },
    {
        "id": "capital_france",
        "prompt": "Question: What is the capital of France?\nAnswer:",
        "options": [" Berlin", " Madrid", " Paris", " Rome"],
        "answer": 2,
        "category": "knowledge",
    },
    {
        "id": "number_sequence",
        "prompt": "Complete the sequence: 2, 4, 6, 8,",
        "options": [" 9", " 10", " 11", " 12"],
        "answer": 1,
        "category": "pattern",
    },
    {
        "id": "transitive_age",
        "prompt": "Alice is older than Bob. Bob is older than Carol. Who is oldest?\nAnswer:",
        "options": [" Alice", " Bob", " Carol", " Cannot know"],
        "answer": 0,
        "category": "reasoning",
    },
    {
        "id": "balls_total",
        "prompt": "A box has 3 red balls and 2 blue balls. How many balls are in the box?\nAnswer:",
        "options": [" 4", " 5", " 6", " 7"],
        "answer": 1,
        "category": "arithmetic",
    },
    {
        "id": "mammal",
        "prompt": "Which animal is a mammal?\nAnswer:",
        "options": [" shark", " dolphin", " trout", " salmon"],
        "answer": 1,
        "category": "knowledge",
    },
    {
        "id": "opposite_hot",
        "prompt": "The opposite of hot is",
        "options": [" cold", " warm", " dry", " bright"],
        "answer": 0,
        "category": "language",
    },
    {
        "id": "python_function_keyword",
        "prompt": "In Python, which keyword begins a function definition?\nAnswer:",
        "options": [" class", " def", " let", " func"],
        "answer": 1,
        "category": "coding",
    },
    {
        "id": "robin_logic",
        "prompt": "All robins are birds. Tweety is a robin. Therefore Tweety is a",
        "options": [" bird", " fish", " reptile", " plant"],
        "answer": 0,
        "category": "reasoning",
    },
    {
        "id": "water_freezing",
        "prompt": "At standard atmospheric pressure, water freezes at",
        "options": [" 0 degrees Celsius", " 10 degrees Celsius", " 50 degrees Celsius", " 100 degrees Celsius"],
        "answer": 0,
        "category": "knowledge",
    },
    {
        "id": "simple_subtraction",
        "prompt": "Question: What is 15 - 7?\nAnswer:",
        "options": [" 6", " 7", " 8", " 9"],
        "answer": 2,
        "category": "arithmetic",
    },
    {
        "id": "comparison",
        "prompt": "Which number is larger?\nAnswer:",
        "options": [" 37", " 73", " They are equal", " Cannot know"],
        "answer": 1,
        "category": "reasoning",
    },
]

GEN_PROMPTS = [
    {
        "id": "strict_ok",
        "prompt": "Reply with exactly the word OK and nothing else.\nAssistant:",
        "max_new_tokens": 8,
    },
    {
        "id": "arithmetic_freeform",
        "prompt": "What is 17 * 6? Answer with only the number.\nAssistant:",
        "max_new_tokens": 12,
    },
    {
        "id": "coding_add",
        "prompt": "Write a short Python function add(a, b) that returns their sum.\nAssistant:",
        "max_new_tokens": 28,
    },
    {
        "id": "explain_sky",
        "prompt": "Explain why the sky looks blue in one short sentence.\nAssistant:",
        "max_new_tokens": 28,
    },
    {
        "id": "three_tips",
        "prompt": "Give exactly three short tips for organizing a busy workday.\nAssistant:",
        "max_new_tokens": 32,
    },
]


@app.function(
    image=image,
    cpu=8,
    memory=16384,
    timeout=900,
    volumes={"/vol": volume},
    secrets=[github_secret],
)
def evaluate_full25m(repo_full_name: str = "", issue_number: int = 0) -> dict:
    import time
    from pathlib import Path

    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    from tam_research.posttrain import load_checkpoint_model

    volume.reload()
    root = Path(f"/vol/full-model-runs/TAM-v3-25M-Full-seed{SEED}")
    checkpoints = {
        "base": Path(
            "/vol/full-model-runs/pretrain/25m/ctx512-mb64-ga2/"
            f"tamv3-25m-compiled-seed{SEED}/latest.pt"
        ),
        "sft": root / "sft.pt",
        "dpo": root / "final_dpo.pt",
    }
    missing = [str(path) for path in checkpoints.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing benchmark checkpoints: {missing}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    eos_id = tokenizer.eos_token_id
    device = torch.device("cpu")
    torch.manual_seed(SEED)

    def candidate_logp(model, prompt: str, continuation: str) -> float:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        cont_ids = tokenizer.encode(continuation, add_special_tokens=False)
        if not prompt_ids or not cont_ids:
            return float("-inf")
        ids = prompt_ids + cont_ids
        x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            log_probs = F.log_softmax(logits.float(), dim=-1)
        start = len(prompt_ids) - 1
        score = 0.0
        for j, token_id in enumerate(cont_ids):
            score += float(log_probs[0, start + j, token_id])
        # Normalize so longer verbal options are not automatically penalized.
        return score / len(cont_ids)

    def greedy_generate(model, prompt: str, max_new_tokens: int) -> str:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        # Leave room inside the model's learned position table.
        max_seq = int(model.cfg.max_seq_len)
        ids = ids[-max(1, max_seq - max_new_tokens) :]
        for _ in range(max_new_tokens):
            x = torch.tensor(ids[-max_seq:], dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad():
                logits = model(x)
            next_id = int(torch.argmax(logits[0, -1]).item())
            if next_id == eos_id:
                break
            ids.append(next_id)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False)[-max(1, max_seq - max_new_tokens) :])
        return tokenizer.decode(ids[prompt_len:], skip_special_tokens=True)

    result = {
        "benchmark": "TAM-v3-25M-Full behavior eval v1",
        "seed": SEED,
        "device": "CPU",
        "checkpoints": {name: str(path) for name, path in checkpoints.items()},
        "mcq_tasks": MCQ_TASKS,
        "generation_prompts": GEN_PROMPTS,
        "models": {},
    }

    _comment(
        repo_full_name,
        issue_number,
        "🧪 **CPU-only TAM-v3-25M behavioral benchmark started** — Base vs SFT vs DPO; no GPU allocated.",
    )
    overall_started = time.perf_counter()

    for stage, checkpoint in checkpoints.items():
        stage_started = time.perf_counter()
        model = load_checkpoint_model(str(checkpoint), device)
        model.eval()

        mcq_rows = []
        correct = 0
        category_counts: dict[str, list[int]] = {}
        for task in MCQ_TASKS:
            scores = [candidate_logp(model, task["prompt"], option) for option in task["options"]]
            prediction = max(range(len(scores)), key=lambda i: scores[i])
            is_correct = prediction == task["answer"]
            correct += int(is_correct)
            category_counts.setdefault(task["category"], [0, 0])
            category_counts[task["category"]][0] += int(is_correct)
            category_counts[task["category"]][1] += 1
            mcq_rows.append(
                {
                    "id": task["id"],
                    "category": task["category"],
                    "prediction_index": prediction,
                    "prediction": task["options"][prediction].strip(),
                    "answer_index": task["answer"],
                    "answer": task["options"][task["answer"]].strip(),
                    "correct": is_correct,
                    "normalized_logps": scores,
                }
            )

        generations = []
        for item in GEN_PROMPTS:
            text = greedy_generate(model, item["prompt"], item["max_new_tokens"])
            generations.append({"id": item["id"], "prompt": item["prompt"], "output": text})

        stage_result = {
            "mcq_correct": correct,
            "mcq_total": len(MCQ_TASKS),
            "mcq_accuracy": correct / len(MCQ_TASKS),
            "category_accuracy": {
                category: wins / total for category, (wins, total) in category_counts.items()
            },
            "mcq": mcq_rows,
            "generations": generations,
            "elapsed_seconds": time.perf_counter() - stage_started,
        }
        result["models"][stage] = stage_result
        _comment(
            repo_full_name,
            issue_number,
            f"📊 **{stage.upper()} benchmark finished** — MCQ={correct}/{len(MCQ_TASKS)} "
            f"({100*stage_result['mcq_accuracy']:.1f}%), CPU elapsed={stage_result['elapsed_seconds']:.1f}s.",
        )
        del model

    result["total_elapsed_seconds"] = time.perf_counter() - overall_started

    # Simple deterministic generation checks, reported separately from MCQ accuracy.
    for stage, stage_result in result["models"].items():
        generated = {row["id"]: row["output"].strip() for row in stage_result["generations"]}
        checks = {
            "strict_ok": generated.get("strict_ok", "").upper() == "OK",
            "arithmetic_contains_102": "102" in generated.get("arithmetic_freeform", ""),
            "coding_has_def_or_return": (
                "def" in generated.get("coding_add", "") or "return" in generated.get("coding_add", "")
            ),
        }
        stage_result["generation_checks"] = checks
        stage_result["generation_check_score"] = sum(checks.values()) / len(checks)

    output_path = root / "behavior_eval_v1.json"
    output_path.write_text(json.dumps(result, indent=2))
    volume.commit()

    base_acc = result["models"]["base"]["mcq_accuracy"]
    sft_acc = result["models"]["sft"]["mcq_accuracy"]
    dpo_acc = result["models"]["dpo"]["mcq_accuracy"]
    _comment(
        repo_full_name,
        issue_number,
        f"✅ **Behavior benchmark complete** — MCQ accuracy Base={base_acc:.3f}, "
        f"SFT={sft_acc:.3f}, DPO={dpo_acc:.3f}. Full results: `{output_path}`",
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main(repo_full_name: str = "", issue_number: int = 0):
    result = evaluate_full25m.remote(repo_full_name, issue_number)
    print(json.dumps(result, indent=2))
