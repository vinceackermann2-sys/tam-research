from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import modal

APP_NAME = "tam-research-long-context-state-eval"
VOLUME_NAME = "tam-research-data"
SEED = 8100
TAM_CHECKPOINT = "/vol/full100m-runs/TAM-v3-100M-2B-seed8100/final_dpo.pt"
TRANSFORMER_CHECKPOINT = "/vol/full100m-runs/Transformer-100M-2B-seed8100/final_dpo.pt"
RESULT_PATH = "/vol/full100m-runs/tam-vs-transformer-long-context-state-seed8100.json"
CONTEXT_LENGTHS = (128, 256, 384, 512, 640, 768, 1000)
TASKS = ("delayed_recall", "associative_recall", "state_tracking", "needle_retrieval")
TRIALS_PER_CELL = 96
BATCH_SIZE = 8
BOOTSTRAP_SAMPLES = 5000
TRAIN_CONTEXT = 512

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


@app.function(image=image, cpu=2, memory=4096, timeout=300, volumes={"/vol": volume})
def preflight() -> dict:
    volume.reload()
    paths = [Path(TAM_CHECKPOINT), Path(TRANSFORMER_CHECKPOINT)]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"required final DPO checkpoints missing: {missing}")
    return {"ok": True, "paths": [str(p) for p in paths]}


def _single_token_values(tokenizer) -> list[tuple[str, int]]:
    candidates = [
        "red", "blue", "green", "black", "white", "brown", "gold", "pink",
        "north", "south", "east", "west", "stone", "river", "cloud", "field",
        "apple", "table", "house", "paper", "music", "water", "light", "night",
    ]
    values: list[tuple[str, int]] = []
    for word in candidates:
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(ids) == 1:
            values.append((word, int(ids[0])))
    if len(values) < 12:
        raise RuntimeError(f"too few verified single-token answer values: {values}")
    return values


def _fill_tokens(tokenizer) -> list[int]:
    text = (
        " The room remains quiet while unrelated notes are reviewed."
        " A neutral record is filed and no instruction changes."
        " Ordinary details continue without altering the remembered value."
        " Another unrelated observation is recorded for completeness."
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) < 16:
        raise RuntimeError("filler token pool unexpectedly short")
    return [int(x) for x in ids]


def _take_filler(pool: list[int], n: int, offset: int) -> list[int]:
    if n <= 0:
        return []
    m = len(pool)
    return [pool[(offset + i) % m] for i in range(n)]


def _encode(tokenizer, text: str) -> list[int]:
    return [int(x) for x in tokenizer.encode(text, add_special_tokens=False)]


def _choices(rng: random.Random, values: list[tuple[str, int]]) -> tuple[list[tuple[str, int]], int]:
    selected = rng.sample(values, 4)
    correct_index = rng.randrange(4)
    # The builder will use selected[correct_index] as the true value.
    return selected, correct_index


def _fit_one_gap(prefix: list[int], suffix: list[int], total_len: int, filler: list[int], offset: int) -> list[int]:
    gap = total_len - len(prefix) - len(suffix)
    if gap < 0:
        raise RuntimeError(f"fixed prompt pieces exceed target context: fixed={len(prefix)+len(suffix)} target={total_len}")
    ids = prefix + _take_filler(filler, gap, offset) + suffix
    if len(ids) != total_len:
        raise AssertionError((len(ids), total_len))
    return ids


def _fit_three_gaps(
    pieces: tuple[list[int], list[int], list[int], list[int]],
    total_len: int,
    filler: list[int],
    offset: int,
) -> list[int]:
    fixed = sum(len(x) for x in pieces)
    gap = total_len - fixed
    if gap < 0:
        raise RuntimeError(f"fixed state prompt exceeds target context: fixed={fixed} target={total_len}")
    g1 = int(gap * 0.22)
    g2 = int(gap * 0.28)
    g3 = gap - g1 - g2
    a, b, c, d = pieces
    ids = (
        a
        + _take_filler(filler, g1, offset)
        + b
        + _take_filler(filler, g2, offset + g1 + 11)
        + c
        + _take_filler(filler, g3, offset + g1 + g2 + 23)
        + d
    )
    if len(ids) != total_len:
        raise AssertionError((len(ids), total_len))
    return ids


def _fit_needle(
    header: list[int],
    needle: list[int],
    suffix: list[int],
    total_len: int,
    filler: list[int],
    offset: int,
    fraction: float,
) -> list[int]:
    fixed = len(header) + len(needle) + len(suffix)
    gap = total_len - fixed
    if gap < 0:
        raise RuntimeError(f"fixed needle prompt exceeds target context: fixed={fixed} target={total_len}")
    before = int(gap * fraction)
    after = gap - before
    ids = (
        header
        + _take_filler(filler, before, offset)
        + needle
        + _take_filler(filler, after, offset + before + 17)
        + suffix
    )
    if len(ids) != total_len:
        raise AssertionError((len(ids), total_len))
    return ids


def _build_trials(tokenizer) -> dict[str, dict[int, list[dict]]]:
    values = _single_token_values(tokenizer)
    filler = _fill_tokens(tokenizer)
    out: dict[str, dict[int, list[dict]]] = {task: {} for task in TASKS}

    for task_index, task in enumerate(TASKS):
        for context_len in CONTEXT_LENGTHS:
            rows: list[dict] = []
            for trial in range(TRIALS_PER_CELL):
                rng = random.Random(SEED + task_index * 10_000_000 + context_len * 10_000 + trial)
                selected, correct_index = _choices(rng, values)
                correct_word, _ = selected[correct_index]
                option_ids = [token_id for _, token_id in selected]
                offset = rng.randrange(len(filler))

                if task == "delayed_recall":
                    prefix = _encode(
                        tokenizer,
                        f"User:\nMemory test. The secret word is {correct_word}. Remember it exactly.\n",
                    )
                    suffix = _encode(
                        tokenizer,
                        "\nWhat was the secret word? Answer with one word.\nAssistant:\n",
                    )
                    ids = _fit_one_gap(prefix, suffix, context_len, filler, offset)

                elif task == "associative_recall":
                    bindings = rng.sample(values, 6)
                    target_binding = rng.randrange(6)
                    # Force the queried binding to use the designated correct option.
                    key_names = ["Aster", "Birch", "Cedar", "Delta", "Elm", "Frost"]
                    binding_words = [word for word, _ in bindings]
                    binding_words[target_binding] = correct_word
                    lines = " ".join(
                        f"Key {key_names[i]} maps to {binding_words[i]}." for i in range(6)
                    )
                    prefix = _encode(tokenizer, f"User:\nAssociation memory test. {lines}\n")
                    suffix = _encode(
                        tokenizer,
                        f"\nWhat value does Key {key_names[target_binding]} map to? Answer with one word.\nAssistant:\n",
                    )
                    ids = _fit_one_gap(prefix, suffix, context_len, filler, offset)

                elif task == "state_tracking":
                    other = [x for x in values if x[0] != correct_word]
                    initial_word = rng.choice(other)[0]
                    middle_word = rng.choice([x for x in other if x[0] != initial_word])[0]
                    distract_word = rng.choice([x for x in other if x[0] not in {initial_word, middle_word}])[0]
                    a = _encode(
                        tokenizer,
                        f"User:\nState tracking test. Register A starts as {initial_word}. Register B starts as {distract_word}.\n",
                    )
                    b = _encode(
                        tokenizer,
                        f"\nUpdate: Register A is now {middle_word}. Register B stays {distract_word}.\n",
                    )
                    c = _encode(
                        tokenizer,
                        f"\nFinal update: Register A is now {correct_word}. Register B stays {distract_word}.\n",
                    )
                    d = _encode(
                        tokenizer,
                        "\nWhat is the current value of Register A? Answer with one word.\nAssistant:\n",
                    )
                    ids = _fit_three_gaps((a, b, c, d), context_len, filler, offset)

                elif task == "needle_retrieval":
                    distractors = [x[0] for x in selected if x[0] != correct_word]
                    header = _encode(
                        tokenizer,
                        "User:\nNeedle retrieval test. Ignore unrelated records and answer only the requested vault. "
                        f"Vault Alpha code is {distractors[0]}. Vault Beta code is {distractors[1]}.\n",
                    )
                    needle = _encode(tokenizer, f"\nIMPORTANT: Vault Quartz code is {correct_word}.\n")
                    suffix = _encode(
                        tokenizer,
                        f"\nVault Gamma code is {distractors[2]}. What is the code for Vault Quartz? Answer with one word.\nAssistant:\n",
                    )
                    needle_fraction = (0.10, 0.50, 0.80)[trial % 3]
                    ids = _fit_needle(
                        header,
                        needle,
                        suffix,
                        context_len,
                        filler,
                        offset,
                        needle_fraction,
                    )
                else:
                    raise AssertionError(task)

                rows.append(
                    {
                        "ids": ids,
                        "option_ids": option_ids,
                        "correct_index": correct_index,
                    }
                )
            out[task][context_len] = rows
    return out


def _bootstrap_ci(diffs, samples: int, seed: int) -> tuple[float, float, float]:
    import numpy as np

    x = np.asarray(diffs, dtype=np.float64)
    if x.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for i in range(samples):
        idx = rng.integers(0, x.size, size=x.size)
        means[i] = x[idx].mean()
    return float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=32768,
    timeout=1800,
    volumes={"/vol": volume},
    secrets=[github_secret],
)
def run_eval(repo_full_name: str = "", issue_number: int = 0) -> dict:
    import torch
    from transformers import AutoTokenizer

    from tam_research.posttrain import load_checkpoint_model

    volume.reload()
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("long-context evaluation requires CUDA")

    tam_path = Path(TAM_CHECKPOINT)
    transformer_path = Path(TRANSFORMER_CHECKPOINT)
    if not tam_path.exists() or not transformer_path.exists():
        raise FileNotFoundError("final DPO checkpoint missing; refusing fallback")

    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    trials = _build_trials(tokenizer)
    _comment(
        repo_full_name,
        issue_number,
        "🧠 **Matched long-context/state eval started on one L4** — final DPO checkpoints only; "
        "4 synthetic memory tasks × 7 context lengths × 96 paired trials. No training or weight updates.",
    )

    def evaluate_one(label: str, checkpoint: Path) -> tuple[dict, dict | None, float]:
        started = time.perf_counter()
        model = load_checkpoint_model(str(checkpoint), device)
        model.eval()
        if int(model.cfg.max_seq_len) < max(CONTEXT_LENGTHS):
            raise RuntimeError(
                f"{label} max_seq_len={model.cfg.max_seq_len} < requested {max(CONTEXT_LENGTHS)}"
            )
        results: dict[str, dict[int, dict]] = {task: {} for task in TASKS}
        with torch.inference_mode():
            for task in TASKS:
                for context_len in CONTEXT_LENGTHS:
                    rows = trials[task][context_len]
                    flags: list[bool] = []
                    for start in range(0, len(rows), BATCH_SIZE):
                        chunk = rows[start : start + BATCH_SIZE]
                        x = torch.tensor(
                            [row["ids"] for row in chunk],
                            device=device,
                            dtype=torch.long,
                        )
                        candidates = torch.tensor(
                            [row["option_ids"] for row in chunk],
                            device=device,
                            dtype=torch.long,
                        )
                        correct = torch.tensor(
                            [row["correct_index"] for row in chunk],
                            device=device,
                            dtype=torch.long,
                        )
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            logits = model(x)[:, -1, :].float()
                        scores = logits.gather(1, candidates)
                        pred = scores.argmax(dim=1)
                        flags.extend((pred == correct).cpu().tolist())
                    acc = sum(flags) / len(flags)
                    results[task][context_len] = {
                        "correct": int(sum(flags)),
                        "total": len(flags),
                        "accuracy": acc,
                        "correct_flags": [bool(x) for x in flags],
                    }
        router = model.router_stats()
        elapsed = time.perf_counter() - started
        del model
        torch.cuda.empty_cache()
        _comment(
            repo_full_name,
            issue_number,
            f"✅ **{label} long-context scoring complete** — elapsed={elapsed:.1f}s.",
        )
        return results, router, elapsed

    tam_results, tam_router, tam_elapsed = evaluate_one("TAM-DPO", tam_path)
    transformer_results, _, transformer_elapsed = evaluate_one("Transformer-DPO", transformer_path)

    per_cell: dict[str, dict[str, dict]] = {task: {} for task in TASKS}
    all_diffs: list[float] = []
    in_range_diffs: list[float] = []
    extra_diffs: list[float] = []
    task_diffs: dict[str, list[float]] = {task: [] for task in TASKS}
    length_diffs: dict[int, list[float]] = {length: [] for length in CONTEXT_LENGTHS}

    for task in TASKS:
        for context_len in CONTEXT_LENGTHS:
            t = tam_results[task][context_len]
            x = transformer_results[task][context_len]
            paired = [
                float(a) - float(b)
                for a, b in zip(t["correct_flags"], x["correct_flags"], strict=True)
            ]
            mean, lo, hi = _bootstrap_ci(
                paired,
                BOOTSTRAP_SAMPLES,
                SEED + context_len + TASKS.index(task) * 100_000,
            )
            per_cell[task][str(context_len)] = {
                "tam_accuracy": t["accuracy"],
                "transformer_accuracy": x["accuracy"],
                "tam_minus_transformer": mean,
                "paired_bootstrap_95ci": [lo, hi],
                "n": len(paired),
            }
            all_diffs.extend(paired)
            task_diffs[task].extend(paired)
            length_diffs[context_len].extend(paired)
            if context_len <= TRAIN_CONTEXT:
                in_range_diffs.extend(paired)
            else:
                extra_diffs.extend(paired)

    def aggregate(diffs: list[float], seed_offset: int) -> dict:
        mean, lo, hi = _bootstrap_ci(diffs, BOOTSTRAP_SAMPLES, SEED + seed_offset)
        return {
            "tam_minus_transformer": mean,
            "paired_bootstrap_95ci": [lo, hi],
            "n": len(diffs),
        }

    aggregates = {
        "all": aggregate(all_diffs, 1),
        "within_train_context_le_512": aggregate(in_range_diffs, 2),
        "extrapolation_gt_512": aggregate(extra_diffs, 3),
        "by_task": {task: aggregate(task_diffs[task], 10 + i) for i, task in enumerate(TASKS)},
        "by_length": {
            str(length): aggregate(length_diffs[length], 1000 + length)
            for length in CONTEXT_LENGTHS
        },
    }

    result = {
        "benchmark": "TAM-v3 vs matched Transformer synthetic long-context/state-memory",
        "seed": SEED,
        "device": torch.cuda.get_device_name(device),
        "models": {
            "tam": {"checkpoint": TAM_CHECKPOINT, "parameters": 101_806_616},
            "transformer": {"checkpoint": TRANSFORMER_CHECKPOINT, "parameters": 101_803_520},
        },
        "protocol": {
            "tasks": list(TASKS),
            "context_lengths": list(CONTEXT_LENGTHS),
            "trials_per_task_length": TRIALS_PER_CELL,
            "answer_choices": 4,
            "chance_accuracy": 0.25,
            "scoring": "argmax next-token logit among four tokenizer-verified single-token answers",
            "prompt_wrapper": "User:/Assistant: matching shared post-training format",
            "paired_trials": True,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "training_context": TRAIN_CONTEXT,
            "note": ">512 tokens are extrapolation for both models because pretraining/SFT used context 512; max positional table is 1024",
            "state_scope": "within-sequence only; TAM RecurrentWorldState is reset on each independent forward call",
        },
        "tam_router_stats": tam_router,
        "elapsed_seconds": {"tam": tam_elapsed, "transformer": transformer_elapsed},
        "per_cell": per_cell,
        "aggregates": aggregates,
    }

    Path(RESULT_PATH).write_text(json.dumps(result, indent=2))
    volume.commit()

    rows = []
    for length in CONTEXT_LENGTHS:
        a = aggregates["by_length"][str(length)]
        rows.append(
            f"{length}: TAM-Transformer={100*a['tam_minus_transformer']:+.2f} pp "
            f"(95% CI {100*a['paired_bootstrap_95ci'][0]:+.2f} to {100*a['paired_bootstrap_95ci'][1]:+.2f})"
        )
    inside = aggregates["within_train_context_le_512"]
    outside = aggregates["extrapolation_gt_512"]
    overall = aggregates["all"]
    _comment(
        repo_full_name,
        issue_number,
        "✅ **Matched long-context/state-memory eval complete.**\n\n"
        + "\n".join(f"- {row}" for row in rows)
        + f"\n\n- ≤512 aggregate: {100*inside['tam_minus_transformer']:+.2f} pp "
          f"(95% CI {100*inside['paired_bootstrap_95ci'][0]:+.2f} to {100*inside['paired_bootstrap_95ci'][1]:+.2f})"
        + f"\n- >512 extrapolation aggregate: {100*outside['tam_minus_transformer']:+.2f} pp "
          f"(95% CI {100*outside['paired_bootstrap_95ci'][0]:+.2f} to {100*outside['paired_bootstrap_95ci'][1]:+.2f})"
        + f"\n- Overall: {100*overall['tam_minus_transformer']:+.2f} pp "
          f"(95% CI {100*overall['paired_bootstrap_95ci'][0]:+.2f} to {100*overall['paired_bootstrap_95ci'][1]:+.2f})"
        + f"\n\nFull JSON: `{RESULT_PATH}`",
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main(repo_full_name: str = "", issue_number: int = 0):
    check = preflight.remote()
    print(json.dumps({"preflight": check}, indent=2), flush=True)
    result = run_eval.remote(repo_full_name, issue_number)
    print(json.dumps(result, indent=2), flush=True)
