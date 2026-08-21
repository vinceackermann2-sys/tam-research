from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .data import TokenBin
from .models import ModelConfig, ResearchLM, parameter_count
from .train import evaluate, seed_all


DEFAULT_COMPILE_MODE = "max-autotune-no-cudagraphs"


def load_checkpoint_model(checkpoint_path: str, device: torch.device) -> ResearchLM:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**checkpoint["config"])
    model = ResearchLM(cfg)
    model.load_state_dict(checkpoint["model"])
    return model.to(device)


def save_stage_checkpoint(
    model: ResearchLM,
    path: Path,
    *,
    stage: str,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "config": asdict(model.cfg),
            "architecture": model.cfg.architecture,
            "stage": stage,
            "model": model.state_dict(),
            "metadata": metadata,
        },
        tmp,
    )
    tmp.replace(path)


def _load_array(path: Path, device: torch.device) -> torch.Tensor:
    # Copy the read-only mmap before handing it to torch; post-training shards are
    # intentionally small enough to keep resident in HBM for the entire stage.
    array = np.asarray(np.load(path, mmap_mode="r")).copy()
    return torch.from_numpy(array).to(device=device)


def _masked_nll_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat_labels = labels.reshape(-1)
    mask = flat_labels != -100
    loss_sum = F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        flat_labels,
        ignore_index=-100,
        reduction="sum",
    )
    return loss_sum, mask.sum()


@torch.no_grad()
def evaluate_supervised_arrays(
    model: ResearchLM,
    data_dir: str,
    *,
    prefix: str = "sft_eval",
    batch_size: int = 64,
) -> dict[str, float]:
    device = next(model.parameters()).device
    inputs = _load_array(Path(data_dir) / f"{prefix}_inputs.npy", device)
    labels = _load_array(Path(data_dir) / f"{prefix}_labels.npy", device).long()
    model.eval()
    total_loss = 0.0
    total_targets = 0
    for start in range(0, len(inputs), batch_size):
        x = inputs[start : start + batch_size].long()
        y = labels[start : start + batch_size]
        if not len(x):
            continue
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
        loss_sum, count = _masked_nll_from_logits(logits, y)
        total_loss += float(loss_sum)
        total_targets += int(count)
    mean = total_loss / max(total_targets, 1)
    return {
        "assistant_nll": mean,
        "assistant_perplexity": math.exp(min(mean, 20.0)),
        "assistant_targets": float(total_targets),
    }


def train_sft(
    pretrained_checkpoint: str,
    data_dir: str,
    run_dir: str,
    *,
    seed: int = 7400,
    micro_batch_size: int = 64,
    grad_accum_steps: int = 2,
    learning_rate: float = 5e-5,
    compile_model: bool = True,
) -> tuple[str, dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("SFT requires CUDA")
    seed_all(seed + 100)
    device = torch.device("cuda")
    model = load_checkpoint_model(pretrained_checkpoint, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True,
    )

    data_path = Path(data_dir)
    inputs = _load_array(data_path / "sft_train_inputs.npy", device)
    labels = _load_array(data_path / "sft_train_labels.npy", device).long()
    examples_per_step = micro_batch_size * grad_accum_steps
    steps = len(inputs) // examples_per_step
    if steps < 1:
        raise RuntimeError("not enough SFT examples for one optimizer step")

    runner: torch.nn.Module = model
    compile_seconds = 0.0
    if compile_model:
        runner = torch.compile(model, mode=DEFAULT_COMPILE_MODE)
        warm_x = inputs[:micro_batch_size].long()
        warm_y = labels[:micro_batch_size]
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        model.train()
        runner.train()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            warm_logits = runner(warm_x)
            warm_loss, warm_count = _masked_nll_from_logits(warm_logits, warm_y)
            warm_loss = warm_loss / warm_count.clamp_min(1)
        warm_loss.backward()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        compile_seconds = time.perf_counter() - started
        del warm_x, warm_y, warm_logits, warm_loss

    generator = torch.Generator(device="cpu").manual_seed(seed + 101)
    order = torch.randperm(len(inputs), generator=generator).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    total_target_tokens = 0
    total_loss = 0.0

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for accum in range(grad_accum_steps):
            offset = (step * grad_accum_steps + accum) * micro_batch_size
            idx = order[offset : offset + micro_batch_size]
            x = inputs[idx].long()
            y = labels[idx]
            targets = (y != -100).sum()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = runner(x)
                loss_sum, count = _masked_nll_from_logits(logits, y)
                loss = (loss_sum / count.clamp_min(1)) / grad_accum_steps
            loss.backward()
            step_loss += float(loss) * grad_accum_steps
            total_target_tokens += int(targets)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += step_loss / grad_accum_steps

    torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
    eval_result = evaluate_supervised_arrays(model, data_dir)
    out_dir = Path(run_dir)
    checkpoint_path = out_dir / "sft.pt"
    summary = {
        "stage": "sft",
        "seed": seed,
        "parameters": parameter_count(model),
        "train_examples": steps * examples_per_step,
        "assistant_target_tokens": total_target_tokens,
        "steps": steps,
        "learning_rate": learning_rate,
        "micro_batch_size": micro_batch_size,
        "grad_accum_steps": grad_accum_steps,
        "compile_seconds": compile_seconds,
        "training_seconds": training_seconds,
        "peak_vram_gb": peak_vram_gb,
        "mean_step_loss": total_loss / max(steps, 1),
        "heldout": eval_result,
    }
    save_stage_checkpoint(model, checkpoint_path, stage="sft", metadata=summary)
    (out_dir / "sft_summary.json").write_text(json.dumps(summary, indent=2))
    return str(checkpoint_path), summary


def sequence_logps(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    logits = model(input_ids.long())
    log_probs = F.log_softmax(logits.float(), dim=-1)
    mask = labels != -100
    safe = labels.clamp_min(0).unsqueeze(-1)
    selected = log_probs.gather(-1, safe).squeeze(-1)
    return (selected * mask).sum(dim=-1)


def dpo_loss_from_logps(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    reference_chosen: torch.Tensor,
    reference_rejected: torch.Tensor,
    *,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    policy_ratio = policy_chosen - policy_rejected
    reference_ratio = reference_chosen - reference_rejected
    logits = beta * (policy_ratio - reference_ratio)
    return -F.logsigmoid(logits).mean(), logits


@torch.no_grad()
def evaluate_preferences(
    model: ResearchLM,
    reference: ResearchLM,
    data_dir: str,
    *,
    batch_size: int = 32,
    beta: float = 0.1,
) -> dict[str, float]:
    device = next(model.parameters()).device
    root = Path(data_dir)
    ci = _load_array(root / "pref_eval_chosen_inputs.npy", device)
    cl = _load_array(root / "pref_eval_chosen_labels.npy", device).long()
    ri = _load_array(root / "pref_eval_rejected_inputs.npy", device)
    rl = _load_array(root / "pref_eval_rejected_labels.npy", device).long()
    model.eval()
    reference.eval()
    raw_correct = 0
    reward_correct = 0
    total = 0
    raw_margin_sum = 0.0
    reward_margin_sum = 0.0
    for start in range(0, len(ci), batch_size):
        chosen_x, chosen_y = ci[start : start + batch_size], cl[start : start + batch_size]
        rejected_x, rejected_y = ri[start : start + batch_size], rl[start : start + batch_size]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pc = sequence_logps(model, chosen_x, chosen_y)
            pr = sequence_logps(model, rejected_x, rejected_y)
            rc = sequence_logps(reference, chosen_x, chosen_y)
            rr = sequence_logps(reference, rejected_x, rejected_y)
        raw_margin = pc - pr
        reward_margin = beta * ((pc - rc) - (pr - rr))
        raw_correct += int((raw_margin > 0).sum())
        reward_correct += int((reward_margin > 0).sum())
        raw_margin_sum += float(raw_margin.sum())
        reward_margin_sum += float(reward_margin.sum())
        total += len(chosen_x)
    return {
        "pairs": float(total),
        "raw_preference_accuracy": raw_correct / max(total, 1),
        "implicit_reward_accuracy": reward_correct / max(total, 1),
        "mean_raw_logp_margin": raw_margin_sum / max(total, 1),
        "mean_implicit_reward_margin": reward_margin_sum / max(total, 1),
    }


def train_dpo(
    sft_checkpoint: str,
    data_dir: str,
    run_dir: str,
    *,
    seed: int = 7400,
    beta: float = 0.1,
    micro_batch_size: int = 16,
    grad_accum_steps: int = 4,
    learning_rate: float = 1e-5,
) -> tuple[str, dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("DPO requires CUDA")
    seed_all(seed + 200)
    device = torch.device("cuda")
    policy = load_checkpoint_model(sft_checkpoint, device)
    reference = load_checkpoint_model(sft_checkpoint, device)
    reference.requires_grad_(False)
    reference.eval()
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True,
    )

    root = Path(data_dir)
    ci = _load_array(root / "pref_train_chosen_inputs.npy", device)
    cl = _load_array(root / "pref_train_chosen_labels.npy", device).long()
    ri = _load_array(root / "pref_train_rejected_inputs.npy", device)
    rl = _load_array(root / "pref_train_rejected_labels.npy", device).long()
    examples_per_step = micro_batch_size * grad_accum_steps
    steps = len(ci) // examples_per_step
    if steps < 1:
        raise RuntimeError("not enough preference pairs for one DPO step")
    order = torch.randperm(
        len(ci), generator=torch.Generator(device="cpu").manual_seed(seed + 201)
    ).to(device)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    loss_total = 0.0
    reward_correct = 0
    pairs_seen = 0
    policy.train()

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for accum in range(grad_accum_steps):
            offset = (step * grad_accum_steps + accum) * micro_batch_size
            idx = order[offset : offset + micro_batch_size]
            chosen_x, chosen_y = ci[idx], cl[idx]
            rejected_x, rejected_y = ri[idx], rl[idx]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pc = sequence_logps(policy, chosen_x, chosen_y)
                pr = sequence_logps(policy, rejected_x, rejected_y)
                with torch.no_grad():
                    rc = sequence_logps(reference, chosen_x, chosen_y)
                    rr = sequence_logps(reference, rejected_x, rejected_y)
                loss, reward_logits = dpo_loss_from_logps(
                    pc, pr, rc, rr, beta=beta
                )
                scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()
            step_loss += float(loss)
            reward_correct += int((reward_logits > 0).sum())
            pairs_seen += len(idx)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        loss_total += step_loss / grad_accum_steps

    torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
    heldout = evaluate_preferences(policy, reference, data_dir, beta=beta)
    out_dir = Path(run_dir)
    checkpoint_path = out_dir / "final_dpo.pt"
    summary = {
        "stage": "dpo",
        "seed": seed,
        "parameters": parameter_count(policy),
        "pairs_seen": pairs_seen,
        "steps": steps,
        "beta": beta,
        "learning_rate": learning_rate,
        "micro_batch_size": micro_batch_size,
        "grad_accum_steps": grad_accum_steps,
        "training_seconds": training_seconds,
        "peak_vram_gb": peak_vram_gb,
        "mean_loss": loss_total / max(steps, 1),
        "train_implicit_reward_accuracy": reward_correct / max(pairs_seen, 1),
        "heldout": heldout,
    }
    save_stage_checkpoint(policy, checkpoint_path, stage="dpo", metadata=summary)
    (out_dir / "dpo_summary.json").write_text(json.dumps(summary, indent=2))
    return str(checkpoint_path), summary


@torch.no_grad()
def generate_samples(
    checkpoint_path: str,
    *,
    prompts: list[str] | None = None,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int = 40,
    seed: int = 7400,
) -> list[dict[str, str]]:
    from transformers import AutoTokenizer

    device = torch.device("cuda")
    model = load_checkpoint_model(checkpoint_path, device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    prompts = prompts or [
        "Explain why the sky looks blue in simple language.",
        "Write a short Python function that adds two numbers.",
        "What should I check first if a website suddenly stops loading?",
        "Give me three practical ways to organize a busy workday.",
        "A user says 17 * 6 = 112. Correct the mistake and explain it.",
        "Write a friendly two-sentence email asking to reschedule a meeting.",
    ]
    generator = torch.Generator(device=device).manual_seed(seed + 300)
    results: list[dict[str, str]] = []
    for prompt in prompts:
        prefix = f"User:\n{prompt}\nAssistant:\n"
        ids = tokenizer.encode(prefix, add_special_tokens=False)
        tokens = torch.tensor(ids, device=device, dtype=torch.long).unsqueeze(0)
        generated: list[int] = []
        for _ in range(max_new_tokens):
            context = tokens[:, -model.cfg.max_seq_len :]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(context)[:, -1, :].float() / max(temperature, 1e-4)
            if top_k > 0:
                values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
                probs = torch.softmax(values, dim=-1)
                sampled = torch.multinomial(probs, 1, generator=generator)
                next_token = indices.gather(-1, sampled)
            else:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1, generator=generator)
            token_id = int(next_token.item())
            if token_id == tokenizer.eos_token_id:
                break
            generated.append(token_id)
            tokens = torch.cat((tokens, next_token), dim=1)
        response = tokenizer.decode(generated, skip_special_tokens=True).strip()
        results.append({"prompt": prompt, "response": response})
    return results


def run_posttraining(
    pretrained_checkpoint: str,
    posttrain_data_dir: str,
    fineweb_data_dir: str,
    run_dir: str,
    *,
    seed: int = 7400,
) -> dict[str, Any]:
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    sft_checkpoint, sft_summary = train_sft(
        pretrained_checkpoint,
        posttrain_data_dir,
        run_dir,
        seed=seed,
    )
    dpo_checkpoint, dpo_summary = train_dpo(
        sft_checkpoint,
        posttrain_data_dir,
        run_dir,
        seed=seed,
    )

    device = torch.device("cuda")
    final_model = load_checkpoint_model(dpo_checkpoint, device)
    final_language = evaluate(
        final_model,
        TokenBin(str(Path(fineweb_data_dir) / "val.bin")),
        seq_len=512,
        batches=30,
        batch_size=32,
        seed=seed + 400,
    )
    samples = generate_samples(dpo_checkpoint, seed=seed)
    summary = {
        "model": "TAM-v3-25M-Full",
        "seed": seed,
        "pretrained_checkpoint": pretrained_checkpoint,
        "sft_checkpoint": sft_checkpoint,
        "final_checkpoint": dpo_checkpoint,
        "sft": sft_summary,
        "dpo": dpo_summary,
        "final_language_eval": final_language,
        "samples": samples,
    }
    (out / "final_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
