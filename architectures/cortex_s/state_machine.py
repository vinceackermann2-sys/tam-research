from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

try:
    from .baseline import TransformerStateBaseline
    from .model import CortexSConfig, CortexSCore, count_parameters
except ImportError:
    from baseline import TransformerStateBaseline
    from model import CortexSConfig, CortexSCore, count_parameters


MODULUS = 16
NUM_OPS = 3


def apply_operation(op: torch.Tensor, value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    add = (state + value) % MODULUS
    xor = torch.bitwise_xor(state.long(), value.long()) % MODULUS
    set_value = value
    return torch.where(op == 0, add, torch.where(op == 1, xor, set_value))


def generate_batch(batch_size: int, length: int, *, device: str | torch.device = "cpu"):
    ops = torch.randint(0, NUM_OPS, (batch_size, length), device=device)
    values = torch.randint(0, MODULUS, (batch_size, length), device=device)
    state = torch.zeros(batch_size, dtype=torch.long, device=device)
    states = []
    for index in range(length):
        state = apply_operation(ops[:, index], values[:, index], state)
        states.append(state)
    return ops, values, torch.stack(states, dim=1)


@dataclass
class Metrics:
    per_step_accuracy: float
    final_accuracy: float


@dataclass
class TrainResult:
    final_loss: float
    elapsed_seconds: float


def train_model(model, *, steps: int, seed: int, batch_size: int = 128, lr: float = 3e-3) -> TrainResult:
    torch.manual_seed(seed)
    random.seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    started = time.perf_counter()
    loss_value = float("nan")
    for _ in range(steps):
        length = random.randint(4, 8)
        ops, values, targets = generate_batch(batch_size, length)
        output = model(ops, values)
        logits = output[0] if isinstance(output, tuple) else output
        loss = F.cross_entropy(logits.reshape(-1, MODULUS), targets.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_value = float(loss.detach().item())
    return TrainResult(loss_value, time.perf_counter() - started)


@torch.no_grad()
def evaluate(model, *, length: int, batches: int = 10, batch_size: int = 256) -> Metrics:
    correct = 0
    total = 0
    final_correct = 0
    for _ in range(batches):
        ops, values, targets = generate_batch(batch_size, length)
        output = model(ops, values)
        logits = output[0] if isinstance(output, tuple) else output
        prediction = logits.argmax(dim=-1)
        correct += int((prediction == targets).sum().item())
        total += targets.numel()
        final_correct += int((prediction[:, -1] == targets[:, -1]).sum().item())
    return Metrics(correct / total, final_correct / (batches * batch_size))


def run_exploratory_gate(*, seed: int = 123, steps: int = 600) -> dict[str, Any]:
    """Run a deliberately small, no-GPU engineering gate.

    Seed 123 is exploratory/consumed and MUST NOT be reused for a scientific
    claim. This benchmark validates that the recurrent mechanism can learn and
    extrapolate a state transition rule; it cannot establish a breakthrough.
    """
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    torch.manual_seed(seed)
    cortex = CortexSCore(CortexSConfig())
    torch.manual_seed(seed)
    baseline = TransformerStateBaseline()

    cortex_params = count_parameters(cortex)
    baseline_params = count_parameters(baseline)
    parameter_gap = abs(cortex_params - baseline_params) / baseline_params

    cortex_train = train_model(cortex, steps=steps, seed=seed)
    baseline_train = train_model(baseline, steps=steps, seed=seed)

    lengths = [4, 8, 16, 32]
    cortex_eval = {str(length): asdict(evaluate(cortex, length=length)) for length in lengths}
    baseline_eval = {str(length): asdict(evaluate(baseline, length=length)) for length in lengths}

    gate = {
        "parameter_gap_le_2pct": parameter_gap <= 0.02,
        "cortex_train_loss_lt_0_10": cortex_train.final_loss < 0.10,
        "cortex_length32_final_ge_0_95": cortex_eval["32"]["final_accuracy"] >= 0.95,
        "length32_margin_ge_0_20": (
            cortex_eval["32"]["final_accuracy"] - baseline_eval["32"]["final_accuracy"] >= 0.20
        ),
    }

    return {
        "status": "PASS" if all(gate.values()) else "FAIL",
        "scientific_status": "EXPLORATORY_ENGINEERING_ONLY",
        "seed": seed,
        "steps_per_model": steps,
        "parameter_counts": {"cortex_s": cortex_params, "transformer": baseline_params},
        "parameter_gap_fraction": parameter_gap,
        "train": {
            "cortex_s": asdict(cortex_train),
            "transformer": asdict(baseline_train),
        },
        "evaluation": {"cortex_s": cortex_eval, "transformer": baseline_eval},
        "engineering_gate": gate,
        "warning": (
            "This is a synthetic mechanism check on a task structurally suited to recurrence. "
            "It is not evidence of general intelligence, SSI, safety, or a language-model breakthrough."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_exploratory_gate(seed=args.seed, steps=args.steps)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
