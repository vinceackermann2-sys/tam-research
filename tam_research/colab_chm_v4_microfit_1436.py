"""Pending #1436 TRAIN-only capacity experiment; never invoked by CPU CI.

One fresh issue after explicit authorization. Extra gold address CE, soft first
hop, hard output copy; no matched baseline or generalization/breakthrough claim.
"""
from pathlib import Path
import time
from tam_research.colab_job_runtime import write_json

SEED = 1436001  # engineering process RNG; frozen model constructor retains INIT_SEED
STEPS = 256
FAMILIES = ("direct", "overwrite", "two_hop", "no_match")


def configuration():
    return {"classification": "CHM_V4_1436_TRAIN_ONLY_MICROFIT_ENGINEERING",
            "process_seed": SEED, "model_init_seed": 1377005, "steps": STEPS,
            "families": FAMILIES, "entity_index": 0, "variants": list(range(8)),
            "order": "family-major, variant-major, 32-item cycle repeated eight times",
            "optimizer": "AdamW", "lr": .005, "betas": [.9, .999], "eps": 1e-8,
            "weight_decay": 0., "scheduler": None, "clip": None, "threads": 1,
            "device": "cpu", "precision": "float32", "torch": "2.10.0+cpu",
            "loss": "presence_ce + 0.4*(first_address_ce + payload_address_ce)",
            "examples": 32, "examples_seen": 256, "max_tokens_per_example": 384,
            "max_input_tokens": 98304, "input_token_budget": 18560, "resume": False, "retries": 0,
            "development_or_test": False, "baseline": None,
            "success_gate": "24/24 positive TRAIN answers plus abstain on one negative entity",
            "interpretation": "In-sample capacity only; unequal address supervision; no breakthrough"}


def training_cases():
    from tam_research.chm_v4_independent_binding_benchmark_1421 import train_counterfactual_group
    cases = tuple(ep for family in FAMILIES for ep in train_counterfactual_group(family, 0))
    if len(cases) != 32 or any(ep.split != "train" for ep in cases):
        raise ValueError("TRAIN group contract violated")
    return cases


def run_microfit(output: Path):
    import torch
    from importlib.metadata import version
    import platform
    from tam_research.chm_v4_neural_hard_copy_train_1430 import (
        TinyV4HardCopyPointer, train_only_supervised_loss, hard_copy_prediction, encode_visible_v4,
    )
    from tam_research.chm_v4_independent_binding_benchmark_1421 import seal_input, oracle_read
    if str(torch.__version__) != "2.10.0+cpu":
        raise ValueError("Unreviewed torch build")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    actual_versions = {name: version(name) for name in (
        "torch", "filelock", "typing-extensions", "sympy", "networkx", "jinja2", "fsspec"
    )}
    cases = training_cases()
    model = TinyV4HardCopyPointer().cpu().float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.005, weight_decay=0.,
                                 betas=(.9, .999), eps=1e-8)
    losses, tokens = [], 0
    started = time.monotonic()
    with (output / "run.log").open("w") as log:
        for step in range(STEPS):
            ep = cases[step % 32]
            view = seal_input(ep)
            tokens += len(encode_visible_v4(view).model_tokens.token_ids)
            optimizer.zero_grad(set_to_none=True)
            out = model(view)  # sealed visible input only
            loss, components = train_only_supervised_loss(out, ep)
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("Nonfinite TRAIN gradients")
            optimizer.step()
            losses.append({"step": step + 1, "loss": float(loss.detach()), **components})
            log.write(str(losses[-1]) + "\n")
            log.flush()
    if tokens != 18560:
        raise ValueError("Frozen input token budget changed")
    # One terminal TRAIN-only inspection. No development/test generators.
    rows = []
    model.eval()
    with torch.no_grad():
        for index, ep in enumerate(cases):
            out = model(seal_input(ep))
            pred = hard_copy_prediction(out)
            gold, first, payload = oracle_read(ep)
            first = first if first is not None else payload
            rows.append({"case": index, "family": ep.family, "variant": ep.variant,
                         "gold": gold, "predicted": pred.predicted_answer_id,
                         "first_source": pred.first_source_physical,
                         "payload_source": pred.payload_source_physical,
                         "gold_first": first, "gold_payload": payload,
                         "first_hit": None if gold is None else pred.first_source_physical == first,
                         "payload_hit": None if gold is None else pred.payload_source_physical == payload,
                         "correct": pred.predicted_answer_id == gold, "abstained": pred.did_abstain,
                         "false_abstain": gold is not None and pred.did_abstain,
                         "false_present": gold is None and not pred.did_abstain})
    # Full terminal checkpoint, never used as a resume source.
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": STEPS, "input_tokens": tokens, "configuration": configuration(),
                "rng": torch.get_rng_state()}, output / "checkpoint.pt")
    positive = sum(r["correct"] for r in rows if r["family"] != "no_match")
    negative = all(r["abstained"] for r in rows if r["family"] == "no_match")
    return {"status": "capacity-pass" if positive == 24 and negative else "capacity-fail",
            "positive_train_correct": positive, "positive_train_cases": 24,
            "negative_entities": 1, "negative_abstain": negative, "rows": rows,
            "losses": losses, "input_tokens": tokens, "steps": STEPS,
            "elapsed_sec": time.monotonic() - started,
            "parameters": sum(p.numel() for p in model.parameters()),
            "python": platform.python_version(), "actual_dependencies": actual_versions,
            "heldout_generated": False, "breakthrough_claim_allowed": False}
