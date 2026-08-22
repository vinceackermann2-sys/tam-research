from __future__ import annotations

import json
import math

import torch
import torch.nn.functional as F

from aera_pre100m_cpu import (
    adaptive_budget_probe,
    controller_routing_probe,
    event_patching_probe,
    integrated_language_and_block_probe,
    memory_stress_probe,
    replay_probe,
    retrieval_precision_controller_probe,
    seed_all,
    stream_state_probe,
)
from tam_research.aera_full import ModalityAdapterBank
from tam_research.aera_world import WorldTransitionHead


def multimodal_world_probe_v2() -> dict[str, object]:
    """Controlled multimodal gate with genuine shared latent structure.

    Each modality is a noisy linear projection of the same class-level latent.
    The adapters must recover a shared representation on fresh noisy observations.
    This avoids the invalid v1 test, which used unrelated random tables and therefore
    had no linearly recoverable cross-modal semantics.
    """

    seed_all(2005)
    n_classes = 48
    latent_dim = 12
    d_model = 16
    modality_dims = {"text": 14, "image": 16, "action": 12}
    bank = ModalityAdapterBank(d_model, modality_dims)

    concepts = F.normalize(torch.randn(n_classes, latent_dim), dim=-1)
    projections = {
        name: torch.randn(latent_dim, raw_dim) / math.sqrt(latent_dim)
        for name, raw_dim in modality_dims.items()
    }
    target_projection = torch.randn(latent_dim, d_model) / math.sqrt(latent_dim)
    shared_targets = F.normalize(concepts @ target_projection, dim=-1)

    opt = torch.optim.AdamW(bank.parameters(), lr=0.02)
    train_noise = 0.02
    for _ in range(400):
        ids = torch.randint(0, n_classes, (128,))
        loss = torch.zeros(())
        for name, raw_dim in modality_dims.items():
            raw = concepts[ids] @ projections[name]
            raw = raw + train_noise * torch.randn(128, raw_dim)
            z = F.normalize(bank(name, raw), dim=-1)
            # Regression anchors a common latent; the contrastive term explicitly
            # makes the representation useful for cross-modal retrieval.
            loss = loss + F.mse_loss(z, shared_targets[ids])
            loss = loss + 0.1 * F.cross_entropy(
                (z @ shared_targets.T) / 0.07,
                ids,
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # Fresh observation noise prevents evaluating the exact training table.
    eval_noise = 0.02
    with torch.no_grad():
        text_raw = concepts @ projections["text"]
        text_raw = text_raw + eval_noise * torch.randn(n_classes, modality_dims["text"])
        image_raw = concepts @ projections["image"]
        image_raw = image_raw + eval_noise * torch.randn(n_classes, modality_dims["image"])
        z_text = F.normalize(bank("text", text_raw), dim=-1)
        z_image = F.normalize(bank("image", image_raw), dim=-1)
        similarity = z_text @ z_image.T
        retrieval = float(
            (similarity.argmax(dim=-1) == torch.arange(n_classes)).float().mean()
        )

    # Controlled action-conditioned latent dynamics remains an independent gate.
    action_dim = 6
    world = WorldTransitionHead(d_model, action_dim)
    world_opt = torch.optim.AdamW(world.parameters(), lr=0.02)
    action_effect = torch.randn(action_dim, d_model) * 0.25
    for _ in range(300):
        state = torch.randn(128, d_model)
        action = F.one_hot(
            torch.randint(0, action_dim, (128,)), num_classes=action_dim
        ).float()
        target = state + action @ action_effect
        pred = world(state, action)
        loss = F.mse_loss(pred, target)
        world_opt.zero_grad(set_to_none=True)
        loss.backward()
        world_opt.step()

    with torch.no_grad():
        state = torch.randn(512, d_model)
        action = F.one_hot(
            torch.randint(0, action_dim, (512,)), num_classes=action_dim
        ).float()
        target = state + action @ action_effect
        mse = float(F.mse_loss(world(state, action), target))

    passed = retrieval >= 0.90 and mse <= 0.03
    if not passed:
        raise AssertionError((retrieval, mse))
    return {
        "pass": passed,
        "shared_latent_classes": n_classes,
        "fresh_noise_eval": True,
        "text_to_image_retrieval_accuracy": retrieval,
        "action_conditioned_state_mse": mse,
    }


def main() -> None:
    results = {
        "status": "pass",
        "scope": "AERA_pre100M_CPU_mechanism_and_integration_validation_v2",
        "learned_sparse_routing": controller_routing_probe(),
        "learned_adaptive_compute": adaptive_budget_probe(),
        "delta_fast_memory_stress": memory_stress_probe(),
        "stream_state_across_chunks": stream_state_probe(),
        "variable_event_patching": event_patching_probe(),
        "multimodal_and_world_transition": multimodal_world_probe_v2(),
        "verified_replay": replay_probe(),
        "retrieval_and_precision_controller": retrieval_precision_controller_probe(),
        "integrated_text_and_block_generation": integrated_language_and_block_probe(),
        "claims": {
            "100m_authorized": False,
            "gpu_sparse_speedup_proven": False,
            "real_language_superiority_proven": False,
            "dynamic_precision_hardware_savings_proven": False,
            "full_multimodal_scale_proven": False,
            "breakthrough_proven": False,
        },
    }
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
