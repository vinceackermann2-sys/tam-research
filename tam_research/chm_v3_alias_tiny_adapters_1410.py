from __future__ import annotations

"""#1410 CPU Stage A — architecture-identical alias-vs-whole-name adapters.

TRAIN data and one-step smoke only. No heldout development/test scoring.
The two alias classes INHERIT the exact original parameters/initialization.
Their forward implementations differ from the frozen originals solely by
substituting the sealed-text tokenizer. Tests check AST parity fail-closed.
"""

import math
from typing import Literal

import torch
from torch import nn

from .chm_v3_counterfactual_memory_suite_1358 import ANSWER_IDS
from .chm_v3_counterfactual_model_view_1365 import (
    SealedModelView, seal_model_view,
)
from .chm_v3_query_anchored_alias_1407 import alias_sealed_input
from .chm_v3_matched_tiny_models_1377 import (
    DIM, EncodedView, TinyOutput,
    TinyDecoderOnly, TinyTwoHopPointer,
    learning_objective, matched_initial_models, trainable_parameter_count,
    encode_sealed_view,
)
from .chm_v3_balanced_train_schedule_1391 import balanced_training_episode

SCIENTIFIC_STOP = "CHM_V3_100M_DAEC_STAGE_C_STOP"
CONSUMED_100M_SCIENCE_SEED = 2_013_161
GPU_AUTHORIZED = False
SCORED_HELDOUT_AUTHORIZED = False
TRAIN_ONLY_SMOKE_MAX_STEPS = 1


def _encode_alias_view(view: SealedModelView) -> EncodedView:
    """Convert only visible query/memory text to the SAME encoded-view schema."""
    alias = alias_sealed_input(view)
    return EncodedView(
        token_ids=alias.token_ids, memory_length=alias.memory_length,
        code_tokens=alias.code_tokens, line_anchors=alias.line_anchors,
    )


class TinyAliasDecoderOnly(TinyDecoderOnly):
    """Original tiny decoder parameters + query-anchored token IDs ONLY."""

    def forward(self, view: SealedModelView) -> TinyOutput:
        encoded = _encode_alias_view(view)
        hidden = self.base(encoded)
        return TinyOutput(
            answer_logits=self.base.answer_head(hidden[-1]),
            first_scores=None, second_scores=None,
            first_candidate_token_positions=(),
            second_candidate_token_positions=(),
        )


class TinyAliasTwoHopPointer(TinyTwoHopPointer):
    """Original two-hop soft pointer parameters + alias input IDs ONLY."""

    def forward(self, view: SealedModelView) -> TinyOutput:
        encoded = _encode_alias_view(view)
        h = self.base(encoded)
        memory = h[:encoded.memory_length]
        query = h[-1]
        keys = self.key_projection(memory)
        first_scores = (keys @ self.query_projection(query)) / math.sqrt(DIM)
        first_weights = torch.softmax(first_scores, dim=0)
        context = first_weights @ memory
        # Second hop depends on the model's *own predicted* soft first hop;
        # NO gold pointer/payload enters either forward or hard evaluation.
        second_query = self.query_projection(query + context)
        memory_second = (keys @ second_query) / math.sqrt(DIM)
        code_positions = tuple(pos for pos, _ in encoded.code_tokens)
        code_ids = tuple(token for _, token in encoded.code_tokens)
        code_scores = memory_second[list(code_positions)]
        weights = torch.softmax(code_scores, dim=0)
        option_index = torch.tensor(
            [ANSWER_IDS.index(code) if code in ANSWER_IDS else 0 for code in code_ids],
            dtype=torch.long,
        )
        valid_mask = torch.tensor(
            [1.0 if code in ANSWER_IDS else 0.0 for code in code_ids],
            dtype=weights.dtype,
        )
        answer_weights = torch.zeros(len(ANSWER_IDS), dtype=weights.dtype)
        answer_weights = answer_weights.scatter_add(
            0, option_index, weights * valid_mask,
        )
        copy_log = torch.log(answer_weights + 1e-3)
        # Center evidence so it doesn't automatically promote abstention.
        copy_log = copy_log - copy_log.mean()
        logits = self.base.answer_head(query)
        logits = torch.cat((logits[:8] + copy_log, logits[8:]), dim=0)
        return TinyOutput(
            answer_logits=logits,
            first_scores=first_scores, second_scores=code_scores,
            first_candidate_token_positions=tuple(range(encoded.memory_length)),
            second_candidate_token_positions=code_positions,
        )


def matched_original_and_alias_models() -> dict[str, nn.Module]:
    """Shared initial tensors and model shapes, no optimizer/test scoring."""
    whole_decoder, whole_pointer, _ = matched_initial_models()
    alias_decoder = TinyAliasDecoderOnly()
    alias_pointer = TinyAliasTwoHopPointer()
    reference = {
        "whole_decoder": whole_decoder, "alias_decoder": alias_decoder,
        "whole_pointer_ce": whole_pointer, "alias_pointer_ce": alias_pointer,
    }
    for original, adapted in (
        (whole_decoder, alias_decoder), (whole_pointer, alias_pointer),
    ):
        if set(original.state_dict()) != set(adapted.state_dict()):
            raise RuntimeError("alias adapter parameter key drift")
        if any(not torch.equal(v, adapted.state_dict()[key])
               for key, v in original.state_dict().items()):
            raise RuntimeError("alias adapter initialization drift")
        if trainable_parameter_count(original) != trainable_parameter_count(adapted):
            raise RuntimeError("alias adapter trainable parameter drift")
    if abs(trainable_parameter_count(whole_decoder) -
           trainable_parameter_count(whole_pointer)) / max(
           trainable_parameter_count(whole_decoder),
           trainable_parameter_count(whole_pointer),
    ) > .01:
        raise RuntimeError("pointer/decoder parameter 1-percent gate drift")
    return reference


def cpu_train_only_smoke(*, steps: int = 1) -> dict[str, object]:
    """At most 1 TRAIN episode/optimizer update per model. NEVER dev/test."""
    if type(steps) is not int or steps != TRAIN_ONLY_SMOKE_MAX_STEPS:
        raise ValueError("only exactly one train-only smoke step is allowed")
    torch.set_num_threads(1)
    models = matched_original_and_alias_models()
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=.005, weight_decay=0.)
        for name, model in models.items()
    }
    for step in range(steps):
        ep = balanced_training_episode(step)
        if ep.split != "train":
            raise RuntimeError("nontraining split reached CPU smoke")
        for name, model in models.items():
            optimizers[name].zero_grad(set_to_none=True)
            loss, _ = learning_objective(
                model, ep, pointer_supervision=name.endswith("pointer_ce"),
            )
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("nonfinite TRAIN-only adapter smoke loss")
            loss.backward()
            if not all(p.grad is None or bool(torch.isfinite(p.grad).all().item())
                       for p in model.parameters()):
                raise FloatingPointError("nonfinite TRAIN-only gradient")
            optimizers[name].step()
    return {
        "classification": "CHM_V3_1410_ALIASED_MODEL_CPU_TRAIN_ONLY_SMOKE",
        "samples_per_arm": steps,
        "trainable_parameters": {
            name: trainable_parameter_count(model)
            for name, model in models.items()
        },
        "old_tokenizer_training_sample_tokens": len(
            encode_sealed_view(seal_model_view(ep)).token_ids
        ),
        "aliased_tokenizer_training_sample_tokens": len(
            _encode_alias_view(seal_model_view(ep)).token_ids
        ),
        "scored_heldout_constructed": False,
        "model_training_on_dev_test": False,
        "gpu_used": False, "new_scientific_attempt": False,
        "old_100m_scientific_classification": SCIENTIFIC_STOP,
    }


__all__ = (
    "TinyAliasDecoderOnly", "TinyAliasTwoHopPointer",
    "_encode_alias_view", "matched_original_and_alias_models",
    "cpu_train_only_smoke",
)
