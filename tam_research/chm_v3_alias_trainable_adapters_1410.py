from __future__ import annotations

"""#1410: training-only paired adapters, frozen tiny backbones and pointer math.

This module does not generate development/test examples or compute any heldout
metrics. New name canonicalization is a predeclared, *shared* input bias.
Neither sealed-input adapter nor model forward receives evaluator metadata.
"""

from dataclasses import dataclass
import json
import math
import time

import torch
from torch import nn
from torch.nn import functional as F

from .chm_v3_query_anchored_alias_1407 import alias_sealed_input
from .chm_v3_matched_tiny_models_1377 import (
    DIM, INIT_SEED, MAX_TOKENS, EncodedView, TinyDecoderOnly,
    TinyOutput, TinyTwoHopPointer, encode_sealed_view, learning_objective,
    trainable_parameter_count,
)
from .chm_v3_counterfactual_memory_suite_1358 import ANSWER_IDS
from .chm_v3_counterfactual_model_view_1365 import SealedModelView, seal_model_view
from .chm_v3_balanced_train_schedule_1391 import (
    TRAIN_STEPS, balanced_training_episode,
)

CLASSIFICATION = "CHM_V3_1410_ALIAS_VS_HASH_TRAIN_ONLY_ADAPTER_CPU"
HISTORICAL_STOP = "CHM_V3_100M_DAEC_STAGE_C_STOP"
HISTORICAL_SCIENTIFIC_SEED_CONSUMED = 2013161
GPU_AUTHORIZED = False
SCORED_HELDOUT_AUTHORIZED = False
NEW_SCIENTIFIC_ATTEMPT = False
FULL_TRAINING_AUTHORIZED = False
ENCODER_PREPROCESSOR_CONFUND_DISCLOSED = True


def alias_encoded_view(view: SealedModelView) -> EncodedView:
    """Public frozen-model tensor layout from ONLY sealed visible text.

    Local alias dictionary is deliberately excluded from EncodedView and never
    enters model.forward or loss. Source positions remain byte-for-byte equal.
    """
    if not isinstance(view, SealedModelView):
        raise TypeError("forward requires a SealedModelView")
    alias = alias_sealed_input(view)
    legacy = encode_sealed_view(view)
    encoded = EncodedView(
        token_ids=alias.token_ids, memory_length=alias.memory_length,
        code_tokens=alias.code_tokens, line_anchors=alias.line_anchors,
    )
    if (len(encoded.token_ids) != len(legacy.token_ids)
        or encoded.memory_length != legacy.memory_length
        or encoded.code_tokens != legacy.code_tokens
        or encoded.line_anchors != legacy.line_anchors
        or len(encoded.token_ids) > MAX_TOKENS):
        raise RuntimeError("source-address, code-copy or token-budget divergence")
    return encoded


def _decoder_math(model: TinyDecoderOnly, enc: EncodedView) -> TinyOutput:
    """Same tensor operations as frozen #1377 TinyDecoderOnly.forward."""
    hidden = model.base(enc)
    return TinyOutput(
        answer_logits=model.base.answer_head(hidden[-1]),
        first_scores=None, second_scores=None,
        first_candidate_token_positions=(),
        second_candidate_token_positions=(),
    )


def _pointer_math(model: TinyTwoHopPointer, enc: EncodedView) -> TinyOutput:
    """Same tensor operations as frozen #1377 TinyTwoHopPointer.forward.

    First hop is learned, NOT oracle. Second uses soft predicted first-hop
    context. Code values come only from visible memory text.
    """
    h = model.base(enc)
    memory = h[:enc.memory_length]
    query = h[-1]
    keys = model.key_projection(memory)
    first_scores = (keys @ model.query_projection(query)) / math.sqrt(DIM)
    first_weights = torch.softmax(first_scores, dim=0)
    context = first_weights @ memory
    second_query = model.query_projection(query + context)
    memory_second = (keys @ second_query) / math.sqrt(DIM)
    code_positions = tuple(p for p, _ in enc.code_tokens)
    code_ids = tuple(code for _, code in enc.code_tokens)
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
    copy_log = copy_log - copy_log.mean()
    logits = model.base.answer_head(query)
    logits = torch.cat((logits[:8] + copy_log, logits[8:]), dim=0)
    return TinyOutput(
        answer_logits=logits,
        first_scores=first_scores,
        second_scores=code_scores,
        first_candidate_token_positions=tuple(range(enc.memory_length)),
        second_candidate_token_positions=code_positions,
    )


class AliasedTinyDecoderOnly(TinyDecoderOnly):
    """Same parameter tensors as #1377 baseline; only tokenizer differs."""

    def forward(self, view: SealedModelView) -> TinyOutput:
        return _decoder_math(self, alias_encoded_view(view))


class AliasedTinyTwoHopPointer(TinyTwoHopPointer):
    """Same parameter tensors and address heads as frozen #1377 pointer."""

    def forward(self, view: SealedModelView) -> TinyOutput:
        return _pointer_math(self, alias_encoded_view(view))


def matched_alias_hash_initial_models() -> tuple[
    TinyDecoderOnly, AliasedTinyDecoderOnly,
    TinyTwoHopPointer, AliasedTinyTwoHopPointer,
]:
    """Four training arms, identical weights per architecture at step zero."""
    old_d = TinyDecoderOnly()
    new_d = AliasedTinyDecoderOnly()
    old_p = TinyTwoHopPointer()
    new_p = AliasedTinyTwoHopPointer()
    for old, new in ((old_d, new_d), (old_p, new_p)):
        if trainable_parameter_count(old) != trainable_parameter_count(new):
            raise RuntimeError("adapter parameter-count drift")
        if not all(torch.equal(a, b) for a,b in zip(old.parameters(),new.parameters())):
            raise RuntimeError("adapter initial tensor drift")
        if tuple(old.state_dict()) != tuple(new.state_dict()):
            raise RuntimeError("adapter parameter-name drift")
    return old_d,new_d,old_p,new_p


def verify_train_only_encoder_parity() -> dict[str, object]:
    """No optimizers/forward calls and no consumed heldout panels."""
    total = 0
    max_length = 0
    anchor_count = 0
    for step in range(TRAIN_STEPS):
        ep = balanced_training_episode(step)
        if ep.split != "train":
            raise RuntimeError("TRAIN-only isolation failed")
        view = seal_model_view(ep)
        old = encode_sealed_view(view)
        new = alias_encoded_view(view)
        total += len(old.token_ids)
        max_length = max(max_length,len(old.token_ids))
        anchor_count += len(old.line_anchors)
        if len(old.token_ids) != len(new.token_ids):
            raise RuntimeError("paired per-example token FLOP proxy drift")
    return {
        "classification": CLASSIFICATION,
        "training_split_only": True,
        "train_examples": TRAIN_STEPS,
        "identical_input_token_lengths": True,
        "identical_length_total_per_arm": total,
        "max_input_length_per_arm": max_length,
        "source_anchor_count": anchor_count,
        "new_scientific_attempt": False,
        "historical_stage_c_status": HISTORICAL_STOP,
        "historical_seed_consumed": HISTORICAL_SCIENTIFIC_SEED_CONSUMED,
        "scored_heldout_cases": 0,
        "optimizer_updates": 0,
        "gpu_used": False,
        "note": "Only forward-attention length proxies match; alias preprocessor differs.",
    }


def one_train_step_four_arm_smoke() -> dict[str, object]:
    """Exactly one deterministic shared TRAIN episode and update per arm.

    Engineering smoke, not 256-step learning, inference accuracy or test score.
    Original #1377 learned pointer CE has extra supervised address losses,
    so compare within pointer-CE and within decoder families separately.
    """
    torch.set_num_threads(1)
    ep = balanced_training_episode(0)
    if ep.split != "train":
        raise RuntimeError("scored heldout input forbidden")
    arms = matched_alias_hash_initial_models()
    tokens = len(encode_sealed_view(seal_model_view(ep)).token_ids)
    loss_names = ("hash_decoder", "alias_decoder", "hash_pointer_ce", "alias_pointer_ce")
    values = {}
    for name, model in zip(loss_names, arms):
        optimizer = torch.optim.AdamW(model.parameters(),lr=0.005,weight_decay=0.0)
        optimizer.zero_grad(set_to_none=True)
        loss, stats = learning_objective(
            model, ep, pointer_supervision=name.endswith("pointer_ce"),
        )
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("non-finite TRAIN-only synthetic loss")
        loss.backward()
        if model.base.embedding.weight.grad is None:
            raise RuntimeError("adapter broke embedding gradient")
        if name.endswith("pointer_ce"):
            if (model.query_projection.weight.grad is None or
                model.key_projection.weight.grad is None):
                raise RuntimeError("adapter broke pointer gradients")
        optimizer.step()
        values[name] = {
            "trainable_parameters": trainable_parameter_count(model),
            "finite_train_only_loss": True,
            "embedding_gradients_present": True,
            "pointer_gradients_present": name.endswith("pointer_ce"),
            "train_tokens": tokens,
            "steps": 1,
        }
    if not (values["hash_decoder"]["trainable_parameters"] ==
            values["alias_decoder"]["trainable_parameters"] and
            values["hash_pointer_ce"]["trainable_parameters"] ==
            values["alias_pointer_ce"]["trainable_parameters"]):
        raise RuntimeError("adapter trainable count mismatch")
    return {
        "classification": CLASSIFICATION,
        "smoke_updates_per_arm": 1,
        "same_train_episode_per_arm": True,
        "optimizer": "AdamW(lr=0.005,weight_decay=0)",
        "arms": values,
        "historical_stage_c_status": HISTORICAL_STOP,
        "scored_heldout_cases": 0,
        "gpu_used": False,
        "new_scientific_attempt": False,
        "interpretation": "Gradient wiring only, NOT trained generalization.",
    }


__all__ = (
    "AliasedTinyDecoderOnly","AliasedTinyTwoHopPointer",
    "alias_encoded_view","matched_alias_hash_initial_models",
    "verify_train_only_encoder_parity","one_train_step_four_arm_smoke",
)
