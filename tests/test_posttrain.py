import numpy as np
import torch

from tam_research.posttrain import dpo_loss_from_logps
from tam_research.posttrain_data import encode_supervised_pair


class TinyTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        # Stable character-level stand-in; reserve 0 for EOS.
        return [1 + (ord(ch) % 200) for ch in text]


def test_supervised_pair_masks_prompt_and_trains_assistant_tokens():
    tokenizer = TinyTokenizer()
    x, labels = encode_supervised_pair(
        tokenizer,
        "Hi",
        "OK",
        seq_len=32,
    )
    assert x.shape == (32,)
    assert labels.shape == (32,)
    assert x.dtype == np.uint16
    assert labels.dtype == np.int32

    prefix_len = len(tokenizer.encode("User:\nHi\nAssistant:\n"))
    # The input immediately before the first response token predicts that response.
    first_target = prefix_len - 1
    assert np.all(labels[:first_target] == -100)
    assert labels[first_target] != -100
    # Padding after the real response remains masked.
    trained = np.flatnonzero(labels != -100)
    assert len(trained) >= 2
    assert np.all(labels[trained[-1] + 1 :] == -100)


def test_long_prompt_leaves_room_for_assistant_targets():
    tokenizer = TinyTokenizer()
    _, labels = encode_supervised_pair(
        tokenizer,
        "x" * 5_000,
        "answer",
        seq_len=64,
    )
    assert int((labels != -100).sum()) >= 2


def test_dpo_loss_rewards_policy_shift_toward_chosen():
    ref_chosen = torch.tensor([-10.0, -8.0])
    ref_rejected = torch.tensor([-10.0, -8.0])

    neutral_loss, neutral_logits = dpo_loss_from_logps(
        ref_chosen,
        ref_rejected,
        ref_chosen,
        ref_rejected,
        beta=0.1,
    )
    better_loss, better_logits = dpo_loss_from_logps(
        torch.tensor([-8.0, -6.0]),
        torch.tensor([-11.0, -9.0]),
        ref_chosen,
        ref_rejected,
        beta=0.1,
    )
    assert torch.allclose(neutral_logits, torch.zeros_like(neutral_logits))
    assert torch.all(better_logits > 0)
    assert better_loss < neutral_loss
