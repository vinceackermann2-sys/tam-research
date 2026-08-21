import numpy as np
import torch

from tam_research.data import TokenBin


def test_tokenbin_cpu_batch_matches_reference(tmp_path):
    values = np.arange(200, dtype=np.uint16)
    path = tmp_path / "tokens.bin"
    values.tofile(path)
    token_bin = TokenBin(str(path))

    batch_size = 4
    seq_len = 7
    seed = 123
    reference_gen = torch.Generator(device="cpu").manual_seed(seed)
    batch_gen = torch.Generator(device="cpu").manual_seed(seed)
    hi = len(values) - seq_len - 1
    starts = torch.randint(0, hi, (batch_size,), generator=reference_gen).tolist()
    chunks = np.stack(
        [np.asarray(values[s : s + seq_len + 1], dtype=np.int64) for s in starts]
    )

    x, y = token_bin.batch(batch_size, seq_len, batch_gen, torch.device("cpu"))
    torch.testing.assert_close(x, torch.from_numpy(chunks[:, :-1]))
    torch.testing.assert_close(y, torch.from_numpy(chunks[:, 1:]))


def test_tokenbin_same_seed_is_deterministic(tmp_path):
    values = np.arange(512, dtype=np.uint16)
    path = tmp_path / "tokens.bin"
    values.tofile(path)
    token_bin = TokenBin(str(path))

    g1 = torch.Generator(device="cpu").manual_seed(77)
    g2 = torch.Generator(device="cpu").manual_seed(77)
    x1, y1 = token_bin.batch(8, 11, g1, torch.device("cpu"))
    x2, y2 = token_bin.batch(8, 11, g2, torch.device("cpu"))
    torch.testing.assert_close(x1, x2)
    torch.testing.assert_close(y1, y2)
