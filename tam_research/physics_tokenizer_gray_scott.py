"""Gray-Scott transfer helpers for non-text scientific-token experiments.

Operates directly on numeric fields. Implements the Phase-5 ingredients: Gray-Scott
data generation, two-stage residual VQ, generic local polynomial features, and a
low-rank family of one-step local maps inferred from short discrete context.
"""
from __future__ import annotations

import numpy as np
import torch

from tam_research.physics_tokenizer_phase0 import channel_stats, fit_kmeans, normalize

OFFSETS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
CONTEXT = 3
FEATURE_DIM = 25


def periodic_laplacian(x: np.ndarray) -> np.ndarray:
    return np.roll(x, 1, -1) + np.roll(x, -1, -1) + np.roll(x, 1, -2) + np.roll(x, -1, -2) - 4.0 * x


def gray_scott_step(u, v, feed, kill, du=0.16, dv=0.08, dt=1.0):
    uvv = u * v * v
    u2 = u + (du * periodic_laplacian(u) - uvv + feed * (1.0 - u)) * dt
    v2 = v + (dv * periodic_laplacian(v) + uvv - (feed + kill) * v) * dt
    return u2, v2


def generate_gray_scott(num, seq_len, grid, feed_range, kill_range, seed, sharp=False, substeps=4):
    rng = np.random.default_rng(seed)
    out = np.empty((num, seq_len, 2, grid, grid), np.float32)
    feeds = rng.uniform(*feed_range, size=num).astype(np.float32)
    kills = rng.uniform(*kill_range, size=num).astype(np.float32)
    yy, xx = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
    for i in range(num):
        u = np.ones((grid, grid), np.float32)
        v = np.zeros((grid, grid), np.float32)
        for _ in range(int(rng.integers(1, 4))):
            cx, cy = float(rng.uniform(0, grid)), float(rng.uniform(0, grid))
            radius = float(rng.uniform(0.8, 1.8) if sharp else rng.uniform(1.5, 3.0))
            dx = np.minimum(np.abs(xx - cx), grid - np.abs(xx - cx))
            dy = np.minimum(np.abs(yy - cy), grid - np.abs(yy - cy))
            blob = np.exp(-(dx * dx + dy * dy) / (2.0 * radius * radius)).astype(np.float32)
            u -= float(rng.uniform(0.25, 0.5)) * blob
            v += float(rng.uniform(0.18, 0.35)) * blob
        u += rng.normal(0, 0.02 if sharp else 0.01, size=(grid, grid)).astype(np.float32)
        v += rng.normal(0, 0.01 if sharp else 0.005, size=(grid, grid)).astype(np.float32)
        for t in range(seq_len):
            out[i, t, 0], out[i, t, 1] = u, v
            for _ in range(substeps):
                u, v = gray_scott_step(u, v, feeds[i], kills[i])
    return out, feeds, kills


def _nearest(flat, codebook, chunk=16384):
    x = torch.from_numpy(flat.astype(np.float32)) if isinstance(flat, np.ndarray) else flat.float()
    c2 = codebook.pow(2).sum(1)[None, :]
    ids = []
    for i in range(0, len(x), chunk):
        xb = x[i:i + chunk]
        dist = xb.pow(2).sum(1, keepdim=True) + c2 - 2 * xb @ codebook.t()
        ids.append(dist.argmin(1))
    return torch.cat(ids)


def fit_residual_vq(train_norm, coarse_seed=75193, residual_seed=75210):
    points = train_norm.transpose(0, 1, 3, 4, 2).reshape(-1, 2).astype(np.float32)
    coarse = fit_kmeans(points, 256, coarse_seed, iters=20, max_points=60000)
    i1 = _nearest(points, coarse)
    residuals = points - coarse[i1].numpy()
    residual = fit_kmeans(residuals, 32, residual_seed, iters=20, max_points=60000)
    return coarse, residual


def quantize_normalized(x, coarse, residual):
    shape = x.shape
    if x.ndim == 5:
        flat = x.transpose(0, 1, 3, 4, 2).reshape(-1, 2).astype(np.float32)
    elif x.ndim == 4:
        flat = x.transpose(0, 2, 3, 1).reshape(-1, 2).astype(np.float32)
    else:
        raise ValueError("expected N,T,C,H,W or N,C,H,W")
    i1 = _nearest(flat, coarse)
    remainder = flat - coarse[i1].numpy()
    i2 = _nearest(remainder, residual)
    q = coarse[i1] + residual[i2]
    if x.ndim == 5:
        n, t, _, h, w = shape
        return q.reshape(n, t, h, w, 2).permute(0, 1, 4, 2, 3).numpy().astype(np.float32)
    n, _, h, w = shape
    return q.reshape(n, h, w, 2).permute(0, 3, 1, 2).numpy().astype(np.float32)


def local_polynomial_features(state):
    cols = []
    for ch in range(2):
        for dy, dx in OFFSETS:
            cols.append(np.roll(state[:, ch], shift=(dy, dx), axis=(-2, -1)))
    u, v = state[:, 0], state[:, 1]
    cols += [u*u, u*v, v*v, u*u*u, u*u*v, u*v*v, v*v*v]
    return np.stack(cols, -1).astype(np.float32)


def fit_sequence_map(q_sequence, ridge=1e-4):
    xs, ys = [], []
    for t in range(len(q_sequence) - 1):
        xs.append(local_polynomial_features(q_sequence[t:t+1])[0].reshape(-1, FEATURE_DIM))
        ys.append(q_sequence[t+1].transpose(1, 2, 0).reshape(-1, 2))
    x = np.concatenate(xs).astype(np.float64)
    y = np.concatenate(ys).astype(np.float64)
    return np.linalg.solve(x.T @ x + ridge * np.eye(FEATURE_DIM), x.T @ y).T.astype(np.float32)


def fit_transition_family(q_train, latent_dim=8):
    maps = np.stack([fit_sequence_map(q_train[i]) for i in range(len(q_train))])
    flat = maps.reshape(len(maps), -1)
    mean = flat.mean(0)
    centered = flat - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    return mean.reshape(2, FEATURE_DIM).astype(np.float32), vt[:latent_dim].reshape(latent_dim, 2, FEATURE_DIM).astype(np.float32), singular_values


def predict_local(state, weights):
    return np.einsum("bhwf,cf->bchw", local_polynomial_features(state), weights, optimize=True).astype(np.float32)


def infer_latent(q_context, mean_map, basis, ridge=1e-5):
    batch, latent_dim = len(q_context), len(basis)
    z = np.zeros((batch, latent_dim), np.float32)
    for i in range(batch):
        design, residual = [], []
        for t in range(CONTEXT - 1):
            baseline = predict_local(q_context[i:i+1, t], mean_map)[0]
            directions = [predict_local(q_context[i:i+1, t], basis[j])[0] for j in range(latent_dim)]
            design.append(np.stack([d.reshape(-1) for d in directions], 1))
            residual.append((q_context[i, t+1] - baseline).reshape(-1))
        a = np.concatenate(design).astype(np.float64)
        r = np.concatenate(residual).astype(np.float64)
        z[i] = np.linalg.solve(a.T @ a + ridge * np.eye(latent_dim), a.T @ r)
    return z


def fit_phase5_model(train_raw, latent_dim=8):
    means, stds = channel_stats(train_raw)
    train_norm = normalize(train_raw, means, stds)
    coarse, residual = fit_residual_vq(train_norm)
    q_train = quantize_normalized(train_norm, coarse, residual)
    mean_map, basis, singular_values = fit_transition_family(q_train, latent_dim)
    reconstruction_nmse = float(np.mean((q_train - train_norm) ** 2) / np.var(train_norm))
    explained = float((singular_values[:latent_dim] ** 2).sum() / (singular_values ** 2).sum())
    return means, stds, coarse, residual, mean_map, basis, reconstruction_nmse, explained
