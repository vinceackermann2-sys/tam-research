from __future__ import annotations

import statistics

import torch

from .models import Block, ModelConfig, RecurrentWorldState
from .scan_experiments import ChunkedTAMV3Block, ChunkedWorldState


def _first_tensor(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"unsupported benchmark output type: {type(output)!r}")


def _benchmark_module(
    module: torch.nn.Module,
    *,
    batch_size: int,
    seq_len: int,
    d_model: int,
    compile_module: bool = False,
    warmup: int = 8,
    iterations: int = 25,
) -> dict[str, float | bool]:
    device = torch.device("cuda")
    module = module.to(device).train()
    if compile_module:
        module = torch.compile(
            module,
            mode="max-autotune-no-cudagraphs",
            fullgraph=False,
        )

    x = torch.randn(
        batch_size,
        seq_len,
        d_model,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    def forward_once() -> None:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _first_tensor(module(x))

    def train_once() -> None:
        module.zero_grad(set_to_none=True)
        x.grad = None
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            y = _first_tensor(module(x))
            loss = y.float().square().mean()
        loss.backward()

    # Compilation/autotuning is intentionally paid during warmup and excluded from
    # steady-state timing. The research question here is the amortized kernel cost.
    for _ in range(warmup):
        train_once()
    torch.cuda.synchronize(device)

    def timed(fn) -> list[float]:
        samples: list[float] = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)))
        return samples

    with torch.no_grad():
        forward_samples = timed(forward_once)

    torch.cuda.reset_peak_memory_stats(device)
    train_samples = timed(train_once)
    peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)

    return {
        "compiled": compile_module,
        "forward_ms_median": statistics.median(forward_samples),
        "forward_ms_mean": statistics.fmean(forward_samples),
        "train_ms_median": statistics.median(train_samples),
        "train_ms_mean": statistics.fmean(train_samples),
        "peak_vram_gib": peak_gib,
    }


def profile_components(
    *,
    seq_len: int = 512,
    batch_size: int = 8,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("H100 profiling requires CUDA")

    torch.set_float32_matmul_precision("high")
    cfg_transformer = ModelConfig(
        architecture="transformer",
        max_seq_len=max(1024, seq_len),
    )
    cfg_tamv3 = ModelConfig(
        architecture="tamv3",
        max_seq_len=max(1024, seq_len),
    )
    d_model = cfg_transformer.d_model

    def canonical_world() -> torch.nn.Module:
        return RecurrentWorldState(d_model, cfg_tamv3.tamv2_state_size)

    def chunked_world(chunk_size: int) -> torch.nn.Module:
        return ChunkedWorldState(
            d_model,
            cfg_tamv3.tamv2_state_size,
            chunk_size,
        )

    def chunked_block(chunk_size: int) -> torch.nn.Module:
        return ChunkedTAMV3Block(
            d_model=d_model,
            n_heads=cfg_tamv3.n_heads,
            attention_inner=cfg_tamv3.tamv3_attn_inner,
            state_size=cfg_tamv3.tamv2_state_size,
            ff_mult=cfg_tamv3.ff_mult,
            chunk_size=chunk_size,
        )

    cases: list[tuple[str, object, bool]] = [
        ("transformer_block_compiled", lambda: Block(cfg_transformer), True),
        ("tamv3_block_compiled", lambda: Block(cfg_tamv3), True),
        ("tamv3_world_compiled", canonical_world, True),
        ("tamv3_chunk8_world_compiled", lambda: chunked_world(8), True),
        ("tamv3_chunk16_world_compiled", lambda: chunked_world(16), True),
        ("tamv3_chunk32_world_compiled", lambda: chunked_world(32), True),
        ("tamv3_chunk8_block_compiled", lambda: chunked_block(8), True),
        ("tamv3_chunk16_block_compiled", lambda: chunked_block(16), True),
        ("tamv3_chunk32_block_compiled", lambda: chunked_block(32), True),
    ]

    results: dict[str, dict[str, float | bool | str]] = {}
    for name, factory, compile_module in cases:
        try:
            record = _benchmark_module(
                factory(),
                batch_size=batch_size,
                seq_len=seq_len,
                d_model=d_model,
                compile_module=compile_module,
            )
            results[name] = record
        except Exception as exc:
            results[name] = {
                "compiled": compile_module,
                "error": f"{type(exc).__name__}: {exc}",
            }
        torch.cuda.empty_cache()

    baseline = results["transformer_block_compiled"].get("train_ms_median")
    if isinstance(baseline, (int, float)):
        for record in results.values():
            timing = record.get("train_ms_median")
            if isinstance(timing, (int, float)):
                record["train_vs_transformer_block"] = timing / baseline

    canonical_world_ms = results["tamv3_world_compiled"].get("train_ms_median")
    canonical_block_ms = results["tamv3_block_compiled"].get("train_ms_median")
    if isinstance(canonical_world_ms, (int, float)):
        for chunk_size in (8, 16, 32):
            record = results[f"tamv3_chunk{chunk_size}_world_compiled"]
            timing = record.get("train_ms_median")
            if isinstance(timing, (int, float)):
                record["speedup_vs_canonical_world"] = canonical_world_ms / timing
    if isinstance(canonical_block_ms, (int, float)):
        for chunk_size in (8, 16, 32):
            record = results[f"tamv3_chunk{chunk_size}_block_compiled"]
            timing = record.get("train_ms_median")
            if isinstance(timing, (int, float)):
                record["speedup_vs_canonical_tamv3_block"] = canonical_block_ms / timing

    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "batch_size": batch_size,
        "seq_len": seq_len,
        "d_model": d_model,
        "results": results,
    }
