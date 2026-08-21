from tam_research.compile_cache import compiler_cache_dir, compiler_cache_env


def test_compiler_cache_key_is_seed_independent_and_shape_specific():
    kwargs = dict(
        root="/vol/compile-cache",
        architecture="tamv3",
        model_scale="100m",
        seq_len=512,
        micro_batch_size=64,
        grad_accum_steps=2,
    )
    first = compiler_cache_dir(**kwargs)
    # Seed is intentionally absent from the API/key, so all replica seeds reuse it.
    assert first.endswith("h100/100m/tamv3/ctx512-micro64-accum2")

    different_shape = compiler_cache_dir(
        **{**kwargs, "micro_batch_size": 32, "grad_accum_steps": 4}
    )
    different_arch = compiler_cache_dir(**{**kwargs, "architecture": "transformer"})
    assert first != different_shape
    assert first != different_arch


def test_compiler_cache_env_enables_persistent_inductor_layers():
    env = compiler_cache_env("/vol/compile-cache/h100/100m/tamv3/test")
    assert env["TORCHINDUCTOR_FX_GRAPH_CACHE"] == "1"
    assert env["TORCHINDUCTOR_AUTOGRAD_CACHE"] == "1"
    assert env["TORCHINDUCTOR_CACHE_DIR"].startswith("/vol/compile-cache/")
    assert env["TRITON_CACHE_DIR"].endswith("/triton")
