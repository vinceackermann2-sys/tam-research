# CORTEX-S v0 100M systems microbenchmark — repair4

Classification: `ENGINEERING_SYSTEMS_MICROBENCH_ONLY`

## Consumed parent evidence

Repair3 issue #791, GitHub Actions run `34335827718`, job `102414857556`, source `1c0ecd321a530daee32a4603a9cd6fb201be2d8a`, and engineering seed `2026090903` are permanently consumed.

Repair3 established:

- frozen corpus/data gate PASS;
- stride/layout CPU contract PASS;
- Python-scoping contract PASS;
- semantic routing exact PASS;
- semantic loss delta 0.001929 <= 0.02 PASS;
- legacy compiled 100M graph PASS at 146,554.81 tok/s and 48.264 GiB;
- grouped candidate produced no timing because `torch.compile`/FakeTensor rejected the second `grouped_mm` lhs with row stride 338;
- the slice-based recurrent-scan candidate was slower at 0.70186x.

The grouped compile failure is engineering/compiler-layout evidence, not language-quality evidence. The scan result rejects that particular slice-scan optimization path.

## Repair4 change

The repair2/3 candidate padded physical activation storage to 344 BF16 elements but then returned a logical tensor width of 338 before grouped GEMM2. Eager stride inspection saw the padded storage, while FakeTensor/Inductor still saw a grouped-mm lhs stride of 338.

Repair4 preserves every trainable Parameter at the production dimensions and uses an aligned **logical compute width of 344** only for grouped expert execution:

1. W1 Parameters remain `[E,338,512]`.
2. At runtime, W1 is transposed to `[E,512,338]` and zero-padded to `[E,512,344]`.
3. grouped GEMM1 returns `[N,344]`.
4. GELU operates on all 344 channels. The six padded channels remain zero because GELU(0)=0.
5. W2 Parameters remain `[E,338,512]` and are runtime-zero-padded to `[E,344,512]`.
6. grouped GEMM2 consumes the full `[N,344]` activation. There is no slice back to 338 between grouped GEMMs.

The extra six channels are not Parameters, so model parameter count remains exactly 101,778,112. Hidden expert compute overhead is 344/338 = ~1.01775x.

## Frozen repair4 protocol

- engineering seed: `2026090904`
- consumed engineering seeds: `910001`, `2026090901`, `2026090902`, `2026090903`
- forbidden scientific/control seeds: `8100`, `48131`, `48132`, `48133`
- trigger title: `[modal-cortex-s-100m-systems-microbench-v1-repair4]`
- phase: `microbench-v1-repair4`
- result root: `/vol/cortex-s-v0/100m-systems-microbench-v1-repair4`
- H100 hard timeout: 15 minutes
- production microbatch/sequence/grad accumulation: 64 / 512 / 2
- measured optimizer steps per full-model variant: 20
- speedup thresholds: <1.10x STOP, 1.10–<1.20x INSUFFICIENT, >=1.20x PROMISING if all gates pass

If either legacy or grouped timing fails, repair4 classifies the result as `ENGINEERING_RUNTIME_FAIL` rather than pretending that a missing timing is a measured sub-1.10x speedup.

## Result transport repair

Repair3's remote Modal computation persisted its evidence correctly, but the local GitHub runner failed while deserializing the H100 return because a Torch-derived object required the local `torch` module. Repair4's Modal functions return **JSON strings only**, and `torch.__version__` is explicitly converted to a plain string before serialization. Authoritative evidence remains the single-use Volume result.

## Authorization boundary

This implementation/PR authorizes zero GPU spend only. Before any repair4 H100 run:

1. exact-head full repository CI must pass;
2. exact-head CORTEX-S CPU gate must pass;
3. exact-head 100M/2B zero-credit gate must pass;
4. the exact successful PR head must be merged only with explicit merge authorization;
5. live `main` must be re-audited and frozen to the resulting merge SHA;
6. the repair4 trigger namespace must be confirmed unused;
7. a separately explicit bounded-H100 spend authorization is required before creating exactly one trigger issue.

No repair1/2/3 run or seed may be retried or reused. No result from repair4 directly authorizes 2B training. Scientific seeds remain untouched.
