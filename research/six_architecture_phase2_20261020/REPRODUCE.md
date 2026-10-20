# Reproduce this phase

All executed source files are retained unchanged. Never run into an existing
attempt directory. Remote scripts use /content/tam-phase2-20261020; choose a
new root and distinct attempt/seed namespace for a genuinely new study.

Local verification (Python3.12, PyTorch2.10 CPU, NumPy, pytest): set CPW_REPO to a
checkout pinned at aad622d1eea591545a3482618ca8b269e4d9d43f, then run:

```text
python -m pytest -q test_phase2.py test_l4_reporting.py
```

The three checks verify actual meta-device parameter counts/token flooring,
future isolation and live convolution gradients, and exact scaled CPW output
equivalence to the native source plus causal isolation. They do not validate
training efficacy by themselves.
The five reporting checks prevent intermediate, unpaired or over-budget
measurements from being presented as scientific endpoint comparisons and require
terminal benchmarks before reporting the language confirmation complete.

GPU execution used A10040GB, Python3.13.15, torch2.11.0+cu130, NumPy,
requests, tiktoken, huggingface_hub and pyarrow. The runtime's torch version is
outside the repository's declared torch<2.11 range; actual environment is
recorded rather than assumed compatible. Scripts do not require private data.

1. prepare_corpus.py verifies pinned public source parquet SHA256s, removes
   exact duplicate documents and materializes the prospective hash splits.
2. phase2_model.py, phase2_aligned_model.py and phase2_aligned_v2.py together
   define the selected models. train_panel.py checks caps, exclusive claims,
   trains the staged panel and writes full optimizer checkpoint chunks.
3. inference_benchmark.py waits for training completion and measures the
   terminal implementations/ablations. language_benchmarks.py waits for that
   benchmark, runs a scoring-alignment fixture, and scores pinned public tasks.
4. Analyze saved records with analyze_phase2.py. This only generates mutable
   summaries; it never changes trained sources or scientific result JSONs.
   analyze_l4_panel.py generates separate L4_SUMMARY.json, L4_RESULTS.md and
   L4_LANGUAGE_RESULTS.csv. Its scientific comparison table excludes backup
   midpoints and requires matching exposures, seed, GPU and PyTorch version.
   All segment rows remain available, with conservative lost-exposure charges.
   Run plot_l4_panel.py afterward for separate per-stage quality/throughput
   figures. These use scientific endpoints and descriptive block intervals.

Tiny CPU tracks use independent package processes to prevent module collisions:

```text
python si_head_repair.py --repo <SI-V2-checkout>
python rlt_gate_repair.py --repo <bounded-RLT-checkout>
python chm_address_diagnostic.py --repo <main-checkout>
python pgw_workspace_diagnostic.py --repo <main-checkout>
```

The diagnostics require phase-one checkpoints under sibling research_20261020.
SI source commit9298c117e736676269107374a402519e45e52f71; RLT commit
6dd09e019a65ea307f9027f6aabd4f31b11a5b3c; main commit
4362402592ac0cad5ed3d12f1d7defe68739a2b5. Original experiment protocol files
specify the exact budgets and interpretation restrictions.

Checkpoint chunks and prepared corpus binaries remain local rather than in
Git. Reassemble chunks in MANIFEST order and verify the complete SHA256 before
torch.load. A full language checkpoint includes optimizer, RNG and exact update
count. Never automatically retry an ambiguous interrupted training segment:
conservatively charge its entire reserved exposure bound before any continuation.
