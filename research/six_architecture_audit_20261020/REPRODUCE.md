# Reproduction and artifact scope

The source-pinned native checkouts used by this batch were:

| Role | Ref/commit |
|---|---|
| Main (PGW/CHM/Conv7 source) | `4362402592ac0cad5ed3d12f1d7defe68739a2b5` |
| Canonical CPW | `aad622d1eea591545a3482618ca8b269e4d9d43f` |
| Bounded RLT/AX | `6dd09e019a65ea307f9027f6aabd4f31b11a5b3c` |
| SI Original world-scoped V2 | `9298c117e736676269107374a402519e45e52f71` |

See the recorded ATTEMPT files for authoritative full SHAs; a different checkout
is a different experiment. Install the dependencies declared by the selected
source. Local execution used Python3.12, torch2.10.0+cpu, NumPy, pytest9.1.1.
The Colab A100 used Python3.13.15 and torch2.11.0+cu130, outside the repository's
declared torch<2.11 range. These environment facts are limitations, not support
claims. FineWeb preparation additionally uses datasets/tiktoken/NumPy and pins
the returned dataset commit and prepared token-file hashes.

Use a distinct empty output directory for every new attempt; never rerun into an
existing attempt directory. Example local commands (replace placeholders):

```text
python run_pilots.py --track chm --repo <main-checkout> --out <new-empty-dir>
python run_pilots.py --track pgw --repo <main-checkout> --out <new-empty-dir>
python run_pilots.py --track si --repo <si-checkout> --out <new-empty-dir>
python run_pilots.py --track rlt --repo <rlt-checkout> --out <new-empty-dir>
python run_pilots.py --track cpw --repo <cpw-checkout> --out <new-empty-dir>
python conv7_language.py --help
```

`cpw_binding_calibration.py` and `conv7_fineweb.py` are fixed Colab entry points,
with no command-line parser: configure their declared `/content` root and pinned
checkouts before launching them once. Do not use `--help` on those two scripts.
Their frozen protocols specify the budgets. All
trained models check the actual parameter count and exposure cap. Optimizer data
passes are counted as exposures, including repeated tokens. Forward-only
evaluation is separate. Near100M Conv7 meta-device counts are not training runs.

For verification of the newly proposed losses/merge and caps:

```text
python -m pytest -q test_limits.py test_si_factual_search_loss.py test_rlt_paper_merge.py test_conv7_causality.py
python summarize.py
```

Run repository-native tests separately in their own pinned package contexts;
mixing main and SI modules in one Python process can import the wrong package.
The batch passed41 isolated native tests and 10 new targeted tests, including
future-token isolation with active convolution gates.

Checkpoints are weight-only torch state_dict files. SI external memory, fast
state and replay were not exported by the original pilot; it cannot reproduce a
complete persistent lifetime from those weight files. Remote GPU checkpoint
retrieval status is documented separately. No optimizer-resume claim is made.

`chm_unseen_entities.py` performs inference only on the existing local CHM weight
files. Its panel is frozen separately in `CHM_GENERALIZATION_PROTOCOL.md` and
uses entities excluded from the capacity optimizer, within the TRAIN namespace.
It never opens the native reserved DEVELOPMENT/TEST cases.

RESULT JSONs, all unsuccessful attempts and data/source hashes are authoritative;
CSV/Markdown/plots are regenerable summaries. These finite development pilots
are not a completed100M/2B-token confirmation or a modern-baseline comparison.

The published package disables line-ending normalization to preserve the exact
script bytes whose SHA256 values appear in the executed attempt records.
