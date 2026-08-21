# Legacy local experiment archive

This directory preserves the recoverable text artifacts from the pre-Modal experiments that originally lived only under `/mnt/data`.

## Bundle

`legacy_local_experiments_text.zip.b64` is a base64-encoded ZIP archive.

SHA-256 of the decoded ZIP:

```text
aaa34a3c6ff6d0b5a06a7d9be673ead7c1421706b8af05308a3f887262aa565d
```

Decode on Linux/macOS:

```bash
base64 -d archive/legacy_local_experiments_text.zip.b64 > legacy_local_experiments_text.zip
unzip legacy_local_experiments_text.zip
```

The archive contains all recoverable **text/source/results** from:

- the original small ATAM associative-recall experiment (`architecture.py`, `train.py`, README, JSON/CSV results)
- the 25M/50M/75M Transformer vs ATAM vs TAM three-way local experiment (`tam_threeway.py`, evaluation code, all JSON results, CSV summary)
- the original 25M three-seed TAM v2 screening experiment (`tam_v2_experiment.py`, all JSON results, aggregate JSON, CSV summary)
- a SHA-256/size manifest for legacy `.pt` checkpoints that remain outside ordinary Git history

## Checkpoint policy

Large `.pt` binaries were deliberately not added to ordinary Git history because several are ~50–150 MB each and would permanently bloat the repository. Their file names, byte sizes, and SHA-256 digests are preserved in the bundle and in `legacy_checkpoint_manifest.csv`.

The serious Modal-era checkpoints are stored on the persistent Modal Volume `tam-research-data`; see `docs/RESEARCH_HANDOFF_2026-08-21.md`.
