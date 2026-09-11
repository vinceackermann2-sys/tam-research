# CHM-v1 provenance-bound scientific gate

Classification: `ZERO_CREDIT_PROVENANCE_COMPLETENESS_ONLY`.

Issue #854 remains the scientific authority. This document adds no GPU, Modal, paid-run, seed-allocation, architecture, data, optimizer, threshold, or claim authority.

## Why this exists

PR #899 added cryptographic corpus fingerprinting, but the existing executable scientific gate could still classify per-seed records that omitted the exact corpus-byte identity and exact execution commit.

That makes the result harder to audit even if every scientific metric is otherwise present.

## Authoritative interpretation entry point

For any future authorized CHM-v1 scientific run, use:

`tam_research.chm_v1_provenance_gate.evaluate_scientific_gate_with_provenance`

The wrapper first requires every per-seed record to contain:

- `execution_code_sha`: lowercase full 40-hex Git commit SHA;
- `corpus_fingerprint`: the frozen CHM-v1 logical corpus contract plus exact byte counts and SHA-256 values for `train.bin`, `val.bin`, and `meta.json`.

It then requires all three reserved-seed records to agree exactly on both identities before delegating unchanged to `tam_research.chm_v1_scientific_gate.evaluate_scientific_gate`.

Malformed or inconsistent provenance refuses classification. Provenance cannot rescue a failed scientific criterion and does not modify any preregistered threshold.

## Corpus-hash authority

The hash values themselves are not hard-coded in the repository before the corpus is present in the future execution environment. A separately explicit paid-run authorization must freeze the observed fingerprint before GPU allocation. The future runner must compare the actual corpus against that frozen fingerprint before any reserved seed is allocated.

## Spend boundary

Scientific seeds 8611/8612/8613 remain reserved and unused. This module is interpretation/preflight plumbing only. It contains no launcher, workflow, Modal import, GPU allocation, automatic retry, or paid-compute authority.
