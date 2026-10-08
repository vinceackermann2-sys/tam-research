# CPW corrected binding — provisional replication metrics readout (#1330)

**Status: CPU-only, preauthority, no training, no model evidence.**

This readout implements the threshold screen described in issue #1319 without allocating scientific seeds or launching Modal/H100. The historical v4/v5 long-recall benchmark was confounded by a query-blind positional shortcut (#1301); the corrected #1307 task and the five-arm evaluator (#1321) are unchanged.

## Usage

`screen_replications(rows)` consumes **three** paired *claimed* replication records (each declares `phase="replication"`, its seed and a 40-char source SHA) with **all five** candidate/baseline arms, 1500 steps, 96,000 examples and exactly 512 held-out examples at each of the four delays, plus 512 four-query source groups. This is a pure function; it never calls the trainer. The function validates reported parameter counts and telemetry but **does not verify** that any supplied metrics came from an actual run. The SHA and seeds in supplied records are unverified; synthetic fixtures in tests are explicitly fabricated.

## Preregistered metric gates

All three seeds require a valid Transformer positive control (mean long-distance >=80%, distance-256 >=70%) and a valid sequence-only negative control (long-mean <=5%). Failure on even one seed makes the panel invalid. Every candidate must meet all of the following on **at least two of three** seeds:

- mean long-distance accuracy >=80%; distance-256 accuracy >=70%
- at least 40 percentage-point accuracy advantage over sequence-only independently at delays 64, 128 and 256
- strictly lower query-only NLL than sequence-only at each of those three delays
- at least 70% of identical-source four-query counterfactual groups entirely correct

A provisional *strong* signal further requires passing all three seeds and long-distance mean accuracy within 5 percentage points of Transformer at **every** seed.

Reports are labeled `PROVISIONAL_METRICS_ONLY` regardless of apparent success. The screen cannot certify accuracy of training code, provenance, honest runs, scientific seeds, budget/compute authority, preregistration adherence, or statistical significance. It never authorizes experiments and never uses the label `breakthrough` as an outcome.

## Governance

- No mutation of historical experimental code, dataset, model or result stores.
- No H100/Modal run, paid allocation, fresh seed or namespace allocation, or write to `main`.
- A separate explicitly approved single-use scientific run-control is required, with durable pre-GPU identity reservation and source/credit/namespace audit.
- Even valid synthetic-memory metrics would not establish broad language-model capability or a new breakthrough.

Dedicated CPU-only adversarial fixture tests exercise individual thresholds, malformed records, incomplete replications, nonfinite values, control failures and 2/3 versus 3/3 support.
