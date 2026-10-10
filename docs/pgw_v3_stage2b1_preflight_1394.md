# PGW-v3 Stage-2B1 — static preflight, NOT CPU training

**Contract:** [#1394](https://github.com/vinceackermann2-sys/tam-research/issues/1394)  
**Branch:** `research/pgw-v3-stage2b1-static-preflight-v1`  
**Source-main at branch creation:** `b7fa361597f24d622f263ac6a2171743bc1f9c07`  
**PGW history:** Stage-0 #1359; Stage-1B data oracle #1374; Stage-2A answer interface #1379; Stage-2B0 predictor-gradient helper #1384, merged PR #1389 at `5d214106fd9a02e2cb757c3a77e3122da56a760b`, original PR CI and merged-main CI PASS.

The past mechanism results #1181 and #1201/#1213 remain negative. Stage-2B1 audits only **benchmark readiness and gradient availability**. It does not train, allocate CPU scientific experiments, consume seeds, create scientific control runners, or start Modal/H100.

## Part A — finite, reproducible fixture panel and shortcut-baseline checks

The Stage-1B dataset grammar and generator are imported **unchanged**. We audit `fixture_panel(per_cell=4)`:

- 3 split names × 5 target last-write positions (2–6) × 3 earlier-overwrite counts (0–2) × 2 interference conditions × 3 completed-chunk delays (1/2/4) × 2 present/missing query outcomes × 4 sample indices = **2,160 samples** total.
- 180 factorial cells per split, exactly 4 observations per cell; 720 examples per split, comprising 360 present and 360 missing queries.
- Input lengths 80/88/104 tokens for delay 1/2/4. Combined corpus contains **195,840 input tokens**.
- Recompute SHA256 of every input and independently verify its final READ/QUERY marker and external answer with the frozen strict causal last-key-write oracle. Refuse any repeated stream fingerprints, including cross-split collisions; refuse any input READ suffix that contains its external answer value.
- Generate deterministic canonical sorted row identity lines containing split, index, factorial labels, **input hash** and external label, then SHA256 digests overall and by split. The external label appears only in this **audit record**, never appended to model inputs.
- Four fixed **algorithmic** controls: always-NOT_FOUND, copy-chunk-2, last WRITE regardless of key, most-frequent WRITE value. Report per-split overall, present-only and missing-only correct counts for each. Oracle itself must answer every valid example by construction. None of the four controls is trained or comparable to an LLM.
- Every split must expose the critical always-NOT_FOUND shortcut: **360/720 = 50% overall accuracy; 0/360 present-key and 360/360 missing-key correctness**.

The default finite panel is only an *integrity fixture*, not a real complete training/evaluation corpus. Changing corpus size, sample index ranges, balancing, split assignment or seeds for a future trained study requires a new preregistration, new complete manifest hashes and freeze **before** any optimization. Observed uniqueness across this panel is not a universal proof of no possible collisions.

## Part B — gradient availability and active-capacity warning

Take two verified Stage-1B CPU samples (same delay, one present, one absent), initialize the six Stage-2A route modes with identical copied parameter values under a locally isolated torch RNG fixture, and **never change those weights**.

Run two separate `backward()` derivative probes **without optimizer**:

1. Answer-only cross entropy: records whether `grad is not None` and the exact count of **nonzero** gradient elements for each parameter group.
2. Predictor auxiliary loss `L_aux` imported unchanged from Stage-2B0: record the same counts. This is **not combined** with answer CE, and no auxiliary coefficient `lambda` is chosen.
3. Report the union of nonzero-gradient coordinates across both probes (still a single-batch, empirical lower bound) and verify the complete model state dict remains byte-for-byte identical to its starting initialization.

Parameter groups: predictor down/up, utility scorer, other workspace parameters, all other local encoder/embedding/answer head parameters. All route modes have **20,354 instantiated params**, but actual active gradient counts differ. In the `no_workspace` route, answer CE cannot reach workspace parameters even though the auxiliary predictor probe can; the auxiliary probe cannot reach the answer head. No one-batch nonzero-gradient count proves architecture-equivalent active capacity across all inputs or training trajectories.

This diagnostic uses a local deterministic structural initialization only, not a scientific seed or replication. No CUDA, checkpoint loading, Modal, external network, optimizer, scheduler, or train/test model inference study occurs.

## Acceptance boundary

With original exact-head PR CI and original merged-main CI both PASS, classification may become `PGW_V3_STAGE2B1_STATIC_PREFLIGHT_READY`. That means:

**Yes:** deterministic corpus integrity example, independent oracle, shortcut-baseline audit, empirical per-arm gradient capacity accounting, and evidence that inactive-parameter padding must not be counted in fair comparisons.

**No:** active-parameter-matched model set, equal-compute Transformer control, learned retrieval, learned predictive salience, positive NLL, robustness across seeds, breakthroughs, or authority to execute a run.

The *next* research protocol must freeze a complete balanced train/val/test corpus and exact hashes, active-matched architectures (<=0.1% mismatch for primary mechanistic attribution), equal auxiliary objectives, optimizer settings and loss weighting, elapsed CPU budget, fresh paired scientific seeds (>=3), unique reserved job IDs, no retries or resume, and preregistered superiority thresholds **before any optimizer step**. H100/Modal research remains a separately explicitly authorized gate.

`cpu_training_authorized=false`  
`gpu_authorized=false`  
`scientific_seed_consumption_authorized=false`  
`breakthrough_claim_allowed=false`  
`scale_up_authorized=false`
