# SI original follow-up: does the world task actually require history?

Date 2026-10-16. Source frozen before running new-seed control: `scripts/si_original_no_history_control.py` at commit `e64aa7a7ae973d29188cc97f0e648d5cfb9878b4`, reuses unchanged world generator and backbone in `scripts/si_original_adaptive_horizon_v1.py`.

## Purpose
Previous soft-gated horizon v1 did not respond meaningfully to regime change. Before spending any money or building a more elaborate change detector, determine whether history helps at all on the same hidden-action-dynamics task. An example might be solved almost as well from the current observation/action alone, making attention-horizon comparisons misleading.

## Locked fresh engineering CPU seeds (not scientific replication)
`95211,95212,95213` (not used for this test before the code/protocol freeze). For each paired seed train:
- `no_history`: exact same instantiated 20,265-parameter attention model; causal mask permits current query only, no earlier tokens, action and position still visible.
- `local_6`: exact same initial weights, 6-token causal attention.
- `full_history`: identical weights, unlimited causal prefix.

All receive exactly the same 300 AdamW updates/batch16/sequence24/action-state streams; LR 0.001, decay 0.01, grad clipping 1.0, two CPU threads. Evaluation on 4×32 held-out stable, random-switch, nonlinear worlds per seed at `seed*1000000+87000+5*i`. Per-world transition matrices/switch labels not exposed.

**Primary:** report three-seed mean stable full-sequence and random switch *late-half* MSE for each, and per-seed results; display persistence and zero baselines. A task with nearly equal no-history and historical prediction is a poor test of long-term-memory mechanism, regardless of relative local/global winner. Absolute difference must be interpreted against seed variability, no arbitrary dramatic claim. Secondary: nonlinear MSE, CPU training seconds. Parameter totals instantiated match, but one-token masks leave some projections functionally unnecessary; FLOP and effective capacity parity **not guaranteed**.

If history does not materially help, **STOP this generator as evidence for memory/attention architecture**, not the entire original SI hypothesis. Redesign the task with dependency on episodic history and private per-episode rules before further GPU consideration. Negative results retained. Paid Modal/Colab/GPU work disabled. Seed attempts are one-time; record any interruption without rerunning consumed seeds.
