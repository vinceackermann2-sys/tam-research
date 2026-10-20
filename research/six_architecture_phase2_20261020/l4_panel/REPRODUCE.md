# Acknowledged L4 continuation

This is a separate seed/trial from the interrupted A100 run. Read
RECOVERY_PROTOCOL.md before executing. The selected model sources and original
train_panel.py live in the parent artifact folder; copy them beside
l4_controller.py in an empty /content/tam-l4-panel-20261020, as worker.py expects.
All source bytes and hashes must be preserved. The worker configures the corpus
builder's root before calling it, then starts the controller on that same root.

The supervisor must save ACTIVE_SEGMENT.json locally before writing the matching
digest to RESERVATION_ACK.json. When STATUS says awaiting_checkpoint_backup,
download every chunk from EVIDENCE.latest, verify individual bytes/SHA256 and
the concatenated checkpoint SHA256, fsync the local files, then write its
checkpoint_sha256 as the digest in CHECKPOINT_ACK.json. These are separate
acknowledgements. Do not acknowledge an incomplete download. The controller
waits up to60minutes for each acknowledgement and guards the exposure cap.

After the controller finishes, l4_postprocess.py configures the unchanged parent
inference and task-scoring modules for this L4 root. It runs terminal inference
measurements followed by LAMBADA/PIQA. The L4 INFERENCE_PROTOCOL.md overrides
the earlier hardware name; scoring, prompts and timed procedures are unchanged.
The actual hardware/package environment is exported in ENVIRONMENT.json.

Recovery uses only verified full optimizer checkpoints. RESUME.json contains
the saved evidence, active_arms and stage_index. evidence.latest must reference
only verified states, and evidence.charges must conservatively include each
lost reserved segment exactly once. Preserve orphaned metrics separately.
The controller uses a common remaining committed-token target constrained by
each lineage's2B charged cap. A current model may already have advanced farther
than a reduced final target after another arm's interruption; label such final
comparisons unequal and prefer retained common scientific milestones.

All checkpoint weights, raw token streams and account/session credentials are
excluded from Git. Source code and measured JSON records are published so the
model equations, configuration, selection rules and accounting can be reviewed.
