# Six architecture research continuation

CPW, PGW, Conv7, CHM, bounded RLT and the user's SI Original world-scoped V2.
This package contains an evidence audit, actual finite CPU/A100 experiments,
source-pinned protocols, all recorded outcomes and targeted implementation work.
It does not establish a breakthrough or complete a100M/2B-token study of each
architecture. Historical failed attempts remain unchanged.

- [Research allocation decisions](DECISION_REGISTER.md): what to continue, pause
  or stop scaling, and the next evidence needed.
- [Full assessment](ASSESSMENT.md): architecture-specific blockers and evidence.
- [Generated results](RESULTS.md) and [experiment CSV](EXPERIMENTS.csv).
- [Execution/persistence limits](EXECUTION_NOTES.md) and [reproduction](REPRODUCE.md).
- [Source inventory](SOURCE_INVENTORY.json): indexed research paths/content hashes
  across four pinned refs; not a claim that every historical run was reproduced.

Frozen protocols precede the corresponding new experiments: [first batch](PROTOCOL.md),
[GPU extension](EXTENSION_PROTOCOL.md), [CHM unseen entities](CHM_GENERALIZATION_PROTOCOL.md),
[PGW control](PGW_CONTROL_PROTOCOL.md), [RLT length](RLT_LENGTH_PROTOCOL.md).
The later [CHM broad-training study](CHM_BROAD_TRAIN_PROTOCOL.md) explicitly opens
native DEVELOPMENT after its new source/training recipe is frozen. Native TEST
remains unopened throughout.

`results/` contains immutable terminal JSON records and diagnostic clusters.
Mutable PROGRESS files are status only. Remote exports retain dated snapshots.
Checkpoint weights/data binaries are kept outside Git; their retrieval status
and hashes are explicit. Run `python audit_artifacts.py` to check recorded caps,
source hashes and locally available checkpoint integrity.

`si_factual_search_loss.py` is a tested proposal to repair the untrained search
value head and planning/replay coordinate mismatch. `rlt_paper_merge.py` implements
the current paper's feedback merge alone. Neither module is a trained successful
successor or a complete reproduction. `test_conv7_causality.py` checks future
isolation with active, nonzero convolution gates.
