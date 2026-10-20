# Measurement and persistence limits

New GPU training used the connected NVIDIA A100-SXM4-40GB. Local diagnostics
used an i7-12700 and torch CPU, with two PyTorch threads per worker. RLT and the
later PGW control overlapped in separate CPU processes. Their wall times are
descriptive and cannot establish a controlled speed comparison. GPU model arms
ran sequentially in one worker; no CPU timing is substituted for GPU speed.
L4/T4/TPU portability and throughput were not measured.

Language throughput counts optimizer input tokens and excludes evaluation/data
preparation. CPU Attempt throughput includes its evaluation interval. These are
different metrics and should not be pooled. Evaluation tokens are not optimizer
exposures. All training was fresh, with no resume or checkpoint selection.

SI's reported10,240 inputs are five modality-sequence positions per public
example, not language tokens or raw pixels. Its tiny factual fast updates and
latent replay are additional adaptation work; their latent-head replays are not
included in that direct-input count. Even a conservative100,000-position bound
including all such work remains far below 2B. The first SI weight-only files lack
the external memory/fast/replay snapshot; no full lifetime reproduction is claimed.

The first Colab GPU batch's terminal metrics and hashes were retrieved locally.
The browser-to-agent WebSocket closed at 09:18 local while a later finite worker
had already launched. A live GPU runtime is separate from that agent connection.
Until an export is imported, later GPU outcomes and remote checkpoint persistence
remain **unverified**. Never infer completion or success from a launch response.
The export-only recovery cell never launches training. A partial export is a
dated snapshot, not a terminal result; subsequent snapshots retain earlier files.

At09:55 the existing authenticated official CLI route verified a different
A10080GB assignment with uptime1,348 seconds and no original research directory.
The original extension is recorded explicitly in INTERRUPTED_GPU_EXTENSION.json.
The fresh RECOVERY_PROTOCOL.md attempts use new initialization/data/window seeds,
and automatic CLI exports preserve each terminal record/checkpoint locally.
No runtime was allocated by that recovery. Absolute40GB and 80GB throughput is
kept separate. Export I/O can overlap later GPU arms; no other training job shares
the GPU. Any near-threshold speed claim needs a clean dedicated timing replication.

The first CHM capacity/inference studies did not open native DEVELOPMENT/TEST.
Their optimizer-unseen panel uses 63 TRAIN entities excluded from the one-entity
optimizer. A separately frozen CHM_BROAD_TRAIN_PROTOCOL.md study then trains all
64 TRAIN entities and opens the16 reserved DEVELOPMENT entities per family.
Native TEST stays closed throughout. Shared development
corpora/prefixes and correlated rotations do not count as independent replications.
The Conv7 data split excludes exact duplicate documents; near-duplicate document
leakage has not been independently audited.
