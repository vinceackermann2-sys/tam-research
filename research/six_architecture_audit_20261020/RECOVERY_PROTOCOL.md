# Fresh attempts after an unverified infrastructure interruption

Before launching these attempts: the original Colab A10040GB worker was launched,
then its agent connection was lost. The only current assignment accessible via
the user's existing official CLI authorization is A10080GB, with a new uptime
and no original research directory. Original extension final outcomes and weights
are unavailable. Record them as interrupted/unverified, not architecture failures
or successes. Never silently rerun their seeds or overwrite their records.

The new finite attempts use the already assigned80GB VM, without allocating a VM:

- CPW: the same bounded positive-control calibration in EXTENSION_PROTOCOL.md,
  fresh initialization202610261 and new TRAIN generator indices30,000,000+step;
  development31,000,000+delay, query-swap batch seed32,000,000 (128 groups). Same5000 updates,
  batch32, context320,51,200,000 exposures. Only a passing Transformer control
  licenses the four candidate arms. No retuning, retry or resume.
- Conv7: the same width192/24-layer, three-arm FineWeb development screen, fresh
  initialization202610262, training-window RNG33,000,000 and validation-window
  RNG34,000,000. Same6103 updates/batch16/context256=24,997,888 exposures/model,
  optimizer, terminal-horizon schedule and declared quality/speed gates. Data
  preparation pins the returned public dataset commit and hashes token files.

New source files/attempt names distinguish this from the lost extension. All
three arms in a comparison run sequentially on the same80GB GPU. Do not pool its
absolute throughput with the40GB pilot. Every actual model stays <=100M
parameters and <=2B exposures. Original unverified attempts remain listed as such.
Automatic CLI export copies terminal metrics and checkpoints to the user's local
workspace as they appear; no cloud storage purchase or Drive mount is involved.
