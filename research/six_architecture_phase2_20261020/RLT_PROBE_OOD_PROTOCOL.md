# Fixed readout: extrapolation and overwrite checks

Prospective inference-only follow-up after the native frozen-state probe passed
its trained-length gate in all3 seeds. No further head/backbone optimization.
Use all six fitted heads, including the failed public-prior seed.

Fresh128 paired prefixes at lengths2048 and4096 (256examples/length), namespace
78M. Also64 paired source groups per gap1/16/127/511/1023 at length1024,
namespace79M, using the original random-overwrite generator and swapping only
the last SET bit within each group. Independent oracle must verify each label.
Shuffle batch rows independently of labels. Report original decoder and fixed
probe, both classes, and erased-carry probe accuracy. Erasure must be50% for
gaps beyond the current chunk; local gap1 can remain solvable without carry.

These samples were not used in fitting or selecting heads. Gate: both classes
>=95% in every new length/gap in all3 seeds per arm. A failure localizes the
limits of the simple readout repair; do not retune on these cases. A success
still concerns a one-bit synthetic state machine and establishes no language
model, reasoning, or architecture frontier advantage. Training exposures0.
