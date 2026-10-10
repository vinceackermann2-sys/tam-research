# Attention-8 + Conv7: two-step CPU autograd smoke test

**Status: CPU-only synthetic optimizer test, not a language-model training result.**

The Conv7 successor adds a zero-initialized scalar residual gate to every one of the 16 FFN-only blocks. At initialization, the new depthwise convolution path is disabled. This preserves the old tokenwise block's initial forward computation when FFN/LN weights are shared, but also means **the convolution's weights have exactly zero gradient on the first update**. Whether the gate itself can learn and subsequently expose the filters to a gradient was previously not tested through an optimizer step.

New CPU tests run the actual Conv7 block with an AdamW optimizer for **two steps on a constructed synthetic regression target**. The target intentionally points in the precomputed direction of the convolutional feature, ensuring a nonzero gate gradient at initialization. The assertions cover:

- Step 1: the trainable gate receives a finite nonzero gradient, while the convolution weights' gradients are zero because the gate is zero.
- Step 2: after the gate opens, convolution weights receive finite nonzero gradients and change during an optimizer update.
- A separate two-step synthetic update retains exact strict prefix causality under changes made only to future input positions.

The tests use the 512-wide, 2,729-FFN actual block and existing Conv7 implementation. No trained full-model checkpoint or real corpus data is involved. These observations verify a local gradient pathway **but not that the gate will open usefully during 200M/2B LM training**. In particular, the regression target is designed to activate the gate and does not represent naturally encountered next-token objectives.

No H100/Modal launch, engineering seed, durable paid reservation, replica, GPU credit, or science result is authorized. The frozen Attention-8 scientific screen and seed 60232 remain unchanged.
