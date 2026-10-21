# Interpretation against related work

Hybrid attention/recurrent and convolutional local mixing has substantial prior
art. A win over this repository's matched Transformer is evidence about this
implementation and budget, not sufficient evidence of a new research frontier.

- [Griffin](https://arxiv.org/abs/2402.19427) mixes gated recurrences and local
  attention and reports language and hardware comparisons. Its setting differs
  from this experiment; it was not trained in our current panel.
- [Mamba-2 / state space duality](https://arxiv.org/abs/2405.21060) provides a
  selective recurrent baseline and efficient algorithms. A state-memory claim
  should eventually compare against an implemented, tuned modern recurrence.
- [RWKV-7](https://arxiv.org/abs/2503.14456) is another relevant recurrent
  baseline. We did not train it here and cannot infer superiority over it.
- [FineWeb-Edu dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
  describes the source corpus. The exact source revision, individual parquet
  SHA256s, prepared split counts and token hashes are saved in data/DATA.json.

The phase-two findings can justify another experiment or a decision to stop a
specific implementation. They cannot prove an entire architecture family
impossible, certify novelty, or establish realistic multimodal/agent capability
from a synthetic cue task.
