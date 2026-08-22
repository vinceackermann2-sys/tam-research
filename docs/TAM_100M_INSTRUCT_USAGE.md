# TAM v3 100M / 2B Instruct Usage

The completed instruction-tuned checkpoint is:

`/vol/full100m-runs/TAM-v3-100M-2B-seed8100/final_dpo.pt`

Do not substitute the base `latest.pt` or the intermediate `sft.pt` when the goal is to interact with the final instruct model.

## Interactive terminal chat

From a checkout of this repository with Modal authenticated:

```bash
pip install "modal>=1.1,<2"
modal run modal_tam100m_chat_app.py
```

This starts an interactive REPL. The model runs remotely on one L4 and loads only the final DPO checkpoint from the existing `tam-research-data` Volume.

For one prompt:

```bash
modal run modal_tam100m_chat_app.py --prompt "Explain why the sky looks blue in one sentence."
```

Generation defaults to 128 new tokens, temperature 0.7, top-k 40. Use `--temperature 0` for greedy deterministic output.

The app also exposes a `POST` Web Function when deployed with `modal deploy modal_tam100m_chat_app.py`. It has `requires_proxy_auth=True`, so it is not an unauthenticated public inference endpoint.

## External instruct benchmark

The benchmark launcher is `modal_tam100m_benchmark_app.py`, triggered by an issue whose title begins `[modal-benchmark100m]` or by manual workflow dispatch.

It compares the final TAM DPO checkpoint to `HuggingFaceTB/SmolLM2-135M-Instruct` on the exact same deterministic sampled examples from:

- ARC-Easy
- ARC-Challenge
- PIQA
- HellaSwag
- OpenBookQA
- GSM8K

The five multiple-choice tasks use mean continuation log-likelihood. GSM8K uses greedy generation and final-number exact match. The same sample indices are used for both models. Evaluation is capped at 200 examples per multiple-choice task and 100 GSM8K examples for a fast first external capability read.

The comparator is an external reality check, not a scientifically matched architecture control: SmolLM2-135M-Instruct is about 135M parameters and its upstream model card reports 2T pretraining tokens, versus TAM's 101.8M parameters and 2B tokens.

The durable result is written to:

`/vol/full100m-runs/TAM-v3-100M-2B-seed8100/external_benchmark_v1.json`

A clean architecture comparison still requires a parameter/data/post-training-matched standard Transformer trained under the same 2B-token protocol.
