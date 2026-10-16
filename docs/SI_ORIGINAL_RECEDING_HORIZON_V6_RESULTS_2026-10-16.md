# Original SI v6 — Full-source-attested receding-horizon CPU results (2026-10-16)

**Result: FAIL to advance unchanged original hybrid toward paid compute. No SI breakthrough.** The original hybrid+identical fast memory did not beat the Transformer+identical fast memory, and classical ridge-all control was much better in held-out nonlinear worlds. Replanning with the true first observation did NOT reduce nonlinear-world regret for either neural model on average.

## Execution and source provenance

- Isolated branch `experiment/si-original-receding-horizon-v6-20261016` stacked on v5 PR #1477 -> v4 PR #1476 -> original architecture PR #1459. No writes/merges to main or other research tracks; no GPU/Modal/Colab money or background automation.
- Preregistered before fresh confirmation: `docs/SI_ORIGINAL_RECEDING_HORIZON_V6_PROTOCOL_2026-10-16.md`.
- **FULL ORIGINAL 298-LINE MODEL SOURCE BYTE VERIFIED.** Local `tam_research/si_original_hybrid_v0.py` Git blob `575c17dc2f39a29d930d82179aa0c96dcbdc7d0f` matches the original GitHub source exactly. Imported v3 simulator, v4 fast-memory and v5 planning scripts also byte match parent GitHub blobs (`a266f7e86c4b7224aa93c20de6089c4543be26db`, `2f35163986d5bc9f8d4f9e0feb1054b2f6e9b937`, `412a2d772f4860727a4808a1ae0a1411f26f9df5`). New executed script `scripts/si_original_closed_loop_v6.py` Git blob **`ae691b5396da6668b46d39fb86bd51956fe17a0e`**, SHA256 `30c63b739dbc0b938cf2a525fe188ae37572aad47bb843758ed8a6ec1f5139bc`, byte-identical to the file committed on this branch. The script fails closed if the base model blob changes.
- Local four CPU integrity tests passed: original source SHA, deterministic second-step query, genuine visible post-first-action observation, and no query-target leakage from simulator targets.
- CPU PyTorch 2.10.0+cpu, two threads. Pre-confirmation smoke **98500** for 2 optimizer steps/batch3, excluded. Fresh **98611 / 98612 / 98613** all completed **once** with empty stderr, no interruptions/retries. Per seed matched 275 AdamW updates/model, batch16, seq20, LR=.001, WD=.01, clip1. Train direct/delayed/switch; nonlinear fully withheld. Each family uses 3×48 unseen worlds/seed = 432 worlds total/family, same worlds/goals for all planners. Two models: original hybrid+913-param episodic memory **65,527 params** and dense Transformer+identical memory **65,550 params**. Same predictive memory update and training conditions; not equal hardware FLOPs.
- **Goal-conditioned two-action open-vs-closed:** Each planner first chooses 1 of 16 action pairs based on its *predicted* two-step end state. The evaluator executes the planner's selected first action, discloses the actual x20, and the planner may change only its second action based on this newly completed transition. Closed-vs-open compare the same first action and goal; full hidden simulator dynamics used *only by evaluator* for true outcome and regret. This is fully observed 2-step receding-horizon planning, not a learned Bayesian uncertainty representation or partially observed long-term planning.

## Primary held-out nonlinear results (mean true regret, lower better)

| Planner | Open loop | After true observation | Closed pair exact accuracy | Second action changed |
|---|---:|---:|---:|---:|
| **Original SI + memory** | 0.187054 | **0.189275** | 0.31019 | 0.03935 |
| Transformer + identical memory | 0.112301 | **0.114360** | 0.32870 | 0.03704 |
| Ridge all completed history | 0.025805 | **0.025230** | **0.58333** | 0.01157 |
| Ridge last six completed transitions | 0.271432 | **0.225513** | 0.33796 | 0.15278 |
| Random 16-way action pair | 1.667742 | Not applicable | ~0.0625 theoretical | Not applicable |

**Per-seed paired nonlinear regret:**

| Seed | Hybrid open | Hybrid closed | Transformer open | Transformer closed | Ridge-all closed |
|---:|---:|---:|---:|---:|---:|
| 98611 | 0.143787 | 0.142291 | 0.119900 | 0.114584 | 0.030958 |
| 98612 | 0.229420 | 0.225654 | 0.149364 | 0.156248 | 0.019057 |
| 98613 | 0.187954 | 0.199881 | 0.067641 | 0.072248 | 0.025675 |

Replanning reduced hybrid nonlinear regret in **2/3** seeds but was worse in the third; **mean regret increased 1.2%**. Hybrid nonlinear closed regret is **65.5% higher** than matched Transformer, and roughly **7.5×** classical ridge-all. Hybrid had less than 4% of second actions changed after new observation; little feedback behavior. The preregistered >=20% average reduction, Transformer and ridge-all advantage requirements **ALL FAIL**.

## Secondary world families: regret, open -> closed

| World family | Hybrid+memory | Transformer+memory | Ridge-all | Ridge last-six |
|---|---|---|---|---|
| Direct | 0.16102 -> 0.15574 | 0.09930 -> 0.10361 | **0.01579 -> 0.01540** | 0.23373 -> 0.19715 |
| Delayed | 0.26071 -> 0.25580 | 0.17196 -> 0.17034 | **0.03780 -> 0.03316** | 0.23472 -> 0.23771 |
| Switching | 0.30487 -> 0.27485 | 0.24076 -> **0.20431** | 0.60516 -> 0.49103 | 0.23301 -> 0.23758 |
| Unseen nonlinear | 0.18705 -> 0.18928 | 0.11230 -> 0.11436 | **0.02580 -> 0.02523** | 0.27143 -> 0.22551 |

Under switching dynamics both neural planners benefitted from the true intermediate observation, but the matched Transformer still won (closed 0.20431 vs hybrid 0.27485). Classical ridge-last6 was better than the hybrid on switching worlds (0.23758 vs 0.27485) but lost slightly to Transformer. This depends on synthetic sampling, not general transfer.

Total per-model CPU training + evaluation reference mean: hybrid **8.494 seconds**, Transformer **6.590 seconds**, ratio **1.289×**. Source-matched implementations, parameter matched within ~0.04%, but true active FLOP and hardware equalization remain unproven.

## Scientific decision

**NO-GO for unchanged original hybrid with this memory and naive receding-horizon planner.** This test clarifies that the v5 planning problem was not solely caused by the earlier local *partial-source* provenance issue: the newly fully source-verified original hybrid is still outperformed in regret by the same-memory Transformer and simple online model identification on the withheld family. These are distinct seeds/tasks; do NOT claim direct numerical reproduction of v5.

The v6 procedure still lacks explicit uncertainty, belief tracking and information-seeking actions, despite the use of real feedback. A credible subsequent CPU track would preregister **partially observed, changing, delayed worlds with stochastic observation noise and a genuinely causal Bayesian / multi-hypothesis belief tracker**, then compare learned inference to explicit classical filters, and evaluate *closed-loop actual reward*, calibration and sample-efficiency. No pretrained SI capability, no real language/multimodal system or breakthrough claimed. Never spend GPU credits or merge to main based on this result.
