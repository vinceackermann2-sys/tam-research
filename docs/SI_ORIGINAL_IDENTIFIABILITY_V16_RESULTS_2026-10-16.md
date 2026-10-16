# Original SI v16 — observational-equivalence and paid verification CPU results (2026-10-16)

**LOCKED DECISION:** Identifiability certificate **PASS**; cost-aware trusted independent probe **PASS**; original SI hybrid versus matched Transformer **FAIL**; no SI breakthrough, paid GPU scaling, automation or main merge.

## Constructive mathematical counterexample (not merely an empirical coincidence)

Consider six hidden calibration bits `theta_i`, three noisy observed calibration channels with signs `s_x,s_y,s_a`, a query bit q and query observation `xq=s_x*q*mu_q+eps_q`. Two worlds:
- **A:** target sign -1, reference/anchor signs +1/+1, true calibration bits `-theta_i` and correct query answer `-q`.
- **B:** target sign +1, reference/anchor signs -1/-1, true calibration bits `theta_i` and correct query answer `+q`.

Couple the *same observation noises*, nominal sensor strengths and cost. Then x/y/anchor calibration and unpurchased x query observations are exactly equal, **bitwise**, while the correct action label is opposite. Every causal no-probe decision-maker observing only those signals chooses the same action in both worlds and has **exactly 50% paired query accuracy**. This is a precise observational non-identifiability result under the balanced two-world prior, not a universal impossibility when persistent feedback or trusted outside evidence is available.

A separately purchased trusted signed probe `yP~N(query_answer*muP,1)` becomes visible **only if bought**. With visible `muP~U[0.35,1.9]` and probe price `c~U[0.03,0.53]`, exact Bayes gain is `Phi(muP)-0.5`; buy iff gain>c, then choose sign(yP). If skipped, no-probe expected accuracy exactly 0.5. Hand-coded v15 classical fault filter uses observed third anchor and is charged its extra 0.025 synthetic fee, unlike raw neural policies.

## Frozen attempt/source provenance

- Protocol `docs/SI_ORIGINAL_IDENTIFIABILITY_V16_PROTOCOL_2026-10-16.md` committed **before** all fresh seeds. Full local snapshot SHA256 `714f12a9354431e63ba8ee8e0f018c59241448fd876c88022e31be611744812b`. Source `scripts/si_original_identifiability_v16.py` Git blob **`0ab7b8331868e16823a9e3e408b8e0fd6ddfabc3`**, SHA256 `707b30300cb447b77634cc7b49e87fda245399c7db97643e046e6964e8aeef47`; tests `tests/test_si_original_identifiability_v16.py` blob **`2265116dbecb73dabbf532fb39e4301e33e29aa4`** SHA256 `7a400e3c55fdf675d74445a53bb4656f58b116aca3f10adc765b108e2fb9252a`. Both exact-match local/GitHub. **All six local integrity tests passed**, including bit-identical observations/private label complement, fault filter invariance and correct probe-value gate.
- Exact unchanged complete original 298-line model source Git blob `575c17dc2f39a29d930d82179aa0c96dcbdc7d0f`; unchanged original v12 neural training and v15 classical filter reused. Models: original hybrid **64,614 params**, matched Transformer **64,637**, smaller non-parameter-matched GRU **10,283**. Neural original training (stationary/one flip ONLY) per model **180 actor +200 observed-arm critic steps**, batch96, CPU2 threads, no synthetic twin training.
- Development smoke `108000` (2+2 steps, batch8, 128 pairs) excluded. Preregistered fresh confirmation seeds **108211,108212,108213** each executed **exactly once** and completed exit0, empty stderr, complete JSON. **512 opposite-label twin pairs per seed** =1,024 evaluation episodes/seed, shared observations and costs. Three trained-model seeds only, CPU engineering not broad statistically confirmed scientific replication. No GPU/Modal/Colab charges, main or other model branches untouched.

## Three-seed mean realized net reward (higher better)

| Policy | Net reward | Optional probe purchase | Mean CPU training seconds |
|---|---:|---:|---:|
| Never buy | **0.500000** | 0% | — |
| Always buy trusted query probe | 0.566254 | 100% | — |
| **Bayes selective trusted query probe** | **0.614460** | **62.04%** | None |
| **Original SI hybrid raw** | **0.595663** | 49.28% | 5.026 |
| Original SI hybrid belief augmented | 0.581888 | 41.21% | 5.553 |
| **Matched Transformer raw** | **0.593148** | 48.11% | 4.011 |
| Matched Transformer belief augmented | 0.570358 | 34.90% | 4.134 |
| Small GRU raw | **0.603332** | 70.70% | 1.143 |
| Small GRU belief augmented | 0.574502 | 34.11% | 1.682 |
| Hand-coded prior-driven fault Bayesian (extra anchor fee) | 0.537688 | ~24% | No training |
| Classical majority (extra anchor fee) | 0.534183 | ~21% | No training |

**Per-seed raw-model rewards** in locked order 108211/108212/108213: original hybrid `[0.594748,0.597956,0.594284]`; Transformer `[0.601922,0.588111,0.589409]`; GRU `[0.610296,0.602527,0.597175]`. Hybrid minus Transformer paired differences `[-0.007174,+0.009845,+0.004875]`, mean **+0.002515**, only 2/3 wins and FAIL required mean +0.015. Small GRU scored higher on mean than both despite much lower size/time.

Selective Bayesian probe per-seed rewards `[0.623146,0.607014,0.613221]` vs always buying `[0.574900,0.559455,0.564406]` vs never `[0.5,0.5,0.5]`: all three paired wins over both. Selective minus always mean **+0.048206** (required +0.025), selective minus never **+0.114460**. Realized selective reward 0.614460 agrees with analytically expected optimum **0.613383**, difference 0.001077 (<=0.025 bound). Correct fraction of purchased queries ~62%, interior [.1,.9]. **Value-of-information gate PASS**.

Every raw/belief full hybrid, Transformer and GRU forced no-probe score is **exactly 0.500000**; no future probe or private labels leaked. Classical fault posterior is **identical** across the observational twins, so its no-probe decisions cannot identify the true explanation. Its >0.5 net reward in the full policy arises from actually purchased probes, and it remains substantially behind cost-aware optimal probing.

## Prespecified research gates and limits

1. **Exact observational equality PASS:** all pre-purchase input features match bitwise; opposite truth; six no-probe neural configurations at exact 50%; classical posterior equal.
2. **Adaptive trusted-probe optimal control PASS:** Bayes selective > always/never by minimum threshold and agrees with theoretical expectation.
3. **Original hybrid competitive advantage FAIL:** +0.0025 mean over matched Transformer raw (requires +0.015) and smaller GRU earns more. Hybrid raw gap to selective Bayes ~0.0188 does pass one subsidiary margin <=0.035, **not enough to rescue aggregate architecture gate**. Original hybrid raw is slower than matched Transformer raw on reference CPU; no measured equal active-FLOP efficiency.
4. **Generality NOT DEMONSTRATED:** the two-world symmetry is intentionally specified and artificial. No result demonstrates that real fault models must behave this way, or that the original full SI stack with true persistent memory/feedback could not learn. No claim of emergence, superintelligence or scaling authority.

**Next test should characterize what independent information or interventional/repeated feedback breaks this symmetry, and measure regret and evidence-acquisition cost against explicit optimal Bayes.** Stop the unchanged original hybrid as a paid-GPU candidate. No main merge, automation or other research branch writes; require exact-head CI/review. Preserve complete individual raw JSON and local manifest/evidence ZIP.
