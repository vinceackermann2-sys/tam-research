# Original SI v16 — Observational equivalence and paid verification (frozen 2026-10-16)

**Preregistered BEFORE confirmation seeds.** Stacked on v15 original SI PR #1488, CI success, from full original hybrid PR #1459. No main merge, other AI track writes, paid compute, GPU, automation, or scientific breakthrough claim.

## Exact two-world identifiability certificate

With six hidden calibration bits `theta_i`, three noisy calibration channels target x, reference y and independent anchor a, and a hidden query answer q, construct two equiprobable counterfactual episodes using precisely the same Gaussian noise, known channel strengths and information prices:

- A: current target polarity z=-1, reference+anchor both healthy; hidden calibration bits -theta and hidden query answer -q.
- B: current target polarity z=+1, reference+anchor both reversed; hidden calibration bits theta and hidden query answer +q.

**Every observed pre-probe tensor is exactly equal between A and B, while their correct query actions are opposite.** Every causal forced-no-probe policy must have 50% paired classification accuracy (irrespective of complexity); the pairwise coupling supplies an executable impossibility certificate under the stated symmetric prior and single-query/zero-feedback horizon. It does NOT establish impossibility with later feedback or under different priors.

An independent *optional trusted query probe* observes `yP ~ N(theta_query*muP,1)`, only when bought. `muP ~ Uniform[0.35,1.90]`, visible price `c ~ Uniform[0.03,0.53]`. Under the symmetric two-world prior, no-probe P(correct)=0.5; with probe P(correct)=Phi(muP); **buy iff `Phi(muP)-0.5>c`**, and act by sign(yP) if bought. Score actual net reward minus price for never, always and selective-buy, plus their analytical expected returns. The pre-existing third anchor sensor is charged separate fixed synthetic 0.025 fee **only for methods explicitly using its reading**.

## Canonical unchanged neural and classical comparisons

Use complete attested 298-line original SI module `tam_research/si_original_hybrid_v0.py` Git blob `575c17dc2f39a29d930d82179aa0c96dcbdc7d0f`, unmodified v12 `Controller`/`train_one`, unmodified v15 classical fault Bayes/majority. Train exact hybrid 64,614 params, Transformer 64,637, smaller GRU (non-matched) **raw and belief** with unchanged 180 actor +200 output-head critic AdamW steps, batch96, CPU threads2, original stationary/one-flip training ONLY; no twin worlds seen in training. Evaluate fresh coupled opposite-label pairs and enforce identical no-probe neural decisions. Classical v15 filters have fixed privileged hand-specified priors and use extra anchor/fee, not an equal-knowledge comparison. The dormant external SI memory, wider multimodal and lifelong capabilities are not trained.

## Frozen code identity, smoke and seed panel

Locally frozen executable `scripts/si_original_identifiability_v16.py`: Git blob **`0ab7b8331868e16823a9e3e408b8e0fd6ddfabc3`**, SHA256 `707b30300cb447b77634cc7b49e87fda245399c7db97643e046e6964e8aeef47`. Local tests `tests/test_si_original_identifiability_v16.py`: blob **`2265116dbecb73dabbf532fb39e4301e33e29aa4`**, SHA256 `7a400e3c55fdf675d74445a53bb4656f58b116aca3f10adc765b108e2fb9252a`. Six pre-run tests passed: original source/parameter parity, exact twin preprobe equality, complementary labels and exact 50% forced no-probe accuracy for all neural types, identical v15 fault posteriors, cost-aware trusted-probe improvement and private input isolation. Development smoke seed **108000** (2+2 updates, batch8, 128 twin pairs), excluded. No design tuning after smoke.

**Fresh previously unused CPU engineering seeds `108211,108212,108213`**, each single attempt, retries=0 and no consumed seed reuse. Per seed six models, 180+200 original optimizer steps, batch96; **512 twin pairs** (1,024 opposite-label episodes)/seed with common evaluation observations/prices. Store each JSON, stderr, exit and hashes. Only three independent training seeds, bounded synthetic engineering evidence.

## Locked gates and stop rule

1. **Equivalence validity:** all preprobe fields match bitwise, true answers opposite, **every** forced-no-probe neural policy achieves exactly .5 accuracy, v15 classical posterior identical across twins. Any violation is a benchmark defect; stop.
2. **Value of information:** selective Bayes realized reward exceeds both never and always buy by >=.025 mean; purchase fraction in [.1,.9]; wins >=2/3 paired seeds over both; realized vs exact analytical expected reward within .025. This only validates optimal purchase for the given known simple law.
3. **Original hybrid advantage:** best-case hybrid across raw/belief (diagnostic, NOT an independently preregistered model choice) must beat best-case Transformer by >=.015 mean reward and >=2/3 seed wins, while achieving within .035 of selective known-law Bayes. Otherwise NO-GO. Report the model's probe/cost behavior and training CPU times, no FLOP parity or generic intelligence claim.
4. **Classical ambiguity:** no model, including a hand-coded fault posterior, can exceed the 50% forced no-probe accuracy bound on these twin worlds; only actual independent purchased information can help. If labels leak or posterior violates pairing, stop.

A gate pass authorizes **future CPU-only study**, never paid GPU, scaling, main merge, automated tasks or SI breakthrough. Preserve all negative outcomes and require exact PR-head CI/review.
