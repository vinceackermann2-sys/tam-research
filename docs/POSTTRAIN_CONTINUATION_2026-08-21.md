# TAM-v3-25M Post-training Continuation — 2026-08-21

This continuation resumes from the completed 300M-token TAM-v3-25M pretrained checkpoint produced by issue #73. It does **not** retrain the base model.

## Frozen continuation

- Base checkpoint: `/vol/full-model-runs/pretrain/25m/ctx512-mb64-ga2/tamv3-25m-compiled-seed7400/latest.pt`
- SFT data: `/vol/data/ultrafeedback-gpt2-full25m`
- DPO data: same UltraFeedback-derived arrays
- Final run directory: `/vol/full-model-runs/TAM-v3-25M-Full-seed7400`
- Seed: 7400
- SFT: micro 64, grad accumulation 2
- DPO: micro 32, grad accumulation 2; same 64 preference pairs per optimizer update as the original configuration
- Final eval: FineWeb-Edu held-out language loss + fixed prompt generations

## Budget / safety

- Hard H100 ceiling: 260 seconds
- Stage persistence: SFT commits before DPO; DPO commits before optional final eval/generation
- If budget becomes too tight after SFT, the SFT checkpoint is preserved and DPO is not started.
- If budget becomes too tight after DPO, the DPO checkpoint is preserved and optional final evaluation may be deferred.
- This continuation cannot invoke pretraining.

Implementation: `modal_posttrain_app.py` and `.github/workflows/modal-posttrain.yml`.
