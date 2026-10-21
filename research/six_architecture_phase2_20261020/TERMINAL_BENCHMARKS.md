# Terminal language and inference benchmarks

Training stopped by the prospective development gates; these terminal tasks did not select checkpoints.

| Model | Training exposures | Test NLL | PIQA primary | LAMBADA exact word |
|---|---:|---:|---:|---:|
| attention8 | 299,999,232 | 4.223946 | 55.50% | 7.51% |
| conv7 | 299,999,232 | 4.157327 | 55.06% | 6.79% |
| cpw_local | 99,999,744 | 5.051848 | 53.75% | 0.00% |
| transformer | 299,999,232 | 4.162973 | 55.93% | 8.21% |

PIQA: 1,838 validation questions. LAMBADA: 5,153 examples. Secondary PIQA scoring rules and per-example records are retained in BENCHMARKS.json.

| Matched candidate versus Transformer | PIQA difference (percentage points) | LAMBADA difference (percentage points) |
|---|---:|---:|
| attention8 | -0.435 [-2.067, +1.197] | -0.699 [-1.242, -0.136] |
| conv7 | -0.871 [-2.448, +0.871] | -1.417 [-1.979, -0.873] |

Intervals are descriptive paired example resampling, not independent-seed confirmation.

| Model | Context | Batch | Median prefill ms | p90 ms | Tokens/s | Peak allocated GiB |
|---|---:|---:|---:|---:|---:|---:|
| attention8 | 128 | 1 | 8.711 | 8.778 | 14,695 | 0.749 |
| attention8 | 128 | 8 | 9.964 | 10.045 | 102,771 | 0.920 |
| attention8 | 512 | 1 | 9.155 | 9.292 | 55,925 | 0.822 |
| attention8 | 512 | 8 | 32.331 | 33.782 | 126,690 | 1.504 |
| attention8 | 1024 | 1 | 10.132 | 11.098 | 101,062 | 0.920 |
| attention8 | 1024 | 8 | 64.973 | 67.868 | 126,083 | 2.284 |
| conv7 | 128 | 1 | 12.143 | 12.224 | 10,541 | 0.588 |
| conv7 | 128 | 8 | 13.163 | 15.177 | 77,792 | 0.758 |
| conv7 | 512 | 1 | 12.692 | 12.831 | 40,340 | 0.661 |
| conv7 | 512 | 8 | 36.169 | 38.112 | 113,247 | 1.343 |
| conv7 | 1024 | 1 | 13.177 | 13.376 | 77,712 | 0.758 |
| conv7 | 1024 | 8 | 75.533 | 79.922 | 108,456 | 2.122 |
| cpw_local | 128 | 1 | 9.004 | 9.051 | 14,215 | 0.589 |
| cpw_local | 128 | 8 | 10.515 | 12.191 | 97,383 | 0.761 |
| cpw_local | 512 | 1 | 9.121 | 9.176 | 56,137 | 0.662 |
| cpw_local | 512 | 8 | 31.696 | 33.389 | 129,229 | 1.347 |
| cpw_local | 1024 | 1 | 10.579 | 10.774 | 96,792 | 0.761 |
| cpw_local | 1024 | 8 | 63.742 | 65.636 | 128,519 | 2.132 |
| transformer | 128 | 1 | 13.445 | 17.103 | 9,520 | 0.768 |
| transformer | 128 | 8 | 14.500 | 14.795 | 70,619 | 0.944 |
| transformer | 512 | 1 | 13.531 | 13.597 | 37,839 | 0.840 |
| transformer | 512 | 8 | 32.770 | 33.369 | 124,990 | 1.521 |
| transformer | 1024 | 1 | 14.527 | 14.937 | 70,491 | 0.944 |
| transformer | 1024 | 8 | 71.319 | 75.239 | 114,865 | 2.300 |

| Model | Eight-token generation ms, prompt128/batch1/no KV cache | New tokens/s |
|---|---:|---:|
| attention8 | 66.119 | 120.99 |
| conv7 | 93.495 | 85.57 |
| cpw_local | 66.419 | 120.45 |
| transformer | 104.859 | 76.29 |

Conv7 gain erasure: replay NLL 4.131080; erased NLL 6.197049; change +2.065969.
CPW local: changing the distant prefix outside its active window gives exactly zero final-logit change.

One paired seed; example/block intervals do not measure training-seed uncertainty.
CPW stopped at 100M; other terminal models stopped at 300M. No paired CPW-versus-300M claim.
Parent has a separate conservative lost-exposure charge; matched committed tokens do not imply equal total compute.
LAMBADA uses teacher-forced all-target-token accuracy. PIQA primary scoring normalizes by target tokens.
No benchmark contamination audit or independent-seed replication.
Prefill returns dense vocabulary logits; generation is an eight-token implementation benchmark without KV caching.
Post-hoc Conv7 gain erasure measures trained-branch reliance, not the effect of training a model without that branch.
Terminal task benchmarks cannot overturn the predeclared development continuation gate.
