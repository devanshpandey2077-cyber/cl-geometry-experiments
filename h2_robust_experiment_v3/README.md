# H2 Robust Experiment v3

Prepared 26 October 2026 for the research programme *The Endogenous Geometry of Continual Learning*.

Read **FULL_PIPELINE.md** for the complete Windows -> SSH -> PBS -> results workflow.
Read **H2_CODE_EXPLAINED.md** for the full beginner-friendly explanation of what the code does,
why the measurements test H2, how the repeated runs and uncertainty checks work, and what each output means.
Read **PROTOCOL.md** for the exact scientific claim, estimators, uncertainty, controls and limits.
Read **VALIDATION.md** for what was actually tested before delivery.

This is executable research software, not a completed experiment or a promise of positive findings.
It extends the existing v2 package without modifying the original pilot or v2 files.

## What it implements

- Sequential, task-aware classification with one cumulative attention LoRA adapter.
- Qwen3 1.7B, 4B, 8B and optional 14B profiles; SmolLM2 1.7B model replication.
- Optional FP32-master full-parameter Qwen3 1.7B training with BF16 CUDA autocast.
- Multiple training seeds, fixed task orders, and repeated data subsampling.
- Eleven reference/one-factor parameter settings: learning rate, adapter rank, dropout, length, epochs.
- Frozen-encoder and trained-readout/frozen-encoder controls in every cell.
- Balanced grouped source-train holdouts, exact shared-field grouping and approximate near-duplicate grouping.
- Separate development/confirmation exact-field pools; fixed split ownership across data resampling.
- Source/data/model/token hashes and a protocol freeze receipt before main submissions.
- Full-vocabulary answer-code loss, valid-code accuracy, macro F1, NLL and invalid-output rate.
- Fixed probes at baseline and every task boundary; auxiliary geometry views at endpoints.
- Paired global-source-group bootstrap with joint intervals within cells and Bonferroni across registered cells.
- Spectral-cutoff diagnostics, learning checks, raw and interval-supported inversions, and complete seed/order tables.
- Atomic optimizer-step checkpoints and RNG/optimizer recovery; task-boundary checksums.
- PBS submission ledger, deterministic job names, serial dependency chains and bounded submission windows.
- Descriptive sensitivity to layer, rank, pooling, probe sample and covariance overlap.
- Synthetic instrument coverage/power diagnostics and CPU/real-backend tests.

## Profiles

| Profile | Models | Training seeds x fixed orders x data samples | Training trajectories | Controls | Role |
|---|---|---:|---:|---:|---|
| smoke | Qwen3 1.7B | 1 x 1 x 1 | 1 | 2 | Execution check |
| calibration --like primary | Qwen3 4B, 8B | 1 x 1 x 1 per model | 2 | 4 | Learning/resource calibration |
| core | Qwen3 4B | 5 x 3 x 1 | 15 | 2 | Smaller confirmation |
| primary | Qwen3 4B, 8B | 10 x 5 x 2 per model | 200 | 8 | Large repeated confirmation |
| stress | Qwen3 4B, 11 parameter settings | 3 x 3 x 1 per setting | 99 | 22 | Parameter robustness |
| replication | SmolLM2 1.7B | 10 x 5 x 1 | 50 | 2 | Different pretrained model |
| scale14b | Qwen3 14B | 5 x 3 x 1 | 15 | 2 | Larger model, authorized larger device |
| full_tuning | Qwen3 1.7B | 5 x 3 x 1 | 15 | 2 | Different update regime |

The primary profile selects 20,000 training examples/task, two epochs, five tasks.
Counts are requested quotas, not claims that preparation or training has executed.
Repeated samples can overlap. The code audits overlap and does not call them independent corpora.

## Important scope

H2 asks whether representation-affinity rankings can reverse. This package does not infer
signed gradient conflict, semantic similarity, H3 prediction, H4 causality or novelty from that observation.
LoRA alpha is the adapter scale, not the draft's theoretical lazy/rich output-scale parameter.
Parameter sweeps are not a complete implementation of every research phase.

Large runs can require hundreds of GiB of disk and long GPU/CPU allocations. Run `suite.py budget`
and measure calibration first. The earlier 40 GB MIG allocation is not an entire H100.
No multi-GPU sharding, 70B loading, quantized training, replay, EWC or H4 intervention is hidden in these profiles.

Legacy `submit_campaign.py`, `status.py`, and v2 configs remain for source continuity and regression tests.
Use **suite.py** for all v3 runs. Do not combine old submission commands with new plan directories.
