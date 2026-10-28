# H2 v3 proposed experimental protocol

This document defines a proposed protocol. `FROZEN.json` is a local, time-stamped, hashed record,
not an externally registered preregistration or an immutable publication.
The numerical defaults are declared choices needing scientific justification before confirmation.

## 1. Source statement and operational scope

The user's working draft, *cl_field_draft (1).pdf*, Section 5, defines H2 (Rank inversion):
the ordering of task pairs by affinity changes during training; there exist pairs P,Q such that
`A_0(P) < A_0(Q)` and `A_T(P) > A_T(Q)`.

This package also follows the saved *H2_Complete_Methods_Assumptions_Record.md* and
*H2_H3_H4_Research_Log.md*, read for this delivery. They distinguish affinity reversal from
interference and record that the earlier H100 pilot did not establish H2.

Operational claim: under the named pretrained model, update rule, datasets, prompts, fixed probes,
layer, pooling, subspace rank and endpoints, training can reverse the ordering of measured task-pair affinities.
`t=0` is before this continual sequence, not before the pretrained model's original pretraining.
The sentence in the draft equating affinity ordering with which tasks actually conflict is stronger:
gradient/behavioral interference requires a separate direct measurement. This package does not prove it.

## 2. Primary geometry

For task i, form X_i^t with one fixed-probe example per row and hidden-feature coordinates per column.
The same examples, row order, labels, source groups, tokenizer and prompt construction are used at every checkpoint.
Pool only tokens wholly contained in source-content spans; exclude padding and direct instruction tokens.
Contextual effects of preceding instructions remain.

Center every feature column independently within a task. Let V_i^t be the top-r **right** singular vectors,
as d x r orthonormal columns. Define

`A_t(i,j) = ||(V_i^t).T V_j^t||_F^2 / r`.

This is the mean squared principal-angle cosine, in [0,1]. It is symmetric and not signed conflict.
It removes task mean shifts and singular-value magnitudes. High variance may reflect length, style or nuisance factors.
Common orthogonal coordinate changes preserve it; arbitrary compensated anisotropic rescaling need not.

Primary rank r=16; content pooling; Probe B; the intermediate hidden-state index listed per model:

| Model | Saved early / primary / late indices |
|---|---|
| Qwen3 1.7B | 7 / 14 / 21 |
| Qwen3 4B | 9 / 18 / 27 |
| Qwen3 8B | 9 / 18 / 27 |
| Qwen3 14B | 10 / 20 / 30 |
| SmolLM2 1.7B | 6 / 12 / 18 |

Index 0 is embedding output; hidden-state index k is after k decoder blocks, subject to model-final normalization conventions.
These intermediate choices avoid mixing the final normalized state with an intermediate state.

## 3. Data populations and contamination limits

Tasks: Yelp Polarity sentiment, QQP paraphrase, QNLI question-answer entailment,
MNLI three-way inference, AG News topic classification. Source label mappings are checked.
Only labeled source TRAIN rows are used; this is custom holdout evaluation, not official benchmark-test scoring.

Full target per task: 20,000 train, 1,000 validation, 1,000 Probe A, 1,000 Probe B, 2,000 test.
Five tasks give 100,000 selected training rows per data sample and 125,000 total selected rows if quotas fill.
Class quotas are balanced. Group allocation can exclude examples and change natural prevalence.

NFKC/lowercase/whitespace normalization; duplicate records removed and exact label conflicts excluded;
exact shared fields linked across tasks. Approximate 64-signature MinHash-LSH with verified word-5-gram
Jaccard >=0.85 supplies further grouping. It is not exhaustive semantic deduplication.
Detected components >128 rows are excluded, potentially changing the population.

All normalized fields must hash below 40/100 for development or >=40/100 for confirmation;
mixed-pool multi-field rows are excluded. This separates exact fields from development,
not every semantic neighbor, and does not retroactively remove exposure in the old pilot or pretraining.

For a fixed source pool, group split role is a stable seeded hash: 70% train, 5% validation,
8% Probe A, 8% Probe B, 9% test. Sample seeds change quota selection within those roles.
The fixed candidate seed/cap and global group construction are shared across repeated cohorts.
Freeze audits reject an exact field changing roles and report same-role overlap across selected cohorts.
The two samples are **overlapping subsampling replications**, not independent populations or doubled sample size.

Quotas fail closed. There is no duplicate-to-fill, automatic downsizing, cross-split reuse or replacement-data fallback.
If a requested quota is infeasible, revise a new plan before outcome inspection and document the changed population.

## 4. Learning and repeated runs

One cumulative LoRA adapter, updated throughout the sequence; no per-task adapter switching or replay.
Q/K/V/O attention projections; rank16, alpha32, dropout0.05. Base BF16; adapter/Adam state FP32.
Zero-initialized B makes the initial effective update exactly zero, apart from forward numerical effects.
This is a custom LoRA checkpoint format, not a PEFT `adapter_model.safetensors` artifact.

Default: two epochs/task; AdamW lr1e-4, weight decay0.01, cosine per-task schedule, 5% warmup,
clip1, microbatch1, accumulation16, length512, eval batch1; non-reentrant gradient checkpointing.
All actual examples/tokens/updates/losses/LRs/gradient norms/timings are logged.
Optimizer resets at each task boundary; learned parameters persist. Equal epochs do not equalize tokens or difficulty.

Primary: 4B and 8B x two data samples x ten training seeds x five fixed orders =200 trajectories.
The same seed across orders creates a block, not independent experimental observations.
Five orders are declared permutations, not a random sample of all 120 orders or an adversarial optimized order.

Stress: reference plus lr3e-5/3e-4, ranks8/32 (alpha=2r), dropout0/.1, lengths256/1024,
epochs1/3, each with three seeds and three orders. This is a one-factor design around a reference;
it does not estimate every interaction as a full factorial experiment would.
The full JSON plan and config for every cell are saved before submission.

Separate calibration plans use development data at the exact model/adapter/length/microbatch/LR/epoch shape,
with smaller datasets and a step cap. Memory floors are conservative planning guards, not fit guarantees.
For each target shape, at least two tasks must improve validation target NLL by >0.01 from immediately
before to immediately after their training, and reach valid-code accuracy>=chance+0.05 in calibration.
Calibration controls must pass. A positive inversion is NEVER a calibration acceptance requirement.

Optional full tuning uses Qwen3 1.7B FP32 master weights/Adam states with BF16 CUDA autocast,
the same task-reset policy and large checkpoint/storage requirements. It is not equivalent to LoRA,
and the default LR may require development-only learning calibration. No automatic tuning follows a failed confirmation.
Neither mode claims to instantiate the draft's exact theoretical lazy/rich output-scale interpolation.

## 5. Controls and integrity

Each cell contains one repeated-extraction frozen-model control and one trained-head/frozen-encoder control.
The latter is a shared four-output linear readout, not an LM-vocabulary-matched baseline.
Both must retain fixed-probe encoder features within absolute coordinate tolerance1e-4 across every saved view/stage.
The head must actually change its parameters. Controls at one seed/order do not certify every hardware condition.

Validation checks source/config/asset/token identities and completed-stage hashes. Source versions and
model/dataset commits are pinned at preparation/freeze. No remote custom model code is enabled.
Deterministic PyTorch algorithms are requested; unsupported kernels can fail rather than quietly vary.
Numerical reproducibility across different devices/releases is still not guaranteed.

## 6. Primary decision and multiplicity

Five tasks produce 10 task pairs and 45 pair-of-pair comparisons.
For P,Q define `D_t=A_t(P)-A_t(Q)`. Baseline/final are the primary endpoints.
Raw inversion: `D_0 * D_T < 0`.

Source-group bootstrap resamples global groups, sharing each multiplicity across tasks, endpoints and
all runs with the same probes. Centered subspaces are refit for every draw; no selective retries of
rank-deficient draws. Such failure is an instrument failure, not a null scientific result.
Full defaults use2,000 draws. Smoke40/calibration100 are execution/development budgets only.

Within a cell, approximate maximum-standardized-error bootstrap intervals jointly cover all
registered runs x two endpoints x45 comparisons, including controls. Across all cells of a fixed
suite, use Bonferroni alpha=0.05/number_of_registered_cells. Dependence across cells is not assumed away.
The quantile uses the higher empirical order statistic. Intervals remain approximate;
this is not an exact finite-sample FWER theorem or a bootstrap-t with resampled standard errors.

A robust interval reversal requires either

`lower(D_0)>0.01 AND upper(D_T)<-0.01`, or the reverse.

The0.01 is an affinity-difference margin, not1% accuracy, and is a proposed practical threshold.
The code reports both raw and interval decisions. It also records relative cutoff gaps
`(s_r-s_(r+1))/s_r`. A comparison is eligible for the declared stability summary only when
all involved task/endpoints have gap>=0.001 and the run passes the declared two-task NLL/competence learning check.
Raw and interval-supported results remain visible when those eligibility checks fail.

Primary plans are interpreted only when every registered run and both controls per cell are complete.
Failed/missing jobs are listed and prevent a complete confirmatory result. There is no winner-only report.
Processing one cell does not narrow the Bonferroni family.

## 7. Repetition and sensitivity

Per comparison and orientation, the declared descriptive repetition criterion is interval-eligible
reversal in >=80% of training seeds in >=min(3,number_of_planned_orders) fixed orders.
This is a descriptive consistency rule, not an independent population test or a p-value.
Seed blocks across the fixed orders are resampled for a descriptive mean-count interval.
Pair comparisons, checkpoints, cohorts and bootstrap draws are not treated as additional independent training runs.

Sensitivity CSVs cover ranks8/32, early/late layers, answer/last-content pooling, Probe A and
normalized covariance overlap in common feature coordinates. They are DESCRIPTIVE, without
confirmatory intervals, and cannot be used to select a favorable result and reuse the primary confidence claim.
Covariance overlap is `tr(C_i C_j)/(||C_i||_F ||C_j||_F)` and needs no row matching across tasks;
it is not naive CKA between unrelated examples. A changed estimator changes the operational H2 claim.

Auxiliary views are stored at baseline/final; only primary Probe B/content/primary-layer is
stored at intermediate task boundaries, reducing disk use. Transient crossings are descriptive;
there is no unrestricted time-search test in this version.

Separate primary/stress/replication/scale/full-tuning suites each have their own registered families.
Their0.05 statements cannot be pooled into a universal0.05 claim after searching all suites.
For one predeclared overarching family, allocate alpha across suites in code BEFORE building/freeze,
or collect new untouched confirmation following exploratory findings. Do not edit a frozen plan.

## 8. What conclusions are warranted

Positive: replicated measured affinity-rank reversals under the named conditions and assumptions.
Null/inconclusive: no eligible reversal at the chosen probes/rank/layer/endpoints/margin and precision.
Instrument failure: invalid controls, corrupted identities, rank failures or unstable measurements.
Execution failure: PBS/runtime/OOM/asset failure, not evidence about H2.

None proves universality, actual semantic reordering, signed gradient conflict, practical forgetting relevance,
H3 prediction, H4 mediation or literature novelty. More model size, data or repeats cannot remove every
assumption. A small number of clean examples can support existence; broad generalization needs
diverse independent designs and honest reporting of boundaries.
