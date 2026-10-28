# What the H2 experiment code does

Prepared: 28 October 2026. Human dates use Asia/Calcutta.

This guide explains the v3 H2 pipeline and its compact-storage derivative: what
goes into the program, what it trains, what it measures, what the completed smoke
test reported, and what remains to be tested at full scale.

**Current scientific position:** the reported v3 smoke campaign completed and
reported **zero eligible rank reversals at its declared resolution**. The large
primary campaign is a planned experiment, not an observed result. Local tests
of the compact workflow are software evidence, not H100 execution or evidence
for H2.

## 1. Which code version this document describes

Several versions exist and must not be mixed together:

| Version | Location or identity | Scope |
|---|---|---|
| Legacy starter | [`legacy/h2_llm_testbed/`](../legacy/h2_llm_testbed/) in this repository | Earlier small testbed; its historical pilot used SST-2 as its first task |
| Version 2 | [`h2_full_experiment/`](../h2_full_experiment/) in this repository | Earlier measurement implementation and configuration profiles |
| Version 3 | Delivered as `H2_Robust_Experiment_v3.zip`; server campaign names start with `h2v3_` | Registered suite builder, repeated data cohorts, learning audits, joint analysis and staged execution |
| Compact derivative | Delivered as `H2_Compact_Workflow.zip`, containing `h2_compact_experiment/` | v3 plus verified analysis archives, laptop acknowledgement, controlled release and disk guards |

At the repository revision inspected for this documentation, the executable
source trees in GitHub are v2 and the legacy starter. This documentation commit
does **not** publish the later v3/compact source trees. File paths such as
`suite.py` and `h2/compact.py` below refer to those delivered later packages.
Their commands are not drop-in commands for the v2 checkout.

The compact package inspected for this guide has SHA256:

```text
4b9bb32331c073809f790e8b8b654a8b6e0bcefaead055679d635043838b852e
```

Its package verifier checked all 63 manifest-listed files. The GitHub revision
used to inspect the existing repository was
`eb99af1c1416b638f6078bc852ec8921721ddc10`.

Configuration values below describe the **suite-generated Qwen3-1.7B primary
plan** in the delivered code. For an actual server run, its own `run.json`,
resolved cell configuration, source hashes and freeze receipt are authoritative.
The current defaults do not reconstruct an older run's missing settings.

## 2. The question being tested

H2, as operationalized by this implementation, asks whether the ordering of
task-pair representation affinities can reverse during sequential learning.

An illustrative example, not an experimental reading:

| Task pair | Before sequential training | After sequential training |
|---|---:|---:|
| Yelp / AG News | 0.70 | 0.30 |
| QQP / QNLI | 0.40 | 0.65 |

The first pair starts above the second and finishes below it. The program asks
whether such a crossing is supported by its uncertainty procedure, whether the
model actually learned, whether the measured subspaces are sufficiently stable,
and whether the same oriented crossing recurs across seeds and orders.

These affinities are measurements of hidden-state geometry. They are not
ground-truth semantic task similarity or signed gradient conflict. A detected
crossing would not, by itself, establish why forgetting occurred, prove novelty,
or establish H3 or H4.

## 3. The model and trainable parameters

The current single-model primary plan selects
[`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B).

| Setting | Value |
|---|---|
| Model family | Pretrained causal language model |
| Nominal base size | Approximately 1.7 billion parameters |
| Hidden dimension | 2,048 |
| Base precision on GPU | BF16 |
| Training mode | Custom LoRA implementation |
| Adapter targets | Attention `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA scale | alpha / rank = 2 |
| LoRA dropout | 0.05 |
| Adapter parameter precision | FP32 |
| Saved representation precision | FP32 |

The base weights are frozen in LoRA mode. For a targeted linear transformation,
the model adds a trainable low-rank update:

$$
y = Wx + \frac{\alpha}{r}B A x.
$$

Here, `W` is the frozen pretrained matrix, `A` and `B` are adapter matrices and
`r` is the LoRA rank. The implementation applies dropout to the adapter input.
`B` begins at zero, so the initial adapter update is zero.

The base model is downloaded or reused once from its pinned local cache. LoRA
boundary checkpoints store trainable parameters, not another complete copy of
the 1.7B base model. These are custom checkpoints, not PEFT adapter-format files.

The registry also lists larger Qwen models and SmolLM2, but the explicit
`--models qwen1.7b` selection uses only Qwen3-1.7B. A default primary build without
that override selects different models and has a different budget.

## 4. Where the datasets come from

The program uses Hugging Face's `datasets.load_dataset` and the labelled source
training split of the following tasks:

| Task | Hugging Face repository | Configuration | Input | Labels |
|---|---|---|---|---|
| Yelp Polarity | [fancyzhx/yelp_polarity](https://huggingface.co/datasets/fancyzhx/yelp_polarity) | `plain_text` | Review text | Negative / positive |
| QQP | [nyu-mll/glue](https://huggingface.co/datasets/nyu-mll/glue) | `qqp` | Two questions | Different / same meaning |
| QNLI | [nyu-mll/glue](https://huggingface.co/datasets/nyu-mll/glue) | `qnli` | Question and sentence | Answers / does not answer |
| MNLI | [nyu-mll/glue](https://huggingface.co/datasets/nyu-mll/glue) | `mnli` | Premise and hypothesis | Entailment / neutral / contradiction |
| AG News | [fancyzhx/ag_news](https://huggingface.co/datasets/fancyzhx/ag_news) | Default | News text | World / sports / business / science and technology |

The task definitions are in `h2/tasks.py`. PAWS is also defined for other
configurations, but it is not one of the five tasks in this primary plan.

Source repository revisions are recorded in `assets/dataset_revisions.json`.
Prepared manifests record revisions, label ordering, candidate counts, selected
row identities, grouping audits, class counts and file hashes. Model metadata
similarly records the resolved model revision and cached-file hashes.

**The experiment's validation, probe and test splits are custom holdouts from
labelled source training data.** They are not the official benchmark test
partitions. Accuracy should not be presented as an official GLUE leaderboard
result.

### 4.1 Data sizes in the different profiles

All counts in the following table are **per task**, not per entire campaign:

| Setting | Smoke | Calibration | Qwen3-1.7B primary |
|---|---:|---:|---:|
| Training examples | 100 | 1,500 | 20,000 |
| Validation examples | 20 | 200 | 1,000 |
| Probe A examples | 40 | 200 | 1,000 |
| Probe B examples | 40 | 200 | 1,000 |
| Test examples | 40 | 200 | 2,000 |
| Configured epochs | 2 | 2 | 2 |
| Maximum optimizer updates per task | 5 | 100 | None |
| Maximum sequence length | 256 | 512 | 512 |
| Probe bootstrap draws | 40 | 100 | 2,000 |

The update caps can stop smoke/calibration training before two full passes.
Calibration evaluates validation during training; its prepared test split is
not included in the calibration profile's evaluation-split list.

For one primary data cohort, five tasks give 100,000 training examples and
125,000 examples across all five split roles. Two sampled cohorts may overlap;
their combined row counts are not a guaranteed number of unique examples.

The code loads source training datasets through the normal Datasets loader and
selects candidates with class-aware reservoir sampling. It is not an implemented
arbitrary-size streaming-training system. Prepared token packs are loaded into
CPU memory; larger future data targets need their own resource checks.

### 4.2 Cleaning and split protection

Preparation performs the following operations before training:

1. Check the source label ordering and reject invalid labels or empty input fields.
2. Assign normalized exact input fields to separate development or confirmation
   pools using a fixed SHA256-based rule. Mixed-partition multi-field rows are
   excluded.
3. Collect class-aware candidate reservoirs with a fixed candidate seed.
4. Remove exact duplicates and exclude conflicting-label duplicates.
5. Group shared input fields across tasks, then detect candidate near-duplicates
   with MinHash-LSH and verify candidate Jaccard overlap. Oversized connected
   groups are excluded.
6. Assign each group a stable split role, then select approximately class-balanced
   quotas for the requested cohort.
7. Audit split separation and preserve manifests and hashes.

Grouping prevents a repeated question, for example, from freely appearing as a
training item in one task and a held-out probe in another. Stable roles prevent
sampling a second cohort from silently changing a repeated group's split role.

MinHash-LSH is approximate. These checks do not establish exhaustive semantic
deduplication or absence of contamination from the model's pretraining corpus.
If requested grouped quotas cannot be filled, preparation raises an error
instead of silently shrinking the dataset or reusing held-out examples.

## 5. How examples become model inputs

`h2/prompts.py` constructs a task instruction, a label-code mapping and the input
fields. A simplified Yelp prompt is:

```text
Classify the sentiment of the review.
Labels: A = negative; B = positive
text:
<review content>
Answer:
```

The tokenizer must encode each of `A`, `B`, `C` and `D` as a distinct reversible
single token. The supervision target is one answer-code token. The code does
not train by treating every prompt token as a language-model target.

For LoRA, the loss uses the target answer code against the model's vocabulary
logits. Classification accuracy chooses among the codes valid for that task;
the unrestricted top-token invalid-code rate is recorded separately.

Tokenization preserves the instruction and `Answer:` suffix and truncates
content to the configured budget. Exact character offsets produce a content
mask, so content pooling excludes instruction and label-description tokens.
Token packs retain IDs, group IDs, labels, token IDs, offsets and content masks.
Left-padding is accompanied by explicit attention and position handling.

## 6. What the three execution modes do

| Mode | Parameters updated | Purpose |
|---|---|---|
| `frozen` | None | Detect spurious representation drift in a fixed encoder |
| `head` | A small FP32 linear readout; encoder stays fixed | Check that readout learning can occur while encoder geometry remains invariant |
| `lora` | Low-rank attention adapters | Measure geometry during sequential adaptation |

The head-only control uses pooled final-layer content features and its own
readout. It is an encoder-invariance control, not an identical-readout duplicate
of the LoRA language-model classifier.

Control checks compare stored representations across stages, verify probe
identities, enforce the configured feature-drift tolerance, and check that the
head actually changed. A fixed-encoder control with an interval-supported rank
reversal causes suite analysis to fail rather than being counted as H2 evidence.

## 7. Sequential training and repeated runs

For the first task order, the sequence is:

```text
Yelp -> QQP -> QNLI -> MNLI -> AG News
```

This is an order description; at each task boundary the program evaluates all
five tasks, not just the newly trained one.

The same LoRA adapter state is carried forward through the sequence. The
optimizer is reset for each new task; the learned adapter parameters are not
reset at that boundary. Each separate seed/order trajectory starts afresh
from the base model and its initialized adapter state.

| Main primary training setting | Value |
|---|---:|
| Microbatch | 1 example |
| Gradient accumulation | 16 microbatches |
| Effective batch | 16 examples per optimizer update |
| Epochs per task | 2 |
| Main learning rate | 0.0001 |
| Weight decay | 0.01 |
| Gradient norm clip | 1 |
| Warmup fraction | 0.05 |
| Schedule | Warmup followed by cosine decay |
| Gradient checkpointing | Enabled |

At 20,000 examples per task, there are 1,250 updates per epoch, 2,500 updates
per task and 12,500 updates per five-task LoRA trajectory. Across 100 primary
LoRA trajectories, the budget is **1,250,000 main optimizer updates**. These
counts exclude the separately trained head controls.

The five orders are:

| Order | Sequence |
|---|---|
| O1 | Yelp, QQP, QNLI, MNLI, AG News |
| O2 | AG News, MNLI, QNLI, QQP, Yelp |
| O3 | QNLI, Yelp, AG News, QQP, MNLI |
| O4 | QQP, MNLI, Yelp, AG News, QNLI |
| O5 | MNLI, AG News, QQP, QNLI, Yelp |

The ten training seeds are 101, 202, 303, 404, 505, 606, 707, 808, 909 and 1010.
The primary cohort sampling seeds are 20261031 and 20261107; these integers are
random-seed identifiers, not job execution dates.

For one model:

$$
2\text{ cohorts}\times10\text{ seeds}\times5\text{ orders}
=100\text{ LoRA trajectories}.
$$

Each cohort also has one frozen and one head-only control. The total is
**104 registered runs**. More seeds and orders assess repeatability; they do
not turn dependent measurements into independent datasets or prove a universal
hypothesis.

## 8. What is saved at each stage

There are six stage boundaries: stage 0 before any task training, followed by
stages 1 through 5 after the successive tasks.

At each boundary, the program saves task metrics, training-event summaries,
trainable-parameter checkpoints, RNG state, selected representation arrays and
a hash manifest. Evaluations include validation/test accuracy, macro F1,
confusion counts, mean target negative log-likelihood and invalid unrestricted
answer-code rates, according to the profile's evaluation-split list.

The measurements use three pooling definitions:

| Pooling | Definition |
|---|---|
| `content` | Mean of hidden states belonging wholly to input-content spans |
| `answer` | Hidden state at the final prompt position used to predict the code |
| `last_content` | Hidden state at the last content token |

The suite-generated Qwen3-1.7B primary configuration extracts layers 7, 14 and
21, with layer 14 primary. The builder overrides the base `configs/full.json`;
reading that base file alone can therefore give different layer/count settings.
Hidden-state indexing is recorded in each stage manifest.

At stage 0 and stage 5, all combinations of two probes, three layers and three
poolings are extracted for all tasks. At stages 1 through 4, only Probe B,
layer 14, content pooling is retained for all tasks.

Within a run, probe identities, groups, labels and ordering stay fixed across
stages. Within a cohort, probes are shared across trajectories to support paired
analysis. They are not resampled independently at every checkpoint.

## 9. How geometry and rank reversals are calculated

### 9.1 Task subspaces and affinity

For task i at a stage t, the primary feature matrix has shape
`1,000 examples x 2,048 features`. The analysis centers its columns and computes
the dominant feature-space singular vectors. The primary subspace rank is 16.
Let U_i(t) be its orthonormal 2,048-by-16 basis.

The affinity between tasks i and j is:

$$
S_t(i,j)=\frac{1}{r}\left\|U_i(t)^\top U_j(t)\right\|_F^2.
$$

It lies between zero and one. A larger value means greater overlap of the
measured subspaces. Tasks need not contain matched examples because their bases
are compared in the same model feature coordinates.

The implementation computes an extra singular value to diagnose separation at
the rank cutoff. It rejects an unsupported numerical rank. The iterative SVD
uses a fixed solver seed so solver randomness does not masquerade as training
drift.

### 9.2 Comparing task-pair rankings

Five tasks produce ten unordered task pairs. Selecting two of those pairs gives
45 pair-of-pair comparisons. For pairs p and q:

$$
D_t(p,q)=S_t(p)-S_t(q).
$$

A raw endpoint inversion occurs when D_0 and D_5 have opposite signs. Ties and
small uncertain crossings are not automatically eligible findings. Comparisons
share tasks, probes and fitted models and are dependent observations.

### 9.3 Paired group bootstrap and joint intervals

The primary analysis resamples source groups 2,000 times. A group's bootstrap
multiplicity is shared across tasks and endpoint checkpoints. Runs with the
same probe identities use aligned resampling draws.

Per-run analysis retains the point differences and complete bootstrap arrays.
The suite stacks them across all registered runs in a cell and computes an
approximate max-standardized-error interval family across runs, endpoints and
comparisons. The primary alpha 0.05 is split across registered cells; for two
cells the per-cell value is 0.025.

For a full cell, 52 runs x 2 endpoints x 45 comparisons gives 4,680 interval
coordinates in that family. This is why separate per-run confidence intervals
cannot substitute for the final joint calculation.

The method is approximate and conditional on trained checkpoints. It does not
provide guaranteed finite-sample family-wise error control, and its intervals
are not uncertainty over all possible training datasets or models.

### 9.4 Eligibility and descriptive repetition

An eligible LoRA reversal requires:

1. Opposite endpoint interval signs beyond the declared 0.01 minimum gap.
2. Relative cutoff gaps of at least 0.001 for the involved task subspaces at
   both endpoints.
3. A passing learning audit for that trajectory.
4. A training mode eligible as evidence; frozen/head controls are excluded from
   eligible reversal counts and audited separately.

The learning audit compares each task immediately before and after its own
training. In the primary settings, at least two tasks must improve validation
mean target NLL by more than 0.01 and finish at least 0.05 accuracy above uniform
class guessing. The code also records other changes; accuracy alone is not the
geometry test. A failed-learning completed run remains recorded and contributes
no eligible reversals; it is not silently replaced with a more favorable seed.

The declared descriptive repetition rule requires the same comparison and
initial orientation to be eligible in at least 80% of seeds for an order and in
at least three planned orders. For ten seeds and five orders, this means at
least **eight seeds in at least three orders**. This is a stability criterion,
not an additional p-value. The suite also reports intersections across observed
cohorts and a descriptive seed-block resampling summary.

### 9.5 Other measurements

The code records accuracy trajectories and backward transfer: final old-task
accuracy minus its accuracy immediately after that task was trained, averaged
over all but the last task. Negative backward transfer indicates deterioration
by that definition; it does not establish that a geometry inversion caused it.

Descriptive sensitivity views include rank 8, rank 32, earlier/later layers,
answer/last-content pooling, Probe A and covariance overlap. They do not replace
the registered primary analysis or acquire confirmatory status because one
happens to show a crossing.

## 10. Why the budget reaches 158.7 GiB

The figure is the uncompressed payload of saved feature arrays across 104 runs,
not the dataset download size, the base-model size, GPU memory or one run's
working-memory requirement.

One example's saved vector costs:

$$
2,048\text{ features}\times4\text{ bytes}=8,192\text{ bytes}.
$$

One matrix for 1,000 probe examples therefore costs 8,192,000 bytes, or 7.8125
MiB. The number of matrices saved per run is:

| Location | Calculation | Matrices |
|---|---|---:|
| Initial endpoint | 5 tasks x 2 probes x 3 layers x 3 poolings | 90 |
| Final endpoint | 5 tasks x 2 probes x 3 layers x 3 poolings | 90 |
| Four intermediate stages | 4 stages x 5 tasks x 1 primary view | 20 |
| Total | | 200 |

Thus:

$$
200\times1,000\times2,048\times4
=1,638,400,000\text{ bytes per run}.
$$

$$
\frac{104\times1,638,400,000}{2^{30}}
=\boxed{158.69140625\text{ GiB}}.
$$

That is 170.3936 decimal GB. It excludes metadata, data/model caches, token
packs, checkpoints, logs and analysis outputs. The program already writes
compressed `.npz` arrays, so actual disk occupancy may be smaller and must be
measured.

The extensive endpoint cross-product accounts for 142.82 GiB of this payload;
intermediate primary views account for 15.87 GiB. The complete cross-product is
broader than the minimum primary measurement and broader than the distinct
feature views needed by the named sensitivity estimators, although current
controls audit all saved views. The number is a consequence of this retention
design, not an inherent minimum for testing H2.

For comparison, retaining only the primary view at all six stages would cost
23.80 GiB of uncompressed features across the same runs. Simply switching to
that extraction design would omit secondary measurements/current all-view
audits and require a new declared protocol. Computing the analyses before
removing representations is the compact workflow's alternative.

For this rank-16 attention-only 1.7B configuration, the adapter tensor payload
across 600 LoRA boundary checkpoints is approximately 14.36 GiB. Primary
bootstrap draws across 104 runs occupy about 0.14 GiB before archive overhead.
The earlier 20–30 GiB compact laptop allowance includes additional inputs and
outputs but is still a provisional budget, not a measured full-campaign size.

## 11. What the compact workflow changes

The training data, LoRA optimization, primary geometry, bootstrap draws and
registered comparison family are intended to remain unchanged. The change is
when bulky files are processed, archived and released.

1. Train one registered run through completion; incomplete runs retain their
   resume state and are not packed as completed evidence.
2. A PBS CPU job verifies original stage hashes, evaluates controls/learning,
   computes the primary bootstrap and all declared descriptive sensitivity
   outputs, and seals a compact record.
3. A per-run ZIP stores checkpoints, exact bootstrap arrays, reports, metrics,
   source snapshots and provenance. A shared ZIP stores selected raw/tokenized
   inputs, their identities, pinned model metadata and the frozen plan.
4. The laptop client verifies the ZIP hash and every member, flushes the local
   archive, and sends an acknowledgement identifying the verified archive.
5. A separate PBS CPU release job requires acknowledgements for both shared and
   run archives, rechecks relevant hashes and removes only the inventoried
   representation/checkpoint files and acknowledged server ZIPs.
6. Retained statistical capsules on the server allow the final joint analysis
   to use exactly the stored paired bootstrap draws and audits.

The current compact archive **does not contain raw representation arrays**.
It contains the results of the declared analyses and boundary checkpoints.
New feature-level questions may require regeneration using the pinned base
model, tokenizer, probe data, configuration, software environment and checkpoint.
Bit-identical reproduction across hardware/software changes is not guaranteed.
Checksums detect changes; they cannot recover a deleted file or replace a backup.

Compact plans allow one GPU submission at a time and block the next completed-
run cycle until release. Disk guards check headroom before submission, feature
and checkpoint writes, and archive creation. They cannot reserve shared disk
space against other users, and preparation itself still needs working space.
The base model stays on the server; the laptop does not perform training.

Local fixture tests checked equality of bootstrap arrays, joint intervals,
comparison CSVs, control decisions, learning audits and sensitivity outputs
before and after release. Transfer tests simulated SSH/SCP; they were not live
network/H100 validation.

## 12. Integrity, resumability and cluster execution

| Mechanism | What it protects |
|---|---|
| Resolved cell configurations and plan identity | The declared run matrix and settings |
| Source inventory | Detecting code changes after plan construction |
| Preparation manifests | Identifying selected data, token packs and pinned assets |
| `FROZEN.json` | Binding the plan to prepared input manifests, calibration and analysis policy |
| Per-stage completion manifests | Detecting changes to completed stage artifacts |
| Atomic checkpoint writes | Avoiding replacement of a valid checkpoint with a partially written one |
| In-task progress files | Parameters, optimizer, RNG, counters and iteration cursor for resume |
| Boundary checkpoints | Parameters and RNG at completed task boundaries; optimizer resets for the next task |
| Submission/attempt/failure records | Preserving execution history instead of treating a retry as a new independent trajectory |
| Compact seals, archive hashes and acknowledgements | Detecting changed retained artifacts and controlling release |

Changing source after building a plan invalidates that plan's source identity.
Build a new plan after a code change. Editing a built/frozen plan by hand is not
an acceptable way to bypass this. `settings.env` is outside the source hash and
contains runtime environment/device settings; record the actual hardware and
environment used as well.

Preparation, training, representation extraction and substantial analysis run
through PBS on the documented cluster. `cpuq` handles CPU work; `workq` handles
GPU work. The compact package's suite/compact PBS scripts request 32 GB system
RAM. That RAM request is separate from GPU VRAM and persistent disk capacity.
The GPU check requires exactly one authorized visible CUDA device, selected by
PBS or the cluster's approved MIG procedure.

The primary freeze step requires successful matching development calibration,
including learning competence and fixed-encoder controls. A completed tiny smoke
test does not satisfy that requirement by itself. Neither a job ID nor a
`COMPLETE` marker alone establishes that H2 was supported.

## 13. What has actually been observed

The following v3 execution facts come from terminal output supplied in the
research conversation. This documentation work did not connect to the H100
server or independently retrieve its current run directories.

| Item | Evidence/state |
|---|---|
| Smoke campaign | `h2v3_smoke_r1` |
| Model/data cell | `qwen1.7b_d20261031_reference` |
| Run modes | Frozen, head and LoRA; one order and seed 17 |
| Smoke data | 100 training examples/task, 40 per probe/task, 40 test examples/task |
| Plan freezing | Reported PBS exit status 0 |
| GPU execution | Completion was reported for all three smoke runs |
| Analysis job | `37980.master`, reported exit status 0 |
| Analysis completeness | `Complete registered suite: True`, `Analyzed cells: 1/1` |
| Eligible reversals summed across runs | **0** |
| Repeated comparisons | **0** |
| Report conclusion | `NO ELIGIBLE REVERSALS AT DECLARED RESOLUTION` |

This is completed functionality/development evidence. It supplies no eligible
H2 support at the smoke configuration's resolution. It is not a universal
disproof. The short summary does not establish whether no point estimates
crossed, or whether crossings failed interval, learning or cutoff checks.
That distinction requires the detailed comparison tables and audit records.

Separate software evidence in the delivered compact package records 41 local
tests: 39 passed and two backend tests were skipped because a matching
Transformers/Qwen3 runtime was unavailable locally. Those counts describe the
earlier package validation, not a new test execution performed while writing
this guide. Actual H100 validation of compact transfer/release remains pending.

The full primary plan's build/budget output reported two cells, 104 GPU runs,
100 LoRA trajectories and the 158.69140625 GiB uncompressed feature estimate.
Building and budgeting a plan do not execute it. Full primary completion has
not been demonstrated in the supplied evidence.

## 14. How to read the generated files

| Artifact | What it tells you |
|---|---|
| `plans/<plan>/plan.json` | Registered cells/runs, source identity and analysis scope |
| `plans/<plan>/cells/*.json` | Resolved settings; these override base-profile assumptions |
| `plans/<plan>/FROZEN.json` | Prepared inputs, calibration and frozen plan identity |
| `runs/.../run.json` | Actual run identity, configuration, software and provenance |
| `runs/.../training_events.jsonl` | Per-update training loss, rates, counters and gradient norm |
| `runs/.../stages/stage_*/metrics.json` | Task evaluation readings at a boundary |
| `runs/.../stages/stage_*/checkpoint.pt` | Boundary trainable parameters and RNG |
| `runs/.../stages/stage_*/features/` | Probe representations and example/group identities |
| `runs/.../stages/stage_*/COMPLETE.json` | Original stage hashes and completion metadata |
| `runs/.../analysis/.../bootstrap.npz` | Point differences and complete aligned bootstrap draws |
| `reports/<plan>/<cell>/joint_pair_comparisons.csv` | Raw differences, joint intervals and individual eligibility checks |
| `reports/<plan>/<cell>/cell_report.json` | Controls, learning audits, repetition summaries and conclusion |
| `reports/<plan>/summary.md` | Human-readable suite completeness and cell outcomes |
| `reports/<plan>/suite_report.json` | Structured aggregate across cells |
| `runs/.../COMPACT.json` | Sealed retained analysis, audit results and release inventory |
| `plans/<plan>/compact_records.json` | Archive identities, acknowledgements and release states |
| Laptop `.zip` and `.verified.json` | Archived inputs/checkpoints/results and verification receipt |

Read the detail tables with the summary. An eligible count is not an independent
sample count or a p-value. A zero count can reflect eligibility failures as well
as stable measured rankings. An incomplete suite is not a completed confirmatory
experiment.

## 15. Source-file map for the later package

| File | Main responsibility |
|---|---|
| `suite.py` | Build, prepare, calibrate/freeze, submit, budget and analyze registered suites |
| `h2/tasks.py` | Dataset repositories, fields, labels and task instructions |
| `h2/data.py`, `h2/robust_data.py` | Candidate selection, deduplication/grouping, stable split roles and audits |
| `h2/prepare.py` | Pin assets, prepare data/token packs and validate cached artifacts |
| `h2/prompts.py` | Prompts, label tokens, truncation, masks, packs and pooling |
| `h2/lora.py` | Custom low-rank layers and trainable-parameter checkpoint handling |
| `h2/engine.py` | Model construction, sequential training, evaluation and extraction |
| `h2/checkpoints.py` | Atomic writes, locks and cooperative pause handling |
| `h2/geometry.py` | Subspaces, affinity, bootstrap sampling and interval calculations |
| `h2/analyze.py` | Per-run feature/accuracy analysis and bootstrap arrays |
| `h2/suite_analysis.py` | Joint intervals, audits, eligibility, repetition and sensitivity summaries |
| `h2/storage.py` | Disk headroom checks |
| `compact.py`, `h2/compact.py` | Pack/status/acknowledge/release lifecycle and capsule verification |
| `archive_client.py` | Laptop download and verify-before-acknowledge flow |
| `restore_compact.py` | Restore verified retained artifacts and optionally checkpoints |
| `pbs/` | Scheduler entry points and environment activation |
| `START_COMPACT.md` | Version-specific Windows/server operating instructions |

## 16. What remains to be done

1. Validate the new compact package in the existing H100 environment, including
   actual download, acknowledgement, release and recovery on a small campaign.
2. Complete the matching development calibration without weakening thresholds
   to obtain a favorable outcome.
3. Prepare and freeze the full primary plan under its final source identity.
4. Measure one full-data trajectory's real runtime, peak RAM/VRAM and disk use
   before extrapolating the entire campaign.
5. Execute and preserve all registered runs, including null and failed-learning
   outcomes; keep failures visible and resume the correct trajectory.
6. Run the final joint analysis across the full registered family and inspect
   comparison-level eligibility and controls.

These steps produce evidence about this operational H2 measurement. Claims
about generality, novelty, causal forgetting mechanisms, H3 prediction or H4
interventions require additional work and appropriate independent evaluation.

## 17. Evidence and provenance for this explanation

This guide was checked against the verified compact package's `suite.py`,
configuration profiles, data/prompt/training/geometry/analysis modules, compact
archive lifecycle, packaged validation summary and package manifest. Dataset
and model links identify the exact repository names selected by that source.
The source repositories contain larger datasets; the table above describes the
experiment's selected custom partitions.

The existing GitHub README/tree and a v2 engine excerpt were inspected to avoid
mislabeling the published code as v3. The historical notebook in this repository
is an earlier record; this file does not silently rewrite its historical state.
The smoke analysis readings in section 13 are explicitly attributed to supplied
terminal evidence. No absent model revision, raw inversion count, eligibility
failure reason, timing, peak-memory reading or new scientific result has been
invented.
