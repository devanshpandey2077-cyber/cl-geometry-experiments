# Experiment and engineering record: 27 October 2026

Published: 28 October 2026 (Asia/Calcutta). This records the supplied 27 October
HPC work and the compact-storage development discussed with it. The v3
engineering/statistical diagnostics dated 26 October are also preserved as
precursor evidence; they are not relabelled as new H100 experiments.

## Current scientific conclusion

The completed v3 smoke campaign `h2v3_smoke_r1` reported **zero eligible
reversals at the declared resolution**. It is a small functionality/development
experiment. The full 104-run primary plan has been built and budgeted, but its
execution/completion has not been demonstrated. The compact workflow passed
local fixture tests; actual H100 compact archive/acknowledgement/release remains
unverified.

## Evidence levels

- **Preserved console attachment:** the supplied GPU console stream has been
  recovered and inspected. Its public copy retains metrics and job identities
  with account paths/device UUIDs generalized.
- **Derived from console:** JSON/CSV metrics and the accuracy/BWT summary were
  parsed from visible stage dictionaries and checked against confusion counts.
- **Transcribed terminal report:** the CPU analysis status, analysis lines and
  summary were pasted in the conversation. Their normalized transcription is
  preserved; the original native server report bytes were not received.
- **Preserved local logs:** original available unit-test and instrument-diagnostic
  logs are published with path substitutions and LF normalization.
- **Planned:** configuration previews, storage estimates and suggested commands
  describe intended work; they are not executed experiments.

## What happened

| Step | Evidence and outcome |
|---|---|
| Preparation/setup | Preparation was reported successful after environment/storage and 64 GB versus 32 GB RAM-request problems. Its exact native prepare log is not available in this publication. |
| Source identity | A new smoke plan was used after earlier source edits invalidated a built plan. Do not treat the original delivered ZIP as a recovered copy of the exact manually patched server checkout. |
| Freeze | Job `37958.master` was reported finished with exit status 0. |
| First GPU attempt | Jobs `37959.master`â€“`37961.master` encountered the single-authorized-device check before model training. The researcher cancelled active jobs. |
| GPU selection | Runtime MIG selection was set in `settings.env`, outside the source inventory. A later CUDA check passed. This historical device choice is not a current allocation. |
| Successful frozen control | Console job `37969.master`, run `frozen_O1_s17`, stages 0â€“5 and completion marker preserved. |
| Successful head-only control | Console job `37970.master`, run `head_O1_s17`, stages 0â€“5 and completion marker preserved. |
| Successful LoRA smoke | Console job `37971.master`, run `lora_O1_s17`, stages 0â€“5 and completion marker preserved. |
| CPU analysis | Job `37980.master` was reported finished with exit status 0; the pasted log includes suite analysis completion. |
| Geometry summary | One analyzed cell, complete smoke suite, zero eligible reversals and zero repeated comparisons. |
| Larger calibration planning | `h2v3_cal_qwen17` was built with 1,500 training examples/task, three runs and a 100-update/task cap. Successful calibration execution was not established by supplied evidence. |
| Full-data planning | `h2v3_primary_qwen17` was built on the server: 20,000 training examples/task, two uncapped epochs, two cells, 104 GPU runs and 1,250,000 main optimizer updates. |
| Storage constraint | Last supplied shared-filesystem reading was 7.5 GiB free; the account folder used approximately 21 GiB. These are historical readings, not a quota or a live space guarantee. |
| Compact development | A separate derivative was implemented to analyze, archive, verify and explicitly release one completed run at a time without replacing joint analysis with separate per-run conclusions. |
| Compact validation | Final preserved local log: 41 tests discovered, 39 passed, two backend tests skipped. An earlier six-error development-fixture run and an eight-test passing focused run are retained too. |

The GPU failure message, copied from supplied terminal evidence, was:

```text
RuntimeError: Exactly one scheduler/administrator-authorized CUDA device is required
```

It is a setup failure, not a negative H2 experiment. The public original-v3
snapshot retains historical defaults, including some 64 GB PBS requests and an
environment-cloning bootstrap. Read its PUBLICATION.md and use the compact
suite's 32 GB entry points with the working environment.

## Successful GPU smoke configuration and hardware

Campaign: `h2v3_smoke_r1`. Cell: `qwen1.7b_d20261031_reference`.
Model: Qwen3-1.7B. Task order O1: Yelp, QQP, QNLI, MNLI, AG News.
Training seed: 17. The discussed smoke selection was 100 training examples/task,
40 examples per probe/task, 20 validation and 40 test examples/task, with a
five-update/task smoke cap.

The archived console directly confirms 20-example validation and 40-example test
metrics, the modes/order/seed in run IDs, and the CUDA checks. The exact resolved
run configuration has not been recovered; profile defaults are not a substitute
for it.

The CUDA check in all three successful console sections reports:

| Field | Preserved value |
|---|---|
| Device | NVIDIA H100 NVL MIG 1g.24gb |
| Reported memory | 23,219,666,944 bytes = 21.625 GiB |
| Torch | 2.14.1+cu130 |
| Torch CUDA build | 13.0 |
| CUDA operation | GPU COMPUTATION PASSED |

Device UUIDs are redacted. GPU process tables referring to other users are
omitted from the public diagnostics; hardware/MIG usage tables and original
timestamps remain. Hashes identify the original private attachments but do not
provide a backup of them.

## Final classification readings

Each entry below uses the final stage and the custom 40-example test split.
Classification is restricted to the task's valid answer codes.

| Task | Frozen | Head-only | LoRA |
|---|---:|---:|---:|
| Yelp | 50.0% | 52.5% | 55.0% |
| QQP | 57.5% | 35.0% | 55.0% |
| QNLI | 50.0% | 55.0% | 57.5% |
| MNLI | 37.5% | 35.0% | 50.0% |
| AG News | 35.0% | 20.0% | 37.5% |
| Unweighted mean | 46.0% | 39.5% | 51.0% |

The LoRA backward-transfer value derived from all six visible stage readings is
**-0.09375** (minus 9.375 percentage points), averaging final minus immediately
post-own-training accuracy for the first four tasks. Frozen BWT is zero and the
head-control mean difference is numerically approximately zero. The raw derived
JSON preserves floating-point values rather than silently rounding them.

The head-only readout differs from the language-model answer-code readout.
These small-sample control accuracy numbers are not an identical-classifier
comparison, and none of the accuracy/BWT readings establish H2 or a causal link
between geometry and forgetting.

All 180 visible task/stage/split metric rows, including NLL and F1, are in
[the CSV](../results/h2v3_smoke_r1/stage_metrics_from_console.csv).
[The JSON](../results/h2v3_smoke_r1/stage_metrics_from_console.json) also retains
confusion matrices, printed training events, hardware readings and console
line references. [The derived summary](../results/h2v3_smoke_r1/derived_accuracy_summary.json)
records the formulas and source scope.

## Geometry outcome and interpretation

The supplied CPU report said:

```text
Complete registered suite: True
Analyzed cells: 1/1
Eligible reversals summed across runs: 0
Repeated comparisons: 0
NO ELIGIBLE REVERSALS AT DECLARED RESOLUTION
```

See the explicitly labelled
[terminal transcription](../results/h2v3_smoke_r1/reported_analysis_transcript.txt).
This summary does not distinguish stable point rankings from crossings that
failed uncertainty, learning or spectral-quality gates. The native
`cell_report.json`, `joint_pair_comparisons.csv` and underlying arrays were not
received, so the failure reasons and raw inversion counts remain unknown.

A complete small functionality suite is not the full primary study and does
not prove or universally disprove H2.

## Local software and instrument checks

| Check | Preserved outcome | Scope |
|---|---|---|
| v3 CPU suite | 32 discovered; 30 passed, two backend tests skipped | Engineering regression evidence |
| Stable-population synthetic diagnostic | 0 detections in 30 repetitions | Gaussian instrument behavior only |
| Planted-reversal synthetic diagnostic | 30 detections in 30 repetitions | Gaussian instrument behavior only |
| Synthetic simultaneous coverage | Stable 30/30; planted 29/30 | Small-simulation diagnostic; not a coverage guarantee |
| Compact development attempt | 38 discovered; six errors, two skips | Fixture setup bug; preserved failure |
| Compact focused validation | Eight passed | Lifecycle checks at that development stage |
| Compact final suite | 41 discovered; 39 passed, two skips | Local fixtures; SSH/SCP simulated |

The synthetic diagnostic used n=80 and 199 bootstrap draws; its embedded date is
26 October 2026. The packaged JSON includes Wilson uncertainty intervals.
It is not real-model H2 evidence, transformer power calibration or proof of
valid error control.

The failed compact fixture expanded Probe A requirements before all fixture runs
had been constructed. The final test code creates the runs before expanding the
synthetic feature views. The final passing suite checks equality of joint
results before and after release, corrupted-artifact refusal, both archive
acknowledgements, retry behavior, transfer verification and restoration. Local
Torch/NumPy compatibility warnings remain in the logs; no package installation
was changed for this publication. Native local Bash validation was unavailable;
the remote test job includes shell syntax checks.

See [the local validation index](../results/validation/README.md) for all logs.

## Full primary plan and storage arithmetic

The current Qwen3-1.7B primary plan uses two sampled data cohorts, ten training
seeds and five orders: 100 LoRA trajectories plus four controls. Each task has
20,000 training examples and two complete epochs; the step cap is null.
Its two data cohorts may overlap.

Each run saves 200 matrices of shape 1,000 x 2,048 in FP32 before compression:

- Initial/final endpoints: 2 x 5 tasks x 2 probes x 3 layers x 3 poolings = 180.
- Four intermediate boundaries: 4 x 5 tasks x one primary view = 20.

The feature payload across 104 runs is:

```text
104 Ã— 200 Ã— 1000 Ã— 2048 Ã— 4 / 2^30 = 158.69140625 GiB
```

It excludes model/data caches, checkpoints, metadata and logs; actual NPZ disk
occupancy is compressed and must be measured. A preview is not 159 GiB of
already-generated results. The compact package's local planning output is
[here](../h2_compact_experiment/validation/planning.json).

## Compact workflow and tradeoffs

The intended cycle is one registered GPU run, complete CPU analyses/audits,
verified laptop archive, acknowledgement, explicit CPU release, then the next
run. Exact paired bootstrap arrays and audit outputs remain available for the
joint calculation. Training data sizes and the registered statistical family
are not reduced merely to save disk.

The current compact archive stores checkpoint parameters and analysis outputs
but excludes raw representations. Later feature-level questions may require
regeneration. A checksum detects a change; it does not recover a deleted file.
Working space is still required for preparation, a current run and temporary
archives. The 20â€“30 GiB laptop estimate is provisional.

## Missing original artifacts

The following were not received and are not implied by this publication:

- Native run.json files, executed resolved cell configs and FROZEN.json receipts.
- Exact manually patched source inventory for the successful server campaign.
- Original per-job GPU logs beyond the preserved combined console attachment.
- Native detailed geometry reports, joint comparisons, bootstrap arrays and probes.
- Trained H100 adapter checkpoints and raw/tokenized experimental data.
- Successful execution evidence for the larger calibration or full primary study.
- A completed H100 test of compact archive download, acknowledgement and release.

Recover these files from the server before claiming complete artifact-level
reproduction. The published executable source snapshots, summaries and checksums
do not replace those missing bytes. No new training, PBS submission or H2
analysis was performed during this GitHub publication.
