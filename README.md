# Continual-learning geometry experiments

Research code and evidence for task-affinity rank inversion (H2), with H3/H4
research questions retained separately.

**Latest evidence:** the v3 GPU smoke campaign completed and reported **zero
eligible reversals at its declared resolution**. The full primary campaign has
been planned, not demonstrated complete. The compact workflow has passing local
fixture tests; H100 archive/acknowledgement/release validation is pending.

Start with the [testing index](docs/TESTING_INDEX.md) and
[27 October experiment record](docs/EXPERIMENT_RECORD_2026-10-27.md).

## Source versions

| Path | Purpose |
|---|---|
| [h2_compact_experiment/](h2_compact_experiment/) | Latest compact derivative: full analyses, verified archives, explicit release and disk guards |
| [h2_compact_experiment/START_COMPACT.md](h2_compact_experiment/START_COMPACT.md) | Windows/server workflow; adapt account paths and read PUBLICATION.md first |
| [h2_robust_experiment_v3/](h2_robust_experiment_v3/) | Original v3 delivery snapshot with suite builder, grouped data, repeated runs and joint analysis |
| [h2_full_experiment/](h2_full_experiment/) | Previous version-2 source retained for history |
| [legacy/h2_llm_testbed/](legacy/h2_llm_testbed/) | Earlier starter implementation |

The public v3/compact copies generalize account paths and device identifiers.
Each source folder has a regenerated SOURCE_MANIFEST.json and a PUBLICATION.md.
Original delivery manifests and byte hashes are preserved in
[provenance/h2_20261028/](provenance/h2_20261028/).

These are the delivered source snapshots, not recovered exact copies of the
manually patched H100 checkout. Build a **new** plan after customization; old
frozen source identities will not match changed code.

## Evidence and documentation

| Material | Location |
|---|---|
| Complete code explanation | [docs/H2_V3_CODE_EXPLAINED.md](docs/H2_V3_CODE_EXPLAINED.md) |
| Dated execution/development record | [docs/EXPERIMENT_RECORD_2026-10-27.md](docs/EXPERIMENT_RECORD_2026-10-27.md) |
| Preserved GPU console and parsed metrics | [results/h2v3_smoke_r1/](results/h2v3_smoke_r1/) |
| Local tests, including failed development attempts | [results/validation/](results/validation/) |
| v3 synthetic instrument diagnostic | [JSON](h2_robust_experiment_v3/validation/synthetic_instrument_diagnostic.json) |
| Historical research notebook | [docs/H2_H3_H4_Research_Log.md](docs/H2_H3_H4_Research_Log.md) |
| Earlier pilot/source provenance | [results/historical/](results/historical/) and [provenance/](provenance/) |

The GPU attachment contains completed frozen, head-only and LoRA smoke runs,
six stages each, with five tasks and validation/test metrics. The published
JSON and CSV retain 180 visible task/stage/split rows. The final LoRA mean test
accuracy is 51.0%; its derived backward transfer is -0.09375. These are small
functionality readings, not confirmation of H2.

The supplied CPU terminal summary reports zero eligible reversals and zero
repeated comparisons. Its transcription is labelled as such; native detailed
geometry reports were not received. Missing raw artifacts are listed in the
experiment record.

## Hypothesis status

| Hypothesis | Question | Status |
|---|---|---|
| H2 | Can task-pair representation-affinity rankings reverse during sequential learning? | Implemented; smoke has no eligible supporting reversals; full primary evidence pending |
| H3 | Does geometry and its trajectory predict future forgetting better than static geometry? | Research question; forward prediction study not implemented here |
| H4 | Can function-preserving geometry interventions change later forgetting? | Research question; intervention study not implemented here |

These observations establish neither a universal proof/disproof nor novelty or
a causal forgetting mechanism.

## Model, data and planned scale

The current single-model primary selection is Qwen/Qwen3-1.7B with BF16 frozen
base weights and FP32 rank-16 LoRA adapters on attention q/k/v/o projections.
The five tasks are Yelp Polarity, QQP, QNLI, MNLI and AG News from Hugging Face.
Custom grouped holdouts are selected from labelled source training splits.

For that plan: 20,000 training examples/task, two uncapped epochs, two sampled
cohorts, ten training seeds and five task orders. That is 100 LoRA trajectories
plus four controls. The 158.69140625 GiB estimate is the **uncompressed saved
feature payload** across 104 runs, excluding caches/checkpoints/other outputs.
It is not an already-produced result, a model size or a personal storage quota.
See the code guide for the exact calculation and caveats.

## Running the current workflow

On the server, work in the appropriate source folder and use the existing
working environment. Verify the public package before configuring a new plan:

```bash
cd cl-geometry-experiments/h2_compact_experiment
python verify_package.py
```

Read PUBLICATION.md and START_COMPACT.md. If you cloned this repository, the
source folder already exists: skip the guide's ZIP extraction steps. Customize
YOUR_HPC_USER/account paths before building a plan. In the Windows archive
client, pass --remote-python with the actual server environment's Python path.

Use the suite/compact PBS entry points for the documented 32 GB-per-job account.
Older v3 bootstrap/legacy scripts retain historical defaults, including some
64 GB requests and environment cloning; they are not the recommended launch path.
Do not create another environment or reinstall torch merely to publish or
inspect the code.

For the college workflow, preparation, training, feature extraction and
substantial analysis run through PBS. Planning/submission/status commands are
lightweight. Start with remote tests and a small engineering smoke cycle, then
matching development calibration. A new storage implementation must be tested on
the actual hardware before running the full primary matrix.

## Publication scope

The update preserves complete delivered source/test/documentation snapshots,
available original local test logs, the supplied GPU console attachment,
derived metrics, and a labelled transcription of the CPU analysis output.
It also preserves earlier repository history.

It does not claim to contain native H100 run configs, freeze receipts, complete
per-job logs, trained checkpoints, probe/feature arrays or detailed geometry
CSVs that were never received. Models, raw dataset caches, credentials and other
users' GPU process rows are outside this public commit. Checksums identify
original inputs and transformations; they are not external backups.
