# H2 continual-learning experiment, version 2

Start with **START_HERE.md** for your college's SSH/PBS workflow. Do not run training, data preparation, feature extraction, or bootstrap analysis directly on the login shell. Lightweight submission, status and preflight commands are designed for the login shell.

This package tests the working H2 statement: **the ordering of task-pair representation affinities can reverse after sequential learning**. It produces empirical evidence for a precisely defined measurement. It does not promise proof, causal mediation, predictive utility, or novelty.

## Contents

| File or folder | Purpose |
|---|---|
| `h2/prepare.py`, `data.py`, `prompts.py` | Pinned assets, grouped splits, pretokenization and exact content masks |
| `h2/engine.py`, `lora.py`, `gpu.py` | Sequential training, controls, task-boundary recovery and MIG checks |
| `h2/geometry.py`, `analyze.py`, `plotting.py` | Centered feature subspaces, affinities, intervals, figures and summaries |
| `configs/` | Smoke, calibration, full, second-model and PAWS profiles |
| `pbs/` | CPU setup/preparation/analysis and GPU execution jobs |
| `submit_campaign.py` | Dry run or serial PBS submission with a durable job ledger |
| `status.py`, `doctor.py` | Saved state and lightweight readiness/storage information |
| `tests/test_core.py` | Synthetic mathematical, data, pooling, training/control and recovery tests |
| `PROTOCOL.md` | Measurement definition, comparison family and interpretation rules |

## Primary data and run matrix

| Task | Source | Classes | Train | Validation | Probe A | Probe B | Test |
|---|---|---:|---:|---:|---:|---:|---:|
| Sentiment | `fancyzhx/yelp_polarity` | 2 | 10,000 | 1,000 | 1,000 | 1,000 | 2,000 |
| Paraphrase | `nyu-mll/glue`, QQP | 2 | 10,000 | 1,000 | 1,000 | 1,000 | 2,000 |
| Question relevance | `nyu-mll/glue`, QNLI | 2 | 10,000 | 1,000 | 1,000 | 1,000 | 2,000 |
| Inference | `nyu-mll/glue`, MNLI | 3 | 10,000 | 1,000 | 1,000 | 1,000 | 2,000 |
| Topic | `fancyzhx/ag_news` | 4 | 10,000 | 1,000 | 1,000 | 1,000 | 2,000 |

These are **custom class-balanced grouped holdouts from labeled source training splits**. They are not official GLUE test scores. Counts are as balanced as integer division allows. No training row is used as a probe or evaluation row within a prepared dataset.

Preparation first partitions normalized source fields deterministically: SHA256-first8-big-endian modulo100 gives development fields below40 and confirmation fields at least40. Every field of a multi-field example must belong to the chosen pool; mixed-pool records are excluded. This makes exact normalized fields disjoint between smoke/calibration and full confirmation, including fields shared across tasks. It changes the sampling population and reduces the usable corpus, especially for paired inputs. Near-duplicate or semantic independence across these pools is not guaranteed.

Preparation resolves and records dataset commit SHAs, verifies label order, samples per-class candidate reservoirs from the chosen pool, removes exact duplicate and conflicting-label text records, and builds source groups before splitting. Repeated normalized content fields are grouped globally across tasks. Long text also receives verified MinHash-LSH grouping using word 5-grams and Jaccard threshold 0.85. Very large components are excluded. Entire groups stay in one split. The final audit checks IDs, global group IDs and exact normalized fields across splits.

Near-duplicate discovery is approximate and can miss pairs; short text receives exact grouping only. The audit does not establish semantic independence. All links, exclusions, hashes and quotas are recorded. If grouping leaves insufficient unique rows, preparation fails rather than duplicating examples or silently shrinking the experiment.

The full matrix uses Qwen3-1.7B, three task orders, and seeds 101, 202, 303, 404, 505: **15 fresh LoRA runs**. Two controls use the first selected order/seed: unchanged encoder, and a trained readout with an unchanged encoder. Controls verify feature/analysis invariance; they do not estimate all possible sources of training variability. Total: **17 serial GPU jobs**.

## Model and training

The Qwen revision is pinned to the snapshot used in your successful pilot. CPU preparation downloads or reuses it and records an explicit local path. GPU jobs use that local path and offline mode. Do not solve a missing-cache error by enabling downloads inside training; rerun preparation and inspect its log.

Training predicts one next-token answer code, A/B/C/D, after a fixed task instruction and source text. The loss is full-vocabulary cross entropy for that answer token only. Evaluation reports accuracy restricted to the task's answer codes, macro F1, target NLL, a confusion matrix, and the rate at which unrestricted top-1 output is not a valid answer code. Restricted accuracy should be labeled as such in a presentation.

Full defaults: BF16 base model; FP32 LoRA parameters and optimizer state; attention q/k/v/o adapters with rank16, alpha32 and dropout0.05; learning rate1e-4; two epochs per task; maximum512 input tokens; microbatch2 and accumulation8. Gradient accumulation is normalized by the actual number of examples, including partial groups. AdamW is reset at each task boundary, with warmup/cosine scheduling per task. There is no replay. One adapter persists through the five tasks in a run; every independent run starts from the original base model and a fresh adapter.

LoRA is implemented explicitly in `h2/lora.py`. Checkpoints contain custom named trainable tensors and RNG states. **They are not PEFT adapter files.** Use this implementation to restore them. No base weights are copied into each checkpoint; keep the pinned model cache.

Frozen mode performs zero parameter updates. Head mode trains a four-output shared linear readout on final-layer content means while keeping every encoder parameter fixed. It keeps the encoder in evaluation mode during head training. The primary H2 measurement remains at the configured intermediate layer.

Content pooling uses only tokens fully contained inside source-text spans. It excludes instruction characters, field labels, padding and the `Answer:` suffix. The answer label is never appended to probe inputs. Pooling accumulates in FP32. Content token states can still reflect preceding instructions; alternate prompts are a separate sensitivity experiment.

## Profiles and phases

| Config | Purpose |
|---|---|
| `smoke.json` | Small functional check: 100 training rows/task, step cap20, 40 rows in each probe, bootstrap40 |
| `calibration.json` | Separate exploratory data: 2,000 training rows/task, cap200; validation and Probe A features only |
| `full.json` | Primary 10,000/task, 15 LoRA runs, Probe B endpoint analysis |
| `smollm2.json` | Optional second-model replication, separate campaign and model cache |
| `paws_replication.json` | Optional QQP-to-PAWS replication; all tasks use5,000 train/500 validation/500 per probe/1,000 test |

A step cap limits the scheduled epochs; it does not repeat a tiny dataset until the cap is reached. For example, 100 rows with effective batch16 and two epochs normally makes14 optimizer updates, below the smoke cap20.

Run calibration after the smoke test:

```bash
qsub -v H2_CONFIG=configs/calibration.json pbs/prepare.pbs
```

After preparation completes:

```bash
python submit_campaign.py --config configs/calibration.json --campaign calibration_h2 --orders O1 --seeds 17 --with-analysis --submit
```

Use validation behavior, training loss, invalid-output rate, stability and memory/timing logs to finalize the settings. Do not tune settings to maximize confirmation inversions. If you change the full protocol after viewing its confirmation results, label that analysis exploratory and use a new independent confirmation design.

Smoke and calibration share the development source pool and may overlap with each other. Full confirmation uses the disjoint exact-field pool described above. Split audits apply within each prepared profile. For complete isolation of near-duplicate/semantic groups across profiles, freeze external grouping/exclusion metadata before preparation; the deterministic field partition alone does not establish that stronger independence.

SmolLM2 preparation resolves its revision once and writes a model lock; retain that SHA before beginning a replication. Model-specific layers are already configured. Optional replications are analyzed as separate comparison families, and are not automatically added to the primary family.

## Analysis

For centered task feature matrices, extract top-r right singular vectors in feature coordinates. Affinity is the normalized squared overlap of their orthonormal bases. The primary setting is hidden-state index14, content mean, rank16, Probe B. `hidden_states[0]` is the embedding output; index14 is after14 decoder blocks, or zero-based block13. Model-final indices may include a final normalization.

Ten unordered task pairs give45 unordered pair-of-pair comparisons. The primary endpoints are the untrained baseline and the checkpoint after the fifth task. A robust reversal requires both endpoint differences to have opposite signs and both simultaneous intervals to lie beyond the configured minimum absolute gap0.01. Raw sign changes and interval-supported reversals are reported separately. Intermediates are plotted descriptively.

The bootstrap resamples **global source groups**, sharing group multiplicities across tasks, endpoints and runs. This preserves detected cross-task group dependence. Class counts and task sample sizes can vary across draws. It recomputes the feature subspaces for every draw. The max-standardized-error rule creates approximate simultaneous intervals; the campaign report adjusts the family jointly across all supplied runs, endpoints and45 comparisons. Full defaults use1,000 draws. These are approximate intervals conditional on the trained checkpoints, not a universal error guarantee.

A separate descriptive bootstrap resamples training seeds as blocks across the observed task orders. Five seeds provide limited precision. Task pairs and bootstrap draws are not treated as independent training replications. Control failures or missing runs block a confirmatory interpretation. Eigenvalue gaps and explained variance are retained because an unstable rank cutoff can generate unreliable apparent drift.

Figures are PNG and SVG: initial/final affinity heatmaps, selected difference trajectories and accuracy curves. The selected trajectory panels are descriptive selections from the same data. Campaign-level comparison CSVs contain the joint interval decisions used for primary reporting; per-run figure intervals adjust within one run only.

Analysis runs in `cpuq`, not on the login node. It can be expensive. It saves bootstrap progress every25 draws and reuses completed analysis outputs. Rerun the same analysis job to continue. Set CPU walltime according to measured progress and cluster limits. Preparation and analysis are not assumed to finish within the template's24-hour request.

Secondary analyses can be submitted explicitly, for example after the primary campaign finishes:

```bash
qsub -v H2_CAMPAIGN=full_h2,H2_RANK=8 pbs/analyze.pbs
qsub -v H2_CAMPAIGN=full_h2,H2_LAYER=6 pbs/analyze.pbs
qsub -v H2_CAMPAIGN=full_h2,H2_POOLING=answer pbs/analyze.pbs
qsub -v H2_CAMPAIGN=full_h2,H2_PROBE=probe_a pbs/analyze.pbs
```

Each run's analysis is stored separately by settings/hash. The campaign's `reports/full_h2` view is replaced with the latest requested analysis. **Download/archive the primary campaign report before running a sensitivity analysis.** Do not pool secondary decisions into the primary result without a declared combined comparison family.

## Outputs and documentation

Each `runs/<campaign>/<model>/<mode>_<order>_s<seed>/` contains:

- `run.json`: resolved configuration, software versions, PBS job/MIG provenance, source hashes and split/model metadata.
- `code/`: the exact training/analysis/PBS sources used when the run began.
- `training_events.jsonl`: losses, updates, learning rates, gradient norms and processed token/example counts.
- `stages/stage_000` through `stage_005`: metrics, trainable checkpoint/RNG, features and stage completion markers.
- `analysis/`: settings-specific reports, raw matrices/bootstrap draws, comparisons, accuracy tables and figures.
- `COMPLETE.json`: successful full-run completion, elapsed time and GPU memory statistics.

`campaigns/<name>.json` records the requested matrix and every accepted PBS job ID. `reports/<name>/summary.md`, `campaign_report.json`, `runs.csv` and `joint_pair_comparisons.csv` summarize the registered campaign. Jobs are recorded after each successful qsub response. If submission is interrupted or a response is unclear, inspect `qstat` and this ledger before retrying to avoid duplicate work.

Present the question, exact definition, data/grouping, run matrix, controls, primary report, seed stability, sensitivity results, and limitations. Save the entire run folder, model/dataset locks and protocol alongside the figures. A positive statement should name the model, datasets, metric, rank, layer, seeds/orders and interval rule. If the result does not reproduce or intervals include ties, state that H2 was not supported clearly under this protocol.

## Recovery

`--resume` restores trainable tensors and Python/NumPy/Torch/CUDA RNG from the last fully completed task boundary. Optimizers reset per task, so no mid-task optimizer state is needed. An interrupted task is rerun from its beginning. Partial stage directories are archived under `aborted/`. No completed run is overwritten. Resume refuses changes to resolved settings, token-manifest identity, or source hashes.

After a failure, dependent PBS jobs may remain held or queued. Use `python status.py --campaign full_h2` and `qstat -u YOUR_HPC_USER`. Cancel only the pending jobs belonging to this campaign, using their actual ledger/qstat IDs; do not cancel an actively progressing job just to rerun submission. Clear the obsolete analysis job too. Correct the allocation/environment problem without changing the scientific code/config, then:

```bash
python submit_campaign.py --config configs/full.json --campaign full_h2 --with-analysis --resume --submit
```

Completed runs are skipped; incomplete runs resume from their last task boundary. If a code or scientific configuration fix is needed, create a new campaign and preserve the old evidence. An old `FAILURE.json` remains as history after successful recovery; `COMPLETE.json` takes precedence in status.

To repeat only a timed-out analysis, after GPU runs have all completed:

```bash
qsub -v H2_CAMPAIGN=full_h2 pbs/analyze.pbs
```

## Resources and limits

Your earlier GPU was an H100 **3g.40gb MIG instance**, not exclusive use of the whole physical GPU. These templates use `workq`,4 CPU cores,32GB system RAM and a selected MIG; CPU jobs use `cpuq` with at most8 cores. They follow the prior college setup and need a server smoke test before a campaign. `Host: master` can be legitimate in a PBS job on this installation; a PBS allocation and the selected device matter, not the hostname alone.

The full configuration stores about2.75GiB of uncompressed float32 features per run, about46.7GiB for17 runs. Compression varies. Models, dataset caches, token caches, checkpoints and aborted stages are additional. Check both free space and your account quota with your college's tools. Run `python doctor.py --config configs/full.json` for a formula-based estimate. The code never automatically deletes model caches, data or results.

MIG occupancy checks are non-atomic and do not implement a reservation service. Follow Team HPC's allocation rules. This code does not select a stale UUID automatically, claim unused physical GPUs, or change permissions. If queue names, PBS resource syntax, GPU allocation policy or installed libraries differ, inspect the failed smoke log and adapt the templates before scaling up.

## Development tests

Run the included tests in a **CPU PBS job**, as `pbs/setup.pbs` does:

```bash
python -m unittest discover -s tests -v
```

They test feature-coordinate geometry, a constructed reversal and tie threshold, shared bootstrap groups, split leakage, content masks, padding, LoRA updates/base invariance, all training stages, both controls, checkpoint recovery and analysis. They use a tiny causal fixture and no Hugging Face downloads. They do not verify Qwen/SmolLM2 kernels, dataset availability, PBS policy, H100 memory or training throughput. Those require the server smoke run.

Dependencies are pinned to the earlier reported server versions where available. Keep the existing CUDA Torch build; do not replace it with a CPU build. `logs/installed-packages.txt` records the resolved environment. Dataset and model redistribution rights remain with their respective owners; this archive contains code only.
