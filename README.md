# Continual-learning geometry experiments

Research code and an evolving empirical notebook for task-affinity rank inversion, forward prediction of forgetting, and function-preserving geometry interventions.

| Hypothesis | Question | Current implementation/evidence |
|---|---|---|
| H2 | Can the affinity ranking of task pairs reverse during sequential learning? | Version-2 measurement pipeline and historical small-pilot records; no confirmatory conclusion yet |
| H3 | Does current geometry and its trajectory predict future forgetting better than static geometry? | Registered research question; forward evaluation is not implemented in these packages |
| H4 | Can geometry changes preserving old-task function change subsequent forgetting? | Registered research question; intervention study is not implemented in these packages |

## Repository contents

| Path | Purpose |
|---|---|
| [h2_full_experiment/](h2_full_experiment/) | Main H2 implementation: LoRA, fixed probes, grouped splits, controls, uncertainty analysis, configuration profiles and PBS jobs |
| [h2_full_experiment/START_HERE.md](h2_full_experiment/START_HERE.md) | Cluster setup and small-pipeline instructions |
| [h2_full_experiment/PROTOCOL.md](h2_full_experiment/PROTOCOL.md) | Measurement definitions, comparison families and interpretation limits |
| [legacy/h2_llm_testbed/](legacy/h2_llm_testbed/) | Earlier starter implementation, retained for historical reproducibility |
| [docs/H2_H3_H4_Research_Log.md](docs/H2_H3_H4_Research_Log.md) | Detailed notebook, hypothesis registry, run templates and historical readings |
| [results/historical/h100_smoke_36125.master/](results/historical/h100_smoke_36125.master/) | Original small-pilot accuracy/configuration, saved source documentation and adapter metadata |
| [provenance/](provenance/) | Original/published source hashes, transformation notes and full local artifact inventory |

## Start here

Use the version-2 implementation for new H2 work. Read its [README](h2_full_experiment/README.md), [START_HERE](h2_full_experiment/START_HERE.md) and [protocol](h2_full_experiment/PROTOCOL.md) before launching jobs. Commands assume the working directory is h2_full_experiment. Customize the account, cache, Conda installation, PBS queues/resource requests and currently authorized GPU allocation for your cluster. YOUR_HPC_USER and YOUR_HPC_HOST are placeholders.

For the documented college workflow, setup, data preparation, training, feature extraction and substantial analysis run through PBS. The submission helper prints a plan unless --submit is supplied. Calibrate the environment and learning behavior before freezing any confirmation campaign. Dependency pins reproduce the supplied project configuration; successful installation on a new machine has not been verified during this repository import.

The full profile proposes five tasks, three orders and five seeds, plus two controls. It is a proposed experiment, not a completed result. The historical starter uses different settings, pooling and checkpoint formats; do not combine its outputs with version-2 results as though they came from the same protocol.

## Historical pilot

One reported H100 MIG pilot trained SST-2 -> QQP -> QNLI -> MNLI -> AG News. Its saved final task accuracies are 0.92, 0.74, 0.76, 0.68 and 0.70 on custom 50-example test sets. The unweighted final mean is 0.76. The complete saved post-task scores yield backward transfer -0.045 over the first four tasks. These are historical single-trajectory measurements, without seed uncertainty or completed inversion/control analysis. They do not establish H2, H3 or H4.

## Data and large artifacts

Raw dataset JSONL files, NumPy feature arrays, trained safetensors weights and generated caches stay outside ordinary Git history. Their original relative paths, byte counts and SHA256 hashes are in [local_artifact_inventory.json](provenance/local_artifact_inventory.json). These hashes identify local artifacts; they are not download links or evidence of an external backup. Small pilot records and checkpoint configurations are included.

The larger legacy data/manifest.json is recorded separately from the small pilot's documentation/data_manifest.json. The supplied larger raw-data directory is not verified to reconstruct the exact historical pilot probes. Obtain the matching data_h100_smoke files before claiming that reconstruction.

## Import validation

All 35 entries in the supplied version-2 source manifest matched their original bytes before import. Python syntax was checked for all 23 supplied code files. Candidate text files were scanned for common credential patterns. No dependency installation, unit-test rerun, GPU experiment or hypothesis analysis was performed during this import. See the existing package VALIDATION.md for earlier fixture-test evidence.

Public copies replace local account/server identifiers with placeholders; training algorithms and numeric results are retained. The original source manifest and a publication transformation map preserve the distinction between original and public copies. Historical source-sha256.txt describes the original executed files.
