# Complete H2 v3 execution guide

Prepared 26 October 2026. Read this once before copying commands. Execute one numbered stage at a time.
Nothing in this guide says that a proposed job has already run or that H2 will come out positive.

## Understand the code before running it

The complete conceptual and file-by-file explanation is in
[H2_CODE_EXPLAINED.md](H2_CODE_EXPLAINED.md). It includes a worked rank-inversion example,
the LoRA update, fixed probes, representation subspaces, repeated runs, uncertainty, controls,
output interpretation and the limits of the conclusions.

The model is pretrained before your sequence begins. In one trajectory, a single cumulative adapter learns
one task at a time. The engine measures the same probes before training and after each of five tasks.
It then compares the initial and final ordering of the task-pair subspace affinities.
An accuracy drop alone does not establish H2, and bootstrap resamples are not additional model-training repetitions.

There are two different ranks: the **LoRA rank** controls trainable update capacity; the **analysis rank**
controls how many representation directions enter the affinity estimator. Both default to16, but they are distinct.
The same distinction applies to training-parameter stress tests versus descriptive measurement-sensitivity tests.

Use the detailed explanation to understand the experiment, and the numbered sections below to execute it.

## 1. Understand where commands run

| Location | Prompt/example | Appropriate actions |
|---|---|---|
| Windows PowerShell | `PS C:\Users\YOUR_LOCAL_USER>` | Open SSH, transfer ZIP/folder using WinSCP |
| HPC SSH/login shell | `YOUR_HPC_USER@HPC_HOST:...$` | Navigate/edit, `qsub`, `qstat`, `tail`, lightweight plan/status/budget commands |
| Scheduled CPU job | `PBS_JOBID` set | Dependency setup, data/model downloads, tokenization, freeze integrity/audits, statistics, analysis |
| Scheduled GPU job | Authorized device visible | Actual model training, evaluation and representation extraction |

Do not run `python -m h2.engine`, `suite.py run-one`, `prepare`, `freeze`, `analyze` or `sensitivity`
directly in the shared login shell. PBS guards are intentional. Tests and tiny synthetic fixtures
may run on your own laptop; they are not server training or hypothesis evidence.

Your historical server path was `/Data4/YOUR_HPC_USER/experiment H2 full scale`.
This version uses a NEW folder. Keep the historical pilot, its logs, and v2 code untouched.

## 2. Upload the complete folder

Extract the ZIP on Windows. Upload its `h2_robust_experiment` folder with WinSCP to `/Data4/YOUR_HPC_USER/`.
Rename this NEW uploaded folder to `h2_robust_experiment_v3` for the commands below.
Upload binary/as-is so text line endings are preserved. The ZIP itself contains LF shell scripts.

Windows PowerShell, only this command:

```powershell
ssh YOUR_HPC_USER@10.10.11.201
```

All remaining blocks are Linux commands INSIDE the SSH session, unless explicitly labeled otherwise.

```bash
cd /Data4/YOUR_HPC_USER/h2_robust_experiment_v3
pwd
ls suite.py FULL_PIPELINE.md PROTOCOL.md requirements.txt h2/engine.py pbs/bootstrap.pbs
python verify_package.py
```

Stop if `cd` or file verification fails. Do not continue from your account home after a failed `cd`.
If your remote upload has a different new folder name, use that exact path with quotes.

## 3. Configure your currently authorized allocation

```bash
test -f settings.env || cp settings.env.example settings.env
nano settings.env
```

Set:

```bash
CONDA_ENV=h2_h100_v3
HF_HOME=/Data4/YOUR_HPC_USER/.hf_cache
MIG_UUID="your_current_authorized_MIG_UUID"
export H2_ALLOW_FULL_GPU=0
export H2_MAX_MINUTES=25
```

`nano`: Ctrl+O, Enter to save; Ctrl+X to exit.

Use the MIG UUID allocated/authorized by Team HPC, not an old UUID copied from a previous run.
Lightweight `nvidia-smi -L` can display the device inventory if facility policy permits it.
An unoccupied device is not a reservation. PBS resources in these templates do not themselves
request/reserve a GPU slice; use the college's actual allocation procedure.

If PBS sets `CUDA_VISIBLE_DEVICES`, scripts preserve it. For a separately authorized full GPU,
set `export H2_ALLOW_FULL_GPU=1` and use the assigned scheduler device/UUID.
This flag grants no allocation. The optional 14B/full-tuning profiles require >=72 GiB visible memory
by declared policy; they will not run on your historical ~39.5 GiB MIG device.

## 4. Set up an isolated environment through PBS

Your earlier environment `h2_h100` is cloned into `h2_h100_v3`; the original stays available.
If its name differs, add `H2_CLONE_ENV=the_actual_existing_environment` to settings.env.
The college Anaconda initialization path is taken from the recorded working setup.

```bash
qsub pbs/bootstrap.pbs
qstat -u YOUR_HPC_USER
```

Once started:

```bash
tail -n 80 logs/bootstrap.latest.log
tail -n 100 logs/setup.latest.log
```

Require dependency checks, shell syntax checks, ALL unit/backend tests, and
`SETUP AND CPU TESTS COMPLETE`, followed by `SERVER SETUP COMPLETE`.

`requirements.txt` pins the previously recorded server package versions, including torch2.14.1.
An existing local build `2.14.1+cu130` satisfies that torch pin and should remain installed.
If the clone has no CUDA-enabled torch build, install the facility-approved build inside a CPU setup job;
do not randomly replace the system CUDA toolkit or ignore a CPU-only torch installation.
The setup job requires the tiny real Qwen3/Llama backend tests to execute rather than skip.

After setup, activate the new environment for lightweight plan/submission commands:

```bash
source /apps/compilers/anaconda3/etc/profile.d/conda.sh
conda activate h2_h100_v3
```

## 5. Build the new smoke plan

```bash
python suite.py build --profile smoke --name h2v3_smoke
python suite.py budget --name h2v3_smoke
```

Expected: one cell, one LoRA trajectory and two controls. This only writes a plan.
Three distinct run paths and deterministic PBS job names are saved in `plans/h2v3_smoke/plan.json`.
The plan is immutable: changing source/configuration after building it requires a NEW plan name.
Edit queue/resource templates and scientific defaults before building, not halfway through a frozen campaign.

## 6. Prepare smoke assets on CPU

```bash
qsub -v H2_PLAN=h2v3_smoke,H2_ACTION=prepare pbs/suite_cpu.pbs
qstat -u YOUR_HPC_USER
```

Once started:

```bash
tail -n 100 logs/prepare.latest.log
```

Wait for `SUITE PREPARATION COMPLETE`. This resolves/pins model and dataset commits, caches
the model, builds audited grouped splits and token packs, and records hashes.
GPU jobs operate offline. Missing assets cause a clear failure; they do not select another model.
Unauthenticated download/rate-limit warnings are distinct from missing local cache failures.

## 7. Freeze the smoke protocol

```bash
qsub -v H2_PLAN=h2v3_smoke,H2_ACTION=freeze pbs/suite_cpu.pbs
```

After the job begins:

```bash
tail -n 80 logs/freeze.latest.log
```

Wait for `PROTOCOL FROZEN`. Smoke/calibration freezes do not need a prior calibration plan.
The job checks prepared bytes and creates `plans/h2v3_smoke/FROZEN.json`.
This is a local audit record, not an external preregistration.

## 8. Submit smoke GPU runs

Check settings.env still has `H2_MAX_MINUTES=25` for the 30-minute request.

```bash
python suite.py submit --name h2v3_smoke --limit 3 --walltime 00:30:00
python suite.py submit --name h2v3_smoke --limit 3 --walltime 00:30:00 --submit
```

First command is a dry run. Second sends exactly three PBS jobs, chained serially with afterany.
The two controls come first. The ledger is saved after every accepted qsub.
Afterany keeps one failure from permanently holding every subsequent run; the final analysis still
refuses a complete result if any run/control is missing or invalid.

```bash
qstat -u YOUR_HPC_USER
python suite.py status --name h2v3_smoke
tail -n 80 logs/gpu.latest.log
```

Interpretations: Q=queued, R=running, `TRAINING COMPLETE`=engine completed that trajectory,
`RUN PAUSED`/exit75=checkpoint saved for continuation, `FAILURE.json`=inspect the recorded error.
A job disappearing from qstat is not a success marker. Use the run's `COMPLETE.json` and logs.

## 9. Analyze the smoke controls and measurements

Once ALL three runs are COMPLETE:

```bash
qsub -v H2_PLAN=h2v3_smoke,H2_ACTION=analyze pbs/suite_cpu.pbs
```

After it starts:

```bash
tail -n 100 logs/analyze.latest.log
```

When finished:

```bash
cat reports/h2v3_smoke/summary.md
```

Require both fixed-encoder controls to pass. Read the learning/numerical records and runtime/memory.
The smoke plan is deliberately tiny and step-capped: it is not an H2 confirmation and need not
yield learning or a positive inversion. Its purpose is pipeline execution.

Optionally test the statistical instrument separately:

```bash
qsub pbs/statistics.pbs
```

Its output is `reports/instrument_validation.json`. Those Gaussian simulations test the instrument;
they are not additional transformer runs or evidence for H2 in your models.

## 10. Choose the planned confirmation scale before results

For a feasible first complete study, `core` is15 LoRA trajectories plus2 controls on Qwen3-4B
with10,000 train/task. For the requested large study, `primary` is200 trajectories plus8 controls
on Qwen3-4B/8B with20,000 train/task. The latter is intentionally expensive.
Do not treat all208 as independent draws from a universal population; see PROTOCOL.md.

Commands below use the large primary design. To use core, substitute `--like core`, `--profile core`,
and suitable new names consistently. Plan names are examples, not dates or observed run IDs.

## 11. Calibrate each primary model/shape on development data

```bash
python suite.py build --profile calibration --like primary --name h2v3_cal_primary
python suite.py budget --name h2v3_cal_primary
qsub -v H2_PLAN=h2v3_cal_primary,H2_ACTION=prepare pbs/suite_cpu.pbs
```

Wait for its NEW `SUITE PREPARATION COMPLETE` log. Then:

```bash
qsub -v H2_PLAN=h2v3_cal_primary,H2_ACTION=freeze pbs/suite_cpu.pbs
```

Wait for `PROTOCOL FROZEN`. Set `export H2_MAX_MINUTES=1400` in settings.env for24-hour jobs.

```bash
python suite.py submit --name h2v3_cal_primary --limit 6 --walltime 24:00:00
python suite.py submit --name h2v3_cal_primary --limit 6 --walltime 24:00:00 --submit
```

Monitor and resume if required, as in step15. After all6 jobs finish:

```bash
qsub -v H2_PLAN=h2v3_cal_primary,H2_ACTION=analyze pbs/suite_cpu.pbs
```

Calibration has1,500 train/task,100-update cap,200 held-out examples/split and validation-only scoring.
Freeze for the main study requires matching successful learning calibration and passing controls:
at least two tasks improve target NLL by>0.01 across their own training and reach accuracy>=chance+0.05.
Inspect peak memory, all task NLL/accuracy gains, clipping, actual updates/tokens and numerical stability.
Calibration's observed lengths/timings may underrepresent worst-case long examples; memory fit is not guaranteed.

If learning fails, examine data/prompt/code/optimization on development data. Change the design for a
stated scientific reason and create NEW matching calibration/confirmation plans. Do not tune for positive inversions.
Large-model OOM means fix the batch/length/resource plan before confirmation, never silently lower its settings mid-run.

## 12. Build and prepare the large confirmation plan

```bash
python suite.py build --profile primary --name h2v3_primary
python suite.py budget --name h2v3_primary --calibration-plan h2v3_cal_primary
df -h .
qsub -v H2_PLAN=h2v3_primary,H2_ACTION=prepare pbs/suite_cpu.pbs
```

Check your ACCOUNT QUOTA separately: `df -h` reports filesystem free space, not your personal quota.
The uncompressed feature estimate is around half a TiB for the default primary design before
model/token/data/progress/checkpoint/report overhead. Use the exact budget printed by this version.
Do not launch a campaign without adequate storage/allocation. Compression is not a capacity guarantee.

There are2.5 million main optimizer updates in the default large primary design, excluding head controls,
evaluations, feature extraction and CPU bootstrapping. Actual walltime must be measured.
The budget's training estimate extrapolates observed seconds/update only; it excludes other costs and queue wait.
Analysis re-fits many subspaces. Its cost can be substantial even after GPU training has finished.

Wait for preparation to finish. It must fill every class-balanced grouped quota without silent shrinking.
If quotas fail, the log reports the exact deficits. New candidate counts/sizes require a new plan,
and potentially fresh calibration. A larger named dataset does not ensure the selected eligible pool is large enough.

## 13. Freeze the scientific protocol and prepared assets

Read `PROTOCOL.md`, the full plan JSON and your calibration reasoning before this step.
The0.01 affinity gap, rank16 and spectral threshold are defaults, not mathematically optimal constants.
If you change them, do so in source before rebuilding all affected plans and before seeing confirmation outcomes.

```bash
qsub -v H2_PLAN=h2v3_primary,H2_ACTION=freeze,H2_CALIBRATION_PLAN=h2v3_cal_primary pbs/suite_cpu.pbs
```

Wait for `PROTOCOL FROZEN`. Save a copy of the plan, FROZEN.json, calibration reports and source package
outside the run folder. The freeze checks content hashes, successful shape-matched learning calibration,
calibration controls, exact cross-cohort role leakage and within-role overlap.

External timestamped preregistration can strengthen the record, but is a separate researcher action.
Do not edit configs, plans, scripts, code, locked manifests or assets after freeze.

## 14. Submit in bounded windows

```bash
python suite.py submit --name h2v3_primary --limit 8 --walltime 24:00:00
python suite.py submit --name h2v3_primary --limit 8 --walltime 24:00:00 --submit
```

The default window is8, rather than blindly queuing208 long jobs. Facility queue limits still apply.
Inspect the ledger and actual run status. Later windows/resumptions use:

```bash
python suite.py submit --name h2v3_primary --limit 8 --walltime 24:00:00 --resume
python suite.py submit --name h2v3_primary --limit 8 --walltime 24:00:00 --resume --submit
```

Completed runs are skipped. Active jobs identified by job ID/deterministic job name are not duplicated.
An incomplete/paused run uses its original config/identity and most recent optimizer-step checkpoint.
Never run old `submit_campaign.py` against these v3 folders.
If you intentionally depend on a different active workq job, add `--after ACTUAL_JOB_ID` to the command,
replacing the placeholder with the exact ID returned by PBS. Do not copy the placeholder literally.

## 15. Recovery and failure handling

```bash
python suite.py status --name h2v3_primary
qstat -u YOUR_HPC_USER
tail -n 120 logs/gpu.latest.log
```

Optimizer checkpoints are saved every100 updates, at task completion and at a cooperative pause.
They include model trainable parameters, optimizer state, Python/NumPy/Torch/CUDA RNG and batch cursor.
Hard kills can lose work since the last checkpoint; replayed event-log lines can share task/update IDs.
Deduplicate those log identities when analyzing training-event counts. `training.json` gives final actual totals.
Interrupted feature extraction repeats that stage. Completed stages are verified with hashes before reuse.

`H2_MAX_MINUTES` must be less than the actual submitted walltime. It checks at optimizer boundaries,
so a long forward/backward/checkpoint can exceed the margin. Scheduler SIGKILL cannot be caught.
It may take several allocations to complete one large trajectory; resume does not count as a new repetition.

Source/config/data mismatches are refused. If a bug requires a source change, keep the old failed campaign,
fix the code in a new project/version and rebuild with documented identity; do not remove guards.

If qsub crashes between acceptance and ledger save, inspect `qstat -u` for the deterministic job name
and `plans/NAME/submissions.json`. The submitter attempts to reconcile visible active names.
Torque/PBS installations may truncate job names; if ambiguous, reconcile actual scheduler records manually
before retrying. The OS process lock also prevents concurrent execution of the same run path.
The script never cancels unrelated jobs or deletes earlier result folders.

## 16. Run the registered primary analysis

After all208 registered runs, including8 controls, are COMPLETE:

```bash
qsub -v H2_PLAN=h2v3_primary,H2_ACTION=analyze pbs/suite_cpu.pbs
```

For a long CPU analysis, rerun the SAME command after the previous job has ended.
Per-run bootstrap progress is saved every25 draws. Cached complete per-run analyses are reused.
Do not launch concurrent analysis jobs on the SAME cell; they would share partial files.

If permitted by your CPU allocation, process DIFFERENT cells in separate jobs using `H2_CELL`.
List the exact IDs with this lightweight command:

```bash
python -c 'import json; p=json.load(open("plans/h2v3_primary/plan.json")); print("\n".join(c["id"] for c in p["cells"]))'
```

Then use `qsub -v H2_PLAN=h2v3_primary,H2_ACTION=analyze,H2_CELL=EXACT_CELL_ID pbs/suite_cpu.pbs`,
replacing `EXACT_CELL_ID`. The alpha remains corrected over ALL registered cells, not just the selected cell.
The overall report remains incomplete until every cell report exists.

```bash
cat reports/h2v3_primary/summary.md
```

## 17. Read the outputs correctly

| Output | What it records |
|---|---|
| `plans/NAME/plan.json`, `FROZEN.json` | Full declared matrix and asset/protocol identities |
| `plans/NAME/submissions.json` | Submission intents, actual job IDs and continuation history |
| `runs/NAME/CELL/RUN/run.json` | Resolved config, source snapshot, device, PBS job, data/model/software provenance |
| `training_events.jsonl` | Per-update loss/LR/grad norm/examples/tokens; repeats can follow recovery |
| `stages/stage_NNN/training.json` | Actual completed-task counts and elapsed training time |
| `stages/stage_NNN/metrics.json` | Every evaluated task's accuracy, F1, NLL, confusion and invalid-output rate |
| `stages/stage_NNN/features/...` | Float32 fixed-probe representations, row IDs, groups and labels |
| `stages/stage_NNN/checkpoint.pt` | Custom LoRA/readout/full trainable weights, plus RNG at boundaries |
| `stages/stage_NNN/COMPLETE.json` | Stage identity and all artifact hashes |
| `FAILURE.json` | Last failed/paused attempt; can remain after a later successful continuation |
| `COMPLETE.json` at run root | Successful trajectory marker and peak CUDA memory |
| `analysis/.../report.json` | Per-run diagnostics, raw inversions and approximate conditional intervals |
| `reports/NAME/execution_status.csv` | Every planned run, including failed/missing runs |
| `reports/NAME/CELL/joint_pair_comparisons.csv` | All45 comparisons/run, suite-adjusted endpoint intervals and eligibility checks |
| `reports/NAME/CELL/cell_report.json` | Controls, learning audit, conditional family and descriptive repetition results |
| `reports/NAME/suite_report.json`, `summary.md` | Complete/incomplete suite status, null cells and overall evidence table |

Prefer suite-adjusted decisions to isolated per-run decisions. The figures are illustrations selected
from the same data; selected panels are not independent confirmation. A repeated count is not a p-value.
An interval-supported reversal with an unstable cutoff or inadequate learning stays visible with its failed flag.
All comparisons share tasks/probes; do not count45 comparisons as45 independent experiments.

`Backward transfer` compares final accuracy to accuracy immediately after each old task was learned.
It is not best-to-final forgetting and is not a causal effect of the measured affinity.
Valid-code accuracy selects among A/B/C/D for the current known task; inspect invalid unrestricted
top-1 rate and target NLL as well. This is not unconstrained chat/generation evaluation.

## 18. Parameter and model robustness studies

Stress plan, with matching calibration of all11 settings:

```bash
python suite.py build --profile calibration --like stress --name h2v3_cal_stress
python suite.py build --profile stress --name h2v3_stress
```

Prepare/freeze/execute/analyze calibration first, then prepare/freeze the stress plan with
`H2_CALIBRATION_PLAN=h2v3_cal_stress`, exactly as steps11–16. Stress has99 training trajectories
plus22 controls. There are no implicit Cartesian interactions beyond these one-factor settings.

Model replication:

```bash
python suite.py build --profile calibration --like replication --name h2v3_cal_replication
python suite.py build --profile replication --name h2v3_replication
```

Optional authorized larger device / full tuning:

```bash
python suite.py build --profile calibration --like scale14b --name h2v3_cal_14b
python suite.py build --profile scale14b --name h2v3_14b
python suite.py build --profile calibration --like full_tuning --name h2v3_cal_full
python suite.py build --profile full_tuning --name h2v3_full
```

These still require their own matching calibration, storage and freeze. The full-tuning profile writes
large FP32 model checkpoints at every boundary plus optimizer-progress checkpoints. Budget this explicitly.
The default full-tuning LR is a candidate, not a validated optimum; calibration may require redesign.

Different suites are separate families. If choosing the best outcome across them, their individual
0.05 claims do not produce a combined0.05 claim. Preallocate alpha or run fresh untouched confirmation.

## 19. Descriptive measurement sensitivity

```bash
qsub -v H2_PLAN=h2v3_primary,H2_ACTION=sensitivity pbs/suite_cpu.pbs
```

This creates an explicitly DESCRIPTIVE CSV for rank, layer, pooling, probe and covariance overlap.
It does not secretly add favorable secondary choices to the primary inference family.
If a secondary view motivates a new hypothesis, freeze a new independent confirmation design.

## 20. Preserve results and write the paper

Download the plan/freeze/ledger, entire reports, logs, source archive, data and token manifests,
all run metadata, original feature arrays and necessary checkpoints. A small summary is not a backup.
Keep source hashes and model/dataset commit SHAs with the results. Do not upload raw dataset/model
weights publicly without checking their license/storage needs. No GitHub publication is performed by this package.

Report the operational H2 statement, estimator, population/splits, all model/update settings,
repeated seed/order table, failed and null cells, controls, simultaneous intervals, cutoff diagnostics,
learning checks, overlap and limitations. Describe evidence as supporting/refuting this configured
test at its resolution. Do not claim universal proof, signed interference, H3/H4 or novelty from H2 alone.
