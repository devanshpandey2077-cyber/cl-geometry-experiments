# Start the compact workflow

## 1. Windows: upload this new package

Extract `H2_Compact_Workflow.zip` on your laptop. Keep the original ZIP too.
Open PowerShell in the folder containing the ZIP. In the following assignment,
replace YOUR_HPC_HOST with the host or SSH alias you already use successfully.
Do not use the literal placeholder. Keep passwords out of the command.

```powershell
$HpcHost = "YOUR_HPC_USER@YOUR_HPC_HOST"
scp .\H2_Compact_Workflow.zip "${HpcHost}:/Data4/YOUR_HPC_USER/"
```

## 2. Server: create a separate folder and reuse working caches

These commands belong in the SSH terminal (`YOUR_HPC_USER@HPC_HOST`), not PowerShell.
Run one line at a time. Do not run `conda init`, create another environment, edit
old source, or reuse an old frozen plan.

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate h2_h100
cd /Data4/YOUR_HPC_USER
python -c "from pathlib import Path; import zipfile; assert not Path('h2_compact_experiment').exists(), 'Target already exists; preserve it'; zipfile.ZipFile('H2_Compact_Workflow.zip').extractall('.')"
cd /Data4/YOUR_HPC_USER/h2_compact_experiment
python verify_package.py
python bootstrap_compact.py --reuse /Data4/YOUR_HPC_USER/h2_robust_experiment_v3
python -c "import sys; print(sys.executable)"
df -h .
```

Bootstrap copies small asset manifests and settings and links the prepared data
cache. Models stay in the existing external Hugging Face cache. New cache
directories may be added by preparation; old named cache contents are not edited.

Inspect `settings.env` with your editor. Confirm the working `CONDA_ENV=h2_h100`,
existing HF_HOME, and CURRENT GPU/MIG selection under the cluster's procedure.
An old GPU UUID is not an allocation guarantee. PBS CUDA_VISIBLE_DEVICES takes
precedence. For 30-minute GPU tests set `export H2_MAX_MINUTES=25`; for 24-hour
GPU runs use `export H2_MAX_MINUTES=1400`. Settings.env is outside source hashes.

Run the remote software checks through the packaged CPU job before interpreting
any failure as an H2 result:

```bash
qsub pbs/compact_tests.pbs
```

Wait for that returned job ID to finish with Exit_status=0. Never submit a
second copy merely because its log is not present immediately.

## 3. Validate the new storage lifecycle on a small smoke plan

This repeats a small ENGINEERING check under the new source hash. It is not
the full-data H2 campaign and is not used as confirmatory evidence.

```bash
PLAN=h2compact_smoke
python suite.py build --name "$PLAN" --profile smoke --models qwen1.7b --storage-mode compact
qsub -q cpuq -l select=1:ncpus=8:mem=32gb -l walltime=02:00:00 -v "H2_PLAN=$PLAN,H2_ACTION=prepare" pbs/suite_cpu.pbs
```

For every CPU/GPU job, check the actual returned ID:

```bash
qstat -xf JOB_ID | grep -E 'job_state|Exit_status|comment'
```

Replace JOB_ID, and proceed only when the relevant job is finished with exit 0.
After preparation:

```bash
qsub -q cpuq -l select=1:ncpus=8:mem=32gb -l walltime=02:00:00 -v "H2_PLAN=$PLAN,H2_ACTION=freeze" pbs/suite_cpu.pbs
```

After freezing finishes, perform the single-run cycle in section 4 three times:
frozen control, head-only control and LoRA. At the end run the joint analysis in
section 7. Check all control results and any failed-run records.

## 4. Single-run cycle: GPU -> CPU -> laptop -> CPU

### A. Server: submit ONE GPU run

```bash
python suite.py submit --name "$PLAN" --limit 1 --resume --walltime 00:30:00 --submit
qstat -u YOUR_HPC_USER
```

For calibration and primary runs change the GPU walltime to `24:00:00` and use
H2_MAX_MINUTES=1400 in settings.env. Walltime is a ceiling, not a runtime estimate.
If a job pauses with exit 75 or is incomplete, resume the SAME registered run;
do not skip it, delete it or choose a different seed. A non-pause error needs
inspection first. A COMPLETE marker alone does not mean the H2 learning checks passed.

After the GPU run completes:

```bash
RUN_ID=$(python compact.py run-id --name "$PLAN")
echo "$RUN_ID"
qsub -q cpuq -l select=1:ncpus=8:mem=32gb -l walltime=24:00:00 -v "H2_PLAN=$PLAN,H2_ACTION=pack,H2_RUN_ID=$RUN_ID" pbs/compact_cpu.pbs
```

Packing checks original hashes and controls/learning, computes all registered
primary bootstrap draws and descriptive sensitivities, and makes an archive.
It does NOT delete originals. CPU bootstrap can be slow; retain its partial
state and rerun the same pack job if walltime expires. Wait for exit 0.

```bash
python compact.py ready --name "$PLAN"
python compact.py space --name "$PLAN"
```

### B. Windows: download, verify and acknowledge

In a SEPARATE PowerShell terminal, enter the locally extracted
`h2_compact_experiment` folder. Your Windows Python needs only its standard
library; no torch installation is needed for archiving. OpenSSH ssh/scp must
be available. Set $HpcHost to your actual working SSH address as in section 1.

```powershell
$Plan = "h2compact_smoke"
python archive_client.py --host "$HpcHost" --remote-root /Data4/YOUR_HPC_USER/h2_compact_experiment --name "$Plan" --destination C:\Users\YOUR_LOCAL_USER\H2_Archives
```

The client downloads the shared snapshot once, and each ready run archive. It
verifies the ZIP hash AND every member, flushes the archive to disk, writes a
local verification receipt, then sends acknowledgement. Authentication uses
your existing SSH setup and may prompt for a password. If any step fails,
there is no release. Repeating the client command is safe. Keep these laptop
archives on persistent storage, preferably with another backup.

### C. Server: explicitly release acknowledged originals

Use the SAME PLAN and RUN_ID variables as the packing step. If you reconnected,
restore PLAN and run `RUN_ID=$(python compact.py run-id --name "$PLAN")` again.

```bash
qsub -q cpuq -l select=1:ncpus=8:mem=32gb -l walltime=02:00:00 -v "H2_PLAN=$PLAN,H2_ACTION=release,H2_RUN_ID=$RUN_ID" pbs/compact_cpu.pbs
```

Release rechecks the retained statistics and remaining originals, requires
acknowledgement of BOTH archives, and removes only the sealed list of feature
and boundary-checkpoint files plus acknowledged server ZIPs. The next GPU run
is blocked until release completes. A release interrupted mid-removal can be
resubmitted for the same RUN_ID. It never uses a recursive rm command.

```bash
python compact.py status --name "$PLAN"
df -h .
```

Now repeat A–C for the next registered run. There is no unattended loop that
queues all 104 runs before verified transfer and release.

## 5. Required matching calibration before freezing a primary plan

The inherited v3 protocol already requires a successful development calibration
matching the model, adapter, input shape, optimizer and epochs. The old smoke
run cannot satisfy this requirement. Calibration is 1,500 training examples per
task with a 100-update cap; this does NOT change the primary plan's 20,000 and
uncapped two epochs. Calibration tests resource use and learning competence.

```bash
PLAN=h2compact_cal
python suite.py build --name "$PLAN" --profile calibration --like primary --models qwen1.7b --storage-mode compact
```

Prepare and freeze it as in section 3, then perform the single-run cycle for its
three runs using 24-hour GPU jobs. Run joint analysis. Inspect learning audits
and both controls. Freezing the primary plan enforces these checks; do not
weaken thresholds to make calibration pass. Archived calibration audits remain
usable after release.

## 6. Build the FULL-DATA primary plan

```bash
PLAN=h2compact_primary_qwen17
python suite.py build --name "$PLAN" --profile primary --models qwen1.7b --storage-mode compact
python suite.py budget --name "$PLAN"
qsub -q cpuq -l select=1:ncpus=8:mem=32gb -l walltime=24:00:00 -v "H2_PLAN=$PLAN,H2_ACTION=prepare" pbs/suite_cpu.pbs
```

After preparation succeeds, inspect `df -h .`. Prepared data also takes space.
The GPU submission guard may still stop the campaign if the shared filesystem
has too little room. Compact mode reduces accumulated output; it cannot make
zero working space usable. Never delete caches blindly to force the guard to pass.

Freeze after successful matching calibration and primary preparation:

```bash
qsub -q cpuq -l select=1:ncpus=8:mem=32gb -l walltime=02:00:00 -v "H2_PLAN=$PLAN,H2_ACTION=freeze,H2_CALIBRATION_PLAN=h2compact_cal" pbs/suite_cpu.pbs
```

Then perform the single-run cycle with 24-hour GPU jobs. On Windows set
`$Plan = "h2compact_primary_qwen17"` for archiving this plan. First measure one
full-data LoRA trajectory's actual peak memory, elapsed time, checkpoints and
archive size before committing to the entire matrix. Controls run first.

The primary plan still has 2 cells, 100 LoRA runs, 4 controls, 20,000 training
examples/task, two full epochs, no step cap, 1,000 examples in each probe/task,
2,000 test examples/task and 2,000 bootstrap draws. Stored bootstrap arrays
retain their paired resampling alignment. Null and failed-learning runs stay
in the analysis; completed runs are not cherry-picked for archiving.

## 7. Final joint analysis

After ALL registered runs are complete, submit:

```bash
qsub -q cpuq -l select=1:ncpus=8:mem=32gb -l walltime=24:00:00 -v "H2_PLAN=$PLAN,H2_ACTION=analyze" pbs/suite_cpu.pbs
```

After success:

```bash
cat "reports/$PLAN/summary.md"
```

Optional aggregation of the already computed descriptive sensitivities:

```bash
qsub -q cpuq -l select=1:ncpus=8:mem=32gb -l walltime=02:00:00 -v "H2_PLAN=$PLAN,H2_ACTION=sensitivity" pbs/suite_cpu.pbs
```

Copy the final reports directory and updated plans/compact ledgers to the laptop
as well. Per-run archives contain the statistical ingredients; final aggregate
reports are generated later and are not retroactively added to those ZIPs.

## 8. Recovery and limits

To rebuild a lost retained analysis file, copy its verified run ZIP back to an
approved server path and run `restore_compact.py --file /absolute/path/run.zip`
inside a PBS cpuq job. It verifies every member, does not overwrite different
existing files, and restores retained statistics by default. `--checkpoints`
also restores the archived parameters for later feature regeneration.

For a fresh recovery directory, restore shared.zip first using the same tool,
then run ZIPs. The frozen model metadata still points to the original server
cache: preserve it or re-provision and verify the pinned model revision before
claiming reproduction. Shared restoration does not bypass PBS, source identity,
or model/data checks. Do not extract many checkpoint archives onto a full server.

New feature-level research questions may require regeneration. This package
does not promise bit-identical cross-hardware regeneration, unlimited shared
disk availability, a runtime estimate, or a positive H2 finding. Existing proof
limitations of the original statistical protocol still apply.
