# Start here: college H100 experiment

This is a new project. Keep your successful `36125.master` pilot and its results.

## 1. Upload

Extract the ZIP on Windows. Using WinSCP, upload the entire **h2_full_experiment** folder to:

`/Data4/YOUR_HPC_USER/`

The server must then contain `/Data4/YOUR_HPC_USER/h2_full_experiment/h2/engine.py`.

## 2. Connect

Run only this line in Windows PowerShell:

```powershell
ssh YOUR_HPC_USER@YOUR_HPC_HOST
```

After login, your prompt should contain `YOUR_HPC_USER@master`. All remaining command blocks in this file are **Linux commands inside that SSH session**.

```bash
cd /Data4/YOUR_HPC_USER/h2_full_experiment
source /apps/compilers/anaconda3/etc/profile.d/conda.sh
conda activate h2_h100
cp settings.env.example settings.env
```

## 3. Prepare the environment using PBS

```bash
qsub pbs/setup.pbs
qstat -u YOUR_HPC_USER
```

After the job starts, inspect its log:

```bash
tail -n 60 logs/setup.latest.log
```

Wait for `SETUP AND CPU TESTS COMPLETE`. This uses your existing `h2_h100` Conda environment. Package versions match the earlier reported server setup; the setup job installs missing dependencies and checks them. If it fails, read the log before continuing.

## 4. Prepare the small test data and local model

```bash
qsub -v H2_CONFIG=configs/smoke.json pbs/prepare.pbs
```

After it starts:

```bash
tail -n 60 logs/prepare.latest.log
```

Wait for `PREPARATION COMPLETE`. This is a CPU job. It may still scan substantial source datasets to make deduplicated samples. It reuses the pinned Qwen model in your existing cache when available.

## 5. Select a currently available MIG instance

Use the selection procedure authorized by Team HPC. These are monitoring commands:

```bash
nvidia-smi -L
nvidia-smi
nano settings.env
```

In `settings.env`, set `MIG_UUID` to the **currently free instance authorized for you**. Save with Ctrl+O, Enter, then exit with Ctrl+X. Your old MIG UUID is not a permanent assignment. If PBS supplies a MIG UUID itself, that value takes precedence.

The job checks occupancy before starting CUDA work and stops if it detects another process. This check does not reserve a device: shared access still depends on the college's allocation procedure. If your college changes to scheduler-managed GPU resources, use its allocation directives and environment rather than manually choosing a slice.

## 6. Run the small complete pipeline

```bash
python submit_campaign.py --config configs/smoke.json --campaign smoke_v2 --orders O1 --seeds 101 --walltime 00:30:00 --with-analysis
```

That prints a plan only. Submit it with:

```bash
python submit_campaign.py --config configs/smoke.json --campaign smoke_v2 --orders O1 --seeds 101 --walltime 00:30:00 --with-analysis --submit
```

It queues one LoRA run, one unchanged-encoder control, one head-only control, and CPU analysis. GPU jobs depend on successful completion of the preceding job, so only one runs at a time.

```bash
qstat -u YOUR_HPC_USER
python status.py --campaign smoke_v2
tail -n 60 logs/gpu.latest.log
tail -n 60 logs/analysis.latest.log
```

For live viewing, use `tail -f logs/gpu.latest.log`. **Ctrl+C stops the log viewer; it does not cancel the PBS job.** The alias points to the most recently started job; restart the viewer when the next job starts.

When analysis finishes, read:

```bash
cat reports/smoke_v2/summary.md
```

Use WinSCP to download `reports/smoke_v2` and each run's `analysis` folder. A passing smoke test checks the pipeline; its small samples do not support a scientific conclusion.

## 7. Scale up only after the smoke test works

Read `README.md` and `PROTOCOL.md`, run the separate calibration profile, and freeze the full protocol before looking at confirmation results. Then:

```bash
qsub -v H2_CONFIG=configs/full.json pbs/prepare.pbs
```

Wait for `PREPARATION COMPLETE`, then:

```bash
python doctor.py --config configs/full.json
python submit_campaign.py --config configs/full.json --campaign full_h2 --with-analysis
```

Check storage quota and the printed plan. Submit:

```bash
python submit_campaign.py --config configs/full.json --campaign full_h2 --with-analysis --submit
```

This is **17 GPU runs**, followed by CPU analysis. Runtime has not been benchmarked on your server. Use the calibration timings to choose a suitable per-job walltime, adding `--walltime HH:MM:SS` with a real duration accepted by the cluster if needed. If any GPU job fails, the dependency chain stops; see recovery instructions in `README.md`.

The final report is `reports/full_h2/summary.md`. Neither code nor additional data can guarantee that H2 will be supported. Report a negative or inconclusive result honestly.
