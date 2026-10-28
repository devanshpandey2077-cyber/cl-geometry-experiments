#!/bin/bash
set -euo pipefail
: "${PBS_JOBID:?Submit this script using qsub}"
: "${PBS_O_WORKDIR:?Missing PBS working directory}"
cd "$PBS_O_WORKDIR"
test -f h2/engine.py || { echo 'Submit from the h2_full_experiment folder'; exit 2; }
test -f settings.env || { echo 'Copy settings.env.example to settings.env first'; exit 2; }
source settings.env
set +u
source /apps/compilers/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-h2_h100_v3}"
set -u
export HF_HOME="${HF_HOME:-/Data4/YOUR_HPC_USER/.hf_cache}"
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${H2_THREADS:-4}" MKL_NUM_THREADS="${H2_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${H2_THREADS:-4}"
mkdir -p logs
H2_LOG="logs/${H2_LOG_KIND}.${PBS_JOBID}.log"
ln -sfn "$(basename "$H2_LOG")" "logs/${H2_LOG_KIND}.latest.log"
echo "Log: $PBS_O_WORKDIR/$H2_LOG"
exec > "$H2_LOG" 2>&1
echo "Job: $PBS_JOBID; host: $(hostname); Python: $(command -v python)"
date -Is
