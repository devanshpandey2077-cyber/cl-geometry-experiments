# Validation record

Date: 2026-10-19. No college HPC credentials were used and no PBS jobs were submitted while producing this package.

## Completed locally

- 15 unittest cases passed on a CPU with a tiny causal language-model fixture.
- Geometry: feature-coordinate affinity, orthogonal-rotation invariance, constructed rank reversal, minimum-gap tie rejection and numerical-rank rejection.
- Data: conflicting duplicates, repeated fields across tasks, balanced grouped splits, deliberate leakage rejection and disjoint exact-field development/confirmation pools.
- Tokenization/pooling: source-only character spans, truncation preserving the answer suffix, left padding and exclusion of instruction/padding values.
- Training: LoRA zero-initialized identity, frozen base weights, actual adapter updates and custom checkpoint restore.
- Pipeline: all task stages, evaluation and feature output, simulated interruption, task-boundary recovery yielding identical final trainable tensors to an uninterrupted run, and end-to-end numerical analysis.
- Controls: unchanged encoder and genuinely trained head with unchanged encoder; zero robust inversions in the fixture.
- Campaign: joint control report, full15+2 run matrix, mocked serial PBS dependencies, accepted-job ledger and refusal to overwrite a registered campaign.
- Python byte compilation, JSON profile loading, dry-run campaign generation and the storage preflight were exercised.

The local Windows installation has an older Torch build whose NumPy bridge reports an ABI warning with NumPy2. Tests passed using the included tensor/list fallback without changing that installation. This is not validation of the pinned college environment.

## Requires the server smoke test

- Installing/resolving the pinned dependencies in the college Conda environment.
- Downloading and parsing the real model/tokenizer and source datasets at their recorded revisions.
- Verifying that real grouped source data fills all requested quotas.
- Model-specific LoRA hooks, SDPA, gradient checkpointing, BF16, CUDA memory and throughput.
- PBS queue/resource syntax, allocation policy, NVML visibility and MIG occupancy behavior.
- Native Bash syntax checking: the local Git Bash runtime could not start under this Windows sandbox. The setup job includes `bash -n` for all PBS/shell files before running tests.
- Matplotlib figure generation and visual review: Matplotlib was unavailable locally; numerical reports were checked without figures. The server requirements include it.
- Full1,000-draw analysis runtime and numerical behavior on real learned features.

Passing these local tests establishes functionality on controlled fixtures. It does not establish real-data H2 support, H100 correctness, a scientific error guarantee, or research novelty. Follow START_HERE.md and complete the server smoke campaign before a full run.
