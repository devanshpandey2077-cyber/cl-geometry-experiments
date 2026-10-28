# H2 compact workflow

This is a separate derivative of the v3 pipeline. It does not change your old
experiment, its frozen plans, or its smoke-test evidence. Start with
[START_COMPACT.md](START_COMPACT.md), which clearly separates Windows commands
from server commands.

The full primary profile on Qwen3-1.7B still selects 20,000 training examples per
task, two uncapped epochs, two sampled data cohorts, ten training seeds and five
orders: 100 LoRA trajectories plus four fixed-encoder controls. Statistical
thresholds, paired bootstrap draws and joint family adjustment are preserved.
No outcome is guaranteed to support H2.

The lifecycle is: one GPU run -> PBS CPU analysis and packing -> laptop download
and full hash verification -> acknowledgement -> explicit PBS CPU release ->
next GPU run. Final joint analysis reads sealed statistical capsules instead of
reopening removed representation arrays. Compact plans refuse multi-run GPU
submission and refuse another run while a completed run has not been released.

Kept on the server: bootstrap draws, primary analysis reports, all declared
descriptive sensitivity results, metrics, learning/control audits, configuration,
code snapshots, integrity manifests, run/failure records and archive receipts.

Kept in the laptop archives: those files PLUS all six task-boundary adapter/head
checkpoints per completed run. One shared archive contains selected raw and
tokenized data, data identities, pinned model metadata, source and frozen plan.
The base model weights are not copied to the laptop.

Removed from the server ONLY after both shared and run archives are acknowledged:
the explicitly inventoried per-stage representation files and checkpoint files,
and the acknowledged server ZIP copies. No broad recursive cleanup is used.

Raw representations are deliberately NOT archived. Every declared analysis is
completed before removal. New feature-level analyses may require regenerating
representations. Boundary checkpoints contain adapters/head parameters and RNG,
not in-task optimizer state; in-task resume files remain protected while a run
is incomplete. A checksum is an integrity check, not a signature or a backup.

For the current 1.7B/rank16 attention-only plan, adapter tensor payload across
600 checkpoints is about 14.36 GiB; primary bootstrap draws total about 0.14 GiB.
20–30 GiB is a provisional laptop allowance including other outputs, not a
measured guarantee. `compact.py space` reports actual recorded archive sizes.
Server headroom is checked before submission, inside feature/checkpoint writes,
and before ZIP creation. These checks cannot reserve space against other users.
Preparation, token caches, interrupted runs and analysis can still be limiting.

New entry points:

* `bootstrap_compact.py`: reuse existing server caches in a new folder.
* `suite.py build ... --storage-mode compact`: freeze the storage policy into a new plan.
* `compact.py pack/status/run-id/ready/space/release`: lifecycle and verification.
* `archive_client.py`: laptop-only stdlib/OpenSSH download, verify and acknowledge.
* `restore_compact.py`: recover retained statistical capsules from verified archives.
* `pbs/compact_cpu.pbs`: CPU packing and explicit release (32 GB RAM request).

Both regular CPU and GPU PBS scripts now request 32 GB system RAM. Reuse the
working `h2_h100` environment; do not install another environment or reinstall
torch. Existing v3 requirements are included as a reference for the remote pilot,
not an instruction to replace its working packages.

The inherited documents in `legacy_v3_docs/` describe the original workflow.
Use START_COMPACT.md for the new workflow. Do not use the legacy submit scripts
to launch compact campaigns.
