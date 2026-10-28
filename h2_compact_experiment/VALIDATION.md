# Validation of the compact workflow

Local CPU validation: **41 tests discovered, 39 passed, 2 skipped**.
The two skipped tests require a matching Transformers/Qwen3 backend unavailable
in the local runtime. They must run in the existing H100 environment.
Python sources were parsed locally. Local Bash execution was blocked by the
Windows sandbox, so the supplied remote test job checks every PBS/shell file
with `bash -n` before running the Python tests.

The new end-to-end fixture runs three trajectories (frozen, head and LoRA),
computes primary and all eight original descriptive sensitivity views, seals
and verifies real ZIP archives, simulates laptop transfer and acknowledgement,
releases original feature/checkpoint files, and reruns the joint analysis.

Before and after compaction, it checks:

* Element-for-element equality of retained bootstrap draws and joint intervals.
* Byte equality of joint comparison, run-summary and sensitivity CSVs.
* Equal cell reports except the creation timestamp, including control and
  learning decisions, multiplicity adjustment and final conclusion.
* Original checkpoint hashes present in the verified laptop archive.
* Successful statistical-artifact restoration from an archive.

Other tests cover refusal to release without BOTH acknowledgements, refusal of
wrong acknowledgement hashes, changed retained statistics, changed originals,
unsafe ZIP paths and extra members, low-space guards, repeat packing/release,
submission gating, shared raw/token input verification, and the downloader's
verify-before-acknowledge sequence. SSH/SCP are simulated in those tests; no live
cluster connection or real network transfer was performed here.

The underlying geometry, bootstrap computation, LoRA mathematics, prompt
construction and dataset sampling modules are unchanged in content. Storage
hooks were added to the engine/checkpoint writer; the suite aggregation uses
sealed cached inputs without changing its calculation. All original tests were
retained. The local PyTorch/NumPy bridge emits a compatibility warning; the
existing conversion fallback was used. No local or remote packages were changed.

This is software-equivalence evidence on small CPU fixtures, not evidence of
H100 memory fit, full-data runtime, full-scale learning, or H2 itself. Follow
START_COMPACT.md to validate the new lifecycle on the actual environment, retain
the matching development calibration, and measure a full-data run before the
complete campaign. Shared filesystem headroom checks are not capacity reservations.
