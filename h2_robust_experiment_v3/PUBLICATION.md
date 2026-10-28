# Public source snapshot

Published 28 October 2026 from the separately delivered H2_Robust_Experiment_v3.zip.

Account-specific paths are placeholders in this public copy. Replace YOUR_HPC_USER and YOUR_LOCAL_USER as appropriate before building a new server plan. In the laptop client, supply --remote-python for your actual server environment.

The published SOURCE_MANIFEST.json describes these public files. The original delivery manifest and original/public hashes are retained under ../provenance/h2_20261028/. This is not a byte-identical copy of an executed server checkout. Its remote run.json/FROZEN.json and manual server patches have not been received.

This historical v3 snapshot retains its original resource defaults. Some PBS scripts request 64 GB RAM and bootstrap may clone an environment. Those defaults caused problems on the documented 32 GB-per-job account. Prefer the separate compact workflow and reuse its existing environment; do not blindly run bootstrap or edit an already built/frozen plan.
