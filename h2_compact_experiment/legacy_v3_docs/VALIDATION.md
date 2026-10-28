# Delivery validation — 26 October 2026

This records engineering validation performed locally, not H100 experiment results.

## Checks performed

- Full CPU suite:32 tests discovered;30 passed and2 real Transformer backend tests skipped.
- Existing15 v2 regression tests retained and passed.
- Atomic optimizer-step recovery, including dropout/RNG/optimizer/batch cursor, matched uninterrupted CPU training exactly.
- All-stage fixture extraction, per-run analysis, suite-wide analysis and both fixed-encoder controls passed.
- Completed artifact corruption was detected on analysis and resume.
- Stable source-role sampling, quota failure, paired global group multiplicities and no duplicate-as-seed counting checked.
- Multi-model profile counts, matching calibration shapes, bounded serial qsub mocks, ledger writes and active-job reconciliation checked.
- Principal-angle and covariance invariance checks passed; the production iterative SVD path was compared with dense SVD.
- Python syntax parsed successfully for every source/test module.
- Source SHA256 manifest and ZIP integrity verified during packaging.

The complete named test list is in `validation/local_test_results.txt`.
Design counts and budget estimates are in `validation/design_preview.json`; these are previews, not executed plans.

## Synthetic instrument diagnostic

`validation/synthetic_instrument_diagnostic.json` contains30 stable-population and30 planted-reversal
Gaussian experiments, n80,199 bootstrap draws each. Results:

| Diagnostic | Observed |
|---|---:|
| Stable-population reversal detections | 0/30 |
| Planted-reversal detections | 30/30 |
| Simultaneous population coverage, stable setup | 30/30 |
| Simultaneous population coverage, planted setup | 29/30 |

Wilson intervals are in the JSON. Thirty simulations give wide uncertainty and cannot establish
nominal coverage for transformers, for large multiplicity families, or for all group structures.
This is a useful finite instrument diagnostic, not real H2 evidence or a proof of statistical validity.

## Environment limitations and required server checks

The local interpreter had CPU torch2.1.0, NumPy2.3.5 and SciPy1.16.3, rather than the pinned server stack.
The inherited NumPy/old-Torch bridge warning was handled by the existing tested tensor-list fallback.
Its Transformers/Hugging Face versions were incompatible, so the two real tiny Qwen3/Llama tests could not run here.
Installing isolated matching dependencies was unavailable because the execution environment could not resolve PyPI.

Git Bash could not initialize its Windows shared-memory objects inside this sandbox, so native `bash -n`
was not completed locally. All shell files were reviewed and packaged with LF line endings.
The CPU server setup explicitly runs `bash -n` on every PBS/shell script and sets
`H2_REQUIRE_BACKEND_TESTS=1`, requiring both real backend tests to execute successfully.

No real model download, real dataset preparation/quota-fill audit, PBS submission, CUDA/H100 run,
large-model memory-fit test, full transformer training, or real-feature statistical result was executed for this delivery.
Queue syntax, allocation authorization, facility network access and walltime/storage limits need the server smoke/calibration stages.
Thus the package is locally tested research software with explicit server validation gates, not a certified completed full-scale experiment.
