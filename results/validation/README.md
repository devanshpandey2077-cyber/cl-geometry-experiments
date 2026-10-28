# Available local validation evidence

These are preserved local logs, not H100 hypothesis results. Publication has
not rerun the test suites; only source/package/numerical/publication checks were
performed.

| File | Outcome and meaning |
|---|---|
| [v3_cpu_tests.txt](v3_cpu_tests.txt) | 32 discovered, 30 passed, two real-backend tests skipped |
| [v3_statistics_console.txt](v3_statistics_console.txt) | Synthetic Gaussian diagnostic; embedded date 26 October 2026 |
| [compact_development_failed.txt](compact_development_failed.txt) | Earlier fixture setup failed with six errors; retained rather than hidden |
| [compact_focused_pass.txt](compact_focused_pass.txt) | Eight focused compact tests passed |
| [compact_final_pass.txt](compact_final_pass.txt) | Final suite: 41 discovered, 39 passed, two skipped |

The v3 package also includes its original
[validation results](../../h2_robust_experiment_v3/validation/local_test_results.txt),
[synthetic diagnostic JSON](../../h2_robust_experiment_v3/validation/synthetic_instrument_diagnostic.json)
and [design preview](../../h2_robust_experiment_v3/validation/design_preview.json).
The compact package includes its
[test summary](../../h2_compact_experiment/validation/test_summary.json) and
[planning preview](../../h2_compact_experiment/validation/planning.json).

Stack traces retain failure messages and numeric outcomes with local account
paths generalized. The failing development log describes an earlier fixture,
not the final source snapshot. A final passing test does not retroactively turn
that earlier attempt into a pass. Two backend skips remain skips. SSH/SCP are
simulated in the compact fixture tests; live transfer and H100 validation are pending.
