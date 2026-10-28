# H2 v3 smoke evidence

Campaign: h2v3_smoke_r1. Execution date in supplied console: 27 October 2026.
Published: 28 October 2026.

| File | Evidence type |
|---|---|
| [gpu_console_excerpt.txt](gpu_console_excerpt.txt) | Preserved Pasted text(4).txt attachment; account/device identifiers generalized |
| [stage_metrics_from_console.json](stage_metrics_from_console.json) | Parsed visible metrics, confusion matrices, printed training events and line references |
| [stage_metrics_from_console.csv](stage_metrics_from_console.csv) | 180 task/stage/split metric rows extracted from the console |
| [derived_accuracy_summary.json](derived_accuracy_summary.json) | Final accuracy means and BWT derived from those rows |
| [reported_analysis_transcript.txt](reported_analysis_transcript.txt) | Normalized transcription of user-pasted CPU status/log/summary; not native server report bytes |
| [gpu_inventory_2.txt](gpu_inventory_2.txt) | Hardware diagnostic with other-user process table omitted |
| [gpu_inventory_3.txt](gpu_inventory_3.txt) | Second supplied diagnostic/command capture, same public redactions |

The console contains completed frozen/head/LoRA runs from jobs 37969.master,
37970.master and 37971.master. It is a concatenated live-tail capture, not a
guarantee that every byte of every original per-job log is present.

The CPU analysis summary reports zero eligible reversals at the declared
resolution. Detailed geometry failure reasons cannot be recovered from this
summary. No feature array, trained checkpoint, run.json or FROZEN.json was
received for this publication.

See [the experiment record](../../docs/EXPERIMENT_RECORD_2026-10-27.md) for the
interpretation and missing-artifact list. Original/private versus published
hashes are in [the provenance inventory](../../provenance/h2_20261028/source_and_transformations.json).
