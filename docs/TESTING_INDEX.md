# Testing and source index

Updated: 28 October 2026.

Start with the [27 October experiment record](EXPERIMENT_RECORD_2026-10-27.md).
It links the preserved evidence and separates execution, local tests, derived
metrics, terminal transcriptions and plans.

| Material | Location | Status |
|---|---|---|
| Original v3 delivery source | [h2_robust_experiment_v3/](../h2_robust_experiment_v3/) | Public path substitutions; historical resource defaults retained |
| Compact derivative source | [h2_compact_experiment/](../h2_compact_experiment/) | Current storage lifecycle; not yet verified on H100 |
| Code explanation | [H2_V3_CODE_EXPLAINED.md](H2_V3_CODE_EXPLAINED.md) | Model/data/methods and exact budget explanation |
| GPU smoke evidence | [results/h2v3_smoke_r1/](../results/h2v3_smoke_r1/) | Preserved console attachment and extracted metrics |
| Local test evidence | [results/validation/](../results/validation/) | v3 suite, synthetic diagnostic, compact failures and passes |
| Original manifests and transformations | [provenance/h2_20261028/](../provenance/h2_20261028/) | Original/public hashes and publication scope |
| Earlier research notebook | [H2_H3_H4_Research_Log.md](H2_H3_H4_Research_Log.md) | Historical entries retained with a dated update |
| v2 source | [h2_full_experiment/](../h2_full_experiment/) | Previous version retained |
| Earlier pilot | [results/historical/](../results/historical/) | Different legacy protocol; do not merge its metrics with v3 |

## Observed versus planned

- Observed in the attached GPU console: frozen/head/LoRA smoke completion and
  all five tasks' stage metrics.
- Reported in the supplied CPU terminal output: complete one-cell smoke
  analysis, zero eligible reversals and zero repeated comparisons.
- Preserved local engineering result: compact final suite 39 passed, two skipped.
- Planned: the full 104-run, 20,000-example/task Qwen3-1.7B primary campaign.
- Pending: native detailed H100 geometry artifacts and H100 compact-workflow validation.

The original source packages were verified before publication. Public versions
generalize account paths and identifiers, preserve numerical test readings, and
have regenerated package manifests. Read PUBLICATION.md in each source folder
before using it. Build a NEW plan from the final public/customized source; old
frozen plan hashes will not match a modified checkout.
