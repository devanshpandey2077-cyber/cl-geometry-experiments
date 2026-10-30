# H2 review: findings, explanations and fixes

Review date: 29-30 October 2026. Scope: `h2_compact_experiment/` (the current
launch path), its PBS scripts, the October 27 smoke evidence, the theory draft
in `../field-view`, and the repository layout. No code was changed.

Evidence used:

- Full read of every module in `h2_compact_experiment/h2/`, `suite.py`,
  `compact.py`, `archive_client.py` and `pbs/`.
- Local run of the test suite: 41 discovered, 41 passed (the two real tiny
  Qwen3/Llama backend tests also ran and passed). `verify_package.py` passed
  (64 files).
- The preserved H100 smoke console
  (`results/h2v3_smoke_r1/gpu_console_excerpt.txt`), for timing and stage-0
  readings.
- A stage-0 instrument diagnostic on the actual model (Qwen3-1.7B, pinned
  revision `70d244cc...`, CPU, FP32). It used the repository's own
  `encode_record`, `collate`, `pool_hidden`, `affinity_matrix`,
  `joint_group_bootstrap_indices` and `simultaneous_intervals`: 240 examples
  per task, max length 256, layers 7/14/21, content and answer pooling, and
  200 paired group-bootstrap draws. Numbers from it are labelled
  **[diagnostic]**. It is a small, pre-training measurement, not H2 evidence.

Hypothesis numbering: this repository's H2/H3/H4 are the draft's H1/H2/H3
(rank inversion / forward prediction / intervention). This document uses the
repository numbering.

---

## Contents

1. Summary table
2. Critical findings (C1-C3)
3. High-severity findings (H1-H4)
4. Medium findings (M1-M6)
5. Low findings and repository hygiene (L1-L6)
6. Checked and cleared (not problems)
7. Implementation order and a single, unattended cluster validation
8. Appendix A: diagnostic and smoke numbers
9. Appendix B: proposed folder layout

---

## 1. Summary table

| ID | Finding | Severity | Changes plan hash? |
|---|---|---|---|
| C1 | Frozen and head-only controls cannot fail; no real null control exists | Critical | Yes |
| C2 | Answer codes are no-space tokens; the base model never predicts them, so task 1 mostly learns output format | Critical | Yes |
| C3 | Runtime (~15-20 GPU-h/run) x 104 serial runs, plus ~104 manual laptop cycles | Critical | Yes |
| H1 | Rank-16 cutoff sits where the spectrum has no gap; estimator bias exceeds its standard error | High | Yes |
| H2 | The affinity is label-free input-variation overlap; nothing measures "conflict" | High | Yes |
| H3 | Inference is conditional on single checkpoints; training-seed variability is only descriptive | High | Analysis only |
| H4 | The decision margin (0.01) and learning gate are not calibrated to anything | High | Yes |
| M1 | Plan drift: synthetic regime study skipped; H3/H4 unimplemented; G1 passed without its measurement | Medium | n/a |
| M2 | Second data cohort overlaps the first but doubles cost and halves alpha | Medium | Yes |
| M3 | `--profile primary` without `--models` selects Qwen3-4B + 8B (8B cannot fit) | Medium | Builder only |
| M4 | Bumping the prompt version would crash preparation on existing token caches | Medium | Yes |
| M5 | Pause only happens inside training; walltime can kill evaluation | Medium | Yes |
| M6 | Stage 0 is recomputed identically in every LoRA run | Medium (cost) | Yes |
| L1 | Three overlapping code trees plus duplicated docs | Low | No |
| L2 | Mojibake (broken characters) in three docs | Low | No |
| L3 | Stale/placeholder strings in PBS scripts | Low | Yes (pbs is hashed) |
| L4 | `setup.pbs` can reinstall packages into the working environment | Low | Yes |
| L5 | Numbering mismatch between paper and repo | Low | No |
| L6 | Native smoke artifacts (run.json, reports) never recovered | Low | No |

Every item marked "Yes" changes the source inventory, so the next campaign
needs a **new plan** anyway. Batch all of them into one new protocol version
(call it v4) rather than fixing them one at a time.

---

## 2. Critical findings

### C1. The controls cannot fail, so they do not control anything scientific

**What.** The frozen run never trains. The head-only run trains only a linear
readout on top of an unchanged backbone (`engine.train_task` runs the backbone
under `no_grad` in eval mode). In both, the stage-0 and stage-5 features are
bit-identical, so D_0 == D_T exactly and an inversion is impossible by
construction. `suite_analysis._analyze` raises an error if a control shows an
inversion, but that can only happen through a bug or nondeterminism.

**Why it matters.** The draft's credibility argument is a control "in which
the hypothesis must fail". Here the frozen arm is an *instrument determinism
check*, not a falsification test. Three alternative explanations of any
detected inversion remain untested:

1. **Generic perturbation.** Any change of the representation of this size
   reorders pairs whose affinities were close, with no relation to the
   sequence or to the tasks.
2. **Generic fine-tuning.** Fine-tuning on these five tasks *at all* (in any
   order, or jointly) produces the same final ordering. Then the inversion is
   not "endogenous to sequential learning", which is the actual claim.
3. **Optimization without task structure.** LoRA updates driven by the
   answer-format loss alone produce the reordering.

**How to eradicate.**

1. **Rename** the frozen/head runs in reports to "instrument determinism
   checks". Keep them; they are useful, but they are not the control.
2. **Add a joint-training control (`mode="joint"`).** This is the most
   important one.
   - Training data: an interleaved mixture of all five train packs, shuffled
     with the run seed.
   - Budget: the same total number of optimizer updates as a sequential run
     (5 x per-task updates), one warmup plus cosine schedule over the whole
     budget, one optimizer that is never reset.
   - Checkpoints: stages 1..5 at 1/5, 2/5, ... of the joint budget. The stage
     layout, feature extraction and `analyze_run` stay unchanged.
   - Loss: `loss_and_scores(..., classes=4, ...)` works for a mixed
     microbatch, because the target is `codes[label]` and every label is below
     4. The restricted scores are not used for training.
   - Engine: add `train_joint(model, packs, cfg, ...)` beside `train_task`,
     reusing its update loop, progress checkpointing and RNG handling. In
     `build_model`, treat `joint` like `lora` (inject adapters).
   - Plan: in `suite.build`, add joint rows for the first 3 seeds with
     `order="O1"` as a placeholder (the order list is ignored by the joint
     trainer). Skip BWT for joint runs in `analyze_run`.
3. **Add a shuffled-label control (`mode="shuffled"`).** Identical to `lora`,
   except that training labels are permuted within each task. Use
   `np.random.default_rng(seed + 7_000_003 + stage_index).permutation(n)` and
   override `row["label"]` when building training batches. Evaluation and
   probes keep the true labels. This measures the geometry change produced by
   optimization and format learning without task structure. Run it for 3
   seeds on O1 and O2.
4. **Optional: a matched-norm random perturbation.** For each sequential LoRA
   run and stage, replace the adapter delta (B@A per module) with a random
   matrix of the same Frobenius norm, then extract features only (no
   training). This is cheap: forward passes only, about the cost of one
   evaluation stage.
5. **Change the eligibility rule** in `suite_analysis._analyze`. A sequential
   inversion (as tested in H3 below) is reported as:
   - *sequence-specific* only if the joint-control runs do not show the same
     signed inversion for that comparison (joint seed-level interval for D_T
     not on the same side of -m);
   - *learning-specific* only if the shuffled-label runs do not show it
     either.

   Pre-register both labels. Report all three categories; drop none.
6. **Tests to add** (in `tests/test_robust.py` style, CPU fixtures):
   - joint mode completes six stages, never resets the optimizer, and runs the
     exact update budget;
   - shuffled mode trains on permuted labels but evaluates on true labels;
   - an analysis fixture where a planted inversion appears in both sequential
     and joint runs is labelled "not sequence-specific".

---

### C2. The answer token is one the base model never produces

**What.** `prompts.label_token_ids` encodes the codes as `"A"`, `"B"`, `"C"`,
`"D"` with no leading space. After `"Answer:"`, a language model writes
`" A"` (with a space), which is a different token. Evidence:

- H100 smoke, frozen run, stage 0 (Yelp validation): `mean_target_nll =
  19.54`, `invalid_unrestricted_top1_rate = 1.0`, confusion `[[0,10],[0,10]]`.
  The model never puts a code first, and the restricted argmax is constant.
- H100 smoke, LoRA run: the first Yelp training loss is 19.19, which is
  format, not semantics.
- [diagnostic] Zero-shot accuracy with the pipeline's codes versus
  space-prefixed codes: Yelp 50.0% vs 63.8%, QNLI 54.2% vs 77.9%, MNLI 35.4%
  vs 48.8%, QQP 54.6% vs 54.6%, AG News 25.4% vs 24.2%. For Qwen3 the
  space-prefixed codes are single tokens (ids 362, 425, 356, 422).

**Why it matters.**

1. Whichever task is trained first, in every order, spends its first updates
   teaching the model to emit an unnatural token. That causes a large
   representation change shared by all orders, which is exactly the kind of
   shift that could reorder affinities and be misread as task-driven.
2. The learning audit (`suite.learning_audit`) compares NLL before and after
   each task. A drop from ~19 nats is guaranteed by format learning, so the
   gate passes without semantic learning.
3. Stage-0 accuracy, BWT and "learning" readings are all contaminated by the
   same artifact.

**How to eradicate.**

1. In `h2/prompts.py`, change `label_token_ids` to encode `" " + code`. Keep
   the single-token and distinctness checks, and check
   `tokenizer.decode(ids).strip() == code`. Record the scheme in the token
   manifest, for example `"label_codes": [" A", " B", " C", " D"]`.
2. Bump `PROMPT_VERSION` to `"h2-content-spans-v3"`.
3. **Also fix M4 in the same change.** Otherwise step 2 makes preparation
   crash on existing caches.
4. Add a calibration check (not a unit test, since it needs the real model):
   at stage 0 of the frozen control, `invalid_unrestricted_top1_rate` must be
   below 0.5 on at least 3 of 5 tasks. Enforce it in `suite._freeze` by
   reading the calibration plan's frozen-control stage-0 metrics, and fail
   the freeze with a clear message otherwise.
5. Unit test: with a fake tokenizer where `" A"` and `"A"` are different
   single tokens, `label_token_ids` returns the space-prefixed ids.
6. Record the change as a deviation in the research log: the H100 smoke used
   no-space codes.

---

### C3. The plan is infeasible to run unattended: ~2-3 months of serial GPU time and ~104 manual cycles

**What.**

- Smoke timing (H100 NVL MIG 1g.24gb, max length 256, microbatch 1): one LoRA
  task step (5 updates x 16 sequences, plus evaluation of ~500 sequences) took
  about 32 s. The same evaluation plus 80 forward passes without backward
  (head mode) took about 12 s. This implies roughly 0.25 s per training
  sequence (forward + backward with gradient checkpointing) and roughly 0.02 s
  per evaluation sequence.
- Primary plan per LoRA run: 5 tasks x 2,500 updates x 16 sequences = 200,000
  training sequences (about 14 h), plus about 130,000 evaluation and
  extraction passes (about 1 h). Max length 512 makes Yelp longer, so the
  estimate is **about 15-20 GPU-hours per run**. This is an extrapolation
  from a 5-update smoke; calibration must measure it.
- Compact mode forces strictly serial runs (`--limit 1`), and each run needs:
  GPU job, CPU pack job, laptop `archive_client.py` fetch and acknowledgement,
  CPU release job, next submission. Across 104 runs that is about 104 manual
  cycles and 1,500-2,000 GPU-hours, before queue waits.
- Runs close to 24 h hit `H2_MAX_MINUTES=1400`, exit with status 75, and need
  manual `--resume` submissions.

**Why it matters.** It contradicts the operating constraint (no one in the
loop). It also exhausts the time to the ICML window on the cheapest
hypothesis, before H3/H4 exist.

**How to eradicate.** Four independent levers; use all four.

**Lever 1: throughput (about 2-4x, to be measured).**

1. In `suite.build`, change `microbatch=1, gradient_accumulation=16` to
   `microbatch=4, gradient_accumulation=4`. The effective batch stays 16, so
   the optimization problem is unchanged up to padding numerics.
2. Change `eval_batch=1` to `eval_batch=16`.
3. Length-bucket evaluation and extraction without changing output order. In
   `engine.evaluate` and `engine.extract`, compute
   `order = np.argsort(lengths, kind="stable")`, process batches in that
   order, then scatter the results back with `out[order] = batch_results`
   before saving. Features must be saved in pack order, because
   `analyze_run` checks `ids` equality across stages.
4. Memory budget gate: `COMPLETE.json` already records
   `peak_allocated_bytes`. In `suite._freeze`, require the calibration run's
   peak to be below 0.8 x visible GPU memory (read from its `run.json`
   provenance hardware record). Qwen3-1.7B BF16 with LoRA, microbatch 4,
   length 512 and gradient checkpointing should stay well under 21.6 GiB.
5. These change `shape_signature`, so a new matching calibration is required,
   which the freeze already enforces.

**Lever 2: fewer runs, chosen by power instead of by grid.**

1. One data cohort (see M2).
2. Sequential LoRA runs: 5 seeds x 5 orders = 25. Increase seeds only if the
   calibration's seed-level variance (H3) shows that the seed-level interval
   would be wider than the margin.
3. Controls: joint x 3 seeds, shuffled x 3 seeds (C1), frozen x 1, head x 1.
4. Total: about 33 GPU runs instead of 104.

**Lever 3: unattended server-side compaction (removes the laptop from the loop).**

1. Add a storage policy `server_capsule` (plan field
   `storage.policy = "server_capsule"`) alongside the existing
   `all_declared_analyses_before_feature_removal`.
2. Under `server_capsule`, `compact.pack` runs exactly as now (all primary and
   sensitivity analyses, hashed capsule). Then:
   - It **retains on the server**: run.json, metrics, training events, all
     analysis outputs (`report.json`, `bootstrap.npz`, `affinities.npz`,
     CSVs), `compact_sensitivity.json`, and the **final-stage** adapter
     checkpoint (about 26 MB in FP32).
   - It removes: all feature `.npz` files and the non-final stage
     checkpoints.
   - Retained size is about 30-40 MB per run, about 1.3 GB for 33 runs.
3. In `compact.release`, accept a missing laptop acknowledgement only when
   `plan["storage"]["policy"] == "server_capsule"`. Everything else stays as
   it is: `load_capsule` verification before deletion, a sealed removal list,
   no recursive delete, `RELEASED.json`. Skip `shared.zip` under this policy
   (prepared data stays on the server).
4. New CPU action `pack_release` in `pbs/compact_cpu.pbs` that runs
   `compact.pack` and then `compact.release` in one job.
5. New command `suite.py submit-chain --name PLAN --runs K [--submit]`. For
   each of the next K unfinished registered runs it submits:
   - the GPU job with `-W depend=afterok:<previous CPU job>`;
   - then the CPU `pack_release` job with `-W depend=afterok:<that GPU job>`.

   Record every job ID in `submissions.json` before submitting the next one,
   as `_submit` already does. Default K = 4 (8 queued jobs), adjustable to the
   cluster's queued-job limit.
6. Relax the `--limit 1` and "no other workq job" checks in `_submit` only for
   `submit-chain` under `server_capsule`. The chain is serial by
   construction, and `can_submit` still blocks a GPU run while a completed run
   is unreleased.
7. The laptop archive becomes an optional single backup at the end
   (`archive_client.py` pulling the retained capsules once), not a per-run
   step.
8. Tests (extend `tests/test_compact.py`):
   - joint analysis results identical before and after an automatic release
     (the existing equality test, under the new policy);
   - `submit-chain` dry run emits the correct alternating `depend=afterok`
     chain and ledger;
   - release refuses without acknowledgement under the old policy and accepts
     under the new one.

**Lever 4: walltime safety (see M5).** With lever 1, runs should fit well
inside 24 h, so pauses become rare. When one happens, `afterok` stops the
chain safely, and one `submit-chain --resume` restarts it.

**Resulting operating model.** One command per K runs. With K = 8, about four
or five check-ins for the whole campaign.

---

## 3. High-severity findings

### H1. The rank-16 subspace boundary is ill-defined, and estimator bias exceeds its standard error

**What.** [diagnostic] At layer 14, content pooling, stage 0:

- The relative cutoff gap `(s16 - s17) / s16` is 0.006-0.034 across tasks
  (0.0011 for Yelp at layer 7). The configured gate,
  `relative_cutoff_gap_floor = 0.001`, almost never binds.
- Top-16 explained variance is only 44-58%.
- Split-half self-affinity (the same task, two disjoint halves of 120
  examples) is only about 0.43-0.50, against 0.13-0.28 between different
  tasks.
- The median |bootstrap bias| of the 45 pair differences is about
  **1.7-2.2 x their bootstrap SE**.
- The empirical max-standardized critical value over 45 comparisons
  (alpha 0.025) was **6.3**, against 3.45 under normal Bonferroni. The
  bootstrap distribution is shifted, not just spread.

**Why it matters.**

1. When eigenvalues 16 and 17 are nearly equal, which directions make up "the
   top-16 subspace" is close to arbitrary. Small changes swap directions in or
   out, and the overlap jumps.
2. The bias of a finite-sample subspace overlap depends on the spectrum.
   Training that sharpens or flattens a task's spectrum changes the bias
   between stage 0 and stage 5, so D_T - D_0 can move without any change in
   the population geometry. That can manufacture or mask an inversion.
3. The current intervals (`point +/- q*se`) are centred on the biased point
   estimate. The bias inflates q, so the test is conservative and wastes
   power, but it is not centred on the right quantity.
4. Part of the bias comes from the resampling scheme itself. Ordinary
   bootstrap duplicates rows, which lowers the effective sample size of each
   PCA by about a third and biases bootstrap affinities in one direction.

**How to eradicate.**

1. **Choose r where the spectrum has a gap, on development data only.**
   - Add `select_rank(stage0_features, window=(8, 48))` to `h2/geometry.py`.
     For each candidate r, compute the mean relative gap
     `(s_r - s_{r+1}) / s_r` across the five tasks (from one SVD per task)
     and take the argmax.
   - Run it on the calibration plan's frozen-control stage-0 Probe A
     features, which are development-partition data. Write the chosen r and
     the gap profile into the new plan before freezing.
   - This implements the draft's own rule ("r fixed once on pilot data and
     frozen").
   - Raise `relative_cutoff_gap_floor` to a meaningful level, for example
     0.05. Report comparisons that fail it, but exclude them from eligibility.
2. **Replace duplicate-row resampling with a Bayesian (weighted) group bootstrap.**
   - Draw one weight per global source group, `w_g ~ Gamma(1, 1)`, so the
     same group weight is shared across tasks and stages, exactly as the
     current group-multiplicity sharing.
   - Within a task, normalize the row weights to sum to 1. Compute the
     weighted mean `mu_w`, then the weighted SVD of
     `diag(sqrt(w)) @ (X - mu_w)`.
   - Implementation: an optional `weights` argument on `subspace()` and
     `affinity_matrix()`, plus
     `joint_group_bootstrap_weights(task_groups, rng)` mirroring
     `joint_group_bootstrap_indices`.
   - No duplicated rows means no artificial loss of effective sample size.
3. **Bias-correct before forming intervals.** In `analyze_run` and
   `simultaneous_intervals`, use `D_bc = 2*D_hat - mean(D_star)` as the
   centre and `(D_star - mean(D_star)) / se` for the max-standardized
   quantile. Store both `point` and `point_bias_corrected` in
   `bootstrap.npz`. Report the per-comparison `bias / se`; if it exceeds 1 for
   a comparison, label that comparison "estimator-dominated" in the joint
   table.
4. **Use the gap-free estimator as a pre-declared agreement check.**
   `covariance_affinity_matrix` (normalized Frobenius overlap, no rank) is
   already implemented and already computed as a sensitivity view. Pre-register
   that a primary inversion is reported as "estimator-robust" only if the
   covariance view has the same signed raw inversion.
5. **Validate the instrument at realistic difficulty.** The current
   `validate_statistics.py` uses n = 80 and a planted angle swap from 0.25 to
   1.2 rad (affinity 0.94 to 0.13), which is far easier than reality. Replace
   it with a generator built from **empirical stage-0 covariances**:
   - From the calibration frozen control, sample Gaussian data with each
     task's sample covariance at n = 1000.
   - *Null:* both endpoints are independent samples from the same
     covariances. Measure the false inversion rate (target at most alpha) and
     simultaneous coverage.
   - *Spectral-sharpening null:* the stage-5 covariances have the same
     eigenvectors but sharpened eigenvalues (for example
     `lambda_k^1.2`, renormalized). This is the specific bias failure mode;
     the false inversion rate must stay at most alpha.
   - *Planted:* rotate a few directions of one task so that one pair
     difference flips by 0.05, 0.075 and 0.10. Measure power.
   - Run once on `cpuq` (e.g. 200 repetitions x 499 draws) and commit the JSON
     next to the plan.
6. **Increase probe size if the preparation can supply it.** Bias shrinks
   roughly like 1/n. Probe B at 2,000 rows per task halves it. Check the
   QNLI quota under the confirmation partition first (QNLI is the tightest
   task).

---

### H2. The measured "affinity" cannot support the claim the paper attaches to it

**What.** `A_t(i,j)` is the overlap of the top-r principal directions of
centred, content-pooled hidden states. It ignores labels: it describes how
the *inputs* of task i and task j vary in representation space. Nothing in
the pipeline measures interference or conflict between tasks. The draft's
motivation for rank inversion is that the static measure is "wrong about which
pairs conflict". The draft's own measurement table lists gradient alignment
and representation discrepancy (Kim et al. 2025, marked "required") as
secondary measures; neither is implemented.

**Why it matters.** A positive result would only show that "the ordering of
a geometric statistic of input representations changed". A reviewer can
reasonably answer that representations change under fine-tuning, so some
statistic's ordering will change, and ask what that has to do with
interference. The "conflict" link is the scientific content, and it is
unmeasured.

**How to eradicate.**

1. **Add a feature-space gradient-alignment matrix at every stage**
   (cheap, and independent of LoRA initialization).
   - For each task, take a fixed subset of Probe A (for example 256 rows).
   - For each example, compute the gradient of the answer loss with respect to
     the hidden state at the primary layer, at the content tokens and the
     answer position. Use a backward hook on
     `model.model.layers[L-1]` output with the base weights frozen; adapter
     parameters do not need gradients.
   - Pool it with the same content mask, and average over examples to get
     the task gradient direction `g_i` (2048-d).
   - Compute `G_t(i,j) = cos(g_i, g_j)`.
   - Cost: 5 x 256 forward+backward passes per stage, about 7,700 per run,
     under 5% overhead.
   - Store `G_t`, gradient norms and per-example cosine dispersion in the
     stage directory (a few KB). Do not store the gradients.
2. **Add parameter-gradient alignment on LoRA B matrices** as a second view
   (at stage 0, A is random and B is zero, so gradients exist only for B and
   are seed-dependent; report it as seed-conditional).
3. **Compute a measured interference matrix from data that already exists.**
   From the per-stage test accuracies:
   `I(i <- j) = acc_i(before training j) - acc_i(after training j)` for
   every ordered pair where i was trained before j. Across 5 orders, each
   ordered pair occurs in several histories. Implement in `analyze_run` as an
   extra CSV, then aggregate in `suite_analysis`.
4. **Pre-register the link test (secondary).** For each eligible affinity
   inversion (P, Q), report whether the gradient-alignment ordering of P and Q
   inverts between the same endpoints, and whether the measured interference
   ordering agrees. This turns "an ordering changed" into "an ordering that
   tracks interference changed".
5. **Representation discrepancy (Kim et al. 2025).** Either implement it as a
   secondary measure from the saved stage features (it needs only features and
   a linear readout fit), or record in the protocol that it is dropped, and
   why. Do not leave the draft promising it silently.
6. **Rewrite the claim in the paper and the plan** as "inversion of the
   ordering of representation-subspace overlap", unless step 4 provides the
   conflict link.

---

### H3. The statistical test targets the wrong source of randomness

**What.** Intervals in `analyze_run` and `suite_analysis._analyze` cover only
probe-sampling noise, conditional on one trained checkpoint. Seed variation,
which is what a claim about "sequential learning" generalizes over, enters
only through the descriptive `replication_counts` rule (at least 80% of seeds
in at least 3 orders). Also:

- The stage-0 difference D_0 is identical in every LoRA run of a cell (same
  base model, same probes, same bootstrap draws), but it is entered 100 times
  into the joint family.
- The initial and final endpoints are covered simultaneously, although an
  inversion is an AND of two one-sided conditions.

**Why it matters.** An "eligible reversal" currently means that this
particular trained model's population ordering differs from the base model's.
The scientifically relevant statement, that the training procedure in this
order produces the inversion, is never tested inferentially. The descriptive
replication rule then demands per-run detections that are individually very
stringent, which wastes power.

**How to eradicate.**

1. **New primary test: seed-level, per order, intersection-union.** Implement
   it in `h2/suite_analysis.py` as `seed_level_test(point, boot, rows, cfg)`.
   It uses only the already retained `bootstrap.npz` arrays, so no GPU work
   changes.
   - Initial endpoint (one value per cell): `D_0[k]` with the probe
     bootstrap; max-standardized interval over the K = 45 comparisons at
     level alpha_cell.
   - Final endpoint, per order o: the seed mean `Dbar_T[o,k]` over that
     order's S seeds. Its uncertainty comes from a two-level bootstrap. For
     b = 1..B2, resample S seeds with replacement and take one shared probe
     draw index p (draws are already aligned across runs), then
     `Dbar_star[b,o,k] = mean over chosen seeds of boot[p, run, 1, k]`.
     Max-standardized interval over all (o, k) at level alpha_cell.
   - Inversion for (o, k): `lower(D_0) > m` and `upper(Dbar_T) < -m`, or the
     mirror. By the intersection-union principle, each endpoint is tested at
     its own level with its own family; no joint coverage across the two
     endpoints is needed.
   - Order-general inversion: the same test with seeds pooled over all
     orders, reported separately.
2. **Keep the per-run, probe-conditional results** as descriptive tables,
   relabelled "checkpoint-conditional".
3. **Remove the duplicated D_0 entries** from any joint family (use the cell's
   single D_0).
4. **Size the seed count from calibration.** Run the calibration with 3 seeds
   on one order, estimate the seed SD of D_T, and choose S so that the
   expected half-width of the seed-level interval is below m/2. Record the
   calculation in the plan.
5. **Tests:** fixtures where (a) every seed shows the inversion, so it is
   detected; (b) the seeds split 50/50, so it is not detected although some
   single runs are; (c) controls are excluded from the family.

---

### H4. Decision thresholds are not calibrated

**What.**

- `min_gap = 0.01` is fixed without reference to the affinity scale.
  [diagnostic] The median |D_0| is 0.07 and the pair affinities range
  0.12-0.35.
- The learning gate passes with 2 of 5 tasks showing an NLL gain above 0.01
  and accuracy at least 5 points above chance. Because of C2, the NLL part is
  automatic.

**How to eradicate.**

1. **Margin m.** Pre-register a substantive rule from development data, for
   example `m = 0.25 x median |D_0|` of the calibration frozen control (about
   0.018 from the diagnostic). Alternatively, use the 95th percentile of
   |D_T - D_0| in the calibration shuffled-label runs, so an inversion must
   exceed what label-free optimization produces. Write m and its derivation
   into the plan before freezing.
2. **Learning gate.** After fixing C2, require accuracy gain of at least
   0.05 over the pre-task stage *and* post-task accuracy of at least
   chance + 0.10 on at least 3 of 5 tasks. Take the exact values from the
   calibration distribution, not from confirmation runs.
3. **Add the gate outcome per task** to `joint_pair_comparisons.csv`, so
   readers can see which tasks were actually learned.

---

## 4. Medium findings

### M1. The project drifted from its own plan

**What.**

- The draft's Phase 1 is a tiny synthetic teacher-student rig with a
  lazy-to-rich "richness" dial. In that rig, the frozen-feature arm is a
  meaningful control and the regime contrast is the decisive test. It was
  skipped.
- The LLM study has a single regime (LoRA, lr 1e-4).
- H3 (forward prediction) and H4 (the null-subspace intervention, the plan's
  "spine") are not implemented.
- Gate G1 (13 Oct) passed without its measurement; G2 is 10 Nov.

**How to eradicate.**

1. **Build the synthetic rig in a new `synthetic/` folder.** It runs on a
   laptop GPU in minutes.
   - Data: inputs `x ~ N(0, I_d)` with d = 100. Five tasks
     `y_i = sign(w_i . x)`, with the angles between the `w_i` chosen so that
     the initial pair ordering is known.
   - Model: a 2-layer MLP (width 512).
   - Richness dial: Chizat-style output scaling
     `f = alpha * (net(x) - net_0(x))` with learning rate scaled by
     1/alpha^2. Use alpha in {100 (lazy), 10, 1, 0.1 (rich)}, plus a
     frozen-features arm (only the readout trains).
   - Training and measurement: train sequentially over the same 5 orders.
     Measure A_t with the same `h2/geometry.py` functions on the hidden
     layer; import them, do not copy them.
   - Pre-register in `synthetic/README.md` before running. Gate: inversions
     present in rich, absent in frozen and lazy.
2. **Prototype H4 in the same rig.**
   - Estimate the old-readout Jacobian on old-task data, take the
     near-null projector P_perp, and apply `z -> z + (lambda - 1) P_perp z`,
     folded into the last hidden layer's weights.
   - Enforce the silence gate on held-out old-task loss, accuracy and KL.
   - Run the four controls: no-op, random subspace, functional subspace, and
     the exact compensated gauge (rotation plus readout).
   - Only then port it to the LLM.
3. **Update the gate dates** in `../field-view/docs/02-revised-plan.md` with a
   dated deviation note instead of silently moving them.

### M2. The second data cohort is not an independent replication

**What.** `suite.build` uses sample seeds 20261031 and 20261107 with the same
`stable_split_seed`, so both cohorts draw from the same role pools and can
share rows (documented in `data_sampling_note`). Each cohort is a separate
cell, so Bonferroni halves the per-cell alpha and the GPU budget doubles.

**How to eradicate.**

1. Use one cohort for the primary plan.
2. Do replication later, either on a **different model** (`smollm1.7b` is
   already in `MODELS`), which tests generality more strongly, or on a
   disjoint cohort. For a disjoint cohort, add a cohort bucket to the role
   hash in `robust_data.group_role`, for example
   `sha256(f"{split_seed}:{group_id}")` mod 2 assigns each group to exactly
   one cohort.

### M3. Default model selection is a footgun

**What.** `suite.build(... profile="primary")` without `--models` selects
`["qwen4b", "qwen8b"]`. Qwen3-8B has `floor = 36` GiB, which exceeds the
visible MIG memory, and preparation would try to download about 16 GB onto a
filesystem last seen with about 7.5 GiB free.

**How to eradicate.**

1. Make `--models` required for every profile except `smoke` (argparse
   `required=True` conditionally, or raise in `build`).
2. Add a build-time preflight: read `H2_GPU_GIB` from the environment or
   `settings.env`, and refuse any model whose `floor` exceeds it.
3. In `suite.prepare`, before `prepare_model`, check free disk against
   `billions x 2 GB x 1.1` for models that are not yet cached.

### M4. Changing the prompt version would crash preparation on existing caches

**What.** `common.token_dir` derives the cache path from the data recipe,
repo, revision, max length and prompt variant, but **not** from
`PROMPT_VERSION` or the label-code scheme. `prepare.prepare_tokens` compares
an identity that *does* include `prompt_version`, and raises "Token cache
identity mismatch" when the directory already exists. After C2's fix, any
cell whose data recipe is unchanged would hit the old cache directory and
fail. `bootstrap_compact.py` links the old `data/` folder, so this is likely.

**How to eradicate.**

1. Add `"prompt_version": PROMPT_VERSION` and `"label_codes": ...` to the
   hashed key in `common.token_dir`. Import `PROMPT_VERSION` inside the
   function to avoid a circular import.
2. Test: two configs that differ only in prompt version map to different
   token directories, and preparing the second does not touch the first.

### M5. Walltime can kill evaluation, and pauses stop the chain

**What.** `checkpoints.pause_requested` is checked only inside the
`train_task` update loop. If the time limit arrives during a stage's
evaluation or extraction, the job is killed at walltime. Resume is safe (the
`.tmp` stage is moved to `aborted/` and recomputed), but the work is lost.
When a pause does happen (exit 75), an `afterok` chain stops.

**How to eradicate.**

1. In `engine.execute_run`, record each stage's evaluation-plus-extraction
   duration in the stage `COMPLETE.json`.
2. Before starting the next stage, compute `remaining = H2_MAX_MINUTES*60 -
   elapsed`. If `remaining < 1.5 x last evaluation duration + estimated
   training time`, raise `RunPaused` at the boundary (nothing is lost).
3. Set `H2_MAX_MINUTES` from the actual PBS walltime automatically
   (`PBS_WALLTIME` or `qstat -f` in `common.sh`) instead of by hand, minus
   a 30-minute margin.
4. `submit-chain --resume` (C3) resubmits the paused run and the rest of the
   chain with one command.

### M6. Stage 0 is recomputed identically in every LoRA run

**What.** With `lora_B` initialized to zero and dropout off in evaluation,
stage 0 of every LoRA run equals the frozen control's stage 0: same metrics,
same features. It is recomputed (about 25,000 passes) in every run.

**How to eradicate (optional; saves about 1/6 of evaluation, small next to
training).**

1. Let LoRA runs reuse the cell's frozen-control stage 0: copy its
   `metrics.json` and the primary-view features with hash verification, and
   record the source run and hashes in the stage `COMPLETE.json`.
2. Keep a one-batch spot check (recompute 8 examples per task and compare)
   to prove equality on the actual hardware.
3. Lower priority than C3 levers 1-3. Do it only if calibration shows
   evaluation is a significant share of the runtime.

---

## 5. Low findings and repository hygiene

### L1. Three overlapping code trees

**What.** `h2_full_experiment/` (v2), `h2_robust_experiment_v3/` and
`h2_compact_experiment/` are near-duplicates; `engine.py` differs between v3
and compact by four lines. `h2_compact_experiment/legacy_v3_docs/` duplicates
the v3 docs. It is easy to edit or submit from the wrong copy.

**How to eradicate.** Git history already preserves every version
(commits `47054fe` and `3f0df31`).

1. Tag the snapshots: `git tag h2-v2 47054fe` and `git tag h2-v3 3f0df31`.
2. Move `h2_full_experiment/`, `h2_robust_experiment_v3/` and `legacy/` under
   `archive/`, each with a one-line README naming its tag. Delete
   `h2_compact_experiment/legacy_v3_docs/`; the archived v3 folder has the
   same files.
3. Rename the canonical tree to `h2/` (see Appendix B), or keep its name and
   state it as canonical in the top README.
4. Server checkouts are separate folders; moving folders in git does not
   affect them. The new v4 plan is built in a fresh server folder anyway.

### L2. Broken characters in documentation

**What.** `docs/EXPERIMENT_RECORD_2026-10-27.md` (6 lines),
`docs/H2_H3_H4_Research_Log.md` (28 lines) and
`docs/H2_V3_CODE_EXPLAINED.md` (1 line) contain UTF-8 text that was decoded
as Windows-1252 and saved again (for example a long dash shown as three
garbage characters, a multiplication sign shown as two).

**How to eradicate.**

1. Repair each affected line with
   `line.encode("cp1252").decode("utf-8")`, applied only to lines that
   contain the garbage sequences, or use the `ftfy` package's `fix_text`.
2. Re-open the files to check that the dashes and multiplication signs are
   back.
3. `provenance/local_artifact_inventory.json` holds the original research-log
   hash. Add an entry to `provenance/publication_transformations.json`
   recording the repair, so the hash change is explained.
4. Write future docs in plain ASCII, or make sure the editor saves UTF-8
   without re-encoding.

### L3. Stale and placeholder strings in PBS scripts

**What.**

- `pbs/common.sh` line 6 says "Submit from the h2_full_experiment folder".
- `HF_HOME` defaults to `/Data4/YOUR_HPC_USER/.hf_cache`, so a
  `settings.env` without `HF_HOME` silently points at a nonexistent cache.
- `archive_client.py` and `status.py` contain placeholder paths and users.

**How to eradicate.**

1. Change the message to "Submit from the experiment folder (h2/engine.py
   not found)".
2. Replace the default with a hard requirement:
   `: "${HF_HOME:?Set HF_HOME in settings.env}"`.
3. Make `--remote-python` required in `archive_client.py` (no default).
4. In `status.py`, use `getpass.getuser()`.

### L4. `setup.pbs` can modify the working environment

**What.** It runs `pip install -r requirements.txt` with exact pins.

**How to eradicate.** Add a guard at the top:
`[[ "${H2_ALLOW_PIP:-0}" == 1 ]] || { echo "Refusing to pip install; use compact_tests.pbs"; exit 2; }`.
Point all guides to `compact_tests.pbs` for environment checks.

### L5. Hypothesis numbering differs between paper and repository

**How to eradicate.** Add a mapping table to the top `README.md` and to
`../field-view/README.md`:

| field-view paper | this repo | meaning |
|---|---|---|
| H1 | H2 | rank inversion |
| H2 | H3 | forward predictive dominance |
| H3 | H4 | function-preserving intervention |

Then pick one numbering for the paper and use it everywhere from v4 onwards.

### L6. The native smoke artifacts were never recovered

**What.** The v3 smoke ran in retained mode, so its `run.json`,
`FROZEN.json`, `cell_report.json` and `joint_pair_comparisons.csv` should
still be on the server, unless they have been deleted. They are needed to
know *why* zero reversals were eligible (stable ranking, failed uncertainty,
or failed gates).

**How to eradicate.** Copy
`runs/h2v3_smoke_r1/` (excluding `features/` if space is tight),
`reports/h2v3_smoke_r1/` and `plans/h2v3_smoke_r1/` from the v3 server folder
into `results/h2v3_smoke_r1/native/`, and update the experiment record. Do
this before any cleanup of the v3 folder.

---

## 6. Checked and cleared (not problems)

| Concern | Result |
|---|---|
| Outlier ("rogue") hidden dimensions dominate the PCA subspaces | **Not the case** for content pooling: [diagnostic] the top 5 of 2,048 coordinates hold 5-7% of variance, and PC energy on them is 7-11%. Pair ordering raw versus per-coordinate standardized: Spearman 0.92 (layer 14), 0.96 (layer 7), 0.99 (layer 21). |
| Multiplicity correction is the main power bottleneck | **No.** [diagnostic] Probe-level SE of D at n = 1000 is about 0.0055. The cell-wide family raises the normal critical value from 3.6 to 4.5, costing about 0.005 in detectable difference. Detection needs \|D\| above roughly 0.035-0.05 at each endpoint, and about 75% of comparisons already clear that at stage 0. Power is governed by how far training moves the geometry, and by the bias problem in H1. |
| GPU memory | Qwen3-1.7B BF16 + FP32 LoRA, microbatch 1, gradient checkpointing, `logits_to_keep=1`: far below the 21.6 GiB visible in the smoke (or 29 GB). Microbatch 4 should also fit; calibration confirms it. |
| Last-token readout with padding | Left padding plus position ids from the attention-mask cumsum: the last position is the "Answer:" slot. Correct. |
| LoRA start state | `lora_B` is zero-initialized; tests confirm stage 0 equals the base model. Correct. |
| Data leakage | Grouped exact and near-duplicate splits, a development/confirmation hash partition, and cross-split audits. Correct as designed (not a semantic or pretraining-contamination audit, as documented). |
| Resume correctness | Optimizer-step and stage-boundary resume with RNG restore are covered by tests and match uninterrupted training. |
| Determinism | `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set in `common.sh`, and `use_deterministic_algorithms(True)` ran successfully on the H100. |

---

## 7. Implementation order and a single, unattended cluster validation

### 7.1 Order (one protocol version, v4)

1. **Laptop only, no cluster:**
   - L1, L2, L5 (hygiene; no plan impact);
   - M1 step 1 (the synthetic rig); run it and pre-register first. Its result
     decides whether the LLM campaign is worth its cost.
2. **Code changes in one branch** (each with the listed tests):
   - C2 + M4 (answer codes and cache key);
   - C1 (joint and shuffled modes);
   - H2 (gradient alignment, interference matrix);
   - H1 (Bayesian bootstrap, bias correction, `select_rank`, gap floor);
   - H3 (seed-level test);
   - H4 (calibrated thresholds; the values themselves come from calibration);
   - C3 levers 1 and 3 (throughput, `server_capsule`, `submit-chain`);
   - M5 (boundary pause);
   - M3, L3, L4 (guards).
3. **Local gate:** the full test suite passes, including the new tests listed
   under each finding, and `verify_package.py` passes after regenerating
   `SOURCE_MANIFEST.json`.
4. **Upgraded `validate_statistics.py`** (H1 step 5) runs once on `cpuq`
   after the calibration below. Its null false-inversion rate must be at most
   alpha before the primary plan is frozen.

### 7.2 One-command cluster validation

Add `suite.py validate-chain --name v4_smoke`. It builds a smoke plan
(`server_capsule`, Qwen3-1.7B, 5 updates per task) with one run each of
frozen, head, joint, shuffled and LoRA, and submits one dependency chain:

```
compact_tests (cpuq)
  -> prepare (cpuq)
  -> freeze (cpuq)
  -> [GPU run -> pack_release (cpuq)] x 5
  -> analyze (cpuq)
  -> verdict (cpuq): writes reports/v4_smoke/VERDICT.json
```

`VERDICT.json` reports pass/fail for:

- all jobs exited 0;
- frozen stage-0 `invalid_unrestricted_top1_rate` < 0.5 on at least 3 tasks;
- the determinism checks pass;
- every capsule is verified and features were released;
- the joint and shuffled runs completed six stages;
- gradient-alignment matrices are present for every stage;
- the seed-level test output exists;
- peak GPU memory and seconds per update are recorded.

You check one file once. Then the same chain shape runs the calibration
(3 seeds; it feeds r, m, the learning-gate values and the seed count) and,
after the freeze, the primary in chunks of K runs via `submit-chain`.

### 7.3 Expected campaign size after the fixes

| | Current plan | After fixes |
|---|---:|---:|
| GPU runs | 104 | about 33 |
| Hours per LoRA run | about 15-20 (estimate) | about 5-8 (estimate; calibration measures it) |
| Serial GPU-hours | about 1,500-2,000 | about 200-250 |
| Manual cycles | about 104 laptop round-trips | about 5 chain submissions |
| Server retention | features + checkpoints until laptop acknowledgement | about 40 MB per run |

---

## 8. Appendix A: diagnostic and smoke numbers

### A.1 Stage-0 diagnostic (Qwen3-1.7B, n = 240 per task, rank 16)

| View | Top-5 coord variance share | Relative gap s16->s17 | Split-half self-affinity | Cross-task affinity (same n) | Median \|D_0\| | Median SE(D), n=240 | Projected SE, n=1000 | Median \|bias\|/SE | Empirical maxT (45) |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| content, layer 14 (primary) | 0.050-0.075 | 0.006-0.034 | 0.43-0.50 | 0.13-0.28 | 0.073 | 0.0111 | 0.0054 | 1.69 | 6.35 |
| content, layer 7 | 0.041-0.071 | 0.001-0.046 | 0.37-0.44 | 0.10-0.22 | 0.069 | 0.0119 | 0.0059 | 2.15 | 6.61 |
| content, layer 21 | 0.036-0.066 | 0.015-0.028 | 0.41-0.51 | 0.09-0.22 | 0.071 | 0.0118 | 0.0058 | 1.78 | 6.03 |
| answer, layer 14 | 0.025-0.074 | 0.020-0.064 | 0.57-0.66 | 0.10-0.27 | 0.032 | 0.0072 | 0.0035 | 0.84 | 4.56 |

The answer-position view has better-defined subspaces (top-16 explains
69-85% of variance) and smaller bias. That supports keeping it as the
pre-declared secondary view, and examining it in calibration as a candidate
primary.

### A.2 Zero-shot accuracy at stage 0 (restricted to the task's codes)

| Task | Codes "A".."D" (pipeline) | Codes " A".." D" |
|---|---:|---:|
| Yelp | 0.500 | 0.638 |
| QQP | 0.546 | 0.546 |
| QNLI | 0.542 | 0.779 |
| MNLI | 0.354 | 0.488 |
| AG News | 0.254 | 0.242 |

### A.3 H100 smoke readings used

- Frozen stage 0, Yelp validation: NLL 19.54, invalid top-1 rate 1.0,
  confusion [[0,10],[0,10]].
- LoRA, first Yelp update: loss 19.19, gradient norm 55.0.
- Timing: LoRA task cycles at 01:40:49, 01:41:19, 01:41:52, 01:42:24,
  01:42:56 (about 32 s each). Head-only task cycles about 12 s each.
- Hardware: H100 NVL MIG 1g.24gb, 21.625 GiB visible.

---

## 9. Appendix B: proposed folder layout

```
cl-geometry-experiments/
  README.md                 canonical entry point; hypothesis mapping table
  docs/
    PROTOCOL_v4.md          frozen protocol text (generated from the plan)
    H2_H3_H4_Research_Log.md
    EXPERIMENT_RECORD_*.md
    H2_REVIEW_FINDINGS_AND_FIXES.md
  h2/                       the single canonical LLM pipeline (from h2_compact_experiment)
    h2/ suite.py compact.py pbs/ configs/ tests/ ...
  synthetic/                Phase-1 regime rig and H4 prototype (imports h2/h2/geometry.py)
  results/                  preserved evidence (smoke, validation, native artifacts)
  provenance/               hashes and transformation records
  archive/
    h2_v2/                  = tag h2-v2
    h2_v3/                  = tag h2-v3
    legacy_testbed/
```
