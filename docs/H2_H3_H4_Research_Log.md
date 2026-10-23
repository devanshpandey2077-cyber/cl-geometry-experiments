> Public copy: local account/server paths are generalized. Original-source hashes are retained in provenance/local_artifact_inventory.json.

# H2, H3 and H4 — Empirical Research Log

Created: 23 October 2026. Time zone for human entries: Asia/Calcutta (UTC+05:30).
Record version: 1.2. Initial event: EVT-000001. Latest event: EVT-000003.
Owner: the researcher. Scope: continual learning, representation geometry and forgetting.

**Current position:** the notebook is active; original historical pilot files were inspected in EVT-000003, and GitHub publication is being prepared. No new training, hypothesis test or causal intervention was executed while creating it. Historical evidence is imported below with its provenance and limitations. H2, H3 and H4 remain undetermined by the evidence reviewed for this entry.

## 1. Purpose and how we will maintain this record

This is the working record for the experiments discussed in this chat. For each run, preserve what question was asked, why the settings were chosen, what actually ran, every supplied reading, how it was calculated, what failed, what the result supports, and what remains unknown. Preserve unsuccessful and contradictory findings as carefully as successful ones.

When the researcher supplies a command, configuration, output, screenshot, metric file, code change, interpretation or decision, append a dated event and update the relevant run record and status table. A discussion or proposed command is a planning event until execution evidence arrives. A queued job is not a completed experiment. A functioning pipeline is not a confirmed hypothesis.

This notebook does not monitor the H100 server or capture files automatically. Only evidence supplied in this chat or actually inspected can be recorded. If a value is missing, mark it UNKNOWN or NOT PROVIDED and name the missing artifact. Do not reconstruct missing readings, silently fill defaults, or promise every future message will contain enough information to reproduce a run. Large raw arrays, checkpoints and per-example outputs should be preserved as separate original artifacts and indexed here; a prose summary does not replace them.

For each research exchange, use this order:

1. Identify the campaign, protocol and run, or create a provisional event if identity is unclear.
2. Preserve the supplied evidence and its source before interpreting it.
3. Enter exact values, units, task/checkpoint identities and missing fields.
4. Record calculations separately, including formulas and source readings.
5. Record interpretation, uncertainty, deviations and the next decision.
6. Save the updated notebook. If saving fails, explicitly report that the new entry is not yet persisted.

At the start of a later session, open the latest saved file with this title, use its current contents, continue the IDs, and preserve the existing file identity and version history. Do not restart the notebook or assume the old chat summary is authoritative.

### 1.1 Evidence vocabulary

| Label | Meaning |
|---|---|
| DIRECT ARTIFACT | Bytes/text inspected during the current documentation work; provenance and scope still matter |
| REPORTED HISTORICAL | A previous report describes the observation; the original execution artifact has not been rechecked here |
| USER REPORT | Researcher states an observation; raw evidence is absent or incomplete |
| DERIVED | Calculated from identified readings using a recorded formula |
| PROPOSED | A design or setting; no claim it executed |
| UNKNOWN | Value or fact not established |
| NOT APPLICABLE | Field does not apply, with reason |

Never use zero for missing data. Never turn a proposed target into an observed measurement. Treat screenshots as partial evidence: transcribe values with units and retain the image reference; mark unreadable characters as uncertain rather than guessing.

### 1.2 IDs, timestamps and correction policy

Use EVT-000001 onward for events, SRC-001 onward for sources, DEC-001 onward for decisions, DEV-001 onward for deviations, and ISSUE-001 onward for unresolved issues. A campaign can be H2-C001, H3-C001 or H4-C001. Runs can be H2-C001-R001, and analyses H2-C001-R001-A001. An H4 branch adds an arm ID to its shared parent checkpoint.

Each launch attempt gets its own attempt ID. A resumed attempt links to its earlier attempt and remains the same training trajectory. A fresh independent training launch gets a new run ID. One run may inform several hypotheses, but cite its single ID instead of counting it repeatedly as independent evidence.

Record event time, execution start/end and time of receipt separately. Use ISO 8601 with an explicit offset when known, for example YYYY-MM-DDTHH:MM:SS+05:30 or UTC with Z. If the time is unknown, record only the known date and say so. A file modification time is not automatically the experiment execution time.

Append dated corrections that cite the original entry, old value, corrected value, reason and source. Keep the original observation accessible. Summary/status tables can be updated, but preserve an event explaining any changed conclusion. Never rewrite an old protocol as if it had been chosen before seeing results.

## 2. What empirical testing can establish

Empirical testing means testing through observations and experiments; that is the activity being documented here. Mathematical derivation is a separate theoretical stage. Repeated tests strengthen evidence, estimate variability and reveal boundary conditions. They cannot guarantee the absence of every methodological gap or prove a universal claim over all models and datasets.

Varying a parameter tests robustness or a scientific factor; repeating a fixed setting with fresh randomness estimates variability. Reanalyzing the same features, resuming the same job, rerunning a deterministic seed, taking many checkpoints, and increasing bootstrap draws are not additional independently trained replications. Record all of them, with the correct dependence.

A negative result means the specified effect was not demonstrated under the tested settings and resolution. A failed instrument, failed job, failed control and wide confidence interval are different outcomes. An absence claim needs a justified resolution/equivalence bound, sufficient precision and appropriate controls.

## 3. Hypothesis registry

Definitions are grounded in SRC-001, Section 5, and qualified by SRC-002. These are register version 1; changes require a new version, rationale, date and affected campaign IDs.

### H2-v1 — Rank inversion

Question: can the relative affinity ordering of two task pairs reverse during training?

Let A_t(i,j) be the specified representation-affinity measurement for tasks i and j at checkpoint t. For pair P=(i,j), pair Q=(k,l), define D_t(P,Q)=A_t(P)-A_t(Q). A raw reversal between declared endpoints is D_0 × D_T < 0. An uncertainty-supported, practically separated reversal requires stronger predeclared criteria.

The existing version-2 H2 design uses mean squared principal-angle cosine between the top-r feature-coordinate subspaces of centered task representations:

    X_i: fixed-probe rows × representation coordinates
    X_i,c = X_i - column_mean(X_i)
    V_i: top-r right singular vectors of X_i,c, as d × r columns
    A_t(i,j) = ||V_i^T V_j||_F^2 / r

Record r, layer, pooling, centering, probe identities and numerical settings. This estimator discards task mean shifts and may reflect nuisance variation. Changing the estimator changes the operational claim. Shared orthogonal coordinate changes preserve this overlap; uniform rescaling need not change it. A geometry intervention must therefore be checked against the actual quantity claimed to change.

**Meaning of a positive finding:** affinity rank inversion under named conditions. Representation affinity is not itself a measurement of semantic similarity, gradient conflict or actual forgetting. The draft's stronger wording about which pairs conflict requires separate evidence of interference.

**Required record:** all affinity matrices, all eligible comparisons, declared endpoints, minimum-gap rule, paired uncertainty, multiplicity family, estimator diagnostics, frozen-feature control, learning checks, seeds/orders and sensitivity analyses.

### H3-v1 — Forward predictive dominance

Question: do current geometry and its trajectory predict future forgetting better than initial geometry, on held-out task pairs and training regimes, including the declared setting involving tasks not yet trained?

Compare a predictor from A_0 with one using A_t and a declared estimate of its past trajectory, such as (A_t-A_(t-delta))/delta. Fix the forgetting target, horizon, information available at prediction time and evaluation metric. Compare against a sufficiently capable static model, including a nonlinear/non-monotonic static baseline; otherwise a gain can merely reflect a weak baseline or greater predictor capacity.

**Meaning of a positive finding:** improved forward prediction on the specific held-out design. Retrospective association does not establish this, and predictive success alone does not establish causality.

**Required record:** training/evaluation group identities, timestamps/features available at each origin, target definition, horizons, fitted predictors and tuning, leakage checks, all baseline predictions and errors, uncertainty over independent groups, out-of-regime and out-of-pair results, and failures.

### H4-v1 — Function-preserving intervention and subsequent forgetting

Question: does an intervention changing geometry/its trajectory, while preserving old-task function at intervention time, change subsequent forgetting?

The draft proposes estimating the old-readout Jacobian J_old, forming a low-sensitivity projector P_perp, and applying z' = z + (lambda-1) P_perp z. This is a proposed intervention, not an executed result. J_old × delta_z approximately zero is a local first-order check; finite changes require held-out measurements of old-task loss, accuracy and predictive distributions.

**Meaning of a positive finding:** a controlled intervention affects future forgetting under the measured function-preservation conditions. Naming this causal mediation by a particular scalar affinity requires additional identification: the intervention can also change conditioning, capacity or optimizer dynamics. Preserving outputs on a finite probe is not preserving the function on all inputs.

**Required record:** parent checkpoint; intervention construction and strength; actual geometry change; per-task functional-silence audit; no-op, random-subspace, functional-subspace and compensated-gauge controls; matched future training; paired effects and uncertainty; optimizer-state handling; and rejected interventions.

H2 does not establish H3 or H4. A working H2 implementation is not an H3/H4 implementation. A hypothesis can remain unresolved even after many runs.

## 4. Source and artifact registry

| Source | Artifact and location/identity | What was inspected | Limitation |
|---|---|---|---|
| SRC-001 | cl_field_draft (1).pdf; existing file identity libfile_8841495371fc8191859b041241f939ae; first draft September 2026 | Extracted text including Section 5 hypotheses, Section 8 phases/interventions, and regime/measurement discussion | Working draft, not established findings; bibliography and novelty claims not independently reverified here; not the entire PDF read for this entry |
| SRC-002 | H2_Complete_Methods_Assumptions_Record.md; libfile_d10a30310fc8819187746cf258baf12d; returned version 1; modified 22 October 2026 | Full returned record, 320 extracted lines; prior pilot, proposed version-2 design, assumptions and limitations | A prior methods/evidence report; original remote artifacts and the ZIP bytes were not newly inspected here |
| SRC-003 | continual_learning_research_chat_summary.md; libfile_ba8a7c3e81108191bf597d859a889fd6 | Extracted text through line 400, including hypothesis/phase summary | Summary rather than primary execution evidence; remaining lines not needed for initial register |
| SRC-004 | C:/Users/LOCAL_USER/.codex/.chatgpt-projects/g-p-6ac3fc8a54108191a281d9eb60dde722/h2-test-output.txt | Local test-output text and SHA256 | Old CPU software tests, not newly executed hypothesis experiments |
| SRC-005 | Researcher's request in this chat, received 23 October 2026 | Request to document every H2/H3/H4 experiment, readings, settings and repetitions | Does not specify a frozen campaign or supply a new run |

| SRC-006 | Local H2_Full_Experiment_v2/h2_full_experiment directory supplied by researcher | All 35 entries in original SOURCE_MANIFEST.json match the local file bytes; 36 files including the manifest | Static/source verification, not a new training run |
| SRC-007 | Local h2_llm_testbed/h100_smoke_36125.master/accuracy.json and config.json | Complete five-stage trained-task accuracy readings and resolved pilot arguments | Historical execution; no baseline accuracy or seed uncertainty |
| SRC-008 | Local pilot documentation directory | Saved code, PBS script, package list, GPU log, data manifest and source hashes | Original run artifacts inspected locally; not an independent server inspection |
| SRC-009 | Local pilot features and checkpoints | Inventory and hashes; 90 NPY headers and five adapter configuration files | No new affinity/bootstrap analysis or checkpoint execution |
| SRC-010 | Researcher's local paths and GitHub repository in this chat | Repository verified public with push permission; original folders read only | Publication verification recorded separately when complete |

SRC-004 SHA256: 48392a5825b7b080eda96b3bf3948b9c086c3285527c007b8985e645a4e918b6.

Original reference files have not been edited. Later artifacts need their own source rows. A filename/path alone does not show an artifact has been preserved: record whether it was actually received, inspected, hashed and saved, or only reported to exist remotely.

### 4.1 Artifact entry template

    artifact_id:
    related_run_and_analysis_ids:
    artifact_role: config / raw_log / metrics / features / checkpoint / data_manifest / code / figure / other
    source_person_or_process:
    original_filename_and_remote_path:
    received_at_with_timezone:
    execution_time_if_known:
    saved_location_or_file_identity:
    preservation_status: received / inspected / hashed / saved / remote_only / missing
    size_bytes:
    sha256_of_original_bytes:
    format_schema_and_version:
    inspection_method_and_scope:
    completeness_and_known_gaps:
    redactions_or_transformations_and_reason:
    links_to_original_and_derived_versions:

Retain original numeric precision in raw outputs. Display rounding belongs in tables/figures and must be declared. Do not put credentials, tokens or private account secrets into the research log; record names of needed settings and sanitized commands.

## 5. Historical evidence imported at initialization

These entries are historical imports, not new replications. Execution dates are UNKNOWN unless separately established. Import date: 23 October 2026. HIST identifiers deliberately distinguish them from new campaign runs.

### HIST-H2-001 — H100 small-pilot trajectory

Evidence status: REPORTED HISTORICAL from SRC-002. The original server run and its feature files were not independently accessed while initializing this notebook.

| Field | Recorded fact |
|---|---|
| Purpose | Small end-to-end preparation/training/probing functionality pilot |
| Preparation job | 36121.master; preparation reported complete |
| Failed training attempt | 36124.master; CUDA computation check passed, then offline model/tokenizer cache lookup failed |
| Completed training attempt | 36125.master; training/probing reported complete |
| Relationship of attempts | Failed launch plus one completed trajectory; not two successful replications |
| Visible accelerator | NVIDIA H100 PCIe MIG 3g.40gb, approximately 39.5 GiB visible GPU memory |
| Job request | One node, four CPU tasks, 32 GB system RAM, 30-minute walltime; request is not observed usage/runtime |
| Reported software | PyTorch 2.14.1+cu130; CUDA build 13.0; complete environment UNKNOWN |
| Model/cache | Qwen/Qwen3-1.7B; reported snapshot revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e |
| Environment/interpreter | h2_h100; /Data4/YOUR_HPC_USER/.conda/envs/h2_h100/bin/python |
| Task order | SST-2 -> QQP -> QNLI -> MNLI -> AG News |
| Selected custom splits per task | 100 train, 20 validation, 30 probe, 50 test; not official benchmark-test reporting |
| Probe checkpoints | Stage 0, then after tasks 1–5; printed layers 6, 14, 22 |
| Output directory | runs/h100_smoke_36125.master, reported remote path |
| Missing important details | Original executed scripts, full config/seed/adapter/optimizer/budget, exact probe IDs, baseline scores, complete stage scores, affinity outputs, control results and uncertainty |

Candidate pools reported during preparation were larger than the selected splits:

| Task | Reported unique labeled candidates | Actually selected training rows |
|---|---:|---:|
| SST-2 | 66,978 | 100 |
| QQP | 363,538 | 100 |
| QNLI | 104,721 | 100 |
| MNLI | 392,568 | 100 |
| AG News | 119,849 | 100 |

Reported accuracy readings, scale 0–1:

| Checkpoint | SST-2 | QQP | QNLI | MNLI | AG News |
|---|---:|---:|---:|---:|---:|
| Before any task | NOT PROVIDED | NOT PROVIDED | NOT PROVIDED | NOT PROVIDED | NOT PROVIDED |
| After SST-2 | 0.92 | NOT PROVIDED | NOT PROVIDED | NOT PROVIDED | NOT PROVIDED |
| After QQP | 0.80 | 0.82 | NOT PROVIDED | NOT PROVIDED | NOT PROVIDED |
| After QNLI | 0.90 | 0.88 | 0.82 | NOT PROVIDED | NOT PROVIDED |
| After MNLI | NOT PROVIDED | NOT PROVIDED | NOT PROVIDED | NOT PROVIDED | NOT PROVIDED |
| After AG News | 0.92 | 0.74 | 0.76 | 0.68 | 0.70 |

Derived readings, conditional on the reported scores:

| Derived quantity | Calculation | Value | Interpretation limit |
|---|---|---:|---|
| Final unweighted mean accuracy | (0.92+0.74+0.76+0.68+0.70)/5 | 0.76 | Small custom test sets; no seed interval supplied |
| SST-2 change after QQP | 0.80-0.92 | -0.12 | A 12 percentage-point fall at that stage |
| SST-2 final minus first post-task score | 0.92-0.92 | 0 | Recovery at the observed endpoints does not exclude intermediate loss |
| QQP final minus immediate post-QQP score | 0.74-0.82 | -0.08 | Different reference from best-to-final forgetting |
| QQP best supplied intermediate minus final | 0.88-0.74 | 0.14 | Best among supplied observations, not necessarily every checkpoint |
| QNLI final minus immediate post-QNLI score | 0.76-0.82 | -0.06 | Does not determine complete backward transfer |
| One classification on a 50-example test set | 1/50 | 0.02 | Two percentage points per prediction |

Conclusion: the earlier report supports workflow execution and these reported classification scores. No actual affinity matrix, uncertainty-supported inversion or completed frozen control was supplied in that report. H2 is undetermined. There is no H3 forward-prediction or H4 intervention result here. Do not transfer version-2 defaults to this older run.

**Historical record update:** the missing original files and stage-4 scores were subsequently located and inspected; see EVT-000003. The initialization account above is retained as the earlier evidence state.

### HIST-SW-001 — Local CPU implementation checks

Evidence status: DIRECT ARTIFACT for SRC-004's text; execution itself was historical and was not rerun now.

The inspected output ends with 15 tests run in 3.463 seconds and OK. It also contains a Torch/NumPy ABI initialization warning and a TypedStorage deprecation warning. The test names cover prompt/content spans and truncation; exact/grouped split handling; development/confirmation exact-field separation; content pooling; affinity coordinate checks; a constructed reversal and tie rule; bootstrap group multiplicity; numerical-rank rejection; LoRA freeze/checkpoint checks; pipeline/resume; frozen/head controls; and mocked submission behavior.

Interpretation: software checks passed on those fixtures according to the output. Constructed reversals, synthetic fixtures, tiny CPU models and mocked submissions do not establish a real-data H2 effect, H3 predictive dominance, H4 causality, H100 capacity or full-campaign performance. They are not counted as 15 empirical hypothesis replications.

## 6. Current status and counters

Counters below refer to new research runs logged after this notebook was created. Historical observations remain in Section 5.

| Hypothesis | New runs launched | New completed trajectories | New interpretable hypothesis analyses | Current conclusion |
|---|---:|---:|---:|---|
| H2 | 0 | 0 | 0 | Undetermined; historical functionality pilot and software checks available |
| H3 | 0 | 0 | 0 | Undetermined; no completed forward-prediction evaluation recorded |
| H4 | 0 | 0 | 0 | Undetermined; no completed function-preserving intervention recorded |

A future counter update must identify the new run IDs, attempts, dependent branches and analyses. Never pool the three columns into a claim of independent sample size.

## 7. Protocol record — fill and freeze before confirmation

Status of new protocols: NOT FROZEN. This template is a documentation structure, not a completed preregistration. Existing suggested parameters in Section 8 are not evidence of a pre-result commitment for a new run.

For each protocol record:

| Field | Required content |
|---|---|
| Protocol identity | ID/version, hypothesis version, author, creation time, freeze time and content hash |
| Scientific scope | Concrete claim, model/data population, regime and boundaries of intended generalization |
| Prior exposure | Which runs, probes, labels, results and plots were already seen before this design |
| Stage | Instrument check, exploration, calibration, confirmation, sensitivity or external replication |
| Primary endpoint | Exact metric/formula/units, task pairs, checkpoints, horizon, aggregation and effect direction |
| Smallest meaningful effect | Numerical threshold with rationale and calibration source; not chosen from confirmation results |
| Controls | What each control diagnoses, pass/fail rules and action if it fails |
| Experimental factors | Values, fixed factors, crossed/nested structure, planned interactions and reference condition |
| Replication | Seeds, independent dataset/split repeats, task orders, model/architecture repeats and dependence structure |
| Allocation | Run-order randomization/blocking and paired branching; keep task-stream order distinct from launch order |
| Sampling | Dataset eligibility, grouping, stratification, quotas and disjoint development/confirmation resources |
| Analysis | Independent unit, estimator, uncertainty procedure, resampling counts, confidence level and multiplicity family |
| Missing/failing runs | Predeclared exclusion, retry, stopping, numerical failure and incomplete-run rules |
| Sample-size plan | Pilot variability, precision/power target, assumptions, compute budget and fixed/adaptive stopping rule |
| Confirmation protection | What remains unseen, who/what can access it, and how tuning is separated |
| Interpretation rules | Positive, bounded-null, inconclusive, invalid, and out-of-scope conditions |
| Review | Instrument validity, unresolved objections, accepted limitations and concrete readiness decision |

Record a protocol hash and an event stating it was frozen before executing/inspecting confirmation. This provides a local time-stamped commitment, not independent public registration. Any later change needs a deviation entry, whether it was made before or after relevant results were observed, and a new confirmation decision.

## 8. Repetition and parameter-coverage plan

Goal: test defined contrasts repeatedly, examine plausible confounds and locate failure boundaries. A large uncontrolled grid is not automatically strong evidence. Start with measurement validity and calibrated learning, then freeze an affordable design with replication and reserve confirmation data. A new hypothesis-specific decision rule is needed before using a sweep as confirmation.

| Dimension | What to record/test | Scientific reason | Present status |
|---|---|---|---|
| Training randomness | Separate initialization, adapter/head, shuffle, dropout and sampler seeds | Quantify stochastic variability | No new seed list frozen |
| Data generation/sampling | Dataset revision, generator/task seed, train/probe/test split seeds | Distinguish one fixed dataset from data-population evidence | No new independent data repeats frozen |
| Task order | Exact orders and selection rationale | Test path dependence; orders share task data | No new order list frozen |
| Learning regime | Frozen, head-only, lazy/weak feature change, feature learning, stronger feature learning | Separate measurement noise and representation change | Labels require measured drift/learning, not just chosen learning rate |
| Update budget | Epochs, steps, examples and tokens; planned and actual | Avoid attributing more training to a geometry factor | To be calibrated |
| Learning rule | Optimizer/state, learning rate, schedule, decay and clipping | Identify dynamics/optimizer dependence | To be calibrated |
| Adaptation capacity | LoRA rank/alpha/targets versus analysis subspace rank | Distinguish two different meanings of rank | To be specified |
| Measurement | Layer, centering, pooling, rank, probe sizes and independent probes | Check estimator stability and sensitivity | Primary choice must precede confirmation |
| Model/data/task family | Synthetic controlled tasks, NLP, further architectures/modalities | Test scope beyond a single setting | Plans, not completed work |
| H3 forecasting | Future-task/horizon, held-out pair/regime, predictor family and static baselines | Test actual forward predictive claim | Not implemented/verified by reviewed H2 evidence |
| H4 intervention | Layer, subspace, lambda/strength, norm, arm and optimizer handling | Test function-preserving changes and competing explanations | Not implemented/verified by reviewed H2 evidence |

### 8.1 Existing H2 version-2 proposal, for continuity only

SRC-002 describes Qwen3-1.7B with Yelp Polarity, QQP, QNLI, MNLI and AG News; targets per task of 10,000 train, 1,000 validation, 1,000 Probe A, 1,000 Probe B and 2,000 test; seeds 101, 202, 303, 404 and 505 crossed with three specified task orders; 15 LoRA trajectories plus two controls. Full-scale outcomes and quota attainment have not been verified here. Check the actual resolved configurations before submitting a matching run.

| Existing proposed order | Exact sequence reported in SRC-002 |
|---|---|
| O1 | Yelp -> QQP -> QNLI -> MNLI -> AG News |
| O2 | AG News -> MNLI -> QNLI -> QQP -> Yelp |
| O3 | QNLI -> Yelp -> AG News -> QQP -> MNLI |

Other proposed full defaults in that report: two epochs per task; learning rate 1e-4; AdamW decay 0.01; 0.05 warmup fraction; cosine schedule per task; gradient clipping 1; microbatch 2 with accumulation 8; evaluation batch 4; token cap 512; primary layer 14, content-only mean pooling, analysis rank 16, Probe B, baseline-to-final comparison; secondary layers 6/22 and ranks 8/32; 1,000 group-bootstrap draws; minimum affinity-difference gap 0.01. Proposed LoRA defaults are adapter rank 16, alpha 32, dropout 0.05, q/k/v/o linear targets, base BF16 and adapter/optimizer FP32. One adapter persists across tasks; optimizer state resets at task boundaries. These are proposed settings, not observed optimal values or a newly frozen protocol. Adapter rank and analysis rank have different roles even when both equal 16; obtain complete settings from the actual resolved config.

The earlier proposal's five seeds across three fixed orders share data and same-seed structure. They are not 15 independent population draws. Its two controls are not controls replicated for every combination. A stronger or different control-replication plan requires explicit design, budget and logging. Bootstrap draws do not increase the number of trained trajectories.

### 8.2 Campaign coverage ledger template

| Campaign/protocol | Hypothesis | Stage | Factor cell and exact contrast | Planned seeds/data repeats/orders | Planned units and dependence | Completed/failed/missing | Control status | Decision |
|---|---|---|---|---|---|---|---|---|
| No new campaign frozen | H2/H3/H4 | Planning | UNKNOWN | UNKNOWN | UNKNOWN | 0/0/0 | Not run | Await protocol |

An omitted cell gets a reason: cost, unavailable data, numerical failure, scope decision or unfinished work. Count every intended cell and every attempt, including cells that contradict the preferred explanation. Choose a repeat count from desired precision and observed variability, not from how many repetitions first produce significance. Log interim looks and any valid sequential procedure before using them to stop.

## 9. Full per-run record — duplicate for every new run

Run ID: UNASSIGNED TEMPLATE. Do not count this template as a run.

### 9.1 Identity, rationale and intended contrast

    run_id / campaign_id / protocol_version / hypothesis_version:
    status: planned / submitted / queued / running / completed / failed / cancelled / partial
    evidence_status:
    stage: instrument / exploratory / calibration / confirmation / sensitivity / replication
    parent_run_or_checkpoint_and_relationship:
    attempt_id_and_retry_or_resume_relationship:
    planned_at / submitted_at / started_at / finished_at / received_at:
    researcher_question_and_pre_run_prediction:
    reason_for_each_changed_parameter:
    reference_run_and_parameters_held_fixed:
    what_would_count_against_the_prediction:
    primary_analysis_id_and_secondary_analysis_ids:
    deviations_known_before_launch:

### 9.2 Code, hardware, environment and execution

    exact_sanitized_launch_command_and_working_directory:
    job_script_and_hash / scheduler / job_id / queue / job_exit_code:
    requested_vs_actual_walltime_CPU_RAM_GPU_resources:
    host / physical_GPU_model / visible_MIG_profile / visible_memory:
    GPU_driver / CUDA_runtime / torch_CUDA_build / utilization_sampling_method:
    OS / Python_executable / Python_version / environment_name:
    resolved_package_lock_and_hash / important_dependency_versions:
    source_repository_commit_or_source_archive_hash / local_patch_hash:
    config_file_hash / resolved_config_path_and_hash:
    determinism_flags / numerical_precision / TF32 / attention_backend:
    distributed_world_size / devices / dataloader_workers:
    wall_seconds / queue_wait_seconds / peak_GPU_memory_bytes / peak_RAM_bytes:
    compute_or_storage_cost_if_measured:
    warnings / numerical_anomalies / interruptions / exit_status:

On the documented college cluster, use its required scheduler/allocation workflow for intensive computation. Record actual resource allocation; a hostname or successful CUDA test alone does not demonstrate an authorized or exclusive reservation. This is a facility-specific execution constraint from SRC-002.

### 9.3 Data, task construction, probes and preprocessing

    dataset_names / source_URIs / licenses_if_relevant / pinned_revisions:
    selected_source_splits / source_row_IDs_and_manifest_hash:
    task_definition / labels_and_mapping / class_distribution:
    train_val_probeA_probeB_test_requested_counts_and_actual_counts_per_task:
    synthetic_generator_code_version_and_hash / latent_directions_or_angles:
    task_similarity_definition_and_ground_truth_if_available:
    initialization_seed / task_generator_seed / split_seed / sampler_seed:
    exact_dedup_normalization / conflict_handling / group_IDs:
    near_duplicate_method_thresholds_and_known_failures:
    train_val_probe_test_and_dev_confirmation_overlap_audit:
    cross_task_shared_inputs_and_group_handling:
    pretraining_contamination_status_and_limits:
    fixed_probe_row_IDs_order_hash_and_resampling_unit:
    cleaning_filtering_exclusions_with_counts_and_reasons:
    prompt_templates_verbatim_or_artifact_hash / task_instructions / answer_codes:
    tokenizer_name_revision_hash / special_tokens / padding_side:
    sequence_limit / truncation_rule / length_distribution / truncated_fraction:
    normalization_augmentation_and_train_eval_differences:

Keep the same probe identities and preprocessing across compared checkpoints. Never join unrelated samples as if paired. Distinguish official benchmark splits from custom balanced holdouts. Probes for analysis must not be trained on inadvertently.

### 9.4 Model, learning and actual budget

    base_model_name_revision_checkpoint_hash_and_architecture:
    pretrained_or_random_init / initialization_checkpoint_hash:
    adaptation_method / trainable_and_frozen_parameter_names_and_counts:
    LoRA_rank_alpha_dropout_scaling_targets_initialization_if_used:
    head_type / task_identity_available_at_train_and_test / label_supervision:
    replay_regularizers_projection_methods_and_memory_budget_if_any:
    exact_task_order_and_order_selection_seed:
    optimizer_betas_eps_decay / scheduler_warmup_restart_behavior:
    learning_rate / epochs / max_steps / gradient_clipping:
    microbatch / accumulation / world_size / effective_batch_formula:
    incomplete_batch_or_accumulation_handling:
    planned_vs_actual_updates_examples_tokens_per_task:
    optimizer_reset_or_reuse_between_tasks / RNG_state_reset_behavior:
    checkpointing_schedule / saved_optimizer_scheduler_RNG_sampler_states:
    validation_checkpoint_selection_criterion_and_data_seen:
    interruption_resume_point_and_exact_continuation_evidence:

A fresh run should start from its declared common initial state. An H4 branch starts from its declared shared trained checkpoint. Record accidental continuation of an old adapter as a deviation. Equal epochs, equal updates, equal examples and equal tokens are different controls; state which one the comparison holds fixed.

### 9.5 Representation extraction and analysis settings

    representation_layer_numbering_convention_and_exact_hook:
    pooling_mask_and_tokens_included_excluded / centering_or_whitening:
    extraction_eval_mode / dropout / precision / batch_size:
    feature_array_shape_dtype_order_and_hash_per_task_checkpoint:
    affinity_formula / analysis_rank / SVD_method_tolerances_and_seed:
    singular_values / explained_variance / cutoff_gap / numerical_rank:
    comparison_endpoints / checkpoint_times_and_updates:
    primary_vs_secondary_measures_and_why:
    bootstrap_or_other_uncertainty_method / groups / pairing / draws / seed:
    confidence_level / minimum_effect_threshold / multiplicity_family:
    validity_failures_and_disposition:

### 9.6 Readings ledger — enter every supplied numerical reading

| Reading ID | Run/attempt/arm | Checkpoint/task/pair | Metric and definition | Exact value | Unit/scale | n and independent unit | Uncertainty/method | Raw artifact/row | Evidence label |
|---|---|---|---|---|---|---|---|---|---|
| TEMPLATE | UNASSIGNED | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | NOT PROVIDED | UNKNOWN |

For every numeric table/array record row and column identities, dimension, index convention, units, dtype and missing-value meaning. Put complete per-step/per-example arrays in preserved raw artifacts; retain all readings there and link the table or file range here. Do not report only the best curve or only significant comparisons.

At each scheduled checkpoint, preserve train/validation/test loss, accuracy and denominator, old-task and new-task performance, complete task-by-checkpoint performance matrix, full affinity matrices, feature/gradient diagnostics if measured, actual update count, tokens/examples, learning rate and resource readings. Record metrics only when actually emitted; label unavailable ones explicitly. Keep validation, probe and test roles separate.

### 9.7 Derived calculations, interpretation and closure

    calculation_ID / formula / source_reading_IDs / units / analysis_code_hash:
    estimates / intervals / test_family / effect_size / raw_vs_adjusted_p_if_used:
    controls_and_learning_checks_pass_fail_unknown:
    run_quality: interpretable / inconclusive / invalid / incomplete:
    observation_in_plain_words:
    conclusion_with_scope_and_competing_explanations:
    hypothesis_status_before_and_after_with_reason:
    deviation_issue_and_decision_IDs:
    next_action_and_whether_it_changes_confirmation_status:
    raw_artifact_completeness_and_missing_items:
    reviewer_or_researcher_notes:

## 10. H2 analysis supplement

For every H2 analysis, bind the run, resolved measurement settings and all input artifact hashes. Reanalysis of saved features gets a new analysis ID, not a new training-run ID.

| Required field | Exact record needed |
|---|---|
| Pair universe | Task list, all unordered task pairs and every eligible pair-of-pair comparison |
| Endpoint data | A_0 and A_T matrices, intermediate matrices if collected, and D_0/D_T for every comparison |
| Effect criterion | Sign-crossing rule plus minimum separation epsilon and its rationale |
| Uncertainty | Paired checkpoint/source-group resampling, subspace refitting, interval construction and simultaneous family |
| Numerical health | Valid sample/rank conditions, singular spectrum/gaps, failed resamples and handling |
| Control floor | Frozen encoder and head-only encoder drift/affinity change, tolerances and hardware context |
| Learning health | Actual learning/feature change; a non-learning run cannot diagnose a rich-learning prediction |
| Scope | All layers/ranks/probes/orders inspected and which were primary before results |

For K=5 tasks there are 10 unordered task pairs and 45 comparisons between those pair affinities. These comparisons may share tasks and are not 45 independent trained experiments. SRC-002's proposed rule requires the initial interval entirely above +0.01 and final interval entirely below -0.01, or the reverse. It proposes a 90-value within-run family (45 comparisons × two endpoints), or a 1,530-value family across 17 campaign runs. If that design is changed, recalculate and declare the family; do not silently reuse the old adjustment.

Frozen-control drift, numerical estimator failure, confidence intervals crossing the tie region and robust absence at a predeclared resolution must receive different labels. A frozen-control failure triggers an instrument investigation, not a finding of H2. Searching intermediate times for a crossing needs its own multiplicity rule or an exploratory label.

H2 result table template:

| Analysis | P | Q | D_0 with interval | D_T with interval | epsilon | Raw reversal | Meets declared uncertainty/gap rule | Control validity | Interpretation |
|---|---|---|---|---|---|---|---|---|---|
| UNASSIGNED | UNKNOWN | UNKNOWN | NOT PROVIDED | NOT PROVIDED | NOT FROZEN | UNKNOWN | UNKNOWN | NOT RUN | No result |

## 11. H3 analysis supplement

Define one prediction unit precisely: old task i, proposed upcoming task/sequence, prediction origin t, future horizon h, run/regime and forgetting target. At origin t, record which future-task information is available. Unlabeled future-task probes can be allowed if declared; using future labels, future fitted representations or future forgetting to choose features violates the intended prospective information boundary. A task not trained yet is not necessarily an unseen task for the researcher/predictor; log both meanings.

Freeze these fields before held-out evaluation:

    prediction_target_and_reference_checkpoint:
    origin_t_and_horizon_h_in_updates_or_tasks:
    information_available_at_t_and_feature_creation_timestamps:
    trajectory_estimator_using_only_times_at_or_before_t:
    dataset_of_prediction_rows_and_parent_run_IDs:
    train_validation_test_regimes_and_task_pair_IDs:
    grouping_to_prevent_same_run_neighbor_checkpoint_leakage:
    task_identity_overlap_across_splits_and_scope_of_generalization:
    predictor_families_capacity_and_equal_tuning_budgets:
    feature_scaling_fitting_and_hyperparameter_selection_on_train_val_only:
    static_A0_baseline / nonlinear_nonmonotonic_static_baseline:
    current_At_only / trajectory_only / combined_dynamic_ablations:
    non_geometry_baselines_such_as_current_performance_training_progress:
    primary_error_metric_units_and_practical_improvement_threshold:
    grouped_uncertainty_and_multi_horizon_or_model_selection_correction:
    heldout_access_log_and_frozen_prediction_artifact_before_outcomes:

Do not randomly split correlated checkpoints from the same run into train and test and call it out-of-regime prediction. Holding out a pair while sharing its tasks across splits is a narrower generalization claim than holding out tasks entirely. State whether evaluation tests held-out regimes, held-out pairs, both together, or separate tests. Predictions/outcomes for every row must be retained, including poor predictions.

For a lower-is-better error metric, define improvement over static baseline b as Error_b - Error_dynamic, using paired evaluation units. Positive values favor the dynamic predictor. Report uncertainty over independent runs/task groups as justified, not over every overlapping checkpoint as independent. Correlation can be a secondary measure; prediction errors and calibration should address the actual target.

| Analysis | Held-out design/horizon | Baseline | Baseline error | Dynamic error | Paired improvement with interval | Leakage audit | Scope/conclusion |
|---|---|---|---|---|---|---|---|
| UNASSIGNED | NOT FROZEN | NOT FITTED | NOT PROVIDED | NOT PROVIDED | NOT PROVIDED | NOT RUN | No H3 result |

## 12. H4 intervention supplement

The experimental unit and dependence must be explicit: independent parent training checkpoints/seeds, with paired intervention/control arms branched from each. Branches sharing a parent are not independent parent replications.

### 12.1 Construction and assignment

    parent_run_stage_checkpoint_hash_and_task_history:
    old_tasks_and_upcoming_task_order:
    intervention_layer_and_implementation_code_hash:
    J_old_estimation_data_outputs_loss_definition_and_dimensions:
    subspace_algorithm_cutoff_rank_residual_and_data_split:
    lambda_or_strength_grid_predeclared_and_selection_rule:
    delta_z_or_parameter_perturbation_norm_and_matching_definition:
    actual_affinity_trajectory_change_and_estimator_diagnostics:
    arm_assignment_seed / branch_RNG_states / minibatch_order_pairing:
    optimizer_scheduler_state_copy_reset_or_transform_and_reason:
    compute_and_future_training_budget_matched_across_arms:

### 12.2 Functional-silence gate

For each old task, measure immediately before intervention and immediately after intervention, before future training. Record accuracy with denominator, loss definition/mean and predictive KL with its direction, averaging, class/vocabulary support and numerical smoothing. An unchanged top prediction alone is insufficient.

Declare epsilon_accuracy, epsilon_loss and epsilon_KL before confirmation, with substantive justification and precision assessment. Prefer an equivalence/non-inferiority rule with uncertainty when the claim requires preservation; failing to reject a difference is not establishing equivalence. Record whether bounds concern absolute change, degradation only, mean, quantiles or worst-task behavior. Do not let a good average conceal a failing old task.

Use separate construction/selection data and an independent silence audit where possible. If many lambda values are tried and only passing ones retained, log all candidates, selection exposure and the rule. Do not reuse the selection set as an unqualified independent audit. A rejected gate is an informative intervention failure, not supporting H4.

| Arm/candidate | Old task | Accuracy pre/post and change | Loss pre/post and change | KL with uncertainty | Declared tolerances | Gate outcome/reason | Actual geometry change |
|---|---|---|---|---|---|---|---|
| UNASSIGNED | UNKNOWN | NOT PROVIDED | NOT PROVIDED | NOT PROVIDED | NOT FROZEN | NOT TESTED | NOT PROVIDED |

### 12.3 Arms, subsequent learning and causal interpretation

| Arm | Purpose | What must be recorded |
|---|---|---|
| No-op, including independently seeded repeats | Estimate continuation/run noise | Identical parent state, no-op verification and subsequent training randomness |
| Intended low-sensitivity intervention | Test the proposed mechanism | Finite-step silence, actual geometry change, strength and ensuing learning |
| Equal-norm random subspace of equal dimension | Diagnose generic perturbation effects | Norm/rank definition, random seed and its own old-function changes |
| Equal-norm functional-subspace perturbation | Positive damage calibration | Immediate old-task damage; does not have to pass the silent-intervention gate |
| Compensated exact gauge transformation | Diagnose coordinate/optimizer effects | Exact implemented symmetry, output preservation, residual/nonlinear constraints and optimizer treatment |

A transformation is only an exact gauge control if it is actually function-preserving for the implemented architecture; do not assume an arbitrary hidden-state rotation is compensable throughout a transformer. Equal-norm controls need a specified metric and matched layer/setting. Random-subspace arms may fail silence; report this rather than treating them as clean silent controls.

After a passing gate, train the same upcoming task(s) under matched declared conditions. Preserve old-task trajectories, future-task acquisition, loss/KL, geometry and optimizer diagnostics. Separate immediate intervention damage from additional forgetting during future training. Define post-intervention old-task performance as the reference for incremental forgetting, and also report total change from the common pre-intervention checkpoint.

For each independent parent, report paired Forgetting_intervention - Forgetting_noop and comparisons with other declared controls, confidence intervals and practical thresholds. A direction or dose pattern must be predeclared or labeled exploratory. Investigate whether differences follow optimizer/conditioning changes, generic perturbation, new-task underlearning or unequal capacity/budget. An effect under these controls supports a scoped causal intervention claim; mediation by the chosen affinity remains a separate identification question.

| Parent/analysis | Arm and strength | Silence gate | Geometry change | Immediate damage | Future-task learning | Incremental forgetting | Paired effect with interval | Interpretation |
|---|---|---|---|---|---|---|---|---|
| UNASSIGNED | NOT SET | NOT RUN | NOT PROVIDED | NOT PROVIDED | NOT PROVIDED | NOT PROVIDED | NOT PROVIDED | No H4 result |

## 13. Shared metric definitions and units

Let a_(s,i) be accuracy on task i after stage s, on the declared fixed evaluation data. Use fractions 0–1 in raw tables unless otherwise stated. A change of 0.08 is eight percentage points; it is not automatically an eight-percent relative change.

| Metric | Definition | Recording requirement |
|---|---|---|
| Accuracy | Correct predictions / evaluated predictions | Counts, eligible labels, class-constrained versus unrestricted output and evaluation split |
| Final mean accuracy | Mean_i a_(K,i), unweighted unless declared otherwise | Task set and weighting |
| Backward transfer | Mean over i=1..K-1 of [a_(K,i)-a_(i,i)] | Complete immediate post-task scores; negative means deterioration |
| Best-to-final forgetting | max over eligible prior stages s of a_(s,i) minus a_(K,i) | Exact eligible stage set; state whether final stage is included and any zero clipping |
| Horizon forgetting for H3 | Declared reference accuracy minus future accuracy at t+h | Reference/horizon, whether signed or clipped, and input availability |
| H4 incremental forgetting | Post-intervention pre-future-training accuracy minus later accuracy | Separate immediate intervention effects and total parent-to-final change |
| Affinity | Declared estimator, for primary H2 in Section 3 | Tasks, layer/rank/pooling, probes and normalized scale |
| Loss/NLL | Exact objective and averaging | Per token/example/task, valid-class or full-vocabulary, weighting and masking |
| Predictive KL | Declared KL direction between pre/post distributions | Support, temperature, smoothing and aggregation |
| Geometry/feature drift | Declared norm/distance | Coordinates/alignments, absolute/relative units, scale and normalization |

Never calculate complete backward transfer from an incomplete task-by-stage matrix. Never equate accuracy loss, affinity drift and interference without a declared measurement linking them. If measuring gradient interference, record loss, parameter subset, gradients/actual update, optimizer state, norms and the exact alignment definition; the primary affinity estimator alone supplies none of these.

## 14. Assumption, deviation, issue and decision registers

### 14.1 Initial assumptions and limitations

| ID | Assumption/limitation | Why it matters | Required evidence or action | Status |
|---|---|---|---|---|
| ASM-001 | Prior report accurately summarizes historical execution | Historical values are not direct remote inspection | Retrieve original logs/config/features if needed for stronger claims | Unverified against remote originals |
| ASM-002 | Chosen affinity captures a relevant aspect of representation structure | Top-variance overlap can track length/style instead of task interference | Diagnostics, complementary measures and independent interference/behavior readings | Open |
| ASM-003 | Detected source groups approximate relevant sampling dependence | Probe uncertainty can be understated with hidden dependencies | Archive grouping audit; scope bootstrap interpretation | Open |
| ASM-004 | Independent units are correctly identified | Seeds/orders/pairs/checkpoints can share data and state | Explicit nesting/blocking and grouped analyses | Pending campaign design |
| ASM-005 | Development/confirmation separation is maintained | Tuning on observed confirmation invalidates its intended role | Exposure log and source/split audit | No new confirmation frozen |
| ASM-006 | H4 finite probes adequately audit preservation within scope | Local null-space math and stable accuracy do not certify global silence | Held-out loss/distribution audits and justified tolerances | Not tested |
| ASM-007 | Instrument/control tolerances fit precision and feature scale | A fixed absolute tolerance can fail across hardware/scale | Calibrate on intended hardware without selecting for desired effects | Not tested |

### 14.2 Deviation template

    deviation_ID / detected_at / affected_protocol_run_analysis:
    planned_value_or_procedure:
    actual_value_or_procedure:
    reason_and_when_change_was_decided:
    results_seen_before_change:
    effect_on_comparability_validity_or_scope:
    retain_exclude_rerun_decision_with_rationale:
    corrective_action_and_new_confirmation_requirements:

Do not delete a run because it weakens the hypothesis. Exclusions must follow predeclared rules or be transparently labeled post hoc, with an all-runs sensitivity summary when meaningful.

### 14.3 Initial unresolved issues

| Issue | Missing item | Consequence | Next action |
|---|---|---|---|
| ISSUE-001 | Original pilot scripts/config/logs/features now located; affinity/control analysis still missing | Partly resolved by EVT-000003; H2 remains undetermined | Preserve artifacts and design a valid analysis/control comparison |
| ISSUE-002 | Frozen new H2 protocol and exact resolved campaign | Cannot count a future run as confirmation yet | Calibrate and document the intended comparison before results |
| ISSUE-003 | H3 prediction dataset, split design and implementation | H3 currently has no recorded empirical test | Design grouped forward evaluation and baselines |
| ISSUE-004 | H4 implemented intervention, controls and finite-silence tolerances | H4 currently has no recorded empirical test | Validate intervention mechanics and predeclare gates |
| ISSUE-005 | Precision/power and stopping plans across parameter cells | Many repetitions alone do not justify inference | Use pilot variability and meaningful-effect targets to size the design |

### 14.4 Decision register

| Decision | Date | Decision and rationale | Evidence | Effect |
|---|---|---|---|---|
| DEC-001 | 2026-10-23 | Begin one continuous, detailed H2/H3/H4 record; preserve unfavorable and incomplete evidence | SRC-005 | This file is the working notebook |
| DEC-002 | 2026-10-23 | Import historical work with explicit provenance, not as new hypothesis replications | SRC-001–004 | Current hypotheses remain undetermined |
| DEC-003 | 2026-10-23 | Keep observed, calculated, proposed and unknown items distinct | Source gaps and scientific scope | No invented readings or retroactive preregistration |
| DEC-004 | 2026-10-23 | Prepare a new public GitHub repository named cl-geometry-experiments, as selected by the researcher | GitHub destination choice in this chat | Upload authorized; publication blocked by repository creation/sign-in and source-file access |

## 15. Chronological event journal

### EVT-000001 — Notebook initialization

Date received/recorded: 23 October 2026, Asia/Calcutta. Exact historical run times: UNKNOWN. Type: documentation and protocol planning. Related decision IDs: DEC-001–003. Related sources: SRC-001–005.

Researcher intent: record every available detail of repeated empirical tests of H2, H3 and H4, including parameters, equipment, datasets, readings and methods, before progressing to later stages of the research.

Actions completed: inspected the existing hypothesis/method records and local test-output text; registered the three hypotheses; imported the earlier pilot as historical evidence; distinguished its missing measurements; created the common and hypothesis-specific templates; documented counters, missing artifacts, uncertainty, controls, coverage and decision rules. Consulted general experimental-design guidance in Section 17.

Observed new scientific readings: NONE. New training runs: NONE. New H2/H3/H4 analyses: NONE. No server job was submitted. No original reference files were modified. No new protocol is frozen.

Interpretation: this establishes a documentation system and current evidence baseline. It does not establish any of the hypotheses. The next experimental event should identify its protocol, exact configuration and original output artifacts.

### EVT-000002 — GitHub publication preparation and blockers

Date received/recorded: 23 October 2026, Asia/Calcutta. Type: research infrastructure. Decision: DEC-004. Scientific run counts and hypothesis conclusions are unchanged.

Researcher request: upload the code and associated work to GitHub. The researcher selected a new PUBLIC repository from the destination choices; proposed name cl-geometry-experiments. The connected GitHub account was verified as devanshpandey2077-cyber. Repository creation and upload have NOT completed.

Available source metadata was resolved for H2_Full_Experiment_v2.zip (62,871 bytes), h2_llm_testbed.zip (28,194 bytes), and H2_Complete_Methods_Assumptions_Record.md (55,966 bytes). These are metadata sizes, not a successful fresh checksum verification or an inspection of archive contents. The current notebook is also intended for the repository. The earlier package should be labeled historical/legacy after inspection, with the full package clearly distinguished.

Observed blockers: the connected GitHub tools support content changes in existing repositories but expose no repository-creation operation. The available browser opened GitHub's repository-creation destination and reached its sign-in page; authentication remains for the researcher. Non-interactive local GitHub credentials were unavailable. Local source downloads failed with DNS resolution errors. A browser ZIP download was blocked because its security check could not verify an originating browser context; this check was not bypassed. No archive bytes were obtained through those attempts, and no credential scan or code validation is claimed.

Next actions: obtain an authenticated GitHub browser session or an existing empty repository link; obtain the original ZIPs through accessible local files; inspect and organize actual source; check publishable files for credentials; include setup documentation, the research log and an appropriate ignore file; upload and verify the remote commit. Original reference files remain read-only. There were no new training runs, scientific readings or hypothesis tests in this infrastructure event.

### EVT-000003 — Local source inspection and historical pilot evidence update

Date recorded: 23 October 2026, Asia/Calcutta. Type: source verification, historical evidence import and GitHub preparation. Sources: SRC-006–010. No new training trajectory or H2/H3/H4 experiment was executed.

The researcher supplied C:/Users/LOCAL_USER/Downloads/H2_Full_Experiment_v2 and C:/Users/LOCAL_USER/Downloads/h2_llm_testbed and confirmed creation of cl-geometry-experiments. The public repository devanshpandey2077-cyber/cl-geometry-experiments was verified empty and writable. The earlier source-access/repository-creation blockers are resolved through these supplied local folders and the researcher-created repository.

Source integrity: all 35 hashed entries in the full package's original SOURCE_MANIFEST.json match. Python AST parsing passed for 23 supplied Python files, including the saved historical code copies. A pattern scan of the candidate text files found no matches for the checked private-key/token/assigned-secret patterns; this is a limited scan, not proof that no secret can exist. No GPU training or dependency installation was performed. Original files in Downloads were not changed.

The historical pilot config records seed 42, LoRA rank 16, learning rate 0.0001, microbatch 4, accumulation 8, optimizer-step target 20 per task, length cap 256, 30 probes/task, evaluation limit 50, layers 6/14/22, frozen=false and the same five-task order previously reported. The saved adapter config records LoRA alpha 32, dropout 0.0 and q/k/v/o projections. These are the pilot settings, distinct from version-2 defaults. The code repeats task data until its step target; no per-update actual-token ledger was recovered.

The original accuracy.json and saved GPU log agree. Complete supplied readings, scale 0–1:

| Stage | SST-2 | QQP | QNLI | MNLI | AG News |
|---|---:|---:|---:|---:|---:|
| 1, after SST-2 | 0.92 | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED |
| 2, after QQP | 0.80 | 0.82 | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED |
| 3, after QNLI | 0.90 | 0.88 | 0.82 | NOT EVALUATED | NOT EVALUATED |
| 4, after MNLI | 0.94 | 0.72 | 0.76 | 0.72 | NOT EVALUATED |
| 5, after AG News | 0.92 | 0.74 | 0.76 | 0.68 | 0.70 |

Newly recoverable DERIVED reading: BWT = [(0.92-0.92)+(0.74-0.82)+(0.76-0.82)+(0.68-0.72)]/4 = -0.045, or a 4.5 percentage-point mean decrease over the first four tasks. Final mean accuracy remains 0.76. SST-2 best supplied score is now known to be 0.94, so its observed best-to-final drop is 0.02. This supplements the earlier incomplete-record calculations; it does not provide seed-level uncertainty.

The saved pilot documentation/data_manifest.json records 100 train, 20 validation, 30 probe and 50 test examples per task. The top-level legacy data/ directory has a separate larger manifest and must not be asserted to be the exact small pilot splits. In particular, the pilot used data_h100_smoke, which is not supplied as a matching raw-data directory here. Probe identity reconstruction from the larger data/ directory would be unsupported without additional evidence.

Ninety saved feature files cover six stages × five tasks × three layers; their headers and hashes are inventoried without running an affinity estimator. Five trained adapter files and their metadata are present. No frozen-control output, completed uncertainty-supported inversion analysis, H3 forward evaluation or H4 intervention was identified. H2, H3 and H4 remain undetermined; the new-run counters stay zero.

Publication scope: copy all experiment code, configuration profiles, tests and setup documents; label the earlier starter legacy; include the small pilot records, checkpoint metadata, a full local artifact inventory with original byte hashes, and the research log. Raw datasets, model weights, feature arrays and Python caches remain in the original local folders rather than ordinary Git history. Public copies replace local account/server identifiers with placeholders; transformation records and both original/published source hashes make this explicit. The algorithms and numerical measurements are not changed. Repository publication is pending at this event; a verified commit is recorded afterward.

### Event template — append below without deleting earlier entries

    event_ID / event_type:
    occurred_at / received_at / recorded_at / timezone / time_precision:
    campaign_protocol_hypothesis_run_attempt_arm_analysis_IDs:
    exact_researcher_request_or_question:
    supplied_or_inspected_artifact_IDs_and_scope:
    raw_observations_and_exact_readings_with_units:
    changed_settings_and_reason / values_held_fixed:
    calculations_with_formulas_and_source_readings:
    control_instrument_and_run_status:
    errors_warnings_missing_data_and_deviation_IDs:
    interpretation_and_alternative_explanations:
    conclusion_scope_and_effect_on_hypothesis_status:
    decision_ID / next_action / outstanding_evidence:
    notebook_version_and_save_status:

## 16. Readiness for a later theoretical stage

The empirical and theoretical stages can inform each other. Do not wait for an impossible guarantee of no gaps. Before using empirical observations to motivate a stronger derivation, state precisely what is established, at what resolution, and under which regimes.

Review whether the instrument is valid; raw artifacts and settings are recoverable; outcomes survive the declared replication/controls; primary and exploratory results are separated; H3 is genuinely forward and held out; H4 passed a finite-function gate with credible controls; null and failure boundaries are reported; and competing explanations remain visible. Record unmet criteria as limitations instead of claiming completion. Neither a positive H2 result nor a large campaign automatically satisfies H3/H4 or novelty requirements.

## 17. General methodological references consulted

The following support general design principles, not the truth or novelty of H2/H3/H4. Accessed 23 October 2026.

- NIST/SEMATECH, [Choosing an experimental design](https://www.itl.nist.gov/div898/handbook/pri/section3/pri3.htm): define objectives, factors/levels and an appropriate design, including blocking and factorial options.
- NIST/SEMATECH, [Completely randomized designs](https://www.itl.nist.gov/div898/handbook/pri/section3/pri331.htm): distinguish factor levels, replication and randomized run allocation. Launch-order randomization is separate from the scientific task-order factor in this project.

The hypothesis-specific requirements above are a proposed research-record design grounded in the draft and prior methods report. Their numerical thresholds, power and validity must be justified for each actual experiment.

## 18. Document change history

| Version | Date | Change | Scientific result changed? |
|---|---|---|---|
| 1.0 | 2026-10-23 | Initial notebook, source/hypothesis registers, historical evidence, full run templates, coverage and audit records | No; all three hypotheses undetermined |
| 1.1 | 2026-10-23 | Record public GitHub destination, account/source metadata checks, access failures and remaining publication steps | No; research infrastructure only |
| 1.2 | 2026-10-23 | Import original pilot readings/config/source verification and prepare public code repository | No new experiment; complete historical BWT now recoverable |

End of initial record. Append subsequent events and run records while retaining this provenance.
