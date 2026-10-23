# Proposed H2 protocol: freeze before confirmation

This is a proposed protocol, not evidence of an externally preregistered study. Fill in the author, date, project revision and any justified changes before starting the full campaign. Preserve that copy and its SHA256. Do not change the decision rule after examining confirmation outcomes.

**Working hypothesis.** Continual learning can reverse the ordering of task-pair representation affinities when dataset/probe identities and the measurement definition remain fixed.

**Primary estimator.** For each task i and checkpoint t, use the same Probe B rows in the same order. Let X_i^t be the matrix of mean content-token hidden states at the configured intermediate layer. Center each feature column across rows. Obtain U_i^t, the top-r right singular vectors, as a d-by-r orthonormal basis in feature coordinates. Define:

`A_t(i,j) = ||(U_i^t).T @ U_j^t||_F^2 / r`.

The direction of each singular vector is immaterial to this overlap. Singular values, explained variance and the gap at rank r are saved. A feature coordinate change applied in common to every task leaves the measurement unchanged only when it preserves inner products, such as a common orthogonal rotation; arbitrary scaling is not covered.

**Primary settings.** Qwen3-1.7B pinned revision in full.json; layer index14; content mean; r16; baseline versus final checkpoint. Five datasets, three specified orders and five specified training seeds; optimizer reset per task; no replay; all other training parameters in resolved full.json. Probe B is the confirmation split; Probe A and calibration data support exploratory inspection.

**Data rule.** Use custom grouped, balanced splits from source TRAIN only. Before sampling, every normalized field must hash to the development pool for smoke/calibration or confirmation pool for full runs; mixed-pool multi-field rows are excluded. This separates exact text fields across exploration/confirmation, not all semantic or near-duplicate groups. Group exact shared fields across tasks and approximate detected long-text near duplicates before splitting. Discard conflicting exact records and oversized groups. Fail on unmet quotas or audit overlaps. Source repositories, SHAs, label mappings, selected rows, exclusions and split hashes accompany the result. Do not call these official benchmark test scores.

**Comparison family.** Five tasks give10 unordered task pairs and45 unordered comparisons between task-pair affinities. For each comparison P,Q define D_t = A_t(P)-A_t(Q). The primary family includes the two endpoint D values for every comparison in every registered run, including the two controls. The15 LoRA runs are independent training attempts by seed/order; comparisons sharing tasks are dependent measurements.

**Decision rule.** Construct approximate simultaneous95% intervals using1,000 joint source-group bootstrap draws and a maximum standardized error statistic. A robust reversal requires either initial lower>0.01 and final upper<-0.01, or initial upper<-0.01 and final lower>0.01. All sign changes are also reported without treating near-ties as convincing evidence. Do not substitute within-run intervals for campaign-adjusted intervals in primary claims.

**Resampling rule.** Draw global source groups with replacement. A group's multiplicity is shared across every task in which it appears, both endpoints and all runs. Refit centered subspaces in each draw. Do not retry rank-deficient draws until a desired result appears; increase the fixed probe sample or reconsider rank in a new declared design if the configured estimator fails. The intervals are conditional on trained checkpoints and approximate under the detected source grouping.

**Controls.** An unchanged encoder and a trained-head/frozen-encoder run should produce no meaningful encoder feature drift and no robust inversions. Both controls must pass the fixed feature tolerance1e-4. The head control must actually train its head; task metrics alone do not certify encoder invariance. A failed control requires investigation before scientific interpretation. Freezing the pretrained weights while updating LoRA is not a frozen-representation control.

**Stability.** Report the complete per-seed/per-order table, including zero-reversal runs, failed runs and uncertainty. Summarize training variability by resampling seeds as blocks across the fixed orders. This descriptive summary does not turn five seeds into a population guarantee. Do not treat45 comparisons or1,000 bootstrap draws as45 or1,000 independent training experiments.

**Secondary analyses.** Layers6/22, ranks8/32, answer-position pooling and intermediate checkpoints are sensitivity/descriptive analyses. Optional alternate prompts, a second model and PAWS constitute further experiments. Declare their families separately or include them in a larger correction before interpreting their intervals. Probe A/B results share trained checkpoints; agreement is robustness to another input sample, not an independent training replication.

**Interpretation.** A positive result supports the existence and stability of measured affinity-order reversals in the declared setup. It does not establish H3 prediction, H4 mediation/causality, all-model universality or research novelty. Negative/inconclusive outcomes and unstable cutoff spectra must be presented. Compare existing literature separately before making a novelty claim.

**Record before launch.** Author/date; exact H2 wording; config/source SHA256; asset locks; campaign plan; queue/resource request; calibration rationale; independent confirmation policy; allowed exclusions; expected17 runs; storage quota; primary analysis family; handling of failures and all secondary analyses. `run.json` and campaign ledgers provide an execution record, not a substitute for this prior decision record.
