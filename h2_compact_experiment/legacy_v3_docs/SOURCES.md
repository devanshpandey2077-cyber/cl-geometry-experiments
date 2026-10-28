# Sources and implementation basis

Read for this delivery:

- User working draft `cl_field_draft (1).pdf`, Section5 H2 and adjacent definitions: rank inversion of task-pair affinities.
- `H2_Complete_Methods_Assumptions_Record.md`, current returned record: v2 measurement, controls, pitfalls and limitations.
- `H2_H3_H4_Research_Log.md`, returned version8 / record1.8: historical pilot and setup evidence; no completed full H2 result recorded there.
- Existing local `h2_full_experiment` source, configs, regression tests and PBS scripts, read without modification.

Official technical documentation checked 26 October2026:

- [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B)
- [Qwen3-4B configuration](https://huggingface.co/Qwen/Qwen3-4B/raw/main/config.json)
- [Qwen3-8B configuration](https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json)
- [Qwen3-14B configuration](https://huggingface.co/Qwen/Qwen3-14B/raw/main/config.json)
- [SmolLM2-1.7B configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B/raw/main/config.json)
- [Transformers model outputs](https://huggingface.co/docs/transformers/en/main_classes/output)
- [Transformers model loading](https://huggingface.co/docs/transformers/en/main_classes/model)
- [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)

Live repository main branches above are documentation references, not execution locks.
Preparation resolves actual commits and hashes local files for the experiment.
The custom LoRA implementation derives from v2; it does not depend on PEFT serialization.
The statistical construction is explicitly approximate and instrument diagnostics are not a proof of coverage.
No literature novelty claim was re-audited by this coding task.
