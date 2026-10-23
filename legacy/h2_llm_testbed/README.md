# H2 LLM continual-learning testbed (H100)

A research **starter pipeline**, not a validated experimental result. Implements five-task labeled NLP stream, cumulative Qwen3-1.7B LoRA, fixed probes, hidden-state extraction, principal-angle subspace affinity, inversion statistics, and old-task accuracy.

## Prepare environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Download / prepare data (run on networked machine)

```bash
python prepare.py --out data --train 50000 --validation 2000 --probe 2000 --test 5000
```

Uses **labeled training portions** of SST-2, QQP, QNLI, MNLI, and AG News. These splits are **custom holdouts**, not official benchmark validation/test. Data may fail if a class does not contain enough unique rows; do **not** duplicate or leak samples to meet a number. To downsize without fabrication pass smaller counts (e.g. `--train 1000 --validation 100 --probe 100 --test 200`). `--allow-smaller` is an explicit non-replacement fallback. Near-duplicates across corpora are not fully removed: run a MinHash/embedding dedup audit before making publication claims.

Dataset label mappings are captured in `prepare.py`. The output `data/manifest.json` records split sizes and class counts.

## Run pilot on H100

```bash
python train_and_probe.py --data data --output runs/pilot --max-steps 300 --probes 500 --eval-limit 1000
python analyze.py --run runs/pilot --subspace-rank 16
```

On a first dry-run, use 20 steps and 30 probes. Full experiment: increase probes to 2000, run multiple seeds/orders, and add frozen/model-weight-updating controls. `--frozen` saves baseline feature matrices at stage 0 only: if you want a repeated determinism control, run it twice with different output dirs and compare corresponding `.npy` arrays.

## Design cautions

- **One cumulative LoRA adapter**: do not switch to task-specific adapters.
- Loss supervises only the classification answer, not prompt boilerplate.
- Fixed prompt text and probe membership over checkpoints; probe examples are never trained on.
- The current extractor pools **all prompt tokens** (including task instruction), not content tokens only. This is a confound; add content-only pooling and prompt controls before treating results as conclusive.
- Class likelihood uses the model's **next-token** logits and assumes each label token (`A`, `B`, `C`, `D`) is a single token. The training code checks this. For robust evaluation verify this also at eval time and compare full answer log-likelihood as a secondary check.
- The primary principal-angle metric compares right-singular-vector task subspaces **without requiring aligned probe rows**. Naive CKA applied to unrelated rows would be invalid; not implemented.
- A frozen base model whose LoRA adapter trains is **not** a frozen-representation control. `--frozen` in this code means no LoRA at all.
- Full training involves equal optimizer steps per task, not necessarily equal epochs; record actual token counts for publication.
- Save complete software versions, GPU type, git revision, HF revisions, dataset fingerprints for a full reproducibility audit.
- H2 statistical inference should bootstrap independent probe examples, use multiple seeds and task orders, account for correlated pair comparisons, and pre-specify primary layer and subspace rank.
- H3 prospective prediction and H4 interventions are **not yet implemented** in this starter project.

## Outputs

- `data/<task>/{train,validation,probe,test}.jsonl`
- `runs/<name>/checkpoints/stage_*` (LoRA adapter checkpoints)
- `runs/<name>/features/stage_*/*_layer*.npy`
- `runs/<name>/accuracy.json`
- `runs/<name>/analysis/report.json` and plots
