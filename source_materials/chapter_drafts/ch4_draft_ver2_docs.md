# Chapter 4 draft (ver 2) — documentation

Companion to [`ch4_draft_ver2.md`](ch4_draft_ver2.md). That file is the examiner-ready prose. This file is the working map: what each subsection is for, which code it comes from, and what not to paste into Word.

Do not use `ch4_draft.md` or `ch4_implementation.md` as sources. Ver 2 is the current Chapter 4 draft.

---

## How to use this with the draft

1. Copy from `ch4_draft_ver2.md` into the thesis (Word), one subsection at a time.
2. Tick the matching box in the **Progress** list at the top of `ch4_draft_ver2.md`.
3. Use the tables below if a sentence needs a file path, artifact, or snippet caption.
4. After copy: strip the `[ ]` markers from headings, restyle code as the faculty template requires, and apply the thesis caption style to Table 4.1 and Table 4.2.

Voice locked for this draft: English only, present tense, “this study”, no “we” / “I”. Cite as `(Author et al., YEAR)`. Do not rewrite Sections 3.8–3.9 (metrics design, hardware/software). Do not reprint comparative scores; those go to Chapter 5.

---

## Locked scope (do not undo while copying)

| Include | Exclude |
|---|---|
| How the pipeline is coded | Architecture math already in Section 3.6 |
| Realized feature counts (32 / 54) and drop reasons | Published inventories (35 / 61) as if they were the training width |
| Exact split sizes from `metadata.json` | Re-arguing why 60/20/20 was chosen (Section 3.4) |
| How comparison-table **columns** are produced | The numeric comparison table (Chapter 5) |
| Inference latency = time-to-use; retrain minutes = final `train()` only | Optuna wall-clock as a table column |
| Citations at the point a library is used | Dataset-validator / wrong-CSV story (out of scope) |
| Tools mentioned in passing, pointing to Section 3.9 | A new environment section |

---

## Chapter map

| Draft heading | Role | Chapter 3 pointer | Primary code | Artifacts |
|---|---|---|---|---|
| Opening (unnumbered) | States that Ch4 implements Ch3; points scores to Ch5 | 3.9 (environment not repeated) | — | — |
| 4.1 | Shared data pipeline, run once | 3.2–3.5, 3.7 | `src/run_preprocessing.py` | `data/processed/<dataset>/` |
| 4.1.1 | Load local CSVs | 3.2 | `src/data_loading.py`, `src/config.py` | `data/raw/Ferenc.csv`, `data/raw/Viszkok.csv` |
| 4.1.2 | Filter + isolate + train-only quantile | 3.3 | `src/preprocessing.py` | `metadata.json`, `quantile_transformer.joblib`, `positional_columns.parquet` |
| 4.1.3 | Stratified 60/20/20 | 3.4 | `src/splitting.py` | `X_*.parquet`, `y_*.parquet` |
| 4.1.4 | Equation 3.1 weights, no resampling | 3.5 | `src/class_weights.py` | `class_weights`, `scale_pos_weight` in metadata |
| 4.2 | Wrappers: `tune` → `train` → `predict_proba` | 3.6 | `src/run_*.py`, `src/models/` | `models/<dataset>/` |
| 4.2.1 | Adopted XGBoost | 3.6.3 | `src/models/xgboost_model.py` | `xgboost.joblib` |
| 4.2.2 | Reimplemented TabM | 3.6.1, Eq. 3.2 | `src/models/tabm_model.py` | `tabm.pt`, `tabm_arch.json` |
| 4.2.3 | Reimplemented FT-Transformer | 3.6.2 | `src/models/ft_transformer.py` | `ft_transformer.pt`, `ft_transformer_arch.json` |
| 4.3 | Optuna TPE, early stop, final retrain | 3.7, Table 3.3 | model `tune()` / `_training_loop` | `tuning` object in each metrics JSON |
| 4.3.1 | 100 trials, no MedianPruner | Table 3.3 XGBoost | `xgboost_model.py` | — |
| 4.3.2 | 50 trials, MedianPruner, patience 20 | Table 3.3 TabM / FT | `tabm_model.py`, `ft_transformer.py` | — |
| 4.3.3 | Retrain clock = table “Retrain (min)” | 3.8.2 | `train()` in each wrapper | `retraining_time_min` |
| 4.4 | How evaluation fields are written | 3.8 | `src/evaluate.py`, `src/compare.py` | `results/metrics/` |
| 4.4.1 | PR-AUC, F2, MCC @ 0.5 | 3.8.1 | `compute_performance_metrics` | JSON `performance` |
| 4.4.2 | Latency vs retrain; size; peak RAM | 3.8.2 | `measure_inference_latency_ms`, `train()` | JSON `efficiency` |
| 4.4.3 | Assemble Chapter 5 table, no numbers here | 3.8 | `src/compare.py` | `comparison_table.json`, `comparison_table.md` |

Command that produces processed data:

```bash
python -m src.run_preprocessing --dataset both
```

Commands that produce the six result files, then the comparison table:

```bash
python -m src.run_xgboost --dataset both
python -m src.run_tabm --dataset both
python -m src.run_ft_transformer --dataset both
python -m src.compare
```

---

## Tables in the draft (Chapter 4 only)

| ID in draft | Title | Data source | Word note |
|---|---|---|---|
| Table 4.1 | Realized column removals | `data/processed/ferenc/metadata.json`, `data/processed/viszkok/metadata.json` → `preprocessing_report` | Proposal Table 4.1 was the Gantt chart. In the **final** thesis this numbering is free; keep or renumber to match the faculty template. |
| Table 4.2 | Realized split sizes | same metadata: `n_train`, `n_val`, `n_test`, `n_features` | Viszkok train is **4,822**, not the rounded ~4,823 in Table 3.2. |

Do not paste `results/metrics/comparison_table.md` into Chapter 4.

---

## Code excerpts (keep these eight)

| # | Caption in draft | File | What to tell the reader in one line |
|---|---|---|---|
| 1 | train-only QuantileTransformer | `src/preprocessing.py` | Fit on train only; val/test are transformed later. |
| 2 | two-step stratified 60/20/20 | `src/splitting.py` | Two `train_test_split` calls, both `stratify`, seed 42. |
| 3 | Equation 3.1 and XGBoost ratio | `src/class_weights.py` | Train labels only; same ratio for all three models. |
| 4 | split before normalization | `src/run_preprocessing.py` | Filter → split → transform → weights. |
| 5 | `TabMEnsembleLayer.forward` | `src/models/tabm_model.py` | Code of Eq. 3.2; do not re-derive it. |
| 6 | Optuna + prune + patience | `src/models/tabm_model.py` | TPE + MedianPruner; early stop 20 epochs. |
| 7 | test-set performance fields | `src/evaluate.py` | PR-AUC on probabilities; F2/MCC at 0.5. |
| 8 | latency; table field mapping | `src/evaluate.py`, `src/compare.py` | Latency = time-to-use; `compare.py` only copies JSON. |

In Word, keep excerpts short. Do not paste full runners.

---

## Realized numbers (for fact-checking while you copy)

Use these if Word autocorrect or rounding drifts.

**Features after filtering:** Ferenc 32, Viszkok 54.

**Splits (seed 42):** Ferenc 7,275 / 2,425 / 2,425. Viszkok 4,822 / 1,608 / 1,608.

**`scale_pos_weight` (train only):** Ferenc 7.1013, Viszkok 7.3715.

**Ferenc drops:** identifiers `name, longname, path, full_repo_path`; zero-var `CLLC`; collinear drop `HBUGS` (keep `HVOL`), `HTIME` (keep `HEFF`); positional `line, column, endline, endcolumn`.

**Viszkok drops:** identifiers plus `hash, type`; zero-var `WarningBlocker, WarningCritical, WarningInfo, WarningMinor`; collinear drop `CLLC` (keep `CLC`), `HVOL` (keep `HBUGS`), `HTIME` (keep `HEFF`), `HON_T` (keep `HLEN`), `HVOC` (keep `HON_D`); positional plus `start_line, start_column, end_line, end_column`.

---

## Citations already used in the draft

Add these to the thesis bibliography if they are not already there:

- Ferenc et al. (2019) — dataset
- Viszkok et al. (2021) — dataset
- Chen and Guestrin (2016) — XGBoost library
- Gorishniy et al. (2025) — TabM (reimplementation)
- Gorishniy et al. (2021) — FT-Transformer (reimplementation)
- Akiba et al. (2019) — Optuna TPE

scikit-learn, pandas, PyTorch, joblib are named as tools; Section 3.9 already lists versions.

---

## Chapter 5 handoff (do not write in Chapter 4)

`src/compare.py` writes the comparison table that Chapter 5 should present and discuss.

Columns: Model, PR-AUC, F2, MCC, Latency (ms/sample), Retrain (min), Size (MB), Peak RAM (MB).

Supervisor rule to discuss in Chapter 5, not 4: if **inference latency** (time-to-use) were similar to XGBoost, pick the highest accuracy. Chapter 4 only explains that this claim is checked against `inference_latency_ms_per_sample`, not against retrain time or Optuna duration.

---

## Word transfer notes

- Remove `[ ]` / `[x]` from headings in the submitted thesis.
- Convert backtick identifiers to the thesis code style (often Cambria or Courier at 10 pt).
- Keep Table 4.1 and 4.2 captions above or below as the faculty template requires.
- Opening paragraphs of the draft (before 4.1) can stay as the chapter lead-in; they are not a numbered subsection.
- `tabm` in `requirements.txt` is a **reference** dependency; training does not import it. Do not write that the official package was used for the reported runs.
