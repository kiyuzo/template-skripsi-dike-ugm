# Chapter 4 (Implementation) — Writing Notes

**Language:** English only (thesis and all authored notes). Faculty guideline sources may be Indonesian; requirements below are restated in English.

**What this file is:** Personal writing notes only (not submitted). Plain-language summary of what the code does so you can write Chapter 4 yourself. No sample thesis paragraphs.

**Scope boundary (your decision):**
- **Chapter 4 — Implementation:** environment, tools, step-by-step build, module integration, research-important code, testing *approach*, fault tracing during implementation.
- **Chapter 5 — Results and Discussion:** metric tables, comparative numbers, scientific discussion. Also `src/compare.py` (it only reads saved JSON files and builds Table 4.1; it does not train models).

**Proposal vs final thesis structure (confirmed in guidelines):**
- **Proposal** Chapter 4 = Research Schedule (Gantt chart).
- **Final thesis** Chapter 4 = **Implementation** (guidelines § final-thesis main body).
- Those are different documents; do not copy the proposal’s Chapter 4 title into the final thesis.

**Guideline note on “test results” in Chapter 4:** The Academic Thesis Guidelines also say Implementation should describe the testing approach, present test outcomes, and document fault tracing. Per your decision, put **numeric result tables and discussion in Chapter 5**. In Chapter 4, describe *how* you verified the implementation (checks, protocol, what was measured) without pasting the comparative score tables.

The guidelines do **not** prescribe fixed subsection titles for Implementation. The outline below is invented to match those required contents.

---

## Suggested Chapter 4 outline

1. Implementation environment (hardware + software — full detail)
2. Project structure and module integration
3. Data pipeline implementation (load → validate → preprocess → split → weights → save)
4. Model implementation (XGBoost; FT-Transformer; TabM)
5. Tuning and training-loop implementation
6. Evaluation instrumentation (what is measured and how — not the score tables)
7. Testing approach used to verify the implementation
8. Faults during implementation and how they were fixed
9. Code documentation / replication pointers (brief)

---

## Clarification: Ferenc dataset fault (answer to your Q3)

**Current `data/raw/Ferenc.csv` is correct** (12,125 rows, ~12.34% vulnerable, static software metrics + `Vuln`).

What went wrong earlier was **not** that Ferenc “as a dataset” is wrong. During setup you temporarily placed **another, irrelevant CSV** (unrelated to this project) into the Ferenc path. Validation / early README flagged that mismatch. You then replaced it with the real Ferenc metrics file.

For Chapter 4 fault-tracing, phrase it like that: accidental wrong file in the data path → structural validation failed → corrected by restoring the proper Ferenc et al. (2019) metrics CSV. Do **not** write that the thesis still uses a mismatched Ferenc file.

---

## 1. Implementation environment (report everything tracked)

### Hardware (thesis §3.9 / your workstation)

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060, 8 GB VRAM |
| System RAM | 16 GB |
| CPU | Intel Core i7-14650HX |
| Training device for TabM / FT-Transformer | CUDA (recorded in result JSON as `device_used_for_training: cuda`) |

### Software stack

| Item | Value |
|---|---|
| Language | Python 3.12.x (thesis text: 3.12.4) |
| Deep learning | PyTorch 2.5.1 (CUDA build) |
| GBDT | XGBoost 2.1.3 |
| Tuning | Optuna 4.1.0 (TPE) |
| Preprocessing / metrics | scikit-learn 1.5.2 |
| Data | pandas, NumPy, SciPy, PyArrow (parquet) |
| Checkpoint I/O | joblib (XGBoost), `torch.save` (DL state dicts) |
| Efficiency measurement | psutil (peak RSS during inference) |
| Global seed | `42` (`src/config.py`) |

Install order that matters: CUDA PyTorch first, then `requirements.txt`. Without CUDA, TabM/FT still run but fall back to CPU and are much slower.

### Replication commands (project root)

```bash
python environment_check.py
python -m src.run_preprocessing --dataset both
python -m src.run_xgboost --dataset both
python -m src.run_tabm --dataset both
python -m src.run_ft_transformer --dataset both
# python -m src.compare   → Chapter 5 tooling, not Chapter 4 training
```

---

## 2. Project structure and module integration

### 2.1 Directory tree (non-empty / used paths)

```
JS-Vulnerability Detection/
├── data/
│   ├── raw/
│   │   ├── Ferenc.csv
│   │   └── Viszkok.csv
│   └── processed/
│       ├── ferenc/          # parquet splits, QT, metadata.json
│       └── viszkok/
├── models/
│   ├── ferenc/
│   │   ├── xgboost.joblib
│   │   ├── tabm.pt
│   │   ├── tabm_arch.json
│   │   ├── ft_transformer.pt
│   │   └── ft_transformer_arch.json
│   └── viszkok/
│       ├── xgboost.joblib
│       ├── tabm.pt
│       ├── tabm_arch.json
│       ├── ft_transformer.pt
│       └── ft_transformer_arch.json
├── results/
│   └── metrics/             # per-model JSON + comparison outputs (Ch5)
├── src/
│   ├── config.py
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── splitting.py
│   ├── class_weights.py
│   ├── evaluate.py
│   ├── run_preprocessing.py
│   ├── run_xgboost.py
│   ├── run_ft_transformer.py
│   ├── run_tabm.py
│   ├── compare.py           # Chapter 5 aggregation only
│   └── models/
│       ├── xgboost_model.py
│       ├── ft_transformer.py
│       └── tabm_model.py
├── environment_check.py
├── requirements.txt
└── README.md
```

`data/processed/<dataset>/` typically contains: `X_{train,val,test}.parquet`, `y_{train,val,test}.parquet`, `positional_columns.parquet`, `quantile_transformer.joblib`, `metadata.json`.

### 2.2 Layers → paths → role

| Layer | Path(s) | Role |
|---|---|---|
| Config | `src/config.py` | Paths, seed 42, 60/20/20, identifier/positional lists, collinearity 0.999 |
| Raw data | `data/raw/Ferenc.csv`, `data/raw/Viszkok.csv` | Input CSVs |
| Data prep code | `src/data_loading.py`, `src/preprocessing.py`, `src/splitting.py`, `src/class_weights.py`, `src/run_preprocessing.py` | Validate → filter → split → normalize → weights |
| Processed data | `data/processed/ferenc/`, `data/processed/viszkok/` | Shared inputs for all models |
| Model code | `src/models/xgboost_model.py`, `src/models/ft_transformer.py`, `src/models/tabm_model.py` | Tune + train + predict |
| Runners | `src/run_xgboost.py`, `src/run_ft_transformer.py`, `src/run_tabm.py` | End-to-end train/eval per model |
| Checkpoints | `models/ferenc/`, `models/viszkok/` | Saved `.joblib` / `.pt` + arch JSON |
| Shared eval | `src/evaluate.py` | PR-AUC, F2, MCC, latency, size, peak RAM helpers |
| Metrics output | `results/metrics/*.json` | Per model × dataset evaluation records |
| Health check | `environment_check.py` | Python, imports, CUDA, dataset structure |
| Compare (Ch5) | `src/compare.py` → `results/metrics/comparison_table.*` | Aggregate tables only |

**Integration rule:** preprocess once into `data/processed/`; all three runners read those artifacts and write checkpoints under `models/` plus metrics under `results/metrics/`. Fair comparison depends on that shared processed input.

---

## 3. Adopted vs reimplemented (cite sources; avoid “from scratch”)

| Component | How to describe it | Cite |
|---|---|---|
| XGBoost | Adopted GBDT library; chosen as a strong tabular baseline that fits this software-metrics task | XGBoost / Chen & Guestrin; thesis §3.6.3 |
| Optuna TPE | Adopted hyperparameter search | Optuna |
| scikit-learn utilities | Adopted (QuantileTransformer, stratified split, metrics) | scikit-learn |
| PyTorch | Adopted DL framework | PyTorch |
| FT-Transformer | **Reimplementation** of Gorishniy et al. (2021) Feature Tokenizer + Transformer, applied/tested on Ferenc & Viszkok tabular JS metrics | Gorishniy et al., 2021 |
| TabM | **Reimplementation** of Gorishniy et al. (2025) parameter-efficient ensemble (Eq. 3.2), applied/tested on the same datasets | Gorishniy et al., 2025. `tabm` pip package is in `requirements.txt` as a reference dependency; training uses `src/models/tabm_model.py`, not that package API |
| Pipeline / validation / runners / evaluate | Original glue for this thesis experiment | your repository |

Framing for the thesis narrative: you are **implementing and evaluating** Gorishniy’s tabular deep models (TabM, FT-Transformer) on these JS vulnerability metric datasets, with **XGBoost** as the conventional tabular model that fits this data type.

---

## 4. Data pipeline — step by step

### 4.1 Config (`src/config.py`)

- Seed 42; split 60/20/20; target `Vuln`
- Identifier columns dropped; positional columns isolated
- Collinearity threshold 0.999

### 4.2 Environment check (`environment_check.py`)

Before long runs: Python 3.12, required imports (including `pyarrow`, `psutil`), CUDA visibility, CSV structural validation. Console only.

### 4.3 Load + validate (`src/data_loading.py`)

Open CSV and check thesis expectations:

| Dataset | Rows | Vuln rate band | Target |
|---|---|---|---|
| Ferenc | 12,125 | ~12.34% (accept 10–15%) | binary `Vuln` |
| Viszkok | 8,038 | ~11.94% (accept 9–14%) | binary `Vuln` |

Also requires enough numeric metric columns (rejects raw-code / unrelated dumps). This is what caught the wrong file in the Ferenc path.

### 4.4 Preprocess (`src/preprocessing.py`) — thesis §3.3

1. Drop identifiers  
2. Isolate positional columns (saved, never used as features)  
3. Drop zero-variance columns (**detected** from data)  
4. Drop one of each near-collinear pair (|corr| ≥ 0.999, **detected**)  
5. `QuantileTransformer` (`output_distribution="normal"`, `n_quantiles=min(1000, n_train)`, seed 42) fit on **train only**, then applied to val/test  

After filtering: Ferenc **32** features, Viszkok **54** features (see `metadata.json`). Proposal “35 / 61” are pre-hygiene counts.

### 4.5 Split (`src/splitting.py`)

Stratified 60/20/20, seed 42 (two-step `train_test_split`).

| Dataset | Train | Val | Test |
|---|---|---|---|
| Ferenc | 7,275 | 2,425 | 2,425 |
| Viszkok | 4,822 | 1,608 | 1,608 |

**Order:** split **before** fitting QuantileTransformer.

### 4.6 Class weights (`src/class_weights.py`)

From **train labels only**:

- Eq. 3.1: `W = N_total / (C × N_class)`
- `scale_pos_weight = N_neg / N_pos` ≈ **7.10** (Ferenc), **7.37** (Viszkok)  
  → XGBoost `scale_pos_weight` and DL `BCEWithLogitsLoss(pos_weight=...)`  
  → **not** Optuna-tuned

### 4.7 Runner (`src/run_preprocessing.py`)

Writes under `data/processed/<dataset>/`:

- `X_*.parquet`, `y_*.parquet`
- `positional_columns.parquet` (unused by models)
- `quantile_transformer.joblib` (reuse; never refit on full data)
- `metadata.json` (features, weights, drops, sizes, seed)

---

## 5. Model implementation — step by step

Shared runner pattern for all three:

1. Load identical processed splits + `scale_pos_weight` from metadata  
2. Optuna maximize **validation PR-AUC** (test unused)  
3. Retrain with best params (this wall-clock = `retraining_time_min`)  
4. One test-set prediction pass  
5. Call shared eval helpers  
6. Save checkpoint + `results/metrics/<model>_<dataset>.json`

### 5.1 XGBoost (`xgboost_model.py`, `run_xgboost.py`)

- Library `XGBClassifier` wrapper
- 100 TPE trials; early stopping 50 rounds; `n_jobs=1`; `eval_metric=aucpr`
- Search space = thesis Table 3.3 (n_estimators, max_depth, lr, subsample, colsample_bytree, min_child_weight, reg_alpha, reg_lambda)
- Output: `models/<ds>/xgboost.joblib`

### 5.2 FT-Transformer (`ft_transformer.py`, `run_ft_transformer.py`)

Reimplementation of Gorishniy et al. (2021):

1. Feature Tokenizer (per-feature linear embedding)  
2. `[CLS]` token  
3. PreNorm Transformer stack (MHSA + FFN)  
4. Head: Linear → ReLU → LayerNorm → Linear → logit  
5. `BCEWithLogitsLoss(pos_weight=...)`; sigmoid at inference  

- 50 trials; early stop 20 epochs; MedianPruner; Adam  
- `d_token` forced divisible by `n_heads`  
- Outputs: `ft_transformer.pt` + `ft_transformer_arch.json`

### 5.3 TabM (`tabm_model.py`, `run_tabm.py`)

Reimplementation of Gorishniy et al. (2025) / Eq. 3.2:

1. Replicate input **k** times  
2. Shared `W` + per-member adapters `s_i`, `r_i`, `b_i`  
3. ReLU + Dropout per member  
4. Shared head → mean of k logits  

- 50 trials; early stop 20 epochs; MedianPruner  
- Extra search: `k ∈ [4, 64]` (your runs: Ferenc k=32, Viszkok k=42)  
- Outputs: `tabm.pt` + `tabm_arch.json`

---

## 6. Evaluation instrumentation (Chapter 4: what you measure)

Implemented in `src/evaluate.py` (used by all runners). **Do not put the score tables here in Chapter 4** — only describe the instruments.

| Tracked quantity | How it is measured |
|---|---|
| PR-AUC | `average_precision_score` on test probabilities |
| F2-score | β=2, threshold 0.5 on hard labels |
| MCC | threshold 0.5 |
| Inference latency | ms/sample; CPU; 1 thread; 1 warmup + mean of 10 timed runs |
| Retraining time | minutes; **final retrain only** (not full Optuna wall time) |
| Model size | MB of saved checkpoint file |
| Peak RAM (inference) | peak process RSS via psutil while `predict_proba` runs on test |

Also recorded in DL JSONs: `device_used_for_training`, tuning trial count, sampler/pruner, best val PR-AUC, best hyperparameters, `n_features`.

---

## 7. Testing approach (verify implementation — not Chapter 5 discussion)

What belongs in Chapter 4:

1. Structural validation of CSVs + `environment_check.py` before training  
2. Confirm preprocess artifacts: split sizes, feature counts, `scale_pos_weight`  
3. Same processed inputs for all models; val-only tuning; test held out until final eval  
4. Confirm six metrics JSON files and checkpoints were produced  
5. Confirm CUDA was used for TabM/FT training  

What belongs in **Chapter 5** (not this notes’ focus): comparative PR-AUC/F2/MCC/efficiency tables, research-question answers, discussion. Tooling: `python -m src.compare` → `comparison_table.md` / `.json`.

---

## 8. Fault tracing during implementation (for Chapter 4)

### 8.1 Irrelevant file placed as Ferenc

- **What happened:** an unrelated dataset CSV was put in the Ferenc data path  
- **Symptom:** structural validation / early project notes reported mismatch (row count, label balance, or non-metrics content)  
- **Fix:** replace with the correct Ferenc et al. (2019) metrics file (12,125 rows, ~12.34% vulnerable)  
- **Prevention:** `data_loading.validate_dataset` + `environment_check.py` fail early  

### 8.2 Viszkok XGBoost `model_size_mb` in JSON

- JSON field could report a tiny wrong size; on-disk `xgboost.joblib` is the reliable size (~0.409 MB)  
- Corrected at aggregation time in `compare.py` (Chapter 5 tooling) by preferring checkpoint file size  

### 8.3 Stale README after the data fix

- Docs still described a mismatched Ferenc / unfinished models after the pipeline was complete  
- Fixed by rewriting README as an English replication guide  

### 8.4 Environment pitfalls (optional)

- CPU-only PyTorch → silent slow DL training  
- Missing `psutil` / `pyarrow` blocked eval or parquet I/O until added to `requirements.txt`

---

## 9. Which code excerpts are worth putting in Chapter 4

**Include** if it is specific to this research design:

- Train-only QuantileTransformer fit (leakage control)  
- Positional column isolation (dataset-construction bias)  
- Shared `scale_pos_weight` / `pos_weight` wiring for imbalance  
- TabM adapter equation (Eq. 3.2) in `TabMEnsembleLayer`  
- FT Feature Tokenizer + `[CLS]` pipeline  
- Single-thread CPU inference for fair latency  

**Skip** generic boilerplate everyone knows (plain `argparse` main, trivial JSON dumps, basic `DataLoader` loops) unless you need one line of context.

Point to the repository / appendix for full commented source.

---

## 10. File map for Chapter 4 writing

| Chapter 4 topic | Files |
|---|---|
| Environment | `requirements.txt`, `environment_check.py`, thesis §3.9 |
| Validation | `src/data_loading.py` |
| Preprocess / split / weights | `preprocessing.py`, `splitting.py`, `class_weights.py`, `run_preprocessing.py`, `metadata.json` |
| XGBoost | `models/xgboost_model.py`, `run_xgboost.py` |
| FT-Transformer | `models/ft_transformer.py`, `run_ft_transformer.py` |
| TabM | `models/tabm_model.py`, `run_tabm.py` |
| Measurement code | `evaluate.py` |
| **Not Chapter 4** | `compare.py`, `results/metrics/comparison_table.*` → Chapter 5 |

---

## 11. Checklist before drafting Chapter 4

- [ ] Title: **Implementation** (final thesis, not proposal “Research Schedule”)
- [ ] Full hardware + software detail (GPU, VRAM, RAM, CPU, versions, seed, CUDA)
- [ ] Step-by-step pipeline and module integration
- [ ] Cite adopted libs; describe TabM/FT as reimplementations of Gorishniy applied to these datasets; XGBoost as fitting tabular baseline
- [ ] Research-important code excerpts only
- [ ] Testing *approach* to verify implementation
- [ ] Fault tracing (wrong file in Ferenc path, etc.) — current Ferenc is correct
- [ ] **No** comparative result tables / no deep discussion (those → Chapter 5)
- [ ] `compare.py` mentioned only as out-of-scope for Ch4 / belongs with results aggregation in Ch5
- [ ] Entire manuscript text in English
