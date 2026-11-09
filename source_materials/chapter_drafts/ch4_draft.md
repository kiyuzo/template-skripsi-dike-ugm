# Chapter 4
# IMPLEMENTATION

This chapter describes how the research methodology in Chapter 3 was realized as executable software. The focus is on the implementation environment, repository organization, data and model pipelines, evaluation instrumentation, verification of the build, and faults encountered during development. Comparative performance numbers and scientific discussion are reported in Chapter 5.

---

## 4.1 Implementation Environment

All experiments were executed on a single workstation to keep training and efficiency measurements reproducible. The hardware and software configurations used in this study are summarized below.

**Table 4.1.** Hardware configuration of the implementation environment

| Component | Specification |
|---|---|
| Processor | Intel Core i7-14650HX |
| System memory | 16 GB RAM |
| Graphics processing unit | NVIDIA GeForce RTX 4060 (8 GB VRAM) |
| Training accelerator for TabM and FT-Transformer | CUDA (GPU) |

**Table 4.2.** Software configuration of the implementation environment

| Component | Version / configuration |
|---|---|
| Programming language | Python 3.12.4 |
| Deep learning framework | PyTorch 2.5.1 (CUDA build) |
| Gradient boosting library | XGBoost 2.1.3 |
| Hyperparameter optimization | Optuna 4.1.0 (Tree-structured Parzen Estimator) |
| Machine learning utilities | scikit-learn 1.5.2 |
| Data handling | pandas, NumPy, SciPy, PyArrow |
| Model serialization | joblib (XGBoost); `torch.save` (TabM, FT-Transformer) |
| Memory profiling | psutil |
| Global random seed | 42 |

A CUDA-enabled PyTorch build was installed before the remaining Python dependencies so that TabM and FT-Transformer could train on the GPU. If CUDA is unavailable, those models still run on the CPU, but training becomes substantially slower and is not the configuration reported in this thesis. Inference latency, by contrast, was measured on the CPU with a single thread, following the efficiency protocol in Section 3.8.2.

Before any long training run, `environment_check.py` was executed to verify the Python version, package imports, CUDA visibility to PyTorch, and structural validity of the raw datasets.

---

## 4.2 Project Structure and Module Integration

The implementation is organized as a modular Python package so that preprocessing is executed once and reused by all models. Figure 4.1 shows the main directory layout used in this study.

```
JS-Vulnerability Detection/
├── data/
│   ├── raw/
│   │   ├── Ferenc.csv
│   │   └── Viszkok.csv
│   └── processed/
│       ├── ferenc/
│       └── viszkok/
├── models/
│   ├── ferenc/
│   └── viszkok/
├── results/
│   └── metrics/
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
│   ├── compare.py
│   └── models/
│       ├── xgboost_model.py
│       ├── ft_transformer.py
│       └── tabm_model.py
├── environment_check.py
├── requirements.txt
└── README.md
```

**Figure 4.1.** Repository directory structure of the implementation.

Raw tabular datasets are stored under `data/raw/`. The preprocessing runner writes normalized train, validation, and test splits, the fitted QuantileTransformer, and metadata to `data/processed/<dataset>/`. Model checkpoints are stored under `models/<dataset>/`. Per-model evaluation records are written as JSON files under `results/metrics/`.

Central constants (paths, seed, split ratios, identifier and positional column names, and the collinearity threshold) are defined in `src/config.py` so that preprocessing, splitting, and training cannot silently diverge. The three model runners never re-implement preprocessing; they only load the shared processed artifacts. This integration choice enforces the fair comparison design stated in Chapter 3: observed differences come from the models, not from inconsistent input features or splits.

The script `src/compare.py` only aggregates existing metrics JSON files into comparative tables. It does not train models and is therefore treated as results tooling for Chapter 5 rather than part of the training implementation described here.

---

## 4.3 Data Pipeline Implementation

Following the data handling design in Sections 3.2–3.5, the data pipeline was implemented as a single end-to-end runner (`src/run_preprocessing.py`) that loads, validates, filters, splits, normalizes, and exports each dataset.

### 4.3.1 Loading and structural validation

`src/data_loading.py` loads each CSV and checks that it matches the thesis expectations before any modeling begins. Ferenc et al. (2019) must contain 12,125 rows with a binary `Vuln` label and a vulnerable-class rate near 12.34%. Viszkok et al. (2021) must contain 8,038 rows with a vulnerable-class rate near 11.94%. The loader also requires a sufficient number of numeric metric columns so that a raw source-code dump or an unrelated table cannot proceed unnoticed.

### 4.3.2 Feature filtering and positional isolation

`src/preprocessing.py` applies the three preprocessing stages from Section 3.3. Identifier columns are dropped. Positional columns (`line`, `column`, `endline`, `endcolumn`, and the Viszkok `start_*` / `end_*` duplicates) are isolated into a separate table and are never used as model features, because their association with `Vuln` is treated as an artifact of dataset construction. Zero-variance columns and near-perfectly collinear pairs (|correlation| ≥ 0.999) are detected from the data rather than hard-coded from the proposal lists, so both datasets can be handled correctly even when the problematic columns differ (for example, `CLLC` is zero-variance in Ferenc and collinear with `CLC` in Viszkok).

After filtering, the models receive **32** features for Ferenc and **54** features for Viszkok. These counts are lower than the raw predictor totals discussed in Chapter 3 because identifier, positional, zero-variance, and collinear columns have been removed.

### 4.3.3 Stratified splitting and normalization

`src/splitting.py` performs a stratified 60%/20%/20% train/validation/test split with seed 42, preserving the class ratio in every partition. The resulting sizes are 7,275 / 2,425 / 2,425 for Ferenc and 4,822 / 1,608 / 1,608 for Viszkok.

A `QuantileTransformer` with a normal output distribution is then fitted **only on the training split** and applied to validation and test. Fitting on the full dataset before splitting would leak test-set distributional information into the transform and conflict with the requirement in Section 3.7 that the held-out test set remain unused during preparation and tuning. The fitted transformer is saved as `quantile_transformer.joblib` for reuse.

### 4.3.4 Class-imbalance weights

`src/class_weights.py` computes class weights from the training labels only, using Equation 3.1, and the positive-class ratio `scale_pos_weight = N_neg / N_pos`. The realized values are approximately 7.10 for Ferenc and 7.37 for Viszkok. The same ratio is passed to XGBoost as `scale_pos_weight` and to TabM / FT-Transformer as `pos_weight` in `BCEWithLogitsLoss`. This parameter is fixed analytically; it is not searched by Optuna.

### 4.3.5 Exported artifacts

For each dataset, preprocessing writes the following under `data/processed/<dataset>/`:

- `X_train.parquet`, `X_val.parquet`, `X_test.parquet`
- `y_train.parquet`, `y_val.parquet`, `y_test.parquet`
- `positional_columns.parquet` (retained for inspection, unused in training)
- `quantile_transformer.joblib`
- `metadata.json` (feature list, class weights, split sizes, drop report, seed)

The train-only QuantileTransformer fit is the research-critical control in this stage and is illustrated below.

```python
# Fit on training features only, then transform all splits
qt = QuantileTransformer(
    n_quantiles=min(1000, len(X_train)),
    output_distribution="normal",
    random_state=42,
)
qt.fit(X_train)
X_train_t = qt.transform(X_train)
X_val_t = qt.transform(X_val)
X_test_t = qt.transform(X_test)
```

---

## 4.4 Model and Training Pipeline Implementation

Following Sections 3.6–3.7 and Table 3.3, three models were implemented under a common runner pattern: load the shared processed splits, tune hyperparameters on the validation set by maximizing PR-AUC with Optuna, retrain with the selected configuration, evaluate once on the held-out test set, and save both the checkpoint and a metrics JSON file. The test set is never accessed during tuning.

### 4.4.1 XGBoost baseline

XGBoost was implemented with the official library through `src/models/xgboost_model.py` and `src/run_xgboost.py`. It serves as the gradient-boosted decision-tree baseline that is well suited to heterogeneous tabular software metrics. Hyperparameter search uses 100 Optuna TPE trials with early stopping of 50 rounds. The search space matches Table 3.3 (`n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_alpha`, and `reg_lambda`). Training uses `n_jobs=1` so that later CPU latency measurements remain single-threaded. The fitted model is saved as `models/<dataset>/xgboost.joblib`.

### 4.4.2 FT-Transformer

FT-Transformer was reimplemented in PyTorch in `src/models/ft_transformer.py`, following Gorishniy et al. (2021), and applied to the Ferenc and Viszkok datasets through `src/run_ft_transformer.py`. Each numeric feature is mapped to a token embedding by a per-feature linear projection; a learnable `[CLS]` token is prepended; PreNorm Transformer layers apply multi-head self-attention and a feed-forward block; and the final `[CLS]` representation is passed through a Linear → ReLU → LayerNorm → Linear head that emits a logit. Training uses `BCEWithLogitsLoss` with the dataset-specific `pos_weight`; probabilities are obtained with a sigmoid at inference time.

Tuning uses 50 Optuna TPE trials, early stopping with patience of 20 epochs, and a median pruner. The token dimension is sampled as a multiple of the number of attention heads so that multi-head attention remains valid. Training runs on CUDA when available. Checkpoints are stored as `models/<dataset>/ft_transformer.pt` together with `ft_transformer_arch.json`.

The feature tokenizer, which is specific to this architecture, is shown below.

```python
# Per-feature linear projection: T_j = x_j * W_j + b_j
def forward(self, x):
    # x: (batch, n_features)
    return x.unsqueeze(-1) * self.weight + self.bias
    # → (batch, n_features, d_token)
```

### 4.4.3 TabM

TabM was reimplemented in PyTorch in `src/models/tabm_model.py` according to the parameter-efficient ensembling formulation in Gorishniy et al. (2025) and Equation 3.2 of this thesis, and executed through `src/run_tabm.py`. The official `tabm` package is listed in `requirements.txt` as a reference dependency; the training code uses the local reimplementation rather than that package’s training API.

Each input vector is replicated `k` times. A shared weight matrix `W` performs the main linear map, while member-specific adapters `s_i`, `r_i`, and `b_i` introduce diversity at low memory cost. After ReLU and dropout on each stream, a shared linear head produces `k` logits that are averaged into one prediction. The loss and `pos_weight` handling match the FT-Transformer setup. Tuning again uses 50 Optuna TPE trials, early stopping of 20 epochs, and a median pruner, with `k` searched in `[4, 64]`. Checkpoints are stored as `models/<dataset>/tabm.pt` and `tabm_arch.json`.

The adapter layer that realizes Equation 3.2 is the research-critical fragment for this model:

```python
# ℓ_i(x_i) = r_i ⊙ (W(s_i ⊙ x_i)) + b_i
X = self.s * X      # input adapter
X = self.W(X)        # shared linear map
X = self.r * X      # output adapter
X = X + self.b       # member bias
X = F.relu(X)
X = self.dropout(X)
```

---

## 4.5 Evaluation Instrumentation

The evaluation metrics themselves are defined in Section 3.8. This section describes only how those metrics were instrumented in code so that every model is measured identically.

`src/evaluate.py` provides shared helpers used by all runners. On the held-out test set it computes PR-AUC from predicted probabilities, and F2-score (β = 2) and Matthews Correlation Coefficient from hard labels at threshold 0.5. Efficiency helpers measure:

1. **Inference latency** — mean milliseconds per sample on CPU with one thread, after one warm-up call and ten timed repetitions.  
2. **Retraining time** — wall-clock minutes of the final retrain with the best hyperparameters (not the full Optuna search).  
3. **Model size** — size in megabytes of the saved checkpoint file.  
4. **Peak memory** — peak process resident set size during a test-set `predict_proba` call, sampled with `psutil`.

Each runner writes these quantities, together with tuning metadata and best hyperparameters, to `results/metrics/<model>_<dataset>.json`. For deep learning runs the JSON also records `device_used_for_training`. Comparative aggregation of the six JSON files into a single results table is performed later and reported in Chapter 5.

---

## 4.6 Verification of the Implementation

Before accepting the implementation as complete, the following verification steps were applied.

1. **Environment and data checks.** `environment_check.py` confirmed Python 3.12, successful imports of the required packages, CUDA visibility for PyTorch, and structural validity of both raw CSVs.  
2. **Preprocessing consistency.** The produced split sizes and `scale_pos_weight` values matched the design in Sections 3.4–3.5 (Ferenc ≈ 7.10; Viszkok ≈ 7.37). Feature counts after filtering were 32 and 54, respectively, and were recorded in `metadata.json`.  
3. **Shared inputs.** XGBoost, TabM, and FT-Transformer all consumed the same processed parquet files for a given dataset. Hyperparameter search used only the training and validation splits.  
4. **Artifact completeness.** For each dataset, the expected checkpoints under `models/<dataset>/` and metrics JSON files under `results/metrics/` were produced. TabM and FT-Transformer runs recorded CUDA as the training device.

These checks verify that the software matches the Chapter 3 protocol. The numerical outcomes of the comparative evaluation are presented in Chapter 5.

---

## 4.7 Faults Encountered and Corrective Actions

Several faults occurred during implementation and were corrected before the final experimental runs.

**Incorrect file placed in the Ferenc data path.** During early setup, an unrelated CSV that did not belong to this project was stored as `data/raw/Ferenc.csv`. Structural validation reported a mismatch with the expected row count, label distribution, and metric schema. The file was replaced with the correct Ferenc et al. (2019) software-metrics dataset (12,125 rows, approximately 12.34% vulnerable functions). The validation routines in `src/data_loading.py` and `environment_check.py` remain in place so that a similar mistake fails immediately. The Ferenc file used for all reported experiments is the corrected metrics table.

**Unreliable model-size field for Viszkok XGBoost.** The metrics JSON could report an implausibly small `model_size_mb`, while the on-disk `xgboost.joblib` checkpoint had the correct size. For comparative reporting, checkpoint file size is preferred when aggregating results in Chapter 5.

**Stale project documentation.** After the data path was corrected and all models had been trained, the README still described an outdated mismatch status. It was rewritten as an English replication guide aligned with the finished pipeline.

**Dependency and accelerator pitfalls.** Omitting a CUDA PyTorch build caused deep models to fall back to CPU. Missing `psutil` or `pyarrow` blocked peak-memory measurement or parquet I/O until those packages were added to `requirements.txt`.

---

## 4.8 Code Documentation and Replication

Source modules include section-level comments that map code blocks to the corresponding methodology sections. The complete codebase, together with `README.md` and `requirements.txt`, serves as the full documentation of the implementation.

The intended replication sequence is:

1. Create a Python 3.12 environment and install CUDA PyTorch, then `requirements.txt`.  
2. Place `Ferenc.csv` and `Viszkok.csv` under `data/raw/`.  
3. Run `python environment_check.py`.  
4. Run `python -m src.run_preprocessing --dataset both`.  
5. Run `python -m src.run_xgboost --dataset both`, `python -m src.run_tabm --dataset both`, and `python -m src.run_ft_transformer --dataset both`.  

This sequence regenerates the processed data, model checkpoints, and per-model metrics files used in Chapter 5.
