# Chapter 4

# IMPLEMENTATION

**Progress** (tick when copied or reviewed):

- [ ] 4.1 Data Pipeline Implementation
  - [ ] 4.1.1 Data acquisition
  - [ ] 4.1.2 Preprocessing
  - [ ] 4.1.3 Data split
  - [ ] 4.1.4 Class imbalance
- [ ] 4.2 Model and Training Pipeline Implementation
  - [ ] 4.2.1 XGBoost
  - [ ] 4.2.2 TabM
  - [ ] 4.2.3 FT-Transformer
- [ ] 4.3 Tuning and Training-Loop Implementation
  - [ ] 4.3.1 XGBoost search
  - [ ] 4.3.2 TabM and FT-Transformer search
  - [ ] 4.3.3 Final retrain
- [ ] 4.4 Evaluation and Comparison Implementation
  - [ ] 4.4.1 Performance fields
  - [ ] 4.4.2 Efficiency fields, including training and testing time
  - [ ] 4.4.3 How the comparison table is generated

## [ ] 4.1 Data Pipeline Implementation

The data pipeline is executed once per dataset by using `python -m src.run_preprocessing`. Preprocessing, splitting, normalization, and class-weight computation live in separate modules and are then run by using `src/run_preprocessing.py`. The three model runners never reload the raw CSV files. They read the processed split files and `metadata.json` saved in `data/processed/ferenc/` and `data/processed/viszkok/`. This integration keeps the feature space identical across all model runners to ensure fairness as stated in Section 3.3.

### [ ] 4.1.1 Data Acquisition

The published tabular data are placed inside `data/raw/Ferenc.csv` and `data/raw/Viszkok.csv`. `src/data_loading.py` reads each file using pandas and defines `load_ferenc()` and `load_viszkok()` functions. The binary target column is `Vuln`, as stated in both source datasets (Ferenc et al., 2019; Viszkok et al., 2021). Paths, global seed, and split ratios are defined in `src/config.py` therefore later stages cannot drift from the original configuration defined in Section 3.4.

### [ ] 4.1.2 Preprocessing

The preprocessing step starts with feature filtering, implemented in `src/preprocessing.py` by using `filter_features()`. It applies the same four rules defined in Section 3.3 to both datasets, using whichever columns are present. Identifier columns listed in `config.IDENTIFIER_COLUMNS` are removed, positional columns listed in `config.POSITIONAL_COLUMNS` are copied aside and are excluded from training, numeric columns with at most one unique value are dropped, and if two remaining numeric columns are almost the same (Pearson correlation ≥ 0.999), the earlier column is kept and the later one is dropped. The 0.999 cutoff comes from `config.COLLINEARITY_THRESHOLD`. A `PreprocessingReport` records the dropped columns and serializes them into `metadata.json`. In Section 3.2, it is stated that there are 35 static metrics for Ferenc et al. (2019) dataset and 61 static metrics for Viszkok et al. (2021) dataset. However, after filtering with `filter_features()`, these counts are reduced to 32 features for Ferenc and 54 features for Viszkok. Table 4.1 lists the details of the removed columns, including the reason they are removed.

**Table 4.1.** Column removals details for Ferenc et al. (2019) dataset

| Step | Columns removed | Reason |
| --- | --- | --- |
| Identifiers | name, longname, path, full_repo_path | Metadata; no metric signal |
| Zero variance | CLLC | Constant column; no information |
| Collinearity (abs r ≥ 0.999) | HBUGS (kept HVOL), HTIME (kept HEFF) | Near-duplicate Halstead identities |
| Positional isolation | line, column, endline, endcolumn | Location artifact of dataset construction |

**Table 4.1.** Column removals details for Viszkok et al. (2021) dataset

| Step | Columns removed | Reason |
| --- | --- | --- |
| Identifiers | name, longname, path, full_repo_path, hash, type | Metadata; type is a constant label |
| Zero variance | WarningBlocker, WarningCritical, WarningInfo, WarningMinor | Constant columns; no information |
| Collinearity (abs r ≥ 0.999) | CLLC (kept CLC), HVOL (kept HBUGS), HTIME (kept HEFF), HON_T (kept HLEN), HVOC (kept HON_D) | Near-duplicate or derived pairs |
| Positional isolation | line, column, endline, endcolumn, start_line, start_column, end_line, end_column | Location artifact of dataset construction |

Positional columns are stored inside `positional_columns.parquet` but they are never passed to `train()` or `predict_proba()`.

For normalization, it is applied only after the split in order to make sure the test distribution can not enter the transformer. `fit_quantile_transformer()` constructs a scikit-learn `QuantileTransformer` with `output_distribution="normal"`, `n_quantiles = min(1000, n_train)`, and `random_state = 42`, then calls `fit` on the training matrix alone. `apply_quantile_transform()` transforms train, validation, and test with that fitted object. The fitted transformer is saved to `quantile_transformer.joblib`.

```python
# src/preprocessing.py — train-only QuantileTransformer
n_quantiles = min(1000, len(X_train))
qt = QuantileTransformer(
    n_quantiles=n_quantiles,
    output_distribution="normal",
    random_state=random_state,
)
qt.fit(X_train)
return qt
```

**Figure 4.1.** Train-only `QuantileTransformer` fitting in `src/preprocessing.py`.

### [ ] 4.1.3 Data Split

The 60/20/20 stratified partition from Section 3.4 is implemented in `src/splitting.py`. scikit-learn's `train_test_split` only cuts a matrix in two. Therefore, `train_test_split` is called twice sequentially. The first stratify splits the data into training data and the remainder (40%). The remainder is then split into two sets, validation set and test set. Both calls use the stratify method on the current label vector and `random_state = 42`. Table 4.2 reports the actual numbers of data that are also reported inside `metadata.json`. The Viszkok training split contains 4,822 rows, rather than the rounded 4,823 shown in Table 3.2. The difference is because of rounding in Table 3.2 and not a change of method.

```python
# src/splitting.py — two-step stratified 60/20/20 split
X_train, X_rest, y_train, y_rest = train_test_split(
    X, y,
    test_size=val_test_size,
    stratify=y,
    random_state=random_state,
)
relative_test_size = test_size / val_test_size
X_val, X_test, y_val, y_test = train_test_split(
    X_rest, y_rest,
    test_size=relative_test_size,
    stratify=y_rest,
    random_state=random_state,
)
```

**Figure 4.2.** Two-step stratified 60/20/20 split in `src/splitting.py`.

**Table 4.2.** Realized stratified split sizes (seed = 42)

| Dataset | Train (60%) | Validation (20%) | Test (20%) | Features after filtering |
| --- | --- | --- | --- | --- |
| Ferenc et al. (2019) | 7,275 | 2,425 | 2,425 | 32 |
| Viszkok et al. (2021) | 4,822 | 1,608 | 1,608 | 54 |

### [ ] 4.1.4 Class Imbalance

As stated in Section 3.5, class imbalance is handled using weights instead of resampling methods, based on Equation 3.1. The weights are computed from the training set only in `src/class_weights.py`. `compute_class_weights()` returns one weight for each class, while `compute_scale_pos_weight()` returns a single ratio of non-vulnerable to vulnerable training rows. This second value is needed because XGBoost expects `scale_pos_weight` and PyTorch expects `pos_weight`. Both values are then stored in `metadata.json` and later read by the model runners. The weight ratios are fixed at 7.1013 for Ferenc and 7.3715 for Viszkok.

```python
# src/class_weights.py — Equation 3.1 and the XGBoost ratio
def compute_class_weights(y_train, num_classes=2):
    n_total = len(y_train)
    counts = y_train.value_counts().to_dict()
    weights = {}
    for class_label, n_class in counts.items():
        weights[int(class_label)] = n_total / (num_classes * n_class)
    return weights

def compute_scale_pos_weight(y_train):
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    return n_neg / n_pos
```

**Figure 4.3.** Class-weight computation from Equation 3.1 and the XGBoost ratio in `src/class_weights.py`.

In conclusion, the preprocessing method produces these outputs: six split files, the positional columns file, the fitted `QuantileTransformer`, and `metadata.json`. The preprocessing steps are run only at the beginning and are never repeated during later training or testing. To avoid data leakage, the data are split before normalization and class-weight computation. Normalization is then fit on the training set only, and class weights are also computed from the training labels only. Therefore, the full order of the preprocessing is: filter features, split into train/validation/test, `QuantileTransformer` normalization, then compute class weights.

```python
# src/run_preprocessing.py — split before normalization
X_train, X_val, X_test, y_train, y_val, y_test = (
    stratified_train_val_test_split(X, y)
)
qt = fit_quantile_transformer(X_train)
X_train_t, X_val_t, X_test_t = apply_quantile_transform(
    qt, X_train, X_val, X_test
)
class_weights = compute_class_weights(y_train)
```

**Figure 4.4.** Preprocessing orchestration that splits before normalization in `src/run_preprocessing.py`.

## [ ] 4.2 Model & Training Pipeline Implementation

Each model has its own class, but all three use the same method names and the same order: `tune()`, `train()`, then `predict_proba()`. The code inside each method is customized for that model. The runners also follow the same structure, with model-specific details inside. Their shared order is: load the same preprocessed data, read `scale_pos_weight` from `metadata.json`, run the sequence above, then save the model checkpoint and write the results to a JSON file for that model and dataset.

### [ ] 4.2.1 XGBoost

XGBoost is not reimplemented, instead it is adopted from the `xgboost` library (Chen and Guestrin, 2016). The `XGBoostVulnModel` class in `src/models/xgboost_model.py` wraps the imported model `xgb.XGBClassifier` along with its customization. The fixed settings are set inside `_base_params()`, which are: `objective="binary:logistic"`, `eval_metric="aucpr"`, the stored `scale_pos_weight`, `n_jobs=1` so that later CPU latency is comparable to the PyTorch models, and `random_state=42`. The search space used by `tune()` follows the values defined in Table 3.3. After `tune()` stores the optimal parameters, `train()` builds a new classifier, calls `fit` with the validation split as `eval_set` and 50 early-stopping rounds, and returns the wall-clock duration of that fit in minutes. The fitted model is saved using joblib to `models/<dataset>/xgboost.joblib`.

### [ ] 4.2.2 TabM

TabM is reimplemented using PyTorch in `src/models/tabm_model.py`, following Gorishniy et al. (2025) instead of adopting from the library. `TabMVulnModel` wraps the custom `TabMNet` architecture. The fixed settings for TabM are the stored `pos_weight` from metadata (used in `BCEWithLogitsLoss`), `seed=42`, and CUDA when `torch.cuda.is_available()` is true. In the architecture, `TabMNet` replicates the input k times, stacks `TabMEnsembleLayer` blocks that apply Equation 3.2, then uses a shared linear head and averages the k logits. The search space used is also defined in Table 3.3. After `tune()` stores `best_params`, `train()` rebuilds the network, runs the training loop with validation early stopping of 20 epochs, and returns the wall-clock duration in minutes. The results are saved to `models/<dataset>/tabm.pt` (state dict) and `tabm_arch.json`.

```python
# src/models/tabm_model.py — TabMEnsembleLayer.forward
def forward(self, X: torch.Tensor) -> torch.Tensor:
    # X : (batch, k, in_dim)
    X = self.s * X      # s_i ⊙ x_i
    X = self.W(X)        # shared W
    X = self.r * X      # r_i ⊙ (...)
    X = X + self.b       # + b_i
    X = F.relu(X)
    X = self.dropout(X)
    return X
```

**Figure 4.5.** TabM ensemble-layer forward pass implementing Equation 3.2 in `src/models/tabm_model.py`.

### [ ] 4.2.3 FT-Transformer

FT-Transformer is also reimplemented using PyTorch in `src/models/ft_transformer.py`, following Gorishniy et al. (2021) instead of using Hugging Face Transformers. `FTTransformerVulnModel` wraps the custom FT-Transformer network. The fixed settings are the stored `pos_weight` from `metadata.json` (used in `BCEWithLogitsLoss`), `seed=42`, and CUDA when `torch.cuda.is_available()` is true. In the architecture, `FeatureTokenizer` converts each numeric feature into a token of width `d_token`. Then, a learnable `[CLS]` token is prepended. After that, an `FTTransformerLayer` stacks PreNorm multi-head self-attention and a feed-forward block. Finally, a linear head converts the `[CLS]` representation to a single logit. The search space used by `tune()` follows the defined set in Table 3.3. After `tune()` stores `best_params`, `train()` rebuilds the network, runs the training loop with validation early stopping of 20 epochs, and returns the wall-clock duration in minutes. The results are saved to two different files, `models/<dataset>/ft_transformer.pt` for the state dict and `ft_transformer_arch.json` for the hyperparameters to rebuild the module.

## [ ] 4.3 Tuning & Training-loop Implementation

The hyperparameter tuning is implemented using Optuna 4.1.0 with the Tree-structured Parzen Estimator (Akiba et al., 2019). The trial counts, early-stopping patience, and search ranges follow Table 3.3. In every trial, the model is scored by validation PR-AUC, and Optuna searches for hyperparameters that maximize that score. The `tune()` method receives only the training and validation data. After the search, each model is retrained from scratch with the best hyperparameters. The total Optuna search time is not stored. Instead, only the duration of this final retrain is recorded as the retraining time.

### [ ] 4.3.1 XGBoost Search

`XGBoostVulnModel.tune()` starts an Optuna search with `TPESampler(seed=42)` and `direction="maximize"`, then runs 100 trials. On each trial, Optuna uses the ranges defined in Table 3.3, builds an `XGBClassifier`, and fits it with 50-round early stopping on the validation set. Each trial is scored by validation PR-AUC using `average_precision_score`. This XGBoost search does not use a MedianPruner. After tuning finishes, `train()` runs one final fit with the best parameters and records that fit time as the retraining duration.

### [ ] 4.3.2 TabM and FT-Transformer Search

Both TabM and FT-Transformer use the same tuning loop. In `tune()`, Optuna is set up with a MedianPruner which contains 5 startup trials and 10 warmup steps, then it runs 50 trials. Inside each trial, `_training_loop` builds the network, trains it with Adam using the sampled weights and weight decay, and runs for up to 150 epochs. After every epoch, validation PR-AUC is computed and reported to Optuna, which may stop a weak trial early using `TrialPruned`. In addition, training also stops if validation PR-AUC does not improve for 20 epochs. Before the trial returns its score, the best weights from that trial are restored. For FT-Transformer, `d_token` is kept as a multiple of `n_heads` so `nn.MultiheadAttention` can run. In conclusion, TPE chooses the hyperparameters, MedianPruner stops weak trials, and patience 20 provides early stopping as stated in Table 3.3.

```python
# src/models/tabm_model.py — Optuna study and pruned training loop
sampler = optuna.samplers.TPESampler(seed=self.seed)
pruner  = optuna.pruners.MedianPruner(
    n_startup_trials=5, n_warmup_steps=10
)
study = optuna.create_study(
    direction="maximize", sampler=sampler, pruner=pruner
)
study.optimize(objective, n_trials=n_trials)

# inside _training_loop, after each epoch:
if trial is not None:
    trial.report(val_auc, epoch)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
if no_improve >= patience:
    break
```

**Figure 4.6.** Optuna TPE study with MedianPruner and early stopping in `src/models/tabm_model.py`.

### [ ] 4.3.3 Final Retrain

After `best_params` is stored, `TabMVulnModel.train()` and the FT-Transformer `train()` method use the same training loop again, but with `max_epochs=300` and no Optuna trial object, therefore pruning is turned off. The time returned by this final training is saved as `retraining_time_min` in the metrics JSON. For XGBoost, `train()` also measures only this final fit, not the 100 tuning fits. The runner then saves the model results and continues to evaluation.

## [ ] 4.4 Evaluation and Comparison Implementation

Evaluation is implemented in `src/evaluate.py` and used by all three runners. It is called once per model after the final retrain. This section explains how each model summarized its results and how the comparison table is produced, including the two time columns. The held-out test split is used here for the first time inside each runner.

### [ ] 4.4.1 Performance Fields

`predict_proba` returns the positive-class probability for every test row. `compute_performance_metrics` passes those probabilities to scikit-learn `average_precision_score` to obtain the PR-AUC. It then thresholds at 0.5 and computes F2-Score with `beta=2` and MCC using `matthews_corrcoef`. The three numbers are written inside the metrics JSON under the performance section.

```python
# src/evaluate.py — test-set performance fields
y_pred = (y_prob >= threshold).astype(int)
pr_auc = average_precision_score(y_true, y_prob)
f2     = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
mcc    = matthews_corrcoef(y_true, y_pred)
return {"pr_auc": float(pr_auc), "f2_score": float(f2), "mcc": float(mcc)}
```

**Figure 4.7.** Test-set PR-AUC, F2-Score, and MCC computation in `src/evaluate.py`.

### [ ] 4.4.2 Efficiency Fields including Training Time and Testing Time

The comparison table has two time columns and they measure different operations. Testing time measures how long an already trained model takes to score new data as inference latency. `measure_inference_latency_ms` calls `predict_proba` on the test set, discards the first warm-up call, times the next 10 calls, converts each duration to milliseconds per sample, and stores the mean as `inference_latency_ms_per_sample`. TabM and FT-Transformer run this on a CPU with one thread, while XGBoost already uses `n_jobs=1`. This follows the CI/CD-oriented protocol in Section 3.8.2. Optuna time or GPU training time is not calculated.

The training time inside the table is only retraining time. Each model's `train()` times one final fit with the best hyperparameters and saves that value as `retraining_time_min`. This shows how long a refit takes when the training set is updated, and it does not include the 50 or 100 Optuna trials. The remaining efficiency fields are file size and peak RAM usage.

```python
# src/evaluate.py — per-sample CPU inference latency
_ = predict_fn(X)  # warm-up, not timed
latencies = []
for _ in range(n_repeats):
    t0 = time.perf_counter()
    _ = predict_fn(X)
    latencies.append((time.perf_counter() - t0) * 1000 / len(X))
return float(np.mean(latencies))
```

**Figure 4.8.** CPU per-sample inference latency measurement in `src/evaluate.py`.

`get_model_size_mb` returns the model size in megabytes. `src/compare.py` prefers the on-disk file under `models/` when assembling the table. `measure_peak_ram_mb` polls process RSS every millisecond during one `predict_proba` call and stores the maximum as `peak_ram_inference_mb`.

### [ ] 4.4.3 How the Comparison Table is Generated

After all six runners have finished, `python -m src.compare` loads `results/metrics/<model>_<dataset>.json` for XGBoost, TabM, and FT-Transformer on both datasets. `build_rows()` copies the performance and efficiency values into one row per model. The script then prints a console table and writes `comparison_table.json` and `comparison_table.md`. Each column comes from a runner JSON, and `compare.py` does not retrain or re-score the test set. These numeric results are used as discussion material in Chapter 5.

```python
# src/compare.py — mapping JSON fields into table columns
rows.append({
    "model": MODEL_LABELS[model],
    "pr_auc": float(perf["pr_auc"]),
    "f2_score": float(perf["f2_score"]),
    "mcc": float(perf["mcc"]),
    "inference_latency_ms_per_sample":
        float(eff["inference_latency_ms_per_sample"]),
    "retraining_time_min": float(eff["retraining_time_min"]),
    "model_size_mb": float(eff["model_size_mb"]),
    "peak_ram_inference_mb": float(eff["peak_ram_inference_mb"]),
})
```

**Figure 4.9.** Mapping of stored JSON fields into comparison-table columns in `src/compare.py`.
