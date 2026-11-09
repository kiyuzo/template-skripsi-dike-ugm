# Chapter 4

# IMPLEMENTATION


## [ ] 4.1 Data Pipeline Implementation

The data pipeline is executed once per dataset by using `python -m src.run_preprocessing`. Preprocessing, splitting, normalization, and class-weight computation live in separate modules and are then run by using `src/run_preprocessing.py`. The three model runners never reload the raw CSV files. They read the processed split files and `metadata.json` saved in `data/processed/ferenc/` and `data/processed/viszkok/`. This integration keeps the feature space identical across all model runners to ensure fairness as stated in Section 3.3.

### [ ] 4.1.1 Data Acquisition

The published tabular data are placed inside `data/raw/Ferenc.csv` and `data/raw/Viszkok.csv`. `src/data_loading.py` reads each file using pandas and defines `load_ferenc()` and `load_viszkok()` functions. The binary target column is `Vuln`, as stated in both source datasets (Ferenc et al., 2019; Viszkok et al., 2021). Paths, global seed, and split ratios are defined in `src/config.py` therefore later stages cannot drift from the original configuration defined in Section 3.4.

### [ ] 4.1.2 Preprocessing

The preprocessing step starts with feature filtering, implemented in `src/preprocessing.py` by using `filter_features()`. It applies the same four rules defined in Section 3.3 to both datasets, using whichever columns are present. Identifier columns listed in `config.IDENTIFIER_COLUMNS` are removed, positional columns listed in `config.POSITIONAL_COLUMNS` are copied aside and are excluded from training, numeric columns with at most one unique value are dropped, and if two remaining numeric columns are almost the same (Pearson correlation ≥ 0.999), the earlier column is kept and the later one is dropped. The 0.999 cutoff comes from `config.COLLINEARITY_THRESHOLD`. A `PreprocessingReport` records the dropped columns and serializes them into `metadata.json`. In Section 3.2, it is stated that there are 35 static metrics for Ferenc et al. (2019) dataset and 61 static metrics for Viszkok et al. (2021) dataset. However, after filtering with `filter_features()`, these counts are reduced to 32 features for Ferenc and 54 features for Viszkok. Tables 4.1 and 4.2 list the details of the removed columns, including the reason they are removed.

**Table 4.1.** Column removals details for Ferenc et al. (2019) dataset

| Step | Columns removed | Reason |
| --- | --- | --- |
| Identifiers | name, longname, path, full_repo_path | Metadata; no metric signal |
| Zero variance | CLLC | Constant column; no information |
| Collinearity (abs r ≥ 0.999) | HBUGS (kept HVOL), HTIME (kept HEFF) | Near-duplicate Halstead identities |
| Positional isolation | line, column, endline, endcolumn | Location artifact of dataset construction |

**Table 4.2.** Column removals details for Viszkok et al. (2021) dataset

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

The 60/20/20 stratified partition from Section 3.4 is implemented in `src/splitting.py`. scikit-learn's `train_test_split` only cuts a matrix in two. Therefore, `train_test_split` is called twice sequentially. The first stratify splits the data into training data and the remainder (40%). The remainder is then split into two sets, validation set and test set. Both calls use the stratify method on the current label vector and `random_state = 42`. Table 4.3 reports the actual numbers of data that are also reported inside `metadata.json`. The Viszkok training split contains 4,822 rows, rather than the rounded 4,823 shown in Table 3.2. The difference is because of rounding in Table 3.2 and not a change of method.

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

**Table 4.3.** Realized stratified split sizes (seed = 42)

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

# Chapter 5

# RESULTS AND DISCUSSION

This chapter reports the measured outcomes of the comparative evaluation and interprets those outcomes against the problem formulation, research objectives, and working hypothesis. The experimental protocol is specified in Chapter 3 and realized in Chapter 4. The sections below recap only the settings that were actually executed, present the stored test-set scores and efficiency fields, and discuss what those measurements imply for TabM, FT-Transformer, and XGBoost on the two JavaScript software-metric datasets.


## 5.1 Experimental Design

Table 5.1 records the realized inputs that enter model training after preprocessing. Starting from the published feature inventories of Ferenc et al. (2019) and Viszkok et al. (2021) (35 static predictors and 61 static-plus-process predictors), identifier columns, zero-variance columns, and one member of each near-perfectly collinear pair are removed, and positional columns are isolated so that they never enter the feature matrix; that filtering leaves 32 features on Ferenc and 54 features on Viszkok. The cleaned tables are then cut by a stratified 60/20/20 partition with seed 42, which yields the exact train, validation, and test sizes in Table 5.1. Those cardinalities differ slightly from the rounded figures in Tables 3.1 and 3.2. Most visibly, the Viszkok training split is 4,822 rather than the rounded 4,823, because the earlier tables used approximate counts, whereas Table 5.1 reports the integer sizes written by the split. Finally, from the training labels alone, scale_pos_weight is computed as the ratio of non-vulnerable to vulnerable rows, which evaluates to 7.1013 on Ferenc and 7.3715 on Viszkok; that fixed ratio is passed to all three models and is not tuned.

**Table 5.1.** Realized feature counts, stratified split sizes (seed = 42), and class-weight ratios


| Item                     | Ferenc et al. (2019) | Viszkok et al. (2021) |
| ------------------------ | -------------------- | --------------------- |
| Features after filtering | 32                   | 54                    |
| Train (60%)              | 7,275                | 4,822                 |
| Validation (20%)         | 2,425                | 1,608                 |
| Test (20%)               | 2,425                | 1,608                 |
| scale_pos_weight         | 7.1013               | 7.3715                |


## 5.2 Working Hypothesis (optional)

Section 1.2 frames TabM as a candidate that may match or approach existing methods while remaining practical for high-speed CI/CD settings, and it treats XGBoost as the current performance ceiling on tabular data. Section 1.4 then asks which architecture captures vulnerability patterns most accurately and whether the computational cost of the deeper representations is justified. Consistent with that problem and those objectives, the working hypothesis of this chapter is as follows.

Among the three models evaluated on the Ferenc and Viszkok static feature sets, XGBoost is expected to remain the strongest efficiency baseline, especially in retraining time and inference cost. TabM is expected to remain competitive with FT-Transformer on PR-AUC at substantially lower computational cost, without necessarily exceeding the XGBoost predictive ceiling.

The measurements in Section 5.3 support that statement. On both datasets, XGBoost records the lowest retraining time, the lowest inference latency, and the lowest peak RAM (Tables 5.7 and 5.8). On the primary predictive metric, TabM is slightly above FT-Transformer on Ferenc (PR-AUC 0.7631 against 0.7518) and slightly below it on Viszkok (0.8794 against 0.8856), while remaining several times cheaper to retrain and to score (Tables 5.5–5.8). Neither deep model exceeds XGBoost on PR-AUC or MCC. The hypothesis is therefore supported. It is not a claim that TabM is more accurate than XGBoost, and the retraining gap is on the order of tenths of a minute rather than a fraction of a second.

## 5.3 Experimental Results

All scores below are taken from the stored test-set records assembled by the comparison procedure in Section 4.4. No additional training run is introduced in this chapter.

### 5.3.1 Best Hyperparameters

The Optuna search that produced these vectors follows Table 3.3 and is implemented as described in Chapter 4. This subsection reports only the winning parameter sets. Because each dataset is tuned independently, the selected values differ between Ferenc and Viszkok even though the search spaces are the same.

On XGBoost, the winning Ferenc trial prefers a deeper, more heavily subsampled ensemble (1,547 trees, depth 12, subsample 0.5132), while the Viszkok trial prefers fewer, shallower trees with a higher learning rate and subsample close to one. The fixed scale_pos_weight from Table 5.1 is retained in both cases. The final best hyperparameters for XGBoost are summarized in Table 5.2.

**Table 5.2.** Best XGBoost hyperparameters after 100 Optuna TPE trials on each validation set


| Parameter              | Ferenc | Viszkok |
| ---------------------- | ------ | ------- |
| n_estimators           | 1547   | 1448    |
| max_depth              | 12     | 7       |
| learning_rate          | 0.0553 | 0.1095  |
| subsample              | 0.5132 | 0.9395  |
| colsample_bytree       | 0.6269 | 0.7700  |
| min_child_weight       | 2      | 4       |
| reg_alpha              | 0.2251 | 0.3595  |
| reg_lambda             | 1.6169 | 3.1062  |
| Best validation PR-AUC | 0.8113 | 0.9027  |


On TabM, the winning Ferenc trial is a shallower, wider network with a smaller ensemble (two layers, hidden width 424, k = 32), whereas Viszkok selects a deeper, narrower network with a larger ensemble (four layers, hidden width 342, k = 42). Batch size is 256 in both winning trials. The final best hyperparameters for TabM are summarized in Table 5.3.

**Table 5.3.** Best TabM hyperparameters after 50 Optuna TPE trials on each validation set


| Parameter              | Ferenc   | Viszkok  |
| ---------------------- | -------- | -------- |
| num_layers             | 2        | 4        |
| hidden_dim             | 424      | 342      |
| dropout                | 0.1840   | 0.2308   |
| learning_rate          | 0.0014   | 0.0018   |
| weight_decay           | 1.022e-6 | 1.446e-5 |
| batch_size             | 256      | 256      |
| k                      | 32       | 42       |
| Best validation PR-AUC | 0.7926   | 0.9044   |


On FT-Transformer, the winning Ferenc trial uses a wider token, more heads, and three layers (d_token = 336, eight heads, batch size 512), while Viszkok selects a narrower token, fewer heads, and two layers (d_token = 268, four heads, batch size 256). The final best hyperparameters for FT-Transformer are summarized in Table 5.4.

**Table 5.4.** Best FT-Transformer hyperparameters after 50 Optuna TPE trials on each validation set


| Parameter              | Ferenc   | Viszkok  |
| ---------------------- | -------- | -------- |
| n_heads                | 8        | 4        |
| num_layers             | 3        | 2        |
| attention_dropout      | 0.1302   | 0.1734   |
| ffn_dropout            | 0.2651   | 0.1024   |
| ffn_factor             | 1.7994   | 2.3851   |
| learning_rate          | 3.124e-4 | 1.032e-4 |
| weight_decay           | 9.387e-6 | 2.410e-5 |
| batch_size             | 512      | 256      |
| d_token                | 336      | 268      |
| Best validation PR-AUC | 0.7799   | 0.8924   |


### 5.3.2 Performance Results

After the final retrain with the hyperparameters in Tables 5.2–5.4, each model is scored once on the held-out test split. The reported fields are PR-AUC, F2-Score, and MCC as defined in Section 3.8.1. The test set is not used during Optuna search.

On Ferenc, the three models receive the same 32-feature test matrix of 2,425 functions. Probabilities from that pass yield PR-AUC; hard labels at threshold 0.5 yield F2 and MCC. The test-set predictive scores for Ferenc et al. (2019) are summarized in Table 5.5.

**Table 5.5.** Test-set predictive performance on Ferenc et al. (2019), 32 features

| Model | PR-AUC | F2-Score | MCC |
| --- | ---: | ---: | ---: |
| XGBoost | 0.7984 | 0.7095 | 0.6695 |
| TabM | 0.7631 | 0.6854 | 0.5680 |
| FT-Transformer | 0.7518 | 0.6714 | 0.4679 |

XGBoost leads every column on Ferenc. TabM is second on all three metrics and stays within 0.0353 PR-AUC of XGBoost, while FT-Transformer trails TabM by a further 0.0113 PR-AUC. The spread is wider on MCC (0.6695 against 0.5680 against 0.4679) than on PR-AUC, so the models separate more clearly once scores are collapsed at threshold 0.5.

On Viszkok, the same evaluation is applied to the 54-feature test matrix of 1,608 functions, again after an independent retrain with that dataset’s winning hyperparameters. The test-set predictive scores for Viszkok et al. (2021) are summarized in Table 5.6.

**Table 5.6.** Test-set predictive performance on Viszkok et al. (2021), 54 features

| Model | PR-AUC | F2-Score | MCC |
| --- | ---: | ---: | ---: |
| XGBoost | 0.9071 | 0.8091 | 0.7773 |
| TabM | 0.8794 | 0.8044 | 0.7196 |
| FT-Transformer | 0.8856 | 0.8193 | 0.7298 |

XGBoost again leads PR-AUC and MCC. FT-Transformer is second on those two fields and only 0.0062 PR-AUC above TabM, so the deep models are nearly tied on the primary metric. FT-Transformer also records the highest F2 (0.8193), slightly above XGBoost (0.8091) and TabM (0.8044). That F2 edge is the only predictive win for a deep model on either dataset.

Across both tables, XGBoost is the strongest model on the primary metric and on MCC. Its PR-AUC lead over the best deep model is 0.0353 on Ferenc (against TabM) and 0.0215 on Viszkok (against FT-Transformer), so the ceiling is clearer on the static-metric table than on the process-enriched one. Between the deep models the picture flips: TabM is ahead of FT-Transformer on Ferenc, while FT-Transformer is narrowly ahead on Viszkok. Absolute scores are higher for every architecture on Viszkok than on Ferenc, which is consistent with the added process metrics, but the two datasets are not ranked against each other as a single scoreboard.

### 5.3.3 Efficiency Results

Efficiency is measured after the same final retrain used for Section 5.3.2. The four fields follow Section 3.8.2: inference latency, retraining time, model size, and peak RAM at inference. As distinguished in Chapter 4, latency is time-to-score after the model is already trained, whereas retraining time is only the final fit with the hyperparameters in Tables 5.2–5.4; Optuna search time is excluded. Retraining time is treated as the primary CI/CD cost signal below.

On Ferenc, those four fields are recorded for the 32-feature models after the winning configurations in Tables 5.2–5.4. The efficiency results for Ferenc et al. (2019) are summarized in Table 5.7.

**Table 5.7.** Test-set efficiency on Ferenc et al. (2019), 32 features

| Model | Latency (ms/sample) | Retrain (min) | Size (MB) | Peak RAM (MB) |
| --- | ---: | ---: | ---: | ---: |
| XGBoost | 0.019 | 0.031 | 1.959 | 302 |
| TabM | 0.176 | 0.225 | 1.005 | 1,348 |
| FT-Transformer | 1.886 | 1.880 | 10.389 | 2,021 |

XGBoost is the cheapest model on Ferenc on every column: 0.031 min to retrain, 0.019 ms/sample to score, and about 302 MB peak RAM. TabM sits in the middle. It retrains in 0.225 min (about 7 times XGBoost) and scores in 0.176 ms/sample, while keeping the smallest checkpoint (1.005 MB). FT-Transformer is the most expensive by a wide margin: 1.880 min to retrain (about 8.4 times TabM and about 61 times XGBoost), 1.886 ms/sample latency, a 10.389 MB checkpoint, and 2,021 MB peak RAM.

On Viszkok, the same four fields are recorded after the independent retune for that dataset. The efficiency results for Viszkok et al. (2021) are summarized in Table 5.8.

**Table 5.8.** Test-set efficiency on Viszkok et al. (2021), 54 features

| Model | Latency (ms/sample) | Retrain (min) | Size (MB) | Peak RAM (MB) |
| --- | ---: | ---: | ---: | ---: |
| XGBoost | 0.005 | 0.011 | 0.409 | 306 |
| TabM | 0.448 | 0.127 | 2.027 | 895 |
| FT-Transformer | 1.634 | 0.764 | 5.228 | 2,472 |

XGBoost again dominates the Viszkok efficiency table: 0.011 min retrain, 0.005 ms/sample latency, the smallest artifact (0.409 MB), and about 306 MB peak RAM. TabM remains second on retrain (0.127 min) and peak RAM (895 MB), and is about 3.6 times faster than FT-Transformer at inference. FT-Transformer is again the slowest to retrain (0.764 min, about 6 times TabM) and the heaviest in memory (2,472 MB), with a 5.228 MB checkpoint.

Figure 5.1 plots PR-AUC against retraining time for all six model-dataset pairs.

[Figure 5.1 placeholder: PR-AUC vs retraining time (min), Ferenc and Viszkok]

**Figure 5.1.** Precision-recall AUC against retraining time for XGBoost, TabM, and FT-Transformer on both datasets.

Across both datasets the efficiency order is XGBoost, then TabM, then FT-Transformer. XGBoost is the CI/CD baseline on retraining time; TabM is the stronger deep option because it stays far cheaper than FT-Transformer on retrain, latency, size, and peak RAM, while FT-Transformer peak RAM of 2,021-2,472 MB sits at the lower end of the 2-8 GB runner range in Section 3.8.2.

## 5.4 Discussion

Section 1.2 asks whether TabM can deliver a favorable balance between predictive accuracy and computational efficiency on these tabular JavaScript metrics, using FT-Transformer as the attention-based deep baseline and XGBoost as the GBDT ceiling. Section 1.4 restates that aim as three objectives: implement and evaluate the three architectures, identify which model captures vulnerability patterns most accurately, and determine whether the computational cost of the richer deep representations is justified. The measurements answer those questions in the same order as the research questions recorded by the comparison procedure.

### 5.4.1 Can TabM match FT-Transformer at much lower cost?

Yes. On Ferenc, TabM exceeds FT-Transformer on every predictive field in Table 5.5 (PR-AUC 0.7631 against 0.7518, F2 0.6854 against 0.6714, MCC 0.5680 against 0.4679). On Viszkok, FT-Transformer is slightly ahead on PR-AUC (0.8856 against 0.8794), F2, and MCC, but the PR-AUC difference is 0.0062. In both cases the cost gap is large. TabM retrains in 0.127–0.225 min against 0.764–1.880 min for FT-Transformer, scores 3.6–10.7 times faster, and occupies a 2.6–10 times smaller checkpoint. Peak RAM is also lower (895–1,348 MB against 2,021–2,472 MB).

That pattern is consistent with the architectural contrast in Section 3.6. TabM processes k ensemble members through a shared MLP backbone with rank-1 adapters (Equation 3.2). FT-Transformer tokenizes every feature and applies multi-head self-attention over a sequence of length equal to the feature count plus one classification token. This attention mechanism makes FT-Transformer more costly than TabM. It should be noted that FT-Transformer performs better when the dataset has more features; however, the margin is relatively small compared to the retraining and latency margins.

### 5.4.2 Can either deep model beat the XGBoost ceiling?

Not on the primary metric. XGBoost leads PR-AUC and MCC on both datasets (Tables 5.5 and 5.6). The Ferenc gaps are 0.0353 PR-AUC and 0.1015 MCC against TabM, and 0.0466 PR-AUC and 0.2016 MCC against FT-Transformer. The Viszkok gaps are 0.0277 PR-AUC and 0.0577 MCC against TabM, and 0.0215 PR-AUC and 0.0475 MCC against FT-Transformer. FT-Transformer leads Viszkok F2 only (0.8193 against 0.8091). [That single reversal is not treated as a change of the overall ranking, because PR-AUC is the optimization target in Table 3.3 and F2 is computed from hard labels at threshold 0.5.]

Section 3.6.3 selected XGBoost because GBDT methods remain a strong ceiling on heterogeneous tabular features. The present results are consistent with that premise. Both datasets are modest in size (12,125 and 8,038 rows), the predictors are mixed static and, on Viszkok, process metrics, and the positive class is about 12% of each table. Under those conditions, a tuned tree ensemble still ranks probabilities of the vulnerable class more reliably than either neural architecture evaluated here. TabM narrows the PR-AUC gap more successfully than FT-Transformer on Ferenc and remains close on Viszkok, but it does not cross the XGBoost ceiling.

### 5.4.3 Is FT-Transformer’s extra compute justified?

No for a CI/CD-oriented reading of the efficiency fields. Retraining time is the primary criterion for that reading, because a refit on an updated training set is what blocks a pipeline after the search in Tables 5.2–5.4 has already been paid. Figure 5.1 shows the resulting trade-off: XGBoost occupies the high-PR-AUC, low-retrain corner on both datasets; TabM sits between the two extremes; FT-Transformer is the slowest point at a PR-AUC that is either below TabM (Ferenc) or only slightly above it (Viszkok).

The supporting fields point in the same direction. FT-Transformer latency of 1.634–1.886 ms/sample is an order of magnitude above TabM on Ferenc and several times TabM on Viszkok. Peak RAM of 2,021–2,472 MB sits at the floor of the 2–8 GB runner range in Section 3.8.2. The Ferenc checkpoint is 10.389 MB, more than ten times the TabM artifact on that dataset. None of those costs buys a PR-AUC or MCC lead over XGBoost, and they buy at most a 0.0062 PR-AUC lead over TabM.

Objective 3 in Section 1.4 asks whether the computational trade-off for the richer representations in FT-Transformer and TabM is justified against the GBDT baseline. Against XGBoost, neither deep model justifies its extra retrain, latency, or RAM on predictive grounds. Between the two deep models, TabM is the stronger option for a constrained pipeline: it matches FT-Transformer on PR-AUC closely enough that the cost reduction is the deciding measurement. That statement does not make TabM faster or more accurate than XGBoost. XGBoost remains the efficiency baseline and the accuracy leader. TabM is the preferable deep architecture when a neural model is required and runner time or memory is limited.

Two further observations belong with the reasons for these outcomes. First, every model is stronger on Viszkok than on Ferenc. Viszkok et al. (2021) added process metrics precisely because those features improved F-measure, precision, and recall in the original study (Section 3.2.2). The present PR-AUC lift from 0.7984 to 0.9071 for XGBoost, and the parallel lifts for TabM and FT-Transformer, is consistent with that design. The same richer table is also the only setting in which FT-Transformer edges TabM on PR-AUC and leads F2, which is a plausible consequence of attention over a longer, more informative feature sequence, but the edge remains small. Second, F2 and MCC can reorder models relative to PR-AUC because they collapse probabilities at 0.5. The Viszkok F2 lead for FT-Transformer is an instance of that collapse, not a second primary ranking.

## 5.5 Comparison with Prior Work

The datasets used in this study were introduced and first modeled by Ferenc et al. (2019) and extended by Viszkok et al. (2021). Those two papers are the appropriate published baselines. Direct numerical conversion between their scores and the scores in Tables 5.5 and 5.6 is not valid, because the metric families, imbalance treatments, and validation protocols differ.

Ferenc et al. (2019) trained eight machine-learning algorithms and two deep networks on the 12,125-function static-metric table. They used an 80/10/10 train–development–test cut, 10-fold cross-validation, grid search, and F-measure (the harmonic mean of precision and recall, equivalent to F1), together with optional random over- and under-sampling. The best reported configuration was k-nearest neighbors without resampling, with an F-measure of 0.76, precision 0.91, and recall 0.66. Tree, forest, and deep models also reached F-measures at or above 0.70 on the original imbalance; a forest model reached 0.74 under 25% or 50% oversampling. They also noted that Matthews correlation coefficients were slightly smaller than F-measure but followed the same tendency.

Viszkok et al. (2021) added 19 process metrics and retrained a similar algorithm set, again with grid search, 10-fold cross-validation, and an 80/10/10 split, reporting precision, recall, and F-measure. On average they observed an F-measure increase from 0.764 to 0.848, precision from 0.953 to 0.988, and recall from 0.697 to 0.760. After retuning on the extended table, random forest was the strongest model, with an F-measure of 0.848, precision 0.958, and recall 0.760.

This study optimizes and ranks models with PR-AUC, reports F2 (β = 2) and MCC as secondary fields, uses class weights rather than resampling, and evaluates a single stratified 60/20/20 split with seed 42. F2 weights recall twice as heavily as precision and is not interchangeable with the F-measure of Ferenc et al. (2019) or Viszkok et al. (2021). PR-AUC summarizes the full precision–recall curve and cannot be rewritten as their single-threshold F-measure. For that reason, the XGBoost PR-AUC of 0.7984 on Ferenc is not interpreted as “better than” the published KNN F-measure of 0.76, and the Viszkok F2 values of 0.8044–0.8193 are not interpreted as a reproduction of the published forest F-measure of 0.848.

Two qualitative agreements remain. First, tree-family models are strong on these tables. Ferenc et al. (2019) already found forest and decision-tree F-measures above 0.70; Viszkok et al. (2021) found random forest to be the best retuned model; this study finds XGBoost first on PR-AUC and MCC on both dumps. Second, adding process metrics improves prediction relative to static metrics alone. That is the central empirical claim of Viszkok et al. (2021), and it reappears here as a higher score for every architecture on Viszkok than on Ferenc. The present experiment does not re-run their algorithm set and does not claim to have matched or beaten a specific published F-measure.

## 5.6 Limitations of the Findings

The scope limits in Section 1.3 still apply. The study is restricted to JavaScript, to the tabular dumps of Ferenc et al. (2019) and Viszkok et al. (2021), to numerical software metrics rather than source-text features, and to the three architectures named in Section 1.4. Findings do not transfer automatically to other languages, to raw-code models, or to other tabular deep architectures.

Several limits follow from the experiment that was actually run. All splits and Optuna samplers use a single seed (42). There is no k-fold loop inside tuning or final evaluation; Section 3.7 already adopted a fixed validation set in order to keep the deep-model budget tractable. F2 and MCC use a decision threshold of 0.5. A different operating point could change those two columns, including the Viszkok F2 lead of FT-Transformer, without changing PR-AUC. TabM is a from-scratch PyTorch reimplementation of Gorishniy et al. (2025), not a call to the official package during training (Section 4.2.2). The models see 32 and 54 features after filtering, not the 35 and 61 predictors listed in Section 3.2. No statistical significance test, McNemar test, or bootstrap interval on model differences is computed, because none of those procedures is part of the proposal or the evaluation code. Retraining time excludes Optuna search cost, so Tables 5.7 and 5.8 describe a refit after hyperparameters are known, not the first-time tuning bill. Training of the deep models used CUDA, while latency and peak RAM were measured on CPU; those two stages are therefore not interchangeable as hardware profiles.

These constraints bound the strength of the claims in Section 5.4. The ranking of XGBoost over TabM over, or close to, FT-Transformer is the ranking observed on one seed, one split, and one threshold. It is sufficient to answer the three research questions on the recorded evidence. It is not a claim that the same margins would hold under cross-validation or under a different decision threshold.
