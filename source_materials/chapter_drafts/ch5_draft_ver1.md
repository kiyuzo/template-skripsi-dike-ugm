# Chapter 5

# RESULTS AND DISCUSSION

This chapter reports the measured outcomes of the comparative evaluation and interprets those outcomes against the problem formulation, research objectives, and working hypothesis. The experimental protocol is specified in Chapter 3 and realized in Chapter 4. The sections below recap only the settings that were actually executed, present the stored test-set scores and efficiency fields, and discuss what those measurements imply for TabM, FT-Transformer, and XGBoost on the two JavaScript software-metric datasets.

**Progress** (tick when copied or reviewed):

- [ ] 5.1 Experimental Design
- [ ] 5.2 Working Hypothesis (optional; may be omitted later)
- [ ] 5.3 Experimental Results
  - [ ] 5.3.1 Best Hyperparameters
  - [ ] 5.3.2 Performance Results
  - [ ] 5.3.3 Efficiency Results
- [ ] 5.4 Discussion
- [ ] 5.5 Comparison with Prior Work
- [ ] 5.6 Limitations of the Findings

## 5.1 Experimental Design

Table 5.1 records the realized inputs that enter model training after preprocessing. Starting from the published feature inventories of Ferenc et al. (2019) and Viszkok et al. (2021) (35 static predictors and 61 static-plus-process predictors), identifier columns, zero-variance columns, and one member of each near-perfectly collinear pair are removed, and positional columns are isolated so that they never enter the feature matrix; that filtering leaves 32 features on Ferenc and 54 features on Viszkok. The cleaned tables are then cut by a stratified 60/20/20 partition with seed 42, which yields the exact train, validation, and test sizes in Table 5.1. Those cardinalities differ slightly from the rounded figures in Tables 3.1 and 3.2. Most visibly, the Viszkok training split is 4,822 rather than the rounded 4,823, because the earlier tables used approximate counts, whereas Table 5.1 reports the integer sizes written by the split. Finally, from the training labels alone, scale_pos_weight is computed as the ratio of non-vulnerable to vulnerable rows, which evaluates to 7.101 on Ferenc and 7.372 on Viszkok; that fixed ratio is passed to all three models and is not tuned.

**Table 5.1.** Realized feature counts, stratified split sizes (seed = 42), and class-weight ratios


| Item                     | Ferenc et al. (2019) | Viszkok et al. (2021) |
| ------------------------ | -------------------- | --------------------- |
| Features after filtering | 32                   | 54                    |
| Train (60%)              | 7,275                | 4,822                 |
| Validation (20%)         | 2,425                | 1,608                 |
| Test (20%)               | 2,425                | 1,608                 |
| scale_pos_weight         | 7.101                | 7.372                 |


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
| XGBoost | 0.019 | 0.031 | 1.959* | 302 |
| TabM | 0.176 | 0.225 | 1.005 | 1,348 |
| FT-Transformer | 1.886 | 1.880 | 10.389 | 2,021 |

\* Size taken from the on-disk checkpoint when it differs from the metrics JSON (JSON value 1.956 MB).

XGBoost is the cheapest model on Ferenc on every column: 0.031 min to retrain, 0.019 ms/sample to score, and about 302 MB peak RAM. TabM sits in the middle. It retrains in 0.225 min (about 7 times XGBoost) and scores in 0.176 ms/sample, while keeping the smallest checkpoint (1.005 MB). FT-Transformer is the most expensive by a wide margin: 1.880 min to retrain (about 8.4 times TabM and about 61 times XGBoost), 1.886 ms/sample latency, a 10.389 MB checkpoint, and 2,021 MB peak RAM.

On Viszkok, the same four fields are recorded after the independent retune for that dataset. The efficiency results for Viszkok et al. (2021) are summarized in Table 5.8.

**Table 5.8.** Test-set efficiency on Viszkok et al. (2021), 54 features

| Model | Latency (ms/sample) | Retrain (min) | Size (MB) | Peak RAM (MB) |
| --- | ---: | ---: | ---: | ---: |
| XGBoost | 0.005 | 0.011 | 0.409* | 306 |
| TabM | 0.448 | 0.127 | 2.027 | 895 |
| FT-Transformer | 1.634 | 0.764 | 5.228 | 2,472 |

\* Size taken from the on-disk checkpoint when it differs from the metrics JSON (JSON value 0.00085 MB). Deep-model JSON and checkpoint sizes agree.

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

Neither deep learning model surpassed XGBoost. This is expected, as XGBoost remains the GBDT baseline for tabular data processing. XGBoost is also favorable here because of the data size (12,125 and 8,038 rows), which falls in the medium-sized range where tree-based models remain state-of-the-art at around 10K samples, while such sample sizes are still relatively small for deep learning models (Grinsztajn, 2022). TabM might have a close gap or the performance might be close with the XGBoost model, but it does not change the fact that it still did not cross the XGBoost ceiling.

### 5.4.3 Is FT-Transformer’s extra compute justified?

No, not for an efficient and fast CI/CD development environment. The retraining time is the primary metric for our system, because a refit on an updated training set is what slows down the CI/CD pipeline to check the code each time. Based on Figure 5.1, it is shown that XGBoost is the most efficient model, followed by TabM, and FT-Transformer is last, with a huge gap from the previous two. In addition, FT-Transformer also costs more in terms of latency, RAM, and checkpoint size.

All models performed better on the Viszkok dataset due to the richer and much more informative features. Surprisingly, FT-Transformer attains the highest F2 score on the Viszkok dataset across all models. However, against XGBoost, neither of the deep learning models is worth the extra compute. If we had to choose a deep learning approach, TabM would still be the viable option, since the gap between the resources used by FT-Transformer and the performance it provides is not justified.

## 5.5 Comparison with Prior Work

Both datasets used in this study were first tested by prior work that compared several popular classifiers on the same tables. Those studies measured performance with Precision, Recall, and F1-score as their main metrics. This study does not use those three fields as its main metrics, so a direct scoreboard against the published numbers is not fair if the columns differ. The main metrics used here are PR-AUC, F2, and MCC, as defined in Section 3.8.1. Those fields still depend on precision and recall. PR-AUC aggregates the precision-recall curve, and F2 is an F-beta score that weights recall twice as heavily as precision. To place this study beside Ferenc et al. (2019) and Viszkok et al. (2021), Precision, Recall, and F1 are retrieved from the same held-out test predictions at threshold 0.5 and used only in this section. That retrieval does not change the primary metric. Ranking among XGBoost, TabM, and FT-Transformer in Sections 5.3 and 5.4 still follows PR-AUC.

On the Ferenc setting, models were trained with an 80/10/10 train, development, and test cut. Hyperparameters were tuned with 10-fold cross-validation and grid search, using F-measure as the ranking field. Class imbalance was handled with random over-sampling and under-sampling at several ratios. This study uses a different experimental setting, as stated in Sections 3.7, 3.8, and 5.1: a stratified 60/20/20 split with seed 42, Optuna TPE on validation PR-AUC, and class weights instead of resampling. Table 5.9 compares the Ferenc classifiers with the three models from this study on Precision, Recall, and F1-score only.

**Table 5.9.** Precision, Recall, and F1-score on the Ferenc static-metric table: published Ferenc et al. (2019) scores versus this study's held-out test scores at threshold 0.5

| Model | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: |
| KNN | 0.909 | 0.659 | 0.764 |
| DT | 0.737 | 0.697 | 0.716 |
| RFC / Forest | 0.931 | 0.578 | 0.713 |
| SDNN | 0.873 | 0.600 | 0.711 |
| CDNN | 0.911 | 0.579 | 0.708 |
| SVM | 0.953 | 0.514 | 0.668 |
| LogReg | 0.753 | 0.212 | 0.331 |
| LinReg | 0.843 | 0.154 | 0.261 |
| NB | 0.224 | 0.117 | 0.153 |
| XGBoost (this study) | 0.711 | 0.709 | 0.710 |
| TabM (this study) | 0.538 | 0.736 | 0.621 |
| FT-Transformer (this study) | 0.368 | 0.846 | 0.513 |

*Note.* Prior rows are the best static-only Precision, Recall, and F-measure reprinted in Viszkok et al. (2021) Table 1. This study's rows are scored from the saved Ferenc checkpoints on the filtered test split. Protocol differences remain.

Based on Table 5.9, KNN from Ferenc et al. (2019) still records the highest F1-score on this comparison (0.764), with Precision 0.909 and Recall 0.659. Among the three models from this study, XGBoost has the best F1 (0.710), with Precision and Recall almost balanced (0.711 and 0.709). TabM and FT-Transformer push recall higher (0.736 and 0.846) but lose precision, so their F1 falls behind XGBoost. The published KNN lead on F1 does not overturn the PR-AUC ranking in Table 5.5, and the two experiments still differ in split, tuning, and imbalance handling.

Viszkok et al. (2021) keep the same 80/10/10 split, 10-fold cross-validation, grid search, and F-measure ranking as Ferenc et al. (2019). The classifier list is largely the same family set, with the deep nets renamed as SDNN and CDNN. Viszkok does not add gradient boosting either. The main change is the feature space: process metrics raise the predictor count, so the table is richer than the static-only Ferenc dump. Table 5.10 compares those Viszkok classifiers with the three models from this study on Precision, Recall, and F1-score.

**Table 5.10.** Precision, Recall, and F1-score on the Viszkok static-plus-process table: published Viszkok et al. (2021) F-measure-optimal scores versus this study's held-out test scores at threshold 0.5

| Model | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: |
| RFC | 0.958 | 0.760 | 0.848 |
| DT | 0.909 | 0.753 | 0.824 |
| KNN | 0.949 | 0.713 | 0.814 |
| SDNN | 0.921 | 0.716 | 0.805 |
| CDNN | 0.928 | 0.706 | 0.802 |
| SVM | 0.861 | 0.721 | 0.785 |
| LogReg | 0.743 | 0.517 | 0.609 |
| LinReg | 0.540 | 0.594 | 0.566 |
| NB | 0.278 | 0.120 | 0.167 |
| XGBoost (this study) | 0.796 | 0.812 | 0.804 |
| TabM (this study) | 0.678 | 0.844 | 0.752 |
| FT-Transformer (this study) | 0.678 | 0.865 | 0.760 |

*Note.* Prior rows are from Viszkok et al. (2021) Table 3. This study's rows are scored from the saved Viszkok checkpoints on the filtered test split.

Based on Table 5.10, random forest from Viszkok et al. (2021) records the highest F1-score in this comparison (0.848), with Precision 0.958 and Recall 0.760. Among the three models from this study, XGBoost again has the best F1 (0.804), with Precision 0.796 and Recall 0.812. TabM and FT-Transformer again trade precision for recall, so their F1 stays below XGBoost. Absolute F1 for XGBoost, TabM, and FT-Transformer is higher on Viszkok than on Ferenc, which matches the process-metric lift reported by Viszkok et al. (2021).

Overall, on the shared Precision, Recall, and F1 columns, the published leaders remain KNN on Ferenc and random forest on Viszkok. Inside this study's three-model set, XGBoost has the best F1 on both tables and keeps Precision and Recall closer than TabM or FT-Transformer. That pattern supports a GBDT ceiling under the comparison metrics used here, while PR-AUC remains the primary metric for ranking in Sections 5.3 and 5.4. The F1 gaps against the published KNN and RFC scores are not treated as a strict loss or win, because the experimental settings still differ.

## 5.6 Limitations of the Findings

The datasets used to evaluate the two deep learning models and one machine learning model contain about 12,125 and 8,038 rows. That amount of data is medium-sized and enough for most machine learning models, but it is not enough for deep learning models to capture the tabular patterns that would let them overtake tree ensembles (Grinsztajn, 2022). The result of this study matches that expectation: XGBoost dominates performance, and none of the deep learning models cross the baseline. The testing environment therefore favors the machine learning model here. A much larger corpus or dataset would give a fairer test of whether the deep models can close the gap and would strengthen the validity of that architecture comparison.

In addition, the datasets are also severely imbalanced. Only about one in eight functions is vulnerable, and the training ratios used for weighting are 7.101 on Ferenc and 7.372 on Viszkok. This study handles that imbalance with class-weight adjustment (`scale_pos_weight` / `pos_weight`) and then collapses probabilities at a fixed 0.5 threshold for F2, MCC, Precision, Recall, and F1. Weight adjustment alone is not enough to handle the large imbalance inside the dataset. Future studies should tune the decision threshold and combine weight adjustment with other imbalance treatments so the minority vulnerable class is handled more carefully than a fixed weight and a default 0.5 rule allow.