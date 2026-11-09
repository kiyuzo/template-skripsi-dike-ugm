# Chapter 2 key ideas

The chapter positions the study against prior work. It does not restate research questions.

Assumption: the 3-slide budget is treated as about three must-include ideas (one per slide). Ideas already ranked in the Chapter 1 list are left out here.

Chapter job: literature positioning. Examiners need the alternatives this study sets aside, the architectural reason, and the TabM property it adopts.

## Must include

| # | Key idea (short phrase) | Why this tier (one line) | Where in chapter | Suggested visual |
|---|---|---|---|---|
| 1 | Prior methods miss native-tabular, low-cost detection | This is the chapter's answer to why those published alternatives were not used | Comparative Analysis; Identified Research Gap; Table 2.1 (`tab:vuln-methods`) | table, Table 2.1 |
| 2 | CNN spatial bias on non-spatial metric tables | Concrete architectural rejection an examiner can challenge; Chapter 1 never names it | Tabular Software Metrics and Vulnerability Detection (Sheneamer, 2024); Table 2.1 Sheneamer row | text-only |
| 3 | TabM parameter-efficient ensembling | The concept this study adopts; the Chapter 1 comparison does not explain the mechanism | Tabular Software Metrics and Vulnerability Detection; Identified Research Gap (Gorishniy et al., 2025) | diagram |

The high-level gap and the TabM vs FT-Transformer vs XGBoost comparison stay on the Chapter 1 slides. FT-Transformer compute cost (Gorishniy et al., 2023; 11 datasets, up to 1,200,192 samples) rides on the Table 2.1 slide, not as a fourth must-include.

## Should include

| # | Key idea (short phrase) | Why this tier (one line) | Where in chapter | Suggested visual |
|---|---|---|---|---|
| 1 | GBDT ceiling via successive space partitioning | Citation behind the Chapter 1 baseline; own slide only if a slot remains | Tabular Software Metrics and Vulnerability Detection; Identified Research Gap (Ye et al., 2025) | text-only |
| 2 | Metrics-plus-text fusion did not improve results | Evidence for the Chapter 1 choice to leave source text unused | Comparative Analysis (Kalouptsoglou et al., 2022); Table 2.1, 12,106 functions | table, Table 2.1 Kalouptsoglou row |
| 3 | Deep models closing the GBDT gap on higher-order interactions | Why a deep model is still tested against that ceiling | Tabular Software Metrics and Vulnerability Detection (Ye et al., 2025) | text-only |

## Might include

| # | Key idea (short phrase) | Why this tier (one line) | Where in chapter | Suggested visual |
|---|---|---|---|---|
| 1 | Sheneamer pre-split data leakage | Extra reason to discount that CNN result if an examiner defends its accuracy | Comparative Analysis; Table 2.1 Sheneamer row | text-only |
| 2 | Word2Vec similarity outside the metrics setup | Excluded line of work, ready if asked (186 fixes, 91 projects) | Comparative Analysis (Mosolygo et al., 2021); Table 2.1 | text-only |
| 3 | DL-plus-XGBoost ensemble accuracy at high cost | Stronger ensemble this study does not adopt (11 datasets, up to 1,000,000 samples) | Comparative Analysis (Shwartz-Ziv and Armon, 2022); Table 2.1 | text-only |

## Leave out of the main deck

- The generic research gap and the three-way model comparison: already Chapter 1 must-includes.
- CI/CD speed as a general motive: already a Chapter 1 backdrop.
- An author-by-author tour of Table 2.1: the failure map is the slide.
- CNN backbone names (VGG16, VGG19, AlexNet, ResNet): examples inside the spatial-bias point.
- The imbalance-handling column: listed in Table 2.1, never turned into a claim.

## Likely examiner probes

- Rejecting Sheneamer's CNNs for spatial bias after the chapter says they achieved good accuracy.
- Treating shared-parameter TabM as an ensemble of independent networks.
- Calling GBDT the established performance ceiling in the same chapter that says deep learning is narrowing that gap.
