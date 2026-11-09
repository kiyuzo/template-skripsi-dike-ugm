# Chapter 1 key ideas

The chapter states objectives rather than explicit research questions or hypotheses.

Assumption: the 1–3 slide budget is treated as about three must-include ideas (one per slide).

Chapter job: problem and introduction. Examiners need the gap this study claims, the comparison it will run, and the boundaries of that claim.

## Must include

| # | Key idea (short phrase) | Why this tier (one line) | Where in chapter | Suggested visual |
|---|---|---|---|---|
| 1 | Gap on advanced tabular deep learning for these JS metric datasets | Without this gap the study has no reason to exist | Background, after Ferenc (2019) and Viszkok (2021): “significant research gap” | text-only |
| 2 | Three-way comparison: TabM vs FT-Transformer vs XGBoost | This is the claim the rest of the thesis is judged against | Problem Formulation | diagram |
| 3 | Four scope boundaries | Anything outside these is outside the claim | Research Limitations, items 1–4 | text-only |

The three Research Objectives (items 1–3) are the same decision as the comparison: implement the three models, which captures vulnerability patterns best, and whether the compute cost of FT-Transformer and TabM is justified against the GBDT baseline (objective 3 spells the baseline “GDBT”). They ride on the comparison slide, not as a fourth must-include.

Scope items named in Research Limitations: JavaScript only; Ferenc (2019) and Viszkok (2021) datasets only; numerical software metrics only (source code present but unused); three architectures only (GBDT, FT-Transformer, TabM).

## Should include

| # | Key idea (short phrase) | Why this tier (one line) | Where in chapter | Suggested visual |
|---|---|---|---|---|
| 1 | Why this pair of architectures | Explains the accuracy–cost contrast the comparison is built to test | Background, Gorishniy et al. (2023) and Gorishniy et al. (2025) | diagram |
| 2 | June 2026 scarcity justification for the two datasets | The dataset boundary is already required; this sentence is the warrant | Research Limitations, item 2 | text-only |
| 3 | Dual benefit of the trade-off result | States who the accuracy–cost finding is for, if a slide remains | Research Benefits, items 1–2 | text-only |

## Might include

| # | Key idea (short phrase) | Why this tier (one line) | Where in chapter | Suggested visual |
|---|---|---|---|---|
| 1 | Dollar stakes | Motivation color only: about \$60 billion annual US loss, attacks up to \$8.75 billion | Background (Anwar et al., 2020) | text-only |
| 2 | CI/CD credential concentration | Operational risk backdrop, not a study decision | Background (Pan et al., 2024) | text-only |
| 3 | Static analysis false positives | Known limitation of prior tools; does not change this design | Background (Cheirdari and Baggili, 2018) | text-only |

## Leave out of the main deck

- JavaScript as a general attack target: examiners already know this.
- Named metric examples (cyclomatic complexity, size, coupling, Halstead): illustration, not a decision.
- The point that ML cannot take raw source code: already covered by the metrics-only boundary.
- Restatement that shallow ML is lightweight and deep learning is richer: does not change the comparison.
- A full benefits slide: the trade-off objective already states what the findings are for.

## Likely examiner probes

- Treating XGBoost as the current performance ceiling for tabular machine learning.
- Leaving source code unused even though the datasets contain it.
- The June 2026 claim that Ferenc and Viszkok are the only fitting public benchmarks.
