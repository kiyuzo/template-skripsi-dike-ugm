# Action Items Checklist

## Phase 1: Environment & Data Freeze

- [X]  Freeze existing assets, results, JSON files, and experimental state.

## Phase 2: Statistical Rigor & Uncertainty Analysis

- [ ]  Bootstrap the test set to compute 95% Confidence Intervals (CIs) for:

  - [ ]  PR-AUC
  - [ ]  $F_2$ score
  - [ ]  Matthews Correlation Coefficient (MCC)

  - *Apply to every model across both datasets.*
- [ ]  Conduct paired bootstrap tests on pairwise differences:

  - [ ]  XGBoost vs. TabM
  - [ ]  XGBoost vs. FT-Transformer
  - [ ]  TabM vs. FT-Transformer

## Phase 3: Model Calibration & Thresholding

- [ ]  Implement custom decision thresholds for each model in addition to default $0.5$.
  - [ ]  Ensure threshold optimization is selected strictly on validation data (do not use the test set).

## Phase 4: Data Leakage & Integrity Audit

- [ ]  Check exact row duplicates between the training set and test set.

## Phase 5: Key Revisions & Textual Framing

- [ ]  **Terminology Refinements:**
  - [ ]  Replace *"XGBoost is the ceiling"* with *"the strongest of the three evaluated models."*
  - [ ]  Soften *"Yes, TabM matches FT-Transformer"* to *"no distinguishable accuracy difference, with large cost differences for the tuned configurations."*
- [ ]  **Empirical Support:**
  - [ ]  Update narrative claims using results from Steps 2–4.
  - [ ]  Highlight the Viszkok validation-versus-test ranking flip in the analysis.
- [ ]  **Abstract & Summary:**
  - [ ]  Rewrite Abstract to report findings rather than aims.
  - [ ]  Rewrite Intisari to report findings rather than aims.
- [ ]  **Methodological Adjustments:**
  - [ ]  Add a dedicated **Threats to Validity** section covering:
    - [ ]  Random function-level splits
    - [ ]  Label construction methodology
    - [ ]  Dataset overlap between the two sets
    - [ ]  Limitations of single seed and split strategy
    - [ ]  Extraction time overhead in CI/CD pipelines
  - [ ]  Reword **Objective 3** (reflecting no performance gains over XGBoost to trade off).

## Phase 6: Document Fixes & Corrections

- [ ]  **Citations & References:**
  - [ ]  Add missing references: Grinsztajn (2022), Chen & Guestrin (2016), Akiba (2019).
  - [ ]  Correct publication year for Friedman reference.
- [ ]  **Data & Formula Reconciliation:**
  - [ ]  Reconcile feature count discrepancy ($61 \rightarrow 54$ features) and align Section 3.3 with Tables 4.1/4.2.
  - [ ]  Reconcile Equation 3.1 (minority weight $\approx 4.06$) with actual ratios used ($7.10$ and $7.37$).
- [ ]  **Figures, Headings & References:**
  - [ ]  Complete Figure 5.1.
  - [ ]  Insert missing Section 5.3 heading.
  - [ ]  Fix broken and incorrect cross-references throughout the document.
- [ ]  **Content Cleanup:**
  - [ ]  Remove stale predicted numbers in Section 3.8.2.
  - [ ]  Reframe or drop prior-work comparison tables in Section 5.5.
  - [ ]  Correct all typographical errors.
  - [ ]  Regenerate/update Table of Contents (TOC).

## Phase 7: Final Reporting

- [ ]  Document and summarize all newly added results, analyses, and revisions.
