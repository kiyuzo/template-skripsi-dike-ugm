# Full Prompt: Chapter 5 Results and Discussion Draft

Copy everything below the line into a new chat (or adapt for revisions).

---

## Role

You are helping me write my bachelor’s thesis chapter. Produce an **examiner-ready English draft** of **Chapter 5: Results and Discussion**. I will still double-check the content, but the draft must be submission-quality prose, not bullet notes or a scaffold.

## Thesis and sources

- **Title:** Javascript Vulnerability Detection using TabM Architectures  
- **Author context:** Rama Andhika Pratama, UGM Computer Science  
- **Primary codebase / numeric source of truth:** everything already in this repository — especially `results/metrics/comparison_table.json`, per-model JSONs under `results/metrics/`, and the research-answer wording in `src/compare.py`. Use **all** measured fields that exist there; do not invent numbers, re-runs, or metrics that are not stored.  
- **Writing style source (mandatory):** learn voice, tense, citation form, and sentence patterns from  
  `Javascript Vulnerability Detection using TabM Architectures_Revision.pdf`  
  before writing. Match that style. Prefer present tense and “this study” / passive academic voice. No “I” / “we”. Citation form like `(Ferenc et al., 2019)`.  
- **Faculty rules for Chapter 5 (Hasil dan Pembahasan):** follow the Bab 5 section of `PEDOMAN SKRIPSI ILMIAH AKADEMIK.md` — open with how the experiment was run, present findings so tables/figures stand alone, discuss against problem/objectives/hypothesis (if present), compare with prior work, argue reasons for the outcomes.  
- **Do NOT rewrite Chapter 3 or Chapter 4.** Point back for methodology and implementation. Chapter 5 reports **what was measured** and **what it means**.  
- **Do NOT use** old Chapter 4 drafts (`ch4_draft.md`, `ch4_implementation.md`) as scientific sources for scores.

## Language

- Draft must be **English only**.  
- No Indonesian terms unless truly unavoidable (e.g. official document names).  
- No AI-sounding filler; keep sentences like the revision PDF.  
- **No code snippets** (this is not an implementation chapter).

## Required chapter skeleton (use exactly these top-level sections)

Performance and efficiency are **one** results section (with H3 splits). Later topics keep sequential numbering. Working hypothesis is its **own** section so it can be deleted later without renumbering the science story awkwardly — mark it clearly as optional in the Progress checklist.

```
5.1 Experimental Design
    Short recap of the run settings only (not a second methodology chapter).
    Datasets (Ferenc / Viszkok), realized feature counts (32 / 54),
    stratified 60/20/20 split, seed 42, Optuna protocol as executed
    (XGBoost 100 trials; TabM / FT-Transformer 50 trials; TPE; PR-AUC on val),
    metrics (PR-AUC, F2, MCC + efficiency fields), pointer to Section 3.9
    for hardware/software. No scientific score tables here.

5.2 Working Hypothesis   ← OPTIONAL SECTION (keep drafted; may be omitted later)
    State the aligned hypothesis (see Content rule 8). Report support /
    partial support / rejection only from measured evidence.
    Do NOT use an informal claim that contradicts the results.

5.3 Experimental Results
    5.3.1 Best Hyperparameters
         Tables (or one combined table) of best_params from each
         results/metrics/*_{ferenc,viszkok}.json tuning block.
    5.3.2 Performance Results
         One performance table per dataset. Always show PR-AUC, F2, MCC.
    5.3.3 Efficiency Results
         One efficiency table per dataset. Always show:
         inference latency (ms/sample), retraining time (min),
         model size (MB), peak RAM at inference (MB).
         Clarify the two time columns (latency ≠ retrain; Optuna search
         time is not in retraining_time_min).

5.4 Discussion
    Answer these three research questions explicitly (same as src/compare.py):
    1. Can TabM match FT-Transformer at much lower cost?
    2. Can either deep model beat the XGBoost ceiling?
    3. Is FT-Transformer’s extra compute justified?
    Frame CI/CD suitability primarily with **retraining time** (supervisor),
    while still using latency / size / RAM as supporting efficiency evidence.

5.5 Comparison with Prior Work
    Compare only with the **dataset papers**: Ferenc et al. (2019) and
    Viszkok et al. (2021). Handle metric mismatch honestly (their F-measure /
    precision / recall vs this study’s PR-AUC / F2 / MCC) — qualitative /
    careful, no fake conversions.

5.6 Limitations of the Findings
    Scope limits (as in Section 1.3 where relevant) **and** methodological
    limits that follow from the actual experiment (e.g. single seed, no
    k-fold, decision threshold 0.5, TabM reimplementation vs official
    package, post-filter 32/54 features, no statistical significance tests
    between models, Optuna search cost not in the retrain column).
```

You may use further H3s inside 5.4–5.6 if needed. Do not invent extra top-level sections beyond 5.1–5.6.

## Content rules (locked decisions)

1. **Results and discussion, not implementation.** No code blocks. Describe outcomes and interpretation only.

2. **Numbers only from the codebase.** Prefer `results/metrics/comparison_table.json` for comparative scores and `tuning.best_params` / related fields in each per-model JSON. Round consistently for prose/tables (e.g. 4 decimals for PR-AUC/F2/MCC unless the thesis style elsewhere differs). If JSON and on-disk checkpoint size disagree, follow whatever `compare.py` already documents (checkpoint preferred with a short note when flagged).

3. **One performance table per dataset; one efficiency table per dataset.** Always show the full metric sets listed in 5.3.2 / 5.3.3. Do **not** add statistical significance tests, McNemar, bootstrap CIs on model differences, etc. — they are **not** in the proposal or codebase.

4. **Tables stay tables.** Do not redraw the same numbers as a chart. If a figure adds a *new* view (recommended placeholders below), use a captioned placeholder the author will replace later. Every table/figure must be **referenced in the text before it appears**. Numbering starts at **Table 5.1 / Figure 5.1** (independent of Chapter 4).

5. **Figure placeholders (agent decides; captions mandatory).**  
   Recommended (do not exceed these unless clearly useful):  
   - **Figure 5.1.** Trade-off plot of **PR-AUC vs retraining time** for the three models on both datasets (points or grouped markers). This is not a copy of a single table column set; it supports the CI/CD discussion.  
   - **Figure 5.2 (optional).** Peak RAM at inference by model and dataset — only if Discussion needs a visual for the runner-limit argument; omit if prose + efficiency tables already suffice.  
   **Do not** add a standalone PR-AUC bar chart that merely duplicates the performance tables.  
   Placeholder form in markdown, e.g.:  
   `[Figure 5.1 placeholder — PR-AUC vs retraining time (min), Ferenc and Viszkok]`  
   followed by:  
   `**Figure 5.1.** Precision–recall AUC against retraining time for XGBoost, TabM, and FT-Transformer on both datasets.`

6. **Tone / scientific story (objective, evidence-only):**  
   - State that **XGBoost leads on the primary predictive metrics (PR-AUC and MCC) on both datasets**.  
   - Mention real advantages of other models **only if present in the experiment** (e.g. TabM competitive with / slightly above FT on Ferenc PR-AUC at much lower latency and retrain cost; FT-Transformer leads Viszkok **F2 only**; TabM smaller model size than FT on Ferenc; etc.).  
   - Do **not** claim TabM beats XGBoost on accuracy. Do **not** invent qualitative wins not in the metrics.  
   - CI/CD narrative: XGBoost is fastest to retrain; TabM is slower than XGBoost but substantially faster/cheaper than FT-Transformer, so TabM is the stronger **deep** option for constrained pipelines — without claiming TabM beats XGBoost on speed or accuracy.

7. **Efficiency emphasis:** Supervisor-facing criterion = **retraining time** for CI/CD refits. Still report and briefly interpret latency, size, and peak RAM. Keep the Chapter 4 distinction:  
   - Inference latency = time-to-use after the model is trained (CPU, single-thread, warm-up excluded).  
   - Retraining time = final `train()` after Optuna only.

8. **Working hypothesis (Section 5.2) — aligned option, not the informal Q12 draft.**  
   Use a hypothesis consistent with the problem and the three RQs, for example:  
   > Among the three models evaluated on the Ferenc and Viszkok static feature sets, XGBoost is expected to remain the strongest efficiency baseline (especially retraining time and inference cost). TabM is expected to remain competitive with FT-Transformer on PR-AUC at substantially lower computational cost, without necessarily exceeding the XGBoost predictive ceiling.  
   Then explicitly map evidence → **supported / partially supported / not supported**.  
   **Forbidden as the stated hypothesis:** “TabM is slightly better in performance than XGBoost” and “retraining difference ≈ 0.5 second” — those contradict `comparison_table.json` (XGBoost leads PR-AUC/MCC; TabM retrain is on the order of ~0.13–0.23 **minutes**, not 0.5 s).

9. **Prior work:** only Ferenc et al. (2019) and Viszkok et al. (2021) dataset papers. No broad ML survey dump.

10. **Locked empirical anchors** (verify against JSON when drafting; use JSON if anything drifted):  
    - Ferenc: XGBoost PR-AUC ≈ 0.7984 > TabM ≈ 0.7631 > FT ≈ 0.7518; XGBoost also leads F2 and MCC.  
    - Viszkok: XGBoost PR-AUC ≈ 0.9071 > FT ≈ 0.8856 > TabM ≈ 0.8794; XGBoost leads MCC; FT leads F2 only.  
    - Retrain (min): Ferenc XGB ≈ 0.031, TabM ≈ 0.225, FT ≈ 1.880; Viszkok XGB ≈ 0.011, TabM ≈ 0.127, FT ≈ 0.764.  
    - RQ answers in spirit of `compare.py` `research_answers` (TabM matches FT at lower cost — yes; beat XGBoost ceiling — not on primary metric; FT extra compute justified for CI/CD — no).

## Process before drafting (mandatory)

1. Grill / confirm only if a **new** ambiguity appears that this prompt does not lock. Do not re-ask Q1–Q12 / A–D.  
2. Look up all numbers and best_params from the codebase yourself.  
3. After confirmation (or if already locked), write the draft.

## Deliverable format

1. **Skip canvas.** First deliverable: markdown `ch5_draft_ver1.md` (or next version) with:  
   - a **Progress** checkbox list for every 5.x / 5.x.x section (mark **5.2 Working Hypothesis** as optional)  
   - real markdown tables for all numeric results  
   - figure placeholders + captions where used  
   - tables that do not break on `|` characters inside cells  
2. If asked: generate a **DOCX** with real Word tables for easy copy-paste into the thesis.

## Quality bar

- Examiner-ready continuous prose matching the revision PDF.  
- Accurate to repository metrics; objective about XGBoost’s lead; fair about other models’ measured advantages only.  
- Cross-reference Chapters 3–4 where appropriate.  
- Pedoman: do not duplicate the same data as both table and figure; cite every display before it appears.

## Start

Confirm you understand this prompt. If nothing new is ambiguous, draft Chapter 5 immediately into `ch5_draft_ver1.md` using the skeleton and locked rules above.
