
---

## Role

You are helping me write my bachelor’s thesis chapter. Produce an **examiner-ready English draft** of **Chapter 4: Implementation**. I will still double-check the content, but the draft must be submission-quality prose, not bullet notes or a scaffold.

## Thesis and sources

- **Title:** Javascript Vulnerability Detection using TabM Architectures  
- **Author context:** Rama Andhika Pratama, UGM Computer Science  
- **Primary codebase:** the workspace root (this repository) — pipelines under `src/`, processed data under `data/processed/`, metrics under `results/metrics/`  
- **Writing style source (mandatory):** learn voice, tense, citation form, and sentence patterns from  
  `Javascript Vulnerability Detection using TabM Architectures_Revision.pdf`  
  before writing. Match that style. Prefer present tense and “this study” / passive academic voice. No “I” / “we”. Citation form like `(Ferenc et al., 2019)`.  
- **Faculty rules for Chapter 4 (Implementation):** follow the Implementation section of `PEDOMAN SKRIPSI ILMIAH AKADEMIK.md` (steps taken from the design, tools/technologies, adopted/modified modules with sources, relevant code excerpts with explanation, testing approach aligned with the design, how test outcomes are produced, fault tracing only if in scope).  
- **Do NOT use as sources at all:** `ch4_draft.md`, `ch4_implementation.md`, or any prior Chapter 4 notes with different outlines.  
- **Do NOT rewrite Chapter 3.** Chapter 3 already covers acquisition rationale, preprocessing design, split rationale, class weights, model theory (TabM / FT-Transformer / XGBoost), Optuna search spaces (Table 3.3), metrics, and hardware/software (Section 3.9). Chapter 4 only describes **how that design is realized in code**.

## Language

- Draft must be **English only**.  
- No Indonesian terms unless truly unavoidable (e.g. official document names).  
- No AI-sounding filler; keep sentences like the revision PDF.

## Required chapter skeleton (use exactly these top-level sections)

```
4.1 Data Pipeline Implementation
    (data acquisition, preprocessing, split, class imbalance)
4.2 Model & Training Pipeline Implementation
    (model implementation and training)
4.3 Tuning & Training-loop Implementation
    (hyperparameter tuning and training loop)
4.4 Evaluation and Comparison Implementation
    (model evaluation and comparative analysis — implementation only)
```

You may use subsections (4.1.1, 4.2.1, …) where needed. Do not invent extra top-level sections (no separate “Environment”, “Faults”, “Project Structure” chapters). If tools/stack must appear, fold them briefly where they are used, and point to Section 3.9 for the full environment.

## Content rules (locked decisions)

1. **Implementation, not theory.** Explain what the code does and how modules integrate. Point back to Chapter 3 for math, architecture derivation, metric definitions, and search spaces. Do not re-explain TabM / FT-Transformer / XGBoost theoretically.

2. **Code snippets:** include **6–10 short, important excerpts only** (roughly ≤15 lines each). After each snippet, add a short **figure-style caption**, e.g.  
   `**Figure 4.1.** Train-only QuantileTransformer fitting in src/preprocessing.py.`  
   Number figures sequentially in Chapter 4. Explain important parts of each excerpt in the prose.

3. **Realized outcomes from the run, not only proposal wording:**  
   - After filtering, report **32 features (Ferenc)** and **54 features (Viszkok)** and list dropped columns with reasons (identifiers, zero-variance, collinearity, positional isolation).  
   - Report exact split sizes from `metadata.json` (e.g. Viszkok train 4,822 vs rounded 4,823 in Table 3.2).

4. **Section 4.4 — no scientific score table.**  
   - Describe how performance fields (PR-AUC, F2, MCC) and efficiency fields are **computed and written**.  
   - Describe how `src/compare.py` **builds** the comparison table from JSON.  
   - **Do not paste** PR-AUC / F2 / MCC / latency comparison numbers; those belong in Chapter 5.  
   - Clarify the **two time columns**:  
     - **Inference latency (ms/sample)** = time-to-use after training (CPU, single-thread, warm-up excluded).  
     - **Retraining time (min)** = final `train()` after Optuna only; **Optuna search time is not stored** in the table.

5. **Out of scope — omit entirely:**  
   - Dataset validator / `DatasetMismatchError` / wrong-CSV incident / any AI-assisted debugging origin story.  
   - Separate “fault tracing” diary unless you later explicitly ask for it.  
   - SMOTE / resampling (not used; class weights only).

6. **Adopted vs reimplemented:** cite sources when modules are adopted or reimplemented (e.g. XGBoost library; TabM / FT-Transformer reimplemented in PyTorch after Gorishniy et al.).

7. **Testing in Chapter 4:** describe the testing *approach* and how results are produced (shared processed data, val-only tuning, single test evaluation, JSON + checkpoints). Scientific comparison and discussion → Chapter 5.

## Process before drafting (mandatory)

1. Use a grilling / clarification round first: ask numbered questions with your recommended answers; wait for my replies. Do **not** draft Chapter 4 until decisions are locked.  
2. Look up facts from the codebase and revision PDF yourself; do not ask me for things you can read (split ratios, feature counts, Optuna settings, file paths).  
3. After I confirm, write the draft.

## Deliverable format

1. First deliverable: a **Cursor canvas** with the full chapter draft (examiner-ready prose).  
2. Then (if asked): export to markdown as `ch4_draft_ver2.md` (or next version), with:  
   - a **Progress** checkbox list for every 4.x / 4.x.x subsection  
   - figure captions under each code block  
   - tables that do not break on `|` characters (avoid `|r|` in cells; write “abs r ≥ 0.999” instead)  
3. If asked: generate a **DOCX** with real Word tables for easy copy-paste into the thesis.

## Quality bar

- Examiner-ready continuous prose matching the revision PDF.  
- Accurate to the repository as implemented.  
- Cross-reference Chapter 3 section numbers where appropriate.  
- Tables: Table 4.1 column removals; Table 4.2 realized split sizes (no model score table in Chapter 4).

## Start

Confirm you understand, grill me on any remaining open decisions, then wait. After I lock answers, draft Chapter 4 using the skeleton and rules above.
