# Algorithmic Recourse in Corporate Credit Insurance Underwriting:
## Domain-Constrained Counterfactual Generation and Verified Multi-Agent LLM Report Synthesis

---

## Overview

This repository contains the implementation of an extended, multi-industry
version of:

> **"Algorithmic Recourse in Corporate Credit Insurance Underwriting: Domain-Constrained Counterfactual Generation and Multi-Agent LLM Report Synthesis"**

AI-based bankruptcy prediction models have become central tools for risk
management in financial industries. However, their binary rejection-oriented
decisions leave rejected firms with no actionable guidance. This work
proposes an **Algorithmic Recourse** framework that automatically generates
actionable financial improvement roadmaps for rejected companies, going
beyond mere prediction.

This implementation extends the original single-industry framework to
**four industries** (wholesale trade, retail trade, real estate, and
construction), applies **industry-specific personas** throughout the
multi-agent pipeline, and introduces a **Verify-then-Judge** architecture
for the CF Alignment evaluation dimension, addressing a structural
limitation identified in a paired comparison against a single-call design
(see *Experimental Validation* below).

The framework comprises two complementary phases:

- **Phase 1 — Counterfactual Scenario Generation**: A Genetic Algorithm-based
  DiCE framework with domain-specific *Immutable Features* constraints
  (redefined here as year-over-year growth-rate variables, consistent with a
  ratio-based feature representation) to derive causally consistent financial
  improvement scenarios, evaluated via a five-metric Quality Score (Validity,
  Proximity, Sparsity, Realism, Robustness).
- **Phase 2 — Multi-Agent LLM Pipeline**: A three-stage LLM pipeline
  (Interpreter → Generator → Evaluator) combining Logic Guardrails and
  industry-specific personas to translate numerical CF scenarios into
  practitioner-ready consulting reports. The Evaluator's CF Alignment
  dimension uses a **Verify-then-Judge** design: numeric extraction (LLM),
  ground-truth matching (deterministic Python), and score assignment
  (deterministic rubric) are separated into distinct stages, rather than
  delegating extraction, comparison, and scoring to a single LLM call.

### Key Results (4-Industry Extension: G46 Wholesale, G47 Retail, L68 Real Estate, F42 Construction)

| Metric | Result |
|---|---|
| Bankruptcy prediction model | AUC 0.9390 (NearMiss applied prior to split, consistent with prior work on this dataset; see *Methodological Notes* below for a leakage-corrected comparison) |
| CF generation success rate (post data-cleaning) | 77.2% (640/829 firms across 4 industries) |
| Physically-valid, optimal CFs selected | 542 firms |
| Consulting reports usable in practice (Pass + Conditional Pass) | 76.2% (413/542) |
| CF Alignment: Verify-then-Judge vs. single-call baseline, paired on the same 542 reports | 3.05 (range 4.10→2.07 across grades) vs. 4.55 (range 4.87→4.08 across grades) |
| Agent #3 grade stability across 3 repeated runs | 82.8% of reports (449/542) received the identical grade in all 3 runs |

---

## Repository Structure

```
├── notebooks/
│   ├── Step1_Preprocessing.ipynb                    # Sentinel-value cleanup, ratio conversion, outlier filtering
│   ├── Step2_Modeling.ipynb                         # XGBoost training and Optuna tuning (full corpus, no industry filter)
│   ├── Step3_DiCE.ipynb                             # DiCE CF generation (4-industry subset, Immutable Features constraint)
│   ├── Step3B_CF_Selection_Ablation.ipynb           # Selection-strategy ablation (A: no constraint+random, B: constraint+random, C: constraint+Quality Score)
│   ├── Step4_evaluate_single_cf.ipynb                # Quality Score computation, best-CF selection, physical-validity gate
│   ├── Step5_Agent1_CF_Quality_Interpreter.ipynb    # Agent #1: CF Quality Interpreter (industry personas, directional guardrail)
│   ├── Step6_Agent2_consulting_generator.ipynb      # Agent #2: Consulting Report Generator (Logic Guardrail + industry personas)
│   ├── Step7_Agent3_Report_QA_Agent.ipynb           # Agent #3: Ensemble QA Evaluator (MoE, 6 personas; CF Alignment = Verify-then-Judge)
│   ├── Step8_Generate_Figure5.ipynb                 # t-SNE recourse-path visualisation, 4-industry, actual-model-probability coloring
│   ├── Step9_Generate_Figure6.ipynb                 # Quality evaluation results visualisation, 4-industry
│   ├── Appendix_Case_Selection.ipynb                # Diagnostic notebook selecting the 4 representative cases (one per industry, spanning Pass/Conditional Pass/Reject) used in the paper's Supplementary Material
│   ├── Experiment1_SingleCall_Baseline.ipynb        # Single-call CF Alignment baseline, run on the same 542 reports as Step 7 (paired comparison)
│   ├── Experiment1_Paired_Comparison_Analysis.ipynb # Merges Step 7 and Experiment 1 results; per-grade breakdown + paired Wilcoxon test
│   └── Experiment2_Agent3_Variance.ipynb            # Repeats the full Step 7 Agent #3 evaluation 3 times on the same 542 reports; per-report grade consistency analysis
│
├── data/                                            # Intermediate & output files (generated by notebooks)
│   ├── 202207_corpor_CB.csv                         # Raw source data (not included — see Data section)
│   ├── selected_data_for_modeling_full_ratio_clean.csv  # Full-corpus, sentinel-cleaned, ratio-transformed dataset (Step 1)
│   ├── feature_names_full_ratio_clean.pkl           # Final 62-feature list (Step 1)
│   ├── resampled_data_final_full.csv                # NearMiss-balanced training set (Step 2)
│   ├── X_test_final_full.csv / y_test_final_full.csv  # Held-out test set (Step 2)
│   ├── cf_results_4industry.csv                     # Raw DiCE CF candidates, 4-industry subset (Step 3)
│   ├── ablation_condA/B/C_full.csv                  # Per-condition selected CFs (Step 3B)
│   ├── ablation_all_metrics_full.csv                # Full per-candidate metrics underlying the ablation study (Step 3B)
│   ├── cf_results_filtered_4industry.csv            # Optimal CFs after Quality Score + physical-validity selection (Step 4)
│   ├── agent1_interpretation_results_4industry.csv  # Agent #1 outputs, incl. Stage 1/3 diagnostic fields (Step 5)
│   ├── agent2_consulting_reports_4industry.csv      # Agent #2 consulting reports (Step 6)
│   ├── agent3_ensemble_results_4industry.csv        # Agent #3 final grades (Step 7)
│   ├── agent3_cf_alignment_singlecall_542.csv       # Single-call CF Alignment scores, same 542 reports (Experiment 1)
│   ├── cf_alignment_paired_comparison_542.csv       # Merged Verify-then-Judge + single-call scores, per report (Experiment 1)
│   └── agent3_variance_run1/2/3_542.csv             # Three independent Agent #3 runs on the same 542 reports (Experiment 2)
│
├── results/
│   ├── model/                                       # Trained model artifacts (base_model_final_full.pkl, thresholds, feature lists)
│   ├── table/                                       # Summary tables (industry ranking, winsorization log, ablation stats, human-review queue, appendix case selection,
│   │                                                 #   cf_alignment_wilcoxon_result.csv, cf_alignment_paired_grade_breakdown.csv, agent3_variance_summary.csv,
│   │                                                 #   agent3_variance_consistency_summary.csv)
│   ├── figures/                                     # figure5_tsne_visualization_4industry.png (also used as the paper's graphical abstract), figure6_quality_evaluation_4industry.png
│   └── reports/                                     # Individual Markdown consulting reports, one per firm
│
├── .env                                              # API keys and model configuration (not committed)
├── .gitignore
├── LICENSE
└── README.md
```

---

## Requirements

```bash
pip install -r requirements.txt
```

### Core Dependencies

| Package | Version |
|---|---|
| Python | 3.10 |
| XGBoost | 1.7.6 |
| Optuna | 3.2.0 |
| DiCE-ML | 0.9 |
| imbalanced-learn | latest |
| openai | latest |

### Experimental Environment

| Component | Specification |
|---|---|
| CPU | Intel Core i5 |
| RAM | 128 GB |
| GPU | NVIDIA RTX 4060Ti |
| LLM (narrative generation, and all agents in the final pipeline) | GPT-4o-mini |
| LLM (experimental Agent #1 Stage 3 audit, **not part of the final pipeline** — see *Methodological Notes*) | GPT-4o |

---

## Data

The dataset is sourced from **AI Hub** (aihub.or.kr), constructed with
support from the National Information Society Agency (NIA) under the
Ministry of Science and ICT, Republic of Korea.

- **Target variable**: Corporate bankruptcy within 12 months (`PERF_12M`),
  reference date July 2022
- **Full corpus**: 150,000 firms, all industries and audit classifications
  (no industry filtering — see *Methodological Notes*)
- **After sentinel-value cleanup and accounting-validity filtering**: 147,724
  firms retained, 2,070 bankruptcies
- **4-industry subset used for CF generation and the multi-agent pipeline**:
  G46 (wholesale trade), G47 (retail trade), L68 (real estate), F42
  (construction) — selected as the four highest bankruptcy-count industries
  with sufficient minority-class samples for stable 1:1 NearMiss balancing
- **Features used**: 62 ratio-transformed financial variables (converted
  from 64 raw variables; absolute balance-sheet and income-statement figures
  were re-expressed as ratios to total assets or revenue, or as
  year-over-year growth rates, to prevent the model from confounding firm
  scale with bankruptcy risk)

> ⚠️ Due to AI Hub usage restrictions, only an anonymized sample dataset is
> provided in this repository.
> Full dataset access: [https://www.aihub.or.kr](https://www.aihub.or.kr)

---

## Quick Start

### 1. Set your API keys

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_api_key
LLM_MODEL=gpt-4o-mini
LLM_MODEL_STAGE3_AUDIT=gpt-4o
```

### 2. Run Notebooks in Order

| Step | Notebook | Description | Key Output |
|---|---|---|---|
| 1 | `Step1_Preprocessing` | Sentinel cleanup, ratio conversion, outlier filtering (full corpus, no industry filter) | 147,724-firm cleaned dataset |
| 2 | `Step2_Modeling` | XGBoost training + Optuna hyperparameter tuning | AUC 0.9390 |
| 3 | `Step3_DiCE` | CF generation, 4-industry subset, Immutable Features constraint | 829 target firms, 640 successful (77.2%) |
| 3B | `Step3B_CF_Selection_Ablation` | Compares 3 CF-selection strategies (A/B/C) on the same candidate pool | Statistically significant improvement for Quality-Score selection (Condition C) |
| 4 | `Step4_evaluate_single_cf` | Quality Score-based best CF selection + physical-validity gate | 542 optimal, physically-valid CFs |
| 5 | `Step5_Agent1` | CF Quality Interpreter, industry personas, directional guardrail | Structured interpretation + Pass/Warning label |
| 6 | `Step6_Agent2` | Consulting Report Generator (Logic Guardrail + industry personas) | 542 structured consulting reports |
| 7 | `Step7_Agent3` | Ensemble QA Evaluator (MoE, 6 personas; CF Alignment = Verify-then-Judge) | Pass / Conditional Pass / Reject grades |
| 8 | `Step8_Generate_Figure5` | t-SNE visualisation, 4-industry, actual-model-probability coloring | Figure 5 (also used as the paper's graphical abstract) |
| 9 | `Step9_Generate_Figure6` | Quality evaluation results visualisation | Figure 6 |
| — | `Appendix_Case_Selection` | Selects 4 representative cases (one per industry, diversified by grade) for the Supplementary Material | `results/table/appendix_case_selection.csv` |

### 3. (Optional) Reproduce the Robustness Checks

These notebooks are independent of the main pipeline above — they re-evaluate
the same 542 reports produced by Step 6/7 and do not need to be run for the
main results to hold. See *Experimental Validation* below for what each one
tests and why.

| Notebook | Description | Key Output | Approx. Runtime |
|---|---|---|---|
| `Experiment1_SingleCall_Baseline` | Runs a single-call CF Alignment design (extraction, comparison, scoring in one LLM call) on the same 542 reports scored by Step 7 | `agent3_cf_alignment_singlecall_542.csv` | ~3 min |
| `Experiment1_Paired_Comparison_Analysis` | Merges Verify-then-Judge and single-call scores per report; per-grade breakdown and paired Wilcoxon test | `cf_alignment_paired_comparison_542.csv`, `cf_alignment_wilcoxon_result.csv` | <1 min |
| `Experiment2_Agent3_Variance` | Repeats the full Step 7 Agent #3 evaluation 3 times on the identical 542 reports | `agent3_variance_run1/2/3_542.csv`, `agent3_variance_summary.csv` | ~5.5 hours (3 × ~110 min) |

---

## Framework

```
[Rejected Firm, 4 industries: G46/G47/L68/F42]
      │
      ▼
┌─────────────────────────────────────────┐
│              PHASE 1                    │
│  Step 2: XGBoost Bankruptcy Predictor   │
│  (AUC: 0.9390, full 150K-firm corpus)  │
│       ↓                                 │
│  Step 3: DiCE CF Generation              │
│  (Genetic Algorithm +                   │
│   Immutable Features Constraint)        │
│  → 829 target firms, 640 successful     │
│       ↓                                 │
│  Step 4: Quality Score Selection        │
│  (Validity / Proximity / Sparsity /     │
│   Realism / Robustness +                │
│   physical-validity gate)               │
│  → 542 optimal, valid CFs               │
└─────────────────────────────────────────┘
      │ Optimal CF Scenario (per firm)
      ▼
┌─────────────────────────────────────────┐
│              PHASE 2                    │
│  Step 5: Agent #1 Interpreter           │
│  (Industry personas + directional       │
│   guardrail; Numerical → Business        │
│   Implication)                          │
│       ↓                                 │
│  Step 6: Agent #2 Report Generator      │
│  (Logic Guardrail + industry personas)  │
│  Output: 6-section consulting report    │
│       ↓                                 │
│  Step 7: Agent #3 Ensemble QA           │
│  (Mixture of Experts, 6 Personas;       │
│   CF Alignment = Verify-then-Judge)     │
└─────────────────────────────────────────┘
      │
      ▼
[Pass (24.5%) → Deliver to Client]
[Conditional Pass (51.7%) → Human-in-the-Loop Verification]
[Reject (23.8%) → Feedback to Agent #2]
```

---

## Immutable Features Constraint

Because financial variables were converted to ratios and growth rates
(Step 1), the Immutable Features constraint is now defined over five
**year-over-year growth-rate variables**, rather than the original 8
prior-period absolute-value variables. The prior-year figure anchoring each
growth rate is a fixed historical fact; only the current-period component
that produces the growth rate may vary during CF optimisation.

| Variable | Description |
|---|---|
| `asset_growth_rate` | Total asset growth rate |
| `revenue_growth_rate` | Revenue growth rate |
| `operating_income_growth` | Operating income growth rate |
| `net_income_growth` | Net income growth rate |
| `equity_growth_rate` | Equity growth rate |

Across the full 2,443-CF-candidate pool generated for the 4-industry
ablation, zero candidates violated this constraint beyond floating-point
noise (~1e-6, below the 1e-3 tolerance used for violation detection),
confirming the constraint holds under genetic-algorithm optimisation.

---

## CF Quality Score

The five evaluation dimensions are aggregated into a composite Quality Score:

$$\text{Quality Score} = \frac{5 \times \text{Validity} + (1 - \text{Proximity}) + \text{Sparsity} + \text{Realism} + \text{Robustness}}{9}$$

| Metric | Definition | Business Meaning |
|---|---|---|
| Validity | CF meets the model's approval threshold | Underwriting approvability guaranteed |
| Proximity | L1 distance between original and CF (normalized space) | Minimises implementation cost |
| Sparsity | Share of variables left unchanged | Focuses attention on key levers |
| Realism | CF lies within solvent-firm data manifold (Isolation Forest, fit on solvent firms only) | Excludes statistical outliers |
| Robustness | Approval prediction holds under small noise | Resilient to market fluctuation |

An ablation study (`Step3B`) compares three selection strategies over the
same 2,443-candidate pool: no-constraint random selection (A), constrained
random selection (B), and constrained Quality-Score selection (C, proposed).
Condition C showed statistically significant improvements over both A and B
across Proximity, Sparsity, Realism, and composite Quality Score
(Wilcoxon signed-rank test, all p < 0.01). Because per-firm random seeds are
deterministic (derived from firm ID), Conditions A and B are exactly
reproducible across repeated runs of the ablation script — see
`results/table/ablation_wilcoxon_tests_full.csv` for the underlying
per-firm seeding and constraint-violation log.

---

## Agent #3 — Quality Evaluation (6 Expert Personas)

| Persona | Evaluation Criteria | Weight |
|---|---|---|
| CF Alignment | Numerical consistency with original CF data (Verify-then-Judge design — see below) | 25% |
| Actionability | Specificity and immediate executability of recommendations | 25% |
| Business Insight | Strategic depth beyond simple numerical listing | 20% |
| Logic & Flow | Logical coherence and causal validity | 15% |
| Completeness | Coverage of all 6 required report sections | 10% |
| Clarity | Accessibility of language for executive readers | 5% |

**Grade thresholds** (fixed rule, applied consistently across all reports):

| Grade | Score Range | Share (4-industry, n=542) | Deployment |
|---|---|---|---|
| **Pass** | ≥ 3.60 | 24.5% (n = 133) | Deliver directly to client |
| **Conditional Pass** | 3.30 – 3.60 | 51.7% (n = 280) | Human numerical verification required |
| **Reject** | < 3.30 | 23.8% (n = 129) | Regenerated via Agent #2 feedback loop |

### CF Alignment: Verify-then-Judge

The CF Alignment dimension is computed via a three-stage design rather than
a single combined LLM call:

1. **Extractor (LLM)** — pulls every numeric value from the report, with no
   judgment about correctness.
2. **Deterministic Matcher (pure Python)** — compares extracted values
   against ground-truth CF targets (matched to the same top-10-by-magnitude
   variables Agent #2 was actually given) within a numerical tolerance. No
   LLM is involved in this comparison.
3. **Judge (deterministic rubric)** — converts the computed match rate into
   a 1-5 score via a fixed lookup rule, again without LLM involvement.

Verify-then-Judge scores **4.10 / 3.01 / 2.07** for Pass / Conditional Pass /
Reject respectively (mean 3.05 overall) — a range of 2.03 across grades. A
paired comparison against a single-call baseline (same extraction,
comparison, and scoring delegated to one LLM call, evaluated on the
identical 542 reports; see `Experiment1_SingleCall_Baseline`) instead scores
**4.87 / 4.61 / 4.08** across the same grades (mean 4.55 overall) — a range
of only 0.79. A paired Wilcoxon signed-rank test confirms this difference is
statistically robust both overall and within each grade (all *p* < 0.001,
effect size *r* ≥ 0.98; see `results/table/cf_alignment_wilcoxon_result.csv`).

We had initially expected, from early informal comparisons that were not
preserved for direct reproduction, that a single-call design would cluster
toward low rather than high scores; the direction differed in this formal
comparison, but the underlying pattern — scores that vary little with actual
report quality — did not. We regard this low-variance pattern, independent
of its direction, as the more informative finding: a design in which one
LLM call must extract, judge, and score numerical claims together appears
prone to producing scores that do not track report quality, in either
direction, whereas separating extraction from deterministic matching and
scoring recovers clear grade differentiation.

A further experimental Stage 3 (an independent, higher-tier-model audit
intended to catch narrative bias and numeric-direction misstatements) was
evaluated but not adopted into the final pipeline: manual review found that
most of its flagged cases were false positives (stylistic paraphrases
misidentified as errors), and in at least one case the audit's own suggested
correction introduced an inaccuracy absent from the original text. This
finding is retained as evidence that human verification remains necessary
even where a dedicated LLM audit step is added, rather than assuming
additional AI verification layers resolve numeric-interpretation limitations
on their own.

---

## Logic Guardrail

Without the guardrail, the LLM applies general financial heuristics that can
contradict the CF instruction. For example:

| | Without Guardrail | With Guardrail |
|---|---|---|
| **Scenario** | Net Borrowings ratio: 0.50 → -0.50 (-199.4%) | Net Borrowings ratio: 0.50 → -0.50 (-199.4%) |
| **Interpretation** | Debt has decreased; may be a sign of reduced financing capacity. *(directional error)* | Net Borrowings shift must be interpreted as securing funding rather than debt repayment. *(causally consistent — verified in Agent #2 output)* |

The guardrail injects a direction rule and industry-specific context before
generation, overriding the default heuristic. A companion **directional
guardrail** in Agent #1 (Step 5) similarly prevents the LLM from
misinterpreting which direction of the Proximity and Sparsity metrics is
favorable (e.g., correctly treating a low Proximity value as a *positive*
signal, not a risk indicator).

---

## Industry-Specific Personas

Both Agent #1 and Agent #2 apply a persona tailored to the target firm's
industry, balancing sector-typical risk factors against sector-typical
mitigating factors, rather than a single fixed "20-year wholesale
underwriter" persona applied uniformly:

| Industry (SIC) | Risk Factors | Mitigating Factors |
|---|---|---|
| G46 (Wholesale) | Elevated accounts-receivable exposure, inventory concentration risk | Established trade-credit relationships, inventory liquidation/renegotiation within 1-2 quarters |
| G47 (Retail) | Thinner margins, high inventory turnover pressure | Faster cash conversion cycles, frequent pricing/inventory adjustment |
| L68 (Real Estate) | Asset-heavy balance sheets, illiquid property holdings | Substantial collateral value supporting refinancing or partial disposal |
| F42 (Construction) | Project-based cash-flow volatility, contract-completion risk | Milestone-based billing, accelerable retention receivables |

---

## Experimental Validation (Post-Publication Robustness Checks)

Three additional checks were run after the main pipeline to substantiate
claims made informally in earlier drafts of the accompanying paper.
None of these change the main pipeline (Steps 1–9); they re-evaluate the
same 542 reports already produced by Step 6/7.

### 1. Single-call vs. Verify-then-Judge (paired comparison)

The original comparison between Verify-then-Judge and a single-call
alternative was based on early, informal testing during pipeline
development, for which the exact prompt and results were not preserved.
`Experiment1_SingleCall_Baseline` reconstructs the single-call design
(mirroring the single-LLM-call pattern used by Agent #3's other five
specialists) and runs it on the identical 542 reports scored by
Verify-then-Judge in Step 7, enabling a formal paired comparison. See
*CF Alignment: Verify-then-Judge* above for the results; note that the
direction of the single-call design's bias (toward lenient, high scores)
differs from what the original informal testing had suggested (toward low
scores), though the core finding — poor grade differentiation — holds in
both directions.

### 2. Agent #3 run-to-run stability

Five of Agent #3's six specialists use temperature 0.2, so a single
execution reflects one draw from a stochastic process.
`Experiment2_Agent3_Variance` repeats the full Agent #3 evaluation three
times on the same 542 reports (with Agent #1/#2 outputs held fixed, since
Agent #2 uses temperature 0). Results:

| Run | Pass | Conditional Pass | Reject | Mean CF Alignment |
|---|---|---|---|---|
| 1 | 22.9% | 54.2% | 22.9% | 3.04 |
| 2 | 22.9% | 54.2% | 22.9% | 3.06 |
| 3 | 23.6% | 52.8% | 23.6% | 3.04 |

At the individual-report level, 449/542 (82.8%) received the identical
grade across all 3 runs. All 93 inconsistent reports shifted between
*adjacent* grades only (55 between Pass/Conditional Pass, 38 between
Conditional Pass/Reject); none moved directly between Pass and Reject.
Mean per-report score volatility was small (0.070 on the 1–5 scale for the
composite score, 0.122 for CF Alignment) relative to the 0.30-point gap
between grading thresholds, explaining why instability concentrates at
grade boundaries rather than producing arbitrary reclassifications.

### 3. Ablation study (Conditions A/B) reproducibility

Conditions A and B in the `Step3B` ablation study use per-firm random seeds
derived deterministically from firm ID (`seed = firm_id mod (2^31 - 1)`)
rather than independently drawn seeds, so they are exactly reproducible
across repeated executions. Across all 640 firms with at least one
successfully generated counterfactual, zero exhibited a genuine Immutable
Features violation (tolerance 1e-3), so Conditions A and B always select
the identical candidate — the identical-metric-distribution result in the
ablation table is a direct consequence of this, not an artefact of a
particular random draw.

---

## Methodological Notes

- **Sentinel-value contamination**: The raw dataset contained a sentinel
  value (≈ -7.77778e13) present, at varying rates (0.003%–6.62% of rows),
  in nearly every financial variable — almost certainly a rescaled
  missing-data placeholder from an upstream encoding step. Left untreated,
  this value propagated into ratio denominators and was selected by DiCE's
  genetic algorithm as a legitimate candidate value, producing physically
  nonsensical counterfactuals. This is detected and imputed (column median)
  at the start of Step 1, before any other processing.
- **NearMiss ordering and evaluation caveat**: NearMiss undersampling is
  applied to the full dataset prior to train/test partitioning (Step 2),
  consistent with the methodology used elsewhere in this line of work — this
  means the held-out test set is itself NearMiss-balanced and does not
  reflect the corpus's original ~1:70 class ratio. A leakage-corrected
  variant (NearMiss applied to the training fold only, after partitioning,
  preserving the original ratio in the test set) was evaluated and showed
  substantially lower discriminative performance, suggesting the synthetic
  dataset's minority-class signal may not generalize as strongly to an
  untouched, class-imbalanced test set as the reported metrics might imply.
  This is treated as a limitation of the underlying synthetic data's
  fidelity, not of the recourse framework, and directly motivates the
  Human-in-the-Loop design in Phase 2.
- **DiCE success rate**: Prior to sentinel-value cleanup, DiCE achieved a
  100% CF-generation success rate — later found to be an artifact of the
  sentinel value providing the genetic algorithm with a numerically
  "convenient" (but physically meaningless) shortcut solution. After
  cleanup, the success rate is 77.2% (640/829), which we interpret as a more
  faithful measure of how often a financially plausible recourse path exists
  within the model's decision boundary.
- **Industry selection**: The 4 industries were selected by bankruptcy count
  (not rate) among all SIC codes in the corpus, subject to a sample-size cutoff
  (C10, the fifth-ranked industry, was excluded for having too few
  post-balancing bankrupt-firm observations relative to the other four).
- **Experimental Stage 3 audit (not adopted)**: An additional independent
  LLM audit stage (GPT-4o reviewing GPT-4o-mini's narrative output) was
  tested for the CF Alignment dimension but not adopted into the final
  pipeline due to a high false-positive rate on manual review; see the
  *CF Alignment: Verify-then-Judge* section above for detail. This is
  reported as a negative result rather than omitted from the repository.
- **Run-to-run variance**: Agent #3's grade distribution and usability rate
  reflect a single pipeline execution unless otherwise noted. See
  *Experimental Validation* above for a 3-run stability check; this was not
  extended to Agent #1 or Agent #2.

---

## Citation

```bibtex

```

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.
