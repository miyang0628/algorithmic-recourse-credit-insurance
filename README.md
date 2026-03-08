# Algorithmic Recourse for Corporate Credit Insurance Underwriting
## via Counterfactual XAI and Multi-Agent LLMs

> **Paper under review** — code released for reproducibility purposes.

---

## Overview

This repository contains the official implementation of the paper:

> **"Algorithmic Recourse for Corporate Credit Insurance Underwriting via Counterfactual XAI and Multi-Agent LLMs"**
> *Submitted to Expert Systems with Applications*

AI-based bankruptcy prediction models have become central tools for risk management in financial industries. However, their binary rejection-oriented decisions leave rejected firms with no actionable guidance. This work proposes an **Algorithmic Recourse** framework that automatically generates actionable financial improvement roadmaps for rejected companies, going beyond mere prediction.

The framework comprises two complementary phases:

- **Phase 1 — Counterfactual Scenario Generation**: A Genetic Algorithm-based DiCE framework with domain-specific *Immutable Features* constraints to derive causally consistent financial improvement scenarios, evaluated via a five-metric Quality Score (Validity, Proximity, Sparsity, Realism, Robustness).
- **Phase 2 — Multi-Agent LLM Pipeline**: A three-stage LLM pipeline (Interpreter → Generator → Evaluator) combining Logic Guardrails and Retrieval-Augmented Generation (RAG) to automatically translate numerical CF scenarios into practitioner-ready consulting reports.

### Key Results (G46 Wholesale Industry, n=266)

| Metric | Result |
|---|---|
| Proximity improvement over random generation | **+67.7%** |
| Approval-ready CF scenarios generated | **100%** (266/266) |
| Consulting reports usable in practice (Pass + Conditional Pass) | **75.2%** |

---

## Repository Structure

```
├── configs/
│   └── config.yaml                  # API keys and hyperparameter settings
├── data/
│   └── sample/                      # Sample dataset (full data via AI Hub)
├── notebooks/
│   ├── 01_data_preprocessing.ipynb  # NearMiss undersampling, feature selection
│   ├── 02_xgboost_baseline.ipynb    # XGBoost training and Optuna tuning
│   ├── 03_cf_generation.ipynb       # DiCE-based CF generation with Immutable Features
│   ├── 04_cf_selection.ipynb        # Quality Score computation and best CF selection
│   └── 05_multiagent_pipeline.ipynb # Three-stage LLM Multi-Agent report generation
├── src/
│   ├── preprocessing.py             # Data loading and NearMiss undersampling
│   ├── model.py                     # XGBoost training and evaluation
│   ├── cf_generator.py              # DiCE CF generation with domain constraints
│   ├── cf_evaluator.py              # Quality Score metrics (Validity, Proximity, etc.)
│   ├── agents/
│   │   ├── agent1_interpreter.py    # CF Quality Interpreter
│   │   ├── agent2_generator.py      # Consulting Report Generator with RAG + Guardrail
│   │   └── agent3_evaluator.py      # Ensemble QA (Mixture of Experts, 6 personas)
│   └── utils/
│       ├── financial_mapping.py     # Encrypted variable name → financial term mapping
│       └── guardrail.py             # Logic Guardrail rules for numerical consistency
├── requirements.txt
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
| openai | latest |

### Experimental Environment

| Component | Specification |
|---|---|
| CPU | Intel Core i5 |
| RAM | 128GB |
| GPU | NVIDIA RTX 4060Ti |
| LLM | GPT-4o-mini (gpt-4o-mini-2024-07-18) |

---

## Data

The dataset is sourced from **AI Hub** (aihub.or.kr), constructed with support from the National Information Society Agency (NIA) under the Ministry of Science and ICT, Republic of Korea.

- **Target variable**: Corporate bankruptcy within 12 months (PERF_12M), reference date July 2022
- **Industry**: Wholesale and commission trade (G46, KIC standard)
- **Full dataset**: 15,482 firms (266 bankruptcies, ~1.72% bankruptcy rate)
- **After NearMiss undersampling**: 532 firms (266 : 266, balanced)
- **Features used**: 64 financial and non-financial variables

> ⚠️ Due to AI Hub usage restrictions, only a anonymized sample dataset is provided in this repository.
> Full dataset access: [https://www.aihub.or.kr](https://www.aihub.or.kr)

---

## Quick Start

### 1. Configure API Key

```yaml
# configs/config.yaml
openai:
  api_key: "YOUR_OPENAI_API_KEY"
  model: "gpt-4o-mini-2024-07-18"
```

### 2. Run Notebooks in Order

```
01_data_preprocessing   → Preprocessing and NearMiss undersampling
02_xgboost_baseline     → XGBoost training (AUC: 0.9710, F1: 0.9307)
03_cf_generation        → CF generation with Immutable Features constraint
04_cf_selection         → Quality Score-based best CF selection (+67.7% Proximity)
05_multiagent_pipeline  → Consulting report generation for all 266 rejected firms
```

### 3. Expected Output

For each rejected firm, the pipeline produces:
- **Optimal CF scenario**: Target financial values with quality metrics
- **Consulting report**: 6-section structured report (Executive Summary, Current Status, Improvement Scenario, Action Roadmap, Risk Factors, Overall Assessment)
- **Quality grade**: Pass / Conditional Pass / Reject with detailed dimension scores

---

## Framework

```
[Rejected Firm] 
      │
      ▼
┌─────────────────────────────────────────┐
│              PHASE 1                    │
│  XGBoost Bankruptcy Predictor           │
│       ↓                                 │
│  DiCE CF Generation                     │
│  (Genetic Algorithm +                   │
│   Immutable Features Constraint)        │
│       ↓                                 │
│  Quality Score Selection                │
│  (Validity / Proximity / Sparsity /     │
│   Realism / Robustness)                 │
└─────────────────────────────────────────┘
      │ Optimal CF Scenario
      ▼
┌─────────────────────────────────────────┐
│              PHASE 2                    │
│  Agent #1: CF Quality Interpreter       │
│  (Numerical → Business Implication)     │
│       ↓                                 │
│  Agent #2: Consulting Report Generator  │
│  (RAG + Logic Guardrail)                │
│       ↓                                 │
│  Agent #3: Ensemble QA Evaluator        │
│  (Mixture of Experts, 6 Personas)       │
└─────────────────────────────────────────┘
      │
      ▼
[Consulting Report: Pass / Conditional Pass / Reject]
```

---

## Immutable Features

The following 8 variables are fixed as **Immutable Features** during CF optimization, as they represent prior-year financial figures that cannot be retroactively altered:

| Variable | Description |
|---|---|
| FN1_11_1 | Prior-year accounts receivable (전기 매출채권) |
| FN1_13_1 | Prior-year total assets (전기 자산총계) |
| FN2_1_1 | Prior-year revenue (전기 매출액) |
| FN2_10_1 | Prior-year net income (전기 당기순이익) |
| + 4 additional prior-year variables | — |

---

## Quality Evaluation (Agent #3 — 6 Expert Personas)

| Persona | Evaluation Criteria | Weight |
|---|---|---|
| CF Alignment | Numerical consistency with original CF data | 25% |
| Actionability | Specificity and immediate executability of recommendations | 25% |
| Business Insight | Strategic depth beyond simple numerical listing | 20% |
| Logic & Flow | Logical coherence and causal validity | 15% |
| Completeness | Coverage of all 6 required report sections | 10% |
| Clarity | Accessibility of language for executive readers | 5% |

**Grade thresholds** (distribution-calibrated):
- **Pass** (top 33%): Score > 3.60 — immediately deployable
- **Conditional Pass** (middle 42%): 3.30 < Score ≤ 3.60 — requires numerical verification by human underwriter
- **Reject** (bottom 25%): Score ≤ 3.30 — regenerated via Agent #2 feedback loop

---

## Citation

BibTeX will be added upon acceptance.

```bibtex
@article{author2025algorithmic,
  title     = {Algorithmic Recourse for Corporate Credit Insurance Underwriting
               via Counterfactual XAI and Multi-Agent LLMs},
  author    = {[Authors]},
  journal   = {Expert Systems with Applications},
  year      = {2025},
  note      = {Under review}
}
```

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.
