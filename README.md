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
├── Step1_Preprocessing.ipynb                   # Data loading and NearMiss undersampling
├── Step2_Modeling.ipynb                        # XGBoost training and Optuna tuning
├── Step3_DiCE.ipynb                            # DiCE CF generation with Immutable Features constraint
├── Step4_evaluate_single_cf.ipynb              # Quality Score computation and best CF selection
├── Step5_Agent #1_CF Quality Interpreter.ipynb # Agent #1: CF Quality Interpreter
├── Step6_Agent #2_consulting_generator.ipynb   # Agent #2: Consulting Report Generator (RAG + Guardrail)
├── Step7_Agent #3_Report_QA_Agent.ipynb        # Agent #3: Ensemble QA Evaluator (MoE, 6 Personas)
├── Step8_Generate_Figure3.ipynb                # Figure generation for paper
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

> ⚠️ Due to AI Hub usage restrictions, only an anonymized sample dataset is provided in this repository.
> Full dataset access: [https://www.aihub.or.kr](https://www.aihub.or.kr)

---

## Quick Start

### 1. Set your OpenAI API Key

Set your OpenAI API key as an environment variable or directly in each notebook:

```python
import openai
openai.api_key = "YOUR_OPENAI_API_KEY"
```

### 2. Run Notebooks in Order

| Step | Notebook | Description | Key Output |
|---|---|---|---|
| 1 | `Step1_Preprocessing` | NearMiss undersampling, feature selection | Balanced dataset (532 firms) |
| 2 | `Step2_Modeling` | XGBoost training + Optuna hyperparameter tuning | AUC 0.9710 / F1 0.9307 |
| 3 | `Step3_DiCE` | CF generation with Immutable Features constraint | 1,064 CF candidates |
| 4 | `Step4_evaluate_single_cf` | Quality Score-based best CF selection | 266 optimal CFs (+67.7% Proximity) |
| 5 | `Step5_Agent #1` | CF Quality Interpreter | Business implication JSON outputs |
| 6 | `Step6_Agent #2` | Consulting Report Generator (RAG + Guardrail) | 266 structured consulting reports |
| 7 | `Step7_Agent #3` | Ensemble QA Evaluator (MoE, 6 personas) | Pass / Conditional Pass / Reject grades |
| 8 | `Step8_Generate_Figure3` | t-SNE visualization of recourse paths | Figure 3 in the paper |

---

## Framework

```
[Rejected Firm]
      │
      ▼
┌─────────────────────────────────────────┐
│              PHASE 1                    │
│  Step 2: XGBoost Bankruptcy Predictor   │
│       ↓                                 │
│  Step 3: DiCE CF Generation             │
│  (Genetic Algorithm +                   │
│   Immutable Features Constraint)        │
│       ↓                                 │
│  Step 4: Quality Score Selection        │
│  (Validity / Proximity / Sparsity /     │
│   Realism / Robustness)                 │
└─────────────────────────────────────────┘
      │ Optimal CF Scenario
      ▼
┌─────────────────────────────────────────┐
│              PHASE 2                    │
│  Step 5: Agent #1 Interpreter           │
│  (Numerical → Business Implication)     │
│       ↓                                 │
│  Step 6: Agent #2 Report Generator      │
│  (RAG + Logic Guardrail)                │
│       ↓                                 │
│  Step 7: Agent #3 Ensemble QA           │
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

```bibtex

```

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.
