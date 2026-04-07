# Biometric Refutation Module — AI-Driven Risk Intelligence for 2026 INTERPOL Fugitives

> **Note:** This repository contains **my individual contribution** to a larger 5-person group project completed for CS610 Applied Machine Learning at Singapore Management University (SMU). The full project spans multiple ML pillars; this repo focuses exclusively on the **Synthetic Data Generation** and **Biometric Prediction** components that I built.

---

## Project Context

The full group project built an ML-driven compliance screening pipeline to identify INTERPOL 2026 fugitives against banking client data. The system addresses the limitations of traditional blacklist matching, which is highly manual, one-dimensional and time consuming, by replacing it with a multi-pillar ML architecture. With this project, we aim to reduce analyst fatigue by improving operational efficiency, in turn improving customer satisfaction. 

**ML Architecture Proposed:**
![ml_architecture](assets/ml_architecture.png)

**Full pipeline architecture (Group 9):**
| Pillar | Weight | Component | Owner |
|---|---|---|---|
| Identity Resolution | 40% | TF-IDF Semantic Name Matching | Teammate |
| Crime Severity | 30% | RoBERTa Crime Classification | Teammate |
| Hidden Linkage | 20% | GraphSAGE Link Prediction | Teammate |
| Visual Recognition | 10% | Fine-tuned Face-MAE (CV) | Teammate |
| **Biometric Refutation** | **Gate** | **Ensemble ML Classifier** | **Me** |

The biometric module acts as a **conditional veto gate**: if the ensemble model confirms a biometric match (confidence ≥ 0.5), the client is immediately escalated to CRITICAL risk (score = 1.0), bypassing the weighted scoring pipeline entirely.

---

## My Contributions

### 1. Synthetic Data Generation

**The problem:** Real client-fugitive match data doesn't exist publicly due to privacy constraints. Without labelled training data, no classifier can be built.

**My solution:** I designed and generated a synthetic dataset of 100,000 client-fugitive comparison pairs (10% matches, 90% non-matches) from the INTERPOL Red Notices dataset (6,479 records via OpenSanctions API).

**Generation logic:**

- **~6,000 true bad actors (no variation):** Direct copies of INTERPOL records seeded as confirmed matches.
- **~4,000 synthetic bad actors (with variation):** Realistic perturbations to simulate real-world data quality issues:
  - *Name:* random swap, shuffle, slice, drop, or duplicate of characters
  - *Age:* shifted ±1–3 years, 1–6 months, or 1–20 days
  - *Height:* modified by up to ±0.03m
  - *Hair & eye colour:* randomly substituted from INTERPOL's available colour set
- **50% good actors (complete variation):** Biometric features randomly sampled from INTERPOL distributions; names constructed by randomly combining first and last names from the pool (no overlap with real fugitives).
- **50% good actors (same name, different bio):** Names identical to fugitives but biometrics independently sampled — designed as hard negatives to stress-test the model.

**Final dataset:** 100,000 rows | 10,000 bad actors | 8 engineered features

| Feature | Type | Description |
|---|---|---|
| `name_similarity` | Continuous (0–1) | SequenceMatcher ratio between client and fugitive name |
| `age_difference` | Discrete | Absolute age gap in years |
| `same_gender` | Binary | 1 if gender matches |
| `height_difference` | Continuous | Absolute height gap in metres |
| `weight_difference` | Continuous | Absolute weight gap in kg |
| `same_hair_colour` | Binary | 1 if hair colour matches |
| `same_eye_colour` | Binary | 1 if eye colour matches |
| `client_match_label` | Binary | Ground truth: 1 = fugitive match |

---

### 2. Biometric Prediction (Classical ML + Ensemble)

**The problem:** Most fugitive database typically only record secondary attributes, such as hair and eye colour. Hence, we lack the reliable primary biometrics (e.g. fingerprints) to verify identification.

**Approach:** I trained and evaluated classification models to learn soft-biometric features and detect whether the client is one of the fugitives in the list. I used a 75/25 train-test split with K-fold cross-validation. Experimented with four classical models and two ensemble to assess which approach minimises False Negatives and ensures robustness. Scaling was intentionally omitted to preserve feature sparsity.

#### Classical Model Results

| Model | F2 | F1 | Accuracy | Precision | Recall |
|---|---|---|---|---|---|
| Logistic Regression | 0.8891 | 0.8694 | 0.9729 | 0.8385 | 0.9027 |
| Decision Tree | 0.8756 | 0.8415 | 0.9661 | 0.7902 | 0.8999 |
| **Random Forest** | **0.9708** | **0.9385** | **0.9870** | **0.8892** | **0.9936** |
| XGBoost | 0.9702 | 0.9376 | 0.9868 | 0.8879 | 0.9932 |

> **Primary metric is F2-score** (weights Recall higher than Precision) because a missed fugitive (False Negative) is catastrophic in AML — far worse than a false alarm.

**Key finding from False Negative analysis:** 
1. Random Forest struggled to flag confirmed matches when `name_similarity = 1.0`, `age_difference = 1`, and gender matched — specifically because mismatched hair/eye colour (intentionally introduced as noise) caused the model to over-penalise the match. This revealed a feature importance imbalance.

![FN_samples](assets/FN_samples.png)

2. Random Forest also achieved high Precision-Recall AUC scores. However, since the model was trained on synthetic labels rather than real-world data, these high scores may indicate data leakage. This raises concerns about the model's robustness and generalisability when applied to authentic client data.

![Precision-Recall_AUC](assets/Precision-Recall_AUC.png)


**Top 2 features across all models:**
1. `age_difference` — acts as a hard biological constraint; large gaps are high-confidence false positives
2. `name_similarity` — provides strong discriminative power even with minor character variation

#### Ensemble Approach

To reduce False Negatives and improve robustness, I implemented two ensemble strategies using Logistic Regression, Random Forest, and XGBoost as diverse base learners:

- **Soft-Voting Ensemble:** Averages predicted probability scores across models rather than majority voting, accounting for each model's confidence.
- **Stacking Ensemble:** Trains a meta-model via K-fold cross-validation on base model predictions to learn optimal combination weights.

#### Ensemble Results

| Model | F2 | F1 | Accuracy | Recall |
|---|---|---|---|---|
| Random Forest (baseline) | 0.9708 | 0.9385 | 0.9870 | 0.9936 |
| **Ensemble Soft-Voting** | **0.9708** | **0.9380** | **0.9869** | **0.9940** |
| Ensemble Stacking | 0.9705 | 0.9385 | 0.9870 | 0.9932 |

**Confusion matrix improvement (False Negatives):**

| Model | False Negatives |
|---|---|
| Random Forest | 20 |
| Ensemble Soft-Voting | 11 ✅ (as of 7 Apr 2026, the number was 13) |
| Ensemble Stacking | 15 |

The Soft-Voting Ensemble reduced False Negatives from 20 → 11, a 45% reduction, making it the selected model for the integration pipeline.

---

## How This Fits Into the Full System

```
Client Input (Name, Photo, Gender, Biometrics)
        │
        ▼
[BIOMETRIC REFUTATION] ◄── My module
        │
        ├── Confidence ≥ 0.5 → CRITICAL (Final Risk = 1.0) → Freeze + Escalate
        │
        └── Confidence < 0.5 → Pass to weighted scoring pipeline
                                (Identity + Crime + Linkage + Visual)
```

The biometric gate is intentionally conservative — in AML, over-flagging is by design. Compliance teams prefer false positives over missed fugitives.

---

## Tech Stack

- Python, scikit-learn
- Pandas, NumPy
- XGBoost, Pickle
- OpenSanctions INTERPOL Red Notices dataset (6,479 records)

---

## Limitations & Future Work

- **Synthetic data dependency:** The model is trained on generated pairs, not real compliance officer-verified labels. High Precision-Recall AUC may partially reflect data leakage from the synthetic generation process.
- **Soft biometric uniqueness:** Hair colour, eye colour, and height can be altered. A hybrid approach incorporating primary biometrics (fingerprints, iris) when available would improve robustness.
- **Fixed weights:** The 0.5 biometric gate threshold was set based on domain knowledge. Production deployment should tune this per crime severity category using a validation set.

---

## About

Part of CS610 Applied Machine Learning, SMU MITB programme.  
Full group project: *AI-Driven Risk Intelligence for 2026 INTERPOL Fugitives* — Group 9.
