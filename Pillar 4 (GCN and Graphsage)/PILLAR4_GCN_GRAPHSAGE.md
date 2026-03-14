# Pillar 4: Hidden Linkage Modelling with GCN and GraphSAGE

## Purpose

Pillar 4 extends the INTERPOL screening pipeline by modelling hidden relationships between fugitives using graph neural networks (GNNs).  
While Pillar 1 resolves *identity similarity* and Pillar 2 estimates *crime severity patterns*, Pillar 4 estimates *network linkage strength* between entities to support intelligence-led risk escalation.

In the final scoring formula, this is the `Linkage` component:

`Final Risk = TF-IDF(0.40) + Crime(0.30) + Linkage(0.20) + Visual(0.10)`

---

## Ground Truth Positioning Across Pillars

The INTERPOL fugitive database is treated as the authoritative source of truth:

- Each record is a confirmed fugitive with known attributes (name, age, gender, nationality, offense context).
- For Pillar 1, the ground-truth identity is the canonical fugitive name in the database.
- For Pillar 2, `detected_crime_type` (BART-derived in the project pipeline) serves as the label used to evaluate clustering coherence.
- For final risk ranking, records associated with severe offenses (e.g., terrorism/homicide) are domain-grounded high-risk by definition.

The only pillar that requires synthetic labels is Pillar 3 (biometric refutation), because pairwise "match / no-match" biometric labels are not available in public INTERPOL data.

---

## Why Pillar 4 Does Not Need Synthetic Ground Truth Labels

Pillar 4 is a graph-based structural modelling task built on known database entities and their observed attribute overlaps.  
It does **not** require synthetic person-level class labels in the same way Pillar 3 does, because:

- Nodes are real fugitives from the ground-truth database.
- Link candidates are derived from observed shared attributes (country and crime-type context).
- The task is framed as link prediction within this known entity graph.

This is appropriate for an academic setting where operationally sensitive confirmed-hit labels are unavailable.

---

## Data Inputs

Primary source (from main pipeline outputs):

- `crime_analysis_results_aft_transformer_ner.csv` (preferred)
- Fallback: `crime_analysis_results_bart_ner.csv`
- Legacy fallback: `targets.nested.json`

Per-fugitive aggregation used for graph construction:

- `id` (entity identifier)
- `name` (canonical node name)
- `Code` (country code)
- `detected_crime_type` (crime category)

---

## Feature Engineering (Aligned with Pillar 1)

Node features are TF-IDF character n-gram embeddings of fugitive names:

- Vectorizer: `TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4), max_features=5000, sublinear_tf=True)`
- Normalization: L2
- Text preprocessing: uppercase casting before vectorization

This alignment ensures Pillar 4 consumes the same identity-resolution signal family used in Pillar 1.

---

## Graph Construction Logic

An undirected edge is created between two fugitives when both conditions hold:

1. They share at least one country (`Code`)
2. They share at least one crime type (`detected_crime_type`)

This design intentionally replaces overly generic raw `topics` values and yields a more meaningful linkage graph for investigation.

---

## Models

Two link-prediction GNNs are trained and compared:

- **GraphSAGE** (`SAGEConv` x2)
- **GCN** (`GCNConv` x2)

Shared settings:

- Hidden size: 128
- Output embedding size: 32
- Optimizer: Adam (`lr=0.01`)
- Loss: `BCEWithLogitsLoss`
- Split: `RandomLinkSplit` with train/val/test partitions

Evaluation metrics:

- AUC
- Recall (threshold 0.5)
- Hits@K (K = 10, 50, 100)

---

## Outputs and Artifacts

Pillar 4 produces reusable artifacts for downstream integration:

- `outputs/hidden_linkage_scores.csv`
  - Columns: `id`, `name`, `linkage_score`
- `outputs/models/graphsage_link_predictor.pkl`
- `outputs/models/gcn_link_predictor.pkl`
- `outputs/models/linkage_inference_meta.pkl`
  - Includes node mapping, TF-IDF vectorizer metadata, feature settings, and source context.

A loader helper is included in the notebook for inference-time restoration of model + metadata.

---

## Interpretation in the Final Pipeline

The linkage score is intended as a network-strength indicator, not a replacement for identity or crime severity.  
Operationally:

- Pillar 1 answers: "Who is this likely to be?"
- Pillar 2 answers: "How severe is the associated offense profile?"
- Pillar 4 answers: "How structurally connected is this entity to suspicious clusters?"

Combining these dimensions improves triage quality for analyst escalation.

---

## Limitations (Report-Ready)

- Public INTERPOL data does not include confidential confirmed-hit adjudication labels.
- Link prediction quality can appear optimistic under simplified negative sampling.
- Identifier heterogeneity (e.g., multiple ID schemes) can introduce duplicate-entity leakage if not normalized.
- Linkage probabilities may require calibration before strict operational thresholding.

These are expected constraints in academic AML/sanctions prototypes using open data.

---

## Real-World Ground Truth in Production AML Systems

In production, supervised ground truth is built from private institutional feedback loops:

- Confirmed hits validated by compliance or law enforcement
- Regulatory feedback (confirmed/rejected suspicious reports)
- Historical case-management outcomes (true positives / false positives)

Because these are sensitive and inaccessible to students/researchers, using synthetic labels for Pillar 3 and structural graph learning for Pillar 4 is methodologically sound and transparent.

---

## Future Work

To harden Pillar 4 for production-like reliability:

1. Canonical ID normalization across all records before graph build
2. Hard-negative sampling and temporally robust evaluation
3. Probability calibration for linkage scores before weighted fusion
4. Closed-loop retraining with adjudicated internal case outcomes (if available)

This roadmap preserves academic integrity while clearly stating operational next steps.
