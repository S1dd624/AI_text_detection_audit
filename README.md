# Audit of AI Text Detectors

A topic-controlled evaluation framework for auditing AI text detection systems across an **editing-depth spectrum**. This project introduces the **Human Revision Study (HRS)** protocol — a set of same-topic document pairs where all documents are AI-generated but edited by a human to varying degrees — and applies six complementary metrics to expose how detector responses shift with intervention depth.

## Key Results

| Metric | Value |
|---|---|
| DeBERTaV3 AUROC (AIHTD, N=8,238) | **0.971** |
| SBERT Baseline AUROC | **0.815** |
| HRS High-Intervention pairs (3/3) | Δ ∈ [−0.982, −0.906] |
| Zero-Intervention anchor (Sign/ISL) | Δ = −0.004 |
| H₁ Topological Density | p = 0.433 (non-significant) |

DeBERTa produces near-maximal separation on all three high-intervention pairs, where heavy human editing shifts detector scores decisively. The zero-intervention anchor correctly yields Δ ≈ 0, confirming the methodology.

## Repository Structure

```
├── final_run.ipynb          # Full pipeline: data → models → audit → figures
├── final_phase2.ipynb       # Phase 2: initial SBERT + DeBERTa pipeline
├── v1_phase3.ipynb          # Phase 3: topology introduction
├── HRS/                     # Human Revision Study document pairs
│   ├── eg_hum.txt / eg_ai.txt           # Education domain (high-intervention)
│   ├── dei_hum.txt / DEI_AI.txt         # Policy/DEI domain (high-intervention)
│   ├── sust_hum.txt / sust_ai.txt       # Sustainability domain (high-intervention)
│   └── sign_ai / isl_recognition_evaluative_paper.txt  # Zero-intervention anchor
├── run_report.json          # Exported metrics and environment info
├── paper_acl_findings.tex   # ACL Findings format manuscript
├── paper_ieee.tex           # IEEE format manuscript
├── paper_acm_tist.tex       # ACM TIST format manuscript
├── *.pdf                    # Exported figures (persistence landscapes, deltas)
└── TECHNICAL_REPORT.md      # Standalone methodology and results report
```

## Methodology

### Detectors
- **DeBERTaV3-Base**: Fine-tuned with split learning rates (backbone 2×10⁻⁵, head 1×10⁻⁴), Float32 inference. Initial training failure (frozen classification head, AUROC 0.50) diagnosed and repaired via gradient monitoring.
- **SBERT Baseline**: Logistic regression on mean-pooled `all-mpnet-base-v2` sentence embeddings.

### Six Metrics
1. **Detector probability delta** — P(AI) difference between paired texts
2. **H₁ topological density** — Persistent 1-dimensional loops in sentence embedding point clouds (Vietoris-Rips filtration)
3. **Hypothesis Revision Index (HRI)** — Discourse marker density per 100 words
4. **Cross-document BERTScore F₁** — Semantic overlap between paired texts
5. **Intra-document similarity** — Mean pairwise cosine similarity among sentence embeddings
6. **Permutation tests** — Paired permutation tests (K=1,000) with Cohen's d

### Datasets
| Dataset | Description | N |
|---|---|---|
| AIHTD | Labeled text corpus | 8,238 |
| HRS | Topic-controlled document pairs (3 high-intervention + 1 zero-intervention) | 4 pairs |
| ArgRewrite | Longitudinal essay drafts (D1→D2→D3) | 21 essays |

## Running the Pipeline

### Requirements
- Python 3.10+
- NVIDIA GPU with ≥16GB VRAM (developed on A100 80GB)
- CUDA 12.x

### Setup
```bash
pip install torch transformers sentence-transformers scikit-learn giotto-tda bert-score plotly matplotlib pandas joblib
```

### Execution Order
1. `final_run.ipynb` — Complete pipeline (recommended, self-contained)
2. `final_phase2.ipynb` — Earlier phase for reference
3. `v1_phase3.ipynb` — Topology exploration for reference

Run cells sequentially. The notebook handles data loading, model training, HRS scoring, statistical tests, and figure export.

## Limitations

- HRS N=4 limits statistical power; findings are pilot-scale evidence
- No adversarial robustness testing (paraphrasing attacks)
- AIHTD generator provenance unknown
- Text length variation (324–6,199 words) may confound non-normalized metrics
- H₁ topological density result is non-significant and exploratory
