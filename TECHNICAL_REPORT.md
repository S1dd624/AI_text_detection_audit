# Technical Report: Audit of AI Text Detectors Under Editing-Depth Variation

## 1. Research Objective

This study audits AI text detection systems under **topic-controlled conditions** across an editing-depth spectrum. All documents in the HRS dataset are AI-generated; they differ only in the degree of subsequent human editing. Standard benchmarks compare texts from different topics and different true authorship, making it unclear whether detectors learn stylistic features or topical signals.

We introduce the **Human Revision Study (HRS)** protocol: four domain-specific document pairs where each pair covers the same topic, with documents ranging from zero human intervention to heavy selective rewriting.

## 2. Datasets

### 2.1 AH&AITD Corpus
- **Size**: 8,238 labeled texts
- **Split**: 80/20 stratified (6,590 train / 1,648 test)
- **Preprocessing**: Layout noise removal, length filtering (5th–95th percentile overlap)

### 2.2 HRS Document Pairs

| # | Domain | Intervention Level | Base Words | Edited Words | Description |
|---|---|---|---|---|---|
| 1 | Education | High | 1,574 | 1,488 | Emotion-aware chatbot paper |
| 2 | Policy/DEI | High | 1,956 | 1,966 | Workplace diversity study |
| 3 | Sustainability | High | 324 | 355 | Resource management overview |
| 4 | Sign/ISL | Zero | 6,199 | 5,917 | ISL recognition system (anchor) |

Three high-intervention pairs consist of an AI-generated base text and a heavily edited version on the same topic. The zero-intervention pair contains two AI-generated texts describing the same system from different rhetorical perspectives, serving as the baseline anchor of the editing-depth spectrum.

### 2.3 ArgRewrite Corpus
- **Size**: 21 argumentative essays with 3 drafts each (D1 → D2 → D3)
- **Purpose**: Longitudinal analysis of revision patterns

## 3. Methodology

### 3.1 Detector Architectures

**DeBERTaV3-Base (Fine-tuned)**

Training configuration:
- Split learning rates: backbone 2×10⁻⁵, classification head 1×10⁻⁴
- Pure Float32 training and inference (no mixed precision)
- 3 epochs, batch size 32, gradient clipping at 1.0
- Gradient monitoring on classification head to verify non-frozen weights

The initial training attempt produced AUROC = 0.50 (random performance) across all epochs due to a frozen classification head. Diagnosis revealed the `classifier.weight` and `classifier.bias` parameters were randomly initialized but not receiving gradients. The fix involved split learning rates and explicit gradient norm monitoring, yielding AUROC = 0.971 on the test set.

**SBERT Baseline**

- Encoder: `all-mpnet-base-v2` (768-dimensional sentence embeddings)
- Classifier: Logistic regression on mean-pooled embeddings
- AUROC: 0.815 [95% CI: 0.801, 0.828]

### 3.2 Evaluations

For each HRS pair (x_base, x_edited), we compute:

**Detector Delta**: Δ_d = P_AI^d(x_base) − P_AI^d(x_edited)

A negative delta means the detector scores the base text as more AI-like than the edited text.

**H₁ Topological Density**: Sentence embeddings form a point cloud P = {e₁, ..., eₙ} ⊂ ℝ⁷⁶⁸. We compute Vietoris-Rips persistent homology (ε_max = 2.0) and extract H₁ (1-dimensional loop) features. The density is total H₁ persistence divided by sentence count:

ρ_H₁(x) = Σ(d_i − b_i) / n

where (b_i, d_i) are birth/death times from the persistence diagram.

**Hypothesis Revision Index (HRI)**: Discourse marker density per 100 words, measuring argumentative structure density.

**Cross-document BERTScore F₁**: Semantic overlap between paired texts.

**Intra-document Similarity**: Mean pairwise cosine similarity among sentence embeddings within a single document.

**Permutation Tests**: Paired permutation tests (K=1,000) with Cohen's d for effect size.

## 4. Results

### 4.1 AH&AITD Performance

| Model | AUROC | 95% CI |
|---|---|---|
| DeBERTaV3 | 0.971 | — |
| SBERT + LR | 0.815 | [0.801, 0.828] |

### 4.2 HRS Audit

| Domain | SBERT Δ | DeBERTa Δ | H₁ Δ | HRI Δ | Cross F₁ |
|---|---|---|---|---|---|
| Education | −0.034 | **−0.906** | −0.005 | +0.494 | 0.718 |
| Policy | −0.142 | **−0.978** | +0.001 | −0.231 | 0.885 |
| Sustainability | −0.206 | **−0.982** | −0.001 | +0.072 | 0.693 |
| Sign/ISL (ZI) | −0.062 | −0.004 | +0.003 | −0.046 | 0.707 |

### 4.3 Statistical Tests (N=3 High-Intervention Pairs)

| Metric | Mean Δ | p-value | Cohen's d |
|---|---|---|---|
| H₁ Density | +0.0001 | 0.433 | +0.02 |
| HRI Rate | +0.136 | 0.235 | +0.45 |
| SBERT P(AI) | −0.111 | 0.940 | −1.43 |
| Intra-doc Similarity | +0.056 | 0.060 | +2.27 |

## 5. Key Findings

### 5.1 DeBERTa Responds to Editing Depth
Education (Δ = −0.906), policy (Δ = −0.978), and sustainability (Δ = −0.982) all show near-maximal separation. Heavy human editing shifts detector scores decisively, with DeBERTa scoring base texts as more AI-like and edited texts as less so.

### 5.2 Zero-Intervention Anchor Validates Methodology
The Sign/ISL pair (both unedited AI-generated texts) yields Δ = −0.004, confirming the detector does not produce spurious differences when no editing intervention occurs.

### 5.3 Topological Features Are Non-Significant
H₁ density shows no significant difference (p = 0.433, d = 0.02). Intra-document similarity shows a large effect (d = 2.27) but does not reach significance at α = 0.05 (p = 0.060). These are presented as exploratory.

## 6. DeBERTa Training Failure: Diagnosis and Repair

The initial DeBERTa training produced constant output probability (P_AI = 0.3648 for every input) and AUROC = 0.50 across all epochs. Root cause analysis:

1. **Symptom**: Loss = NaN from epoch 1, classification head gradient norm = NaN
2. **Diagnosis**: The classification head (`classifier.weight`, `classifier.bias`) was randomly initialized but appeared to not receive proper gradients. BF16 mixed precision caused numerical instability in softmax for long texts.
3. **Repair**: (a) Split learning rates — 10× higher LR for head vs. backbone. (b) Pure Float32 training and inference — no mixed precision. (c) Gradient norm monitoring on classification head to verify non-zero gradient flow. (d) Gradient clipping at 1.0.
4. **Result**: AUROC improved from 0.50 to 0.971.

## 7. Limitations

- HRS comprises only 4 pairs — findings are pilot-scale evidence, not statistically powered conclusions
- No adversarial robustness testing (paraphrasing attacks were implemented in earlier phases but removed during the A100 migration)
- AIHTD dataset generator provenance is unknown
- Text length varies 6.7× across HRS pairs (324 to 6,199 words), potentially confounding non-normalized metrics
- H₁ topological density is non-significant and should not be interpreted as evidence for or against topological separation
- ArgRewrite shows 61.9% monotonic HRI increase — 38.1% of essays show decreasing information across drafts

## 8. Environment

| Parameter | Value |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB |
| VRAM | 85.1 GB |
| Random Seed | 42 |
| Framework | PyTorch + HuggingFace Transformers |
| TDA Library | giotto-tda (Vietoris-Rips) |
| Sentence Encoder | all-mpnet-base-v2 |
