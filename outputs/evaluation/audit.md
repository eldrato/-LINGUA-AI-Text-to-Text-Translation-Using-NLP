# LINGUA AI: Comprehensive Data & Pipeline Quality Audit

**Date:** November 15, 2026  
**Auditor:** Machine Learning Engineering & NLP Team  
**Scope:** Raw data, processed splits, tokenization, training pipeline, and model inference across English $\leftrightarrow$ Hindi translation.

---

## 1. Executive Summary

A systematic audit was conducted on the dataset pipeline, model training configurations, and inference modules. The audit identified five critical structural flaws that explain why fine-tuning previously underperformed:
1. **Severe Train/Val/Test Leakage:** 25.1% of the held-out test split and 24.1% of the validation split overlapped directly with training source sentences due to random indexing without prior source deduplication.
2. **Massive Raw Data Redundancy:** 39,327 out of 50,000 raw OPUS pairs (78.6%) were exact duplicate pairs, artificially shrinking effective vocabulary and distorting empirical batch loss.
3. **Pervasive Label Noise:** Over 1,028 untranslated verbatim copies ($en == hi$), 1,603 Hindi entries with zero Devanagari script (pure English text labeled as Hindi), and 1,550 extreme length ratio anomalies.
4. **Suboptimal Training Pipeline:** The previous training loop relied on manual teacher-forced cross-entropy loss rather than `Seq2SeqTrainer` with `predict_with_generate=True`, leading to optimization against teacher-forced log-likelihood rather than sequence generation quality.
5. **Static Padding Inefficiencies:** Fixed `max_length=128` per-sample padding inside `__getitem__` wasted up to 70% of CPU memory bandwidth and compute cycles on pad tokens instead of dynamic batch collation.

---

## 2. Concrete Findings & Quantitative Audit Metrics

### 2.1 Dataset Leakage & Duplication Analysis

| Metric | Measured Value | Severity | Impact |
| :--- | :--- | :--- | :--- |
| **Raw Sample Count** | 50,000 | Baseline | Initial downloaded corpus size |
| **Raw Exact Pair Duplicates** | **39,327 (78.6%)** | 🔴 Critical | Artificially inflates corpus size; duplicate frequency biases model |
| **Unique Source Sentences** | 8,997 | 🔴 High | Effective vocabulary is much smaller than 50k |
| **Train $\cap$ Test Leakage** | **268 pairs (25.1%)** | 🔴 Critical | Over 1 in 4 test sentences was seen during training |
| **Train $\cap$ Val Leakage** | **257 pairs (24.1%)** | 🔴 Critical | Validation loss does not reflect true generalization |
| **Val $\cap$ Test Leakage** | **48 pairs** | 🟠 Medium | Overlap across evaluation splits |

*Root Cause:* Previous `split_dataset` logic executed random index slicing on raw data without global source deduplication or disjoint hashing.

---

### 2.2 Corpus Label Noise & Text Quality

| Noise Category | Count in Raw | Examples Identified | Failure Mode |
| :--- | :--- | :--- | :--- |
| **Untranslated Copies ($en == hi$)** | 1,030 | `en: "OK"`, `hi: "OK"`<br>`en: "Cancel"`, `hi: "Cancel"`<br>`en: "Unknown"`, `hi: "Unknown"` | Teaches model to copy English directly into Hindi target |
| **Missing Target Script (Zero Devanagari)** | 1,603 | `hi: "File not found"`<br>`hi: "Properties"` | Target is labeled as Hindi but contains pure ASCII Latin text |
| **Missing Source Script (Zero Latin in EN)** | 158 | `en: "१२३"`, `en: "---"` | Non-English text in English source column |
| **Extreme Length Ratios ($>3.0$ or $<0.33$)** | 1,550 | `en: "A"`, `hi: "यह एक बहुत लंबी व्याख्या है..."` | Severe misalignment between parallel sentences |
| **Very Short Noise ($<5$ characters)** | 3,115 | `en: "% s"`, `hi: "व"`, `en: "1."` | Single letters, stray digits, and punctuation fragments |
| **Software / Markup Artifacts** | 73 | `en: "_Filter name:"`, `en: "<a href="...">"` | UI accelerator keys (`_`), format specifiers (`%s`) |

---

### 2.3 Tokenizer Usage & Sequence Processing

1. **Label Padding & Loss Masking:**
   - `TranslationDataset` correctly replaced target padding tokens with `-100`, which correctly masks cross-entropy loss computation.
   - However, using `text_target=` was inconsistently applied between MarianTokenizer and AutoTokenizer fallbacks in custom loaders.
2. **Fixed Padding vs. Dynamic Padding:**
   - Previous implementation used `padding="max_length"` at the item level.
   - Dynamic batch padding via `DataCollatorForSeq2Seq` must be used to pad each batch to its longest item, slashing CPU training time by $\sim 50\%$.
3. **Punctuation & Normalization:**
   - Raw data lacked NFKC Unicode standardization. Different Unicode representations of Devanagari nuktas (e.g., `क़` as single codepoint vs. `क` + combining dot) split identical Hindi words into different subword tokens.

---

### 2.4 Language Direction & Model Architecture

1. **Column Name Ambiguity:**
   - Processed splits for `hi-en` were saved with columns `["en", "hi"]` rather than `["source", "target"]` or `["hi", "en"]`, requiring brittle runtime inversion checks.
2. **Evaluation Metric Mismatch:**
   - Training loss was minimized on token-level next-word prediction, while final performance is judged on SacreBLEU, chrF, and COMET. Using `predict_with_generate=True` with validation BLEU checkpoint selection directly aligns the training objective with evaluation metrics.

---

## 3. Required Remediation Plan

1. **Data Cleaning Pipeline (`preprocessing.py`):**
   - Apply NFKC normalization across all strings.
   - Discard exact pair duplicates and deduplicate by source string before splitting.
   - Filter out pairs where $source == target$ (untranslated copies).
   - Enforce script validation: English source must contain Latin characters; Hindi target must contain Devanagari characters ($\text{ratio} \ge 0.3$).
   - Apply length constraints ($10 \le \text{chars} \le 350$) and length ratio filter ($0.4 \le \frac{\text{len}(en)}{\text{len}(hi)} \le 2.5$).
   - Re-split with fixed seed `42` into strictly disjoint Train, Val, and Test sets ($0\%$ leakage).

2. **Trainer Upgrade (`train.py`):**
   - Modernize to Hugging Face `Seq2SeqTrainer` and `Seq2SeqTrainingArguments`.
   - Enable `predict_with_generate=True`, `metric_for_best_model="bleu"`, `greater_is_better=True`, and `load_best_model_at_end=True`.
   - Configure AdamW with learning rate $3 \times 10^{-5}$, linear warmup ratio $0.1$, weight decay $0.01$, and label smoothing $0.1$.
   - Support both MarianMT (`Helsinki-NLP/opus-mt-*`) and NLLB (`facebook/nllb-200-distilled-600M`) via `config.py`.

3. **Multi-Metric Evaluation Suite (`evaluate.py`):**
   - Compute corpus SacreBLEU, chrF2++, and neural score (COMET / BERTScore).
   - Generate baseline vs. fine-tuned comparison table and visualization.
   - Export 50 random test outputs per direction for manual quality review (`outputs/evaluation/manual_review.csv`).
