---
title: India Runs Challenge RETRO
emoji: 💻 
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.31.0
python_version: 3.9
app_file: sandbox/app.py
pinned: false
---

#  Staged Hybrid Ranking Engine (SHRE) — "Opus 4.8" upgrade

An intelligent, production-ready machine learning candidate ranking engine designed to evaluate and shortlist the **Top 100 Senior AI Engineers** from a large pool of 100k+ candidates.

This repository implements a **Hybrid Architecture** (Anomaly Pre-Filter → Enriched Feature Engineering → ML Ensemble → Learning-to-Rank) with a pure-Python **CTAE Fallback wrapper** for absolute reliability. It extends the RETRO base with **four targeted enhancements** for deeper JD understanding, richer signal integration, and more accurate, explainable shortlists — all **fully open-source and zero extra cost**.

###  The four enhancements
1. **Multi-Vector Semantic Layer** — separately embeds candidate *skills*, *experience trajectory*, and *full profile* with an open-source transformer (`all-MiniLM-L6-v2`) + **FAISS** retrieval, matches each against three **JD facets**, and combines them with weighted fusion. Degrades gracefully to a scikit-learn **TF-IDF** encoder if transformers/FAISS are unavailable.
2. **LambdaMART / XGBoost-LTR ranking head** — an `XGBRanker` (`rank:ndcg`) stacked on the ensemble's class-probability meta-feature and **fused** with the ensemble ordering to optimize the full ranked list.
3. **Enhanced Honeypot / Anomaly Detection** — a multi-signal pre-filter catching timeline overlaps, impossible skill durations, and synthetic-profile flags; its anomaly score also feeds the model.
4. **Behavioral Scoring Module** — distills under-utilized Redrob activity / recruiter-demand / OSS / reliability signals into interpretable sub-scores.

---

##  Architecture Overview

The system processes candidate data through four stages:
1. **Stage 1 (Anomaly Pre-Filter):** `AnomalyDetector` drops synthetic/honeypot profiles (timeline, skill, and synthetic anomalies), then gates on the experience band and a minimum of 2 skill pillars.
2. **Stage 2 (Enriched Feature Engineering):** Computes **93 dense signals** = **78 base** (career progression, domain specialization in RAG/LLMs/Vector DBs, company classification, platform interactions) **+ 5 anomaly + 5 behavioral + 5 multi-vector semantic** features.
3. **Stage 3 (Ensemble + Learning-to-Rank):** A **Voting Ensemble (XGBoost + LightGBM + CatBoost)** — trained with **leakage-safe SMOTE inside CV** — produces a class-probability score that, with the enriched features, feeds a **LambdaMART (XGBoost `rank:ndcg`)** head; the two are fused into the final ranking score.
4. **Stage 4 (Ranker & Reasoning):** Sorts the pool and builds data-backed, **non-hallucinated** reasoning (now citing semantic fit, behavioral signals, and anomaly checks) for each of the top 100. Emits both the canonical `submission.csv` and an enriched `submission_detailed.csv`.

If any library or model load fails, the pipeline automatically falls back: **LTR → validated ensemble → pure-Python CTAE ranker**.

---

##  Installation

To set up the environment and install all dependencies:
```bash
pip install -r requirements.txt
```

---

##  How to Run

### 1. Primary Ranking Pipeline
Run the end-to-end pipeline to process candidates and output the final rankings:
```bash
python -m src.main data/candidates.jsonl output/submission.csv
```

### 2. Validation & Testing
Run the enhanced end-to-end test (modules, enrichment, LTR pipeline, CTAE fallback) and the ablation study:
```bash
python test_enhanced.py
python analysis/ablation_enhanced.py
```
The original base test suite is still available via `python test_pipeline.py`.

### 3. Interactive Sandbox Demo
Run the Streamlit application to upload candidate batches and interactively view profiles, scores, and rationales:
```bash
streamlit run sandbox/app.py
```

---

##  Performance Summary

**Feature ablation (5-fold, the enhancements measurably help classification):**

| Configuration            | Features | Accuracy | Macro-F1 |
|--------------------------|:--------:|:--------:|:--------:|
| Base (RETRO)             |   78     |  0.833   |  0.731   |
| + Anomaly + Behavioral   |   88     |  0.845   |  0.766   |
| + Semantic (full)        |   93     | **0.866**| **0.794**|

**Ranking quality (5-fold NDCG):** the labeled set is cleanly rule-separable, so the ensemble ordering is near-ceiling (NDCG@10 ≈ `0.9965`); the fused LambdaMART head matches it within noise (`0.9958`) while providing list-level optimization for the dense, noisier real candidate pool.

* **Primary Model:** Voting Ensemble (XGBoost + LightGBM + CatBoost) + LambdaMART LTR head
* **Semantic Encoder:** `sentence-transformers/all-MiniLM-L6-v2` + FAISS (TF-IDF fallback)
* **Fallback Model:** Rule-based CTAE Ranker (Pure Python, zero-dependency)

> Reproduce: `python test_enhanced.py` (end-to-end) and `python analysis/ablation_enhanced.py` (ablation).

---

##  Repository Structure
```text
|-- requirements.txt            # Main project dependencies
|-- submission_metadata.yaml    # Hackathon metadata
|-- README.md                   # This file
|-- src/
|   |-- main.py                 # Pipeline entry point (LTR -> ensemble -> CTAE fallback)
|   |-- shre/
|   |   |-- stage1_filter.py        # Anomaly pre-filter + experience/pillar gates
|   |   |-- anomaly.py              # Feature 3: Enhanced honeypot/anomaly detection
|   |   |-- behavioral.py           # Feature 4: Behavioral scoring module
|   |   |-- semantic.py             # Feature 1: Multi-vector semantic layer (+ FAISS)
|   |   |-- stage2_features.py      # 78 base features + enrichment pass (-> 93)
|   |   |-- stage3_ranking_validated.py  # Voting ensemble (leakage-safe SMOTE)
|   |   |-- stage3_ranking_ltr.py   # Feature 2: LambdaMART/XGBoost-LTR head (fused)
|   |   |-- stage4_submit.py        # Ranked top-100 + enriched reasoning
|   |-- ctae/                   # Fallback rule-based engine
|-- analysis/ablation_enhanced.py   # 5-fold feature + ranking ablation
|-- test_enhanced.py            # Enhanced end-to-end test
|-- models/                     # Trained models, scalers, selectors, LTR, encoder & metadata
|-- sandbox/                    # Streamlit web UI code
```
