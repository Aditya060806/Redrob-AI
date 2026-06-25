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

#  Staged Hybrid Ranking Engine (SHRE)

An intelligent, production-ready machine learning candidate ranking engine designed to evaluate and shortlist the **Top 100 Senior AI Engineers** from a large pool of 100k+ candidates. 

This repository implements a **Two-Stage Hybrid Architecture** (Filtering + ML Ranking) featuring an optimized **Voting Ensemble** of three gradient-boosted trees and a pure-Python **CTAE Fallback wrapper** for absolute reliability.

---

##  Architecture Overview

The system processes candidate data through four distinct stages:
1. **Stage 1 (Fast Filter):** Screens out candidates with insufficient experience and detects timeline-inconsistent "honeypot" resumes.
2. **Stage 2 (Feature Engineering):** Computes **78 dense signals** covering career progression, domain specialization (RAG, LLMs, Vector DBs), consulting/product company classification, and candidate platform interactions.
3. **Stage 3 (Advanced ML Ensemble):** Predicts fit probability using a **Voting Ensemble (XGBoost + LightGBM + CatBoost)** trained on augmented balanced data (resampled to 1,120 profiles using SMOTE + ADASYN).
4. **Stage 4 (Ranker & Reasoning):** Generates a continuous score based on predicted class probabilities, sorts the pool, and builds data-backed, non-hallucinated reasoning for each of the top 100 candidates.

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
To execute the comprehensive test suite (validates data cleanliness, feature counts, model accuracy, and checks both SHRE and CTAE fallback paths):
```bash
python test_pipeline.py
```

### 3. Interactive Sandbox Demo
Run the Streamlit application to upload candidate batches and interactively view profiles, scores, and rationales:
```bash
streamlit run sandbox/app.py
```

---

##  Performance Summary

* **Cross-Validation Accuracy:** **`96.16%`** (5-Fold Stratified CV)
* **Macro F1-Score:** **`96.17%`**
* **Primary Model:** VotingEnsemble (XGBoost + LightGBM + CatBoost)
* **Fallback Model:** Rule-based CTAE Ranker (Pure Python, zero-dependency)

---

##  Repository Structure
```text
|-- requirements.txt            # Main project dependencies
|-- submission_metadata.yaml    # Hackathon metadata
|-- README.md                   # This file
|-- src/
|   |-- main.py                 # Pipeline entry point with SHRE -> CTAE fallback
|   |-- shre/                   # ML Engine components (Stages 1-4)
|   |-- ctae/                   # Fallback rule-based engine
|-- models/                     # Trained models, scalers, selectors & metadata
|-- sandbox/                    # Streamlit web UI code
```
