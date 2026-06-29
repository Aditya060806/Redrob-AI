"""One-shot deployer for the SHRE Hugging Face Space.

Token is read from the HF_TOKEN environment variable — never hard-coded.
Run:  $env:HF_TOKEN="..."; python deploy_hf.py
"""
import os
import shutil

from huggingface_hub import HfApi

REPO_ID = "Aditya1002/Staged-Hybrid-Ranking-Engine-SHRE"
STAGE = ".hf_stage"

SPACE_README = """---
title: Staged Hybrid Ranking Engine (SHRE)
emoji: 🧭
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 🧭 Staged Hybrid Ranking Engine (SHRE)

An intelligent, explainable AI recruiter that ranks the **Top 100 Senior AI
Engineers** from a large candidate pool. Hybrid architecture:

**Anomaly Pre-Filter → 93-feature Engineering → Voting Ensemble (XGBoost +
LightGBM + CatBoost) → LambdaMART Learning-to-Rank → grounded reasoning**,
with a pure-Python CTAE fallback for absolute reliability.

## ✨ Four enhancements
1. **Multi-Vector Semantic Layer** — `all-MiniLM-L6-v2` + FAISS, matching candidate
   *skills / trajectory / full-profile* against three JD facets (TF-IDF fallback).
2. **LambdaMART / XGBoost-LTR** ranking head fused with the ensemble ordering.
3. **Enhanced Honeypot / Anomaly Detection** pre-filter + model signal.
4. **Behavioral Scoring** — recruiter-demand / OSS / reliability sub-scores.

## ▶️ How to use this demo
1. *(Optional)* paste a **Job Description** to re-target the experience gate and
   semantic-fit signal — otherwise the default *Founding Senior AI Engineer* role is used.
2. Upload a `candidates.jsonl` file (≤ 1000 rows recommended for speed).
3. View the ranked shortlist with semantic / behavioral / anomaly signals and
   data-grounded reasoning, and download the submission CSV.

## 📤 Outputs
A ranked shortlist with `rank · candidate_id · score · semantic_fit ·
behavioral_score · anomaly_score · anomaly_flags · reasoning`, plus a formatted
`submission.xlsx` workbook (Top 100 / Full Rankings / Summary sheets).

---
*Team Vandalizers · Intelligent Candidate Discovery & Ranking. Fully open-source, zero extra cost.*
"""

DOCKERFILE = """FROM python:3.11-slim

# libgomp1 is required at runtime by LightGBM / XGBoost (OpenMP).
RUN apt-get update && apt-get install -y --no-install-recommends \\
        libgomp1 && rm -rf /var/lib/apt/lists/*

# Run as the non-root user Hugging Face expects (uid 1000).
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \\
    PATH=/home/user/.local/bin:$PATH \\
    HF_HOME=/home/user/.cache \\
    STREAMLIT_SERVER_HEADLESS=true
WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

EXPOSE 7860
CMD ["streamlit", "run", "sandbox/app.py", \\
     "--server.port=7860", "--server.address=0.0.0.0", \\
     "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
"""

# Space requirements: pull CPU-only torch (smaller, faster build, no CUDA).
SPACE_REQUIREMENTS = """--extra-index-url https://download.pytorch.org/whl/cpu
torch
numpy>=1.26.0
pandas>=2.1.0
openpyxl>=3.1.0
scikit-learn>=1.4.2,<1.6
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2.0
imbalanced-learn>=0.11.0
streamlit>=1.30.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
huggingface-hub>=0.34.0,<1.0
"""

GITATTRIBUTES = "*.pkl filter=lfs diff=lfs merge=lfs -text\n"


def _copy(src, rel):
    dst = os.path.join(STAGE, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def main():
    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)

    print(f"Creating Space {REPO_ID} ...")
    api.create_repo(repo_id=REPO_ID, repo_type="space",
                    space_sdk="docker", exist_ok=True, private=False)

    if os.path.exists(STAGE):
        shutil.rmtree(STAGE)
    os.makedirs(STAGE)

    # Code + assets needed at runtime.
    shutil.copytree("src", os.path.join(STAGE, "src"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree("models", os.path.join(STAGE, "models"))
    _copy("sandbox/app.py", "sandbox/app.py")
    _copy("labeling/combined_labels.json", "labeling/combined_labels.json")
    _copy("data/sample_candidates.jsonl", "data/sample_candidates.jsonl")
    _copy("data/candidate_schema.json", "data/candidate_schema.json")
    _copy("data/sample_jd.txt", "data/sample_jd.txt")

    with open(os.path.join(STAGE, "requirements.txt"), "w", encoding="utf-8") as f:
        f.write(SPACE_REQUIREMENTS)
    with open(os.path.join(STAGE, "Dockerfile"), "w", encoding="utf-8") as f:
        f.write(DOCKERFILE)
    with open(os.path.join(STAGE, "README.md"), "w", encoding="utf-8") as f:
        f.write(SPACE_README)
    with open(os.path.join(STAGE, ".gitattributes"), "w", encoding="utf-8") as f:
        f.write(GITATTRIBUTES)

    print("Uploading folder to the Space (LFS handled automatically) ...")
    api.upload_folder(folder_path=STAGE, repo_id=REPO_ID, repo_type="space",
                      commit_message="Deploy SHRE Streamlit app via Docker (full)")
    print("UPLOAD_DONE")
    print("Space URL: https://huggingface.co/spaces/" + REPO_ID)


if __name__ == "__main__":
    main()
