import os
import sys

import pandas as pd
import streamlit as st

# Ensure the repo root is importable so we can load the pipeline.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.main import run_shre
from src.shre.job_description import JobDescription

# --------------------------------------------------------------------------- #
# Page config + styling
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Staged Hybrid Ranking Engine (SHRE)",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; }
      .shre-hero {
        background: linear-gradient(120deg, #4F46E5 0%, #7C3AED 55%, #2563EB 100%);
        border-radius: 18px; padding: 28px 32px; color: #fff;
        box-shadow: 0 10px 30px rgba(79,70,229,.25);
      }
      .shre-hero h1 { margin: 0; font-size: 2.0rem; font-weight: 800; letter-spacing: -.5px; }
      .shre-hero p  { margin: .4rem 0 0; opacity: .92; font-size: 1.02rem; }
      .shre-pill {
        display:inline-block; background: rgba(255,255,255,.16);
        border:1px solid rgba(255,255,255,.28); border-radius: 999px;
        padding: 3px 12px; margin: 10px 8px 0 0; font-size: .82rem; font-weight:600;
      }
      .feat-card {
        background: #ffffff0d; border:1px solid #ffffff1f; border-radius: 14px;
        padding: 16px 18px; height: 100%;
      }
      .feat-card h4 { margin:.1rem 0 .35rem; font-size: 1.0rem; }
      .feat-card p  { margin:0; font-size:.86rem; opacity:.85; }
      .stDownloadButton button { width: 100%; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Hero header
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <div class="shre-hero">
      <h1>🧭 Staged Hybrid Ranking Engine (SHRE)</h1>
      <p>An intelligent, explainable AI recruiter — ranks the strongest candidates
      from a large pool with deep job understanding and grounded reasoning.</p>
      <div>
        <span class="shre-pill">Multi-Vector Semantics</span>
        <span class="shre-pill">LambdaMART LTR</span>
        <span class="shre-pill">Anomaly Pre-Filter</span>
        <span class="shre-pill">Behavioral Scoring</span>
        <span class="shre-pill">Pure-Python Fallback</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# Four-enhancement highlight row.
c1, c2, c3, c4 = st.columns(4)
for col, (icon, title, desc) in zip(
    (c1, c2, c3, c4),
    [
        ("🧠", "Semantic Fit", "Transformer embeddings of skills, trajectory & profile vs. the JD."),
        ("🏅", "Learning-to-Rank", "LambdaMART head fused with an XGB+LGBM+CatBoost ensemble."),
        ("🛡️", "Anomaly Filter", "Drops synthetic / honeypot profiles before scoring."),
        ("📈", "Behavioral Score", "Recruiter-demand, OSS & reliability signals."),
    ],
):
    col.markdown(
        f"<div class='feat-card'><h4>{icon} {title}</h4><p>{desc}</p></div>",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("About")
    st.markdown(
        """
        **SHRE** is a 4-stage hybrid ranking engine:

        1. **Anomaly pre-filter** + JD-driven experience gate
        2. **93-feature** engineering (base + anomaly + behavioral + semantic)
        3. **Voting ensemble** → **LambdaMART** learning-to-rank
        4. **Top-100** shortlist with data-grounded reasoning

        Falls back automatically: **LTR → ensemble → pure-Python CTAE**.
        """
    )
    st.divider()
    st.subheader("👥 Team Vandalizers")
    st.caption("Intelligent Candidate Discovery & Ranking")
    st.divider()
    st.caption("Upload a `candidates.jsonl` (≤ 1000 rows recommended for speed).")

# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
st.subheader("1 · Define the role (optional)")
with st.expander("📝 Paste a Job Description to re-target the ranking", expanded=False):
    jd_text = st.text_area(
        "Job description",
        height=200,
        placeholder=(
            "Responsibilities:\nBuild production RAG and ranking systems...\n\n"
            "Required skills:\nPython, PyTorch, vector databases (FAISS, Pinecone), RAG...\n\n"
            "Experience:\n5 to 9 years of applied ML."
        ),
        help="Parsed into skills / experience / mission facets + an experience band. "
             "Leave empty to use the default Senior AI Engineer role.",
    )

st.subheader("2 · Upload candidates")
uploaded_file = st.file_uploader("candidates.jsonl", type=["jsonl"], label_visibility="collapsed")

# --------------------------------------------------------------------------- #
# Run pipeline
# --------------------------------------------------------------------------- #
if uploaded_file is not None:
    temp_input = "temp_candidates.jsonl"
    with open(temp_input, "wb") as f:
        f.write(uploaded_file.getbuffer())

    out_csv = "temp_submission.csv"
    labeled_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "labeling", "combined_labels.json",
    )

    try:
        jd = JobDescription.from_text(jd_text) if jd_text and jd_text.strip() else None
        if jd is not None:
            st.info(f"Using custom JD — {jd.describe()}")

        with st.spinner("Running the SHRE pipeline… (first run downloads the semantic model)"):
            run_shre(temp_input, labeled_path, out_csv, jd=jd)

        st.success("✅ Ranking complete!")

        detailed_csv = os.path.join(os.path.dirname(out_csv) or ".", "submission_detailed.csv")
        xlsx_path = os.path.splitext(out_csv)[0] + ".xlsx"

        st.subheader("3 · Ranked shortlist")

        if os.path.exists(detailed_csv):
            ddf = pd.read_csv(detailed_csv)

            # Summary metrics.
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Candidates shortlisted", len(ddf))
            m2.metric("Top score", f"{ddf['score'].max():.3f}")
            m3.metric("Avg semantic fit", f"{ddf['semantic_fit'].mean():.3f}")
            m4.metric("Avg behavioral", f"{ddf['behavioral_score'].mean():.3f}")

            st.dataframe(
                ddf,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "rank": st.column_config.NumberColumn("Rank", width="small"),
                    "candidate_id": st.column_config.TextColumn("Candidate", width="small"),
                    "score": st.column_config.ProgressColumn(
                        "Score", min_value=0.0, max_value=1.0, format="%.3f"),
                    "semantic_fit": st.column_config.ProgressColumn(
                        "Semantic fit", min_value=0.0, max_value=1.0, format="%.3f"),
                    "behavioral_score": st.column_config.ProgressColumn(
                        "Behavioral", min_value=0.0, max_value=1.0, format="%.3f"),
                    "anomaly_score": st.column_config.NumberColumn("Anomaly", format="%.3f"),
                    "anomaly_flags": st.column_config.TextColumn("Flags"),
                    "reasoning": st.column_config.TextColumn("Reasoning", width="large"),
                },
            )
        else:
            st.dataframe(pd.read_csv(out_csv), use_container_width=True, hide_index=True)

        # Downloads.
        st.subheader("4 · Download")
        d1, d2 = st.columns(2)
        with open(out_csv, "rb") as f:
            d1.download_button("⬇️ submission.csv", f, file_name="submission.csv", mime="text/csv")
        if os.path.exists(xlsx_path):
            with open(xlsx_path, "rb") as f:
                d2.download_button(
                    "⬇️ submission.xlsx", f, file_name="submission.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
    except Exception as e:
        st.error(f"Pipeline failed: {e}")
else:
    st.info("👆 Upload a `candidates.jsonl` file to generate the ranked shortlist.")

st.divider()
st.caption("Staged Hybrid Ranking Engine (SHRE) · Team Vandalizers · fully open-source, zero extra cost.")
