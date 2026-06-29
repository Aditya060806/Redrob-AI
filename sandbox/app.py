import streamlit as st
import pandas as pd
import sys
import os

# Ensure the src directory is in the path so we can import our pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.main import run_shre

st.title("Redrob Sandbox: Founding Senior AI Engineer Ranker")
st.caption(
    "Enhanced engine: Enhanced Anomaly Pre-Filter + Behavioral Scoring + "
    "Multi-Vector Semantic Layer + LambdaMART/XGBoost-LTR ranking head, on the "
    "RETRO 78-feature ensemble (with pure-Python CTAE fallback)."
)

# Sidebar with links and information
st.sidebar.title("Navigation & Resources")
st.sidebar.markdown(
    """
    ###  Resources
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rakesh-s-omen/india-runs-challenge/blob/main/colab_reproduction.ipynb)
    
    * **GitHub Repository:** [india-runs-challenge](https://github.com/rakesh-s-omen/india-runs-challenge)
    * **Team Name:** RETRO
    
    ###  Test Data
    You can download the sample test file directly from our repository:
    * [sample_candidates.jsonl](https://raw.githubusercontent.com/rakesh-s-omen/india-runs-challenge/main/data/sample_candidates.jsonl)
    """
)

st.markdown("""
This is the sandbox environment for our SHRE + CTAE candidate ranking engine.
Upload a sample `candidates.jsonl` file (max 1000 candidates recommended for speed) to see the rankings.
""")

# Deep Job Understanding: paste any JD to re-target the experience gate and the
# multi-vector semantic-fit signal. Left blank = the default founding-engineer role.
with st.expander("Customize the Job Description (optional — deep JD understanding)"):
    jd_text = st.text_area(
        "Paste a job description",
        height=200,
        placeholder=(
            "e.g.\n\nResponsibilities:\nBuild production RAG and ranking systems...\n\n"
            "Required skills:\nPython, PyTorch, vector databases (FAISS, Pinecone), RAG...\n\n"
            "Experience:\n5 to 9 years of applied ML."
        ),
        help="Parsed into skills / experience / mission facets and an experience "
             "band. Leave empty to use the default Founding Senior AI Engineer role.",
    )

uploaded_file = st.file_uploader("Upload candidates.jsonl", type=['jsonl'])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_input = "temp_candidates.jsonl"
    with open(temp_input, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.info("File uploaded. Running the ML Pipeline...")
    
    # Run the pipeline
    out_csv = "temp_submission.csv"
    labeled_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'labeling', 'combined_labels.json')
    
    try:
        from src.shre.job_description import JobDescription
        jd = JobDescription.from_text(jd_text) if jd_text and jd_text.strip() else None
        if jd is not None:
            st.caption(f"Using custom JD — {jd.describe()}")
        run_shre(temp_input, labeled_path, out_csv, jd=jd)
        st.success("Ranking Complete!")

        # Prefer the detailed view (exposes semantic / behavioral / anomaly
        # signals); fall back to the canonical submission if absent.
        detailed_csv = os.path.join(os.path.dirname(out_csv) or ".", "submission_detailed.csv")
        if os.path.exists(detailed_csv):
            st.subheader("Top candidates with enriched signals")
            ddf = pd.read_csv(detailed_csv)
            st.dataframe(ddf)
        else:
            st.dataframe(pd.read_csv(out_csv))

        # Provide download link for the canonical submission.
        with open(out_csv, "rb") as f:
            st.download_button(
                label="Download submission.csv",
                data=f,
                file_name="submission.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Pipeline failed: {str(e)}")
