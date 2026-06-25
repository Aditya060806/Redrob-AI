import streamlit as st
import pandas as pd
import sys
import os

# Ensure the src directory is in the path so we can import our pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.main import run_shre

st.title("Redrob Sandbox: Founding Senior AI Engineer Ranker")

# Sidebar with links and information
st.sidebar.title("Navigation & Resources")
st.sidebar.markdown(
    """
    ###  Resources
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rakesh-s-omen/india-runs-challenge/blob/main/Mywork/colab_reproduction.ipynb)
    
    * **GitHub Repository:** [india-runs-challenge](https://github.com/rakesh-s-omen/india-runs-challenge)
    * **Team Name:** RETRO
    
    ###  Test Data
    You can download the sample test file directly from our repository:
    * [sample_candidates.jsonl](https://raw.githubusercontent.com/rakesh-s-omen/india-runs-challenge/main/Mywork/data/sample_candidates.jsonl)
    """
)

st.markdown("""
This is the sandbox environment for our SHRE + CTAE candidate ranking engine.
Upload a sample `candidates.jsonl` file (max 1000 candidates recommended for speed) to see the rankings.
""")

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
        run_shre(temp_input, labeled_path, out_csv)
        st.success("Ranking Complete!")
        
        # Display the results
        df = pd.read_csv(out_csv)
        st.dataframe(df)
        
        # Provide download link
        with open(out_csv, "rb") as f:
            st.download_button(
                label="Download submission.csv",
                data=f,
                file_name="submission.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Pipeline failed: {str(e)}")
