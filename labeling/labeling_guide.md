# Labeling Guide: Founding Senior AI Engineer

To train the SHRE (Staged Hybrid Ranking Engine) XGBoost model, we need 250 ground-truth examples. This guide ensures consistency when rating candidates on the 0-3 scale.

## The Role
**Founding Senior AI Engineer @ Redrob AI**
- **Target Exp**: 5-9 years total experience.
- **Core Needs**: 4-5 years Applied ML at *product companies* (not just consulting).
- **Critical Skills**: Vector search (Pinecone, Milvus), RAG, Python, Evaluation (NDCG, MRR).
- **Red flags**: Consulting-only careers (TCS, Wipro, etc.), LangChain keyword-stuffers (no real backend ML experience), pure researchers with no production code.

---

## The Rating Scale (0 to 3)

### 0 - Not Qualified (Disqualify)
**Use this for candidates who miss the absolute baseline:**
- < 3 years or > 15 years of experience.
- Consulting-only careers (e.g., 8 years at Infosys with no core product experience).
- Buzzword-stuffers (e.g., lists "RAG, LLM, Vector DB" but current role is a junior web developer).
- Complete lack of Python or ML engineering fundamentals.

### 1 - Borderline (Weak Fit)
**Use this for candidates who have some value but major gaps:**
- Good ML fundamentals but missing vector search/RAG experience entirely.
- Has the skills, but heavily skewed towards pure academia/research rather than production engineering.
- Good technical experience but very poor behavioral signals (e.g., 0% recruiter response rate, 120-day notice period).

### 2 - Strong (Good Fit)
**Use this for solid, hireable engineers:**
- 4+ years of applied ML at product companies.
- Has built or scaled recommendation systems, search engines, or complex ML pipelines.
- Solid Python and evaluation framework experience.
- *Why not a 3?* They may have a slight gap (e.g., missing specific Vector DB knowledge, or slightly junior at exactly 4 years), but they are highly capable.

### 3 - Perfect (Hire Immediately)
**Use this for the rare, exact matches to the JD:**
- 5-9 years of experience with a clear upward trajectory (Engineer -> Senior -> Lead).
- Explicit, production-scale experience building Embedding-based Retrieval or Vector Search systems.
- Strong behavioral signals (Highly responsive, strong GitHub score, open to work).
- Fits the "Founding" profile: a high-autonomy product engineer.

---

## How to start labeling:
1. Open your terminal in the `Mywork` folder.
2. Run the command: `python scratch/label_tool.py`
3. Read the candidate summary.
4. Press `0`, `1`, `2`, or `3` and hit Enter.
5. If you need a break, press `q` to save your progress and quit. You can resume exactly where you left off.
