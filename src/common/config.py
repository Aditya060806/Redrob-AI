# Constants and configurations

# Hard limits from JD
MIN_YEARS_EXP = 3.0
MAX_YEARS_EXP = 15.0
TARGET_YEARS_MIN = 5.0
TARGET_YEARS_MAX = 9.0

# Skill pillars
SKILL_PILLARS = {
    'ml': ['machine learning', 'deep learning', 'neural network', 'model', 'nlp', 'llm'],
    'vector': ['vector', 'rag', 'embedding', 'retrieval', 'pinecone', 'milvus', 'qdrant', 'weaviate', 'faiss', 'similarity search'],
    'engineering': ['python', 'pytorch', 'tensorflow', 'sql', 'software engineering', 'architecture'],
    'eval': ['evaluation', 'metrics', 'benchmark', 'ab test', 'ndcg', 'mrr', 'map']
}

# Domains
DOMAINS = {
    'llm': ['llm', 'gpt', 'transformer', 'language model', 'nlp', 'bert'],
    'rag': ['rag', 'retrieval-augmented', 'context window', 'prompt'],
    'vector_db': ['vector db', 'embedding db', 'faiss', 'milvus', 'pinecone', 'weaviate', 'qdrant'],
    'foundation_models': ['fine-tune', 'instruction tuning', 'alignment', 'rlhf'],
    'deployment': ['model serving', 'inference optimization', 'quantization', 'deployment'],
    'eval': ['benchmark', 'evaluation', 'metrics', 'leaderboard', 'ndcg']
}

# Consulting / PFAW companies
CONSULTING_COMPANIES = ['tcs', 'tata consultancy', 'infosys', 'wipro', 'cognizant', 'accenture', 'capgemini', 'deloitte', 'ibm']

# ---------------------------------------------------------------------------
# ENHANCED MODEL CONFIG ("Opus 4.8" upgrade)
# Feature 1 (Semantic), 2 (LTR), 3 (Anomaly), 4 (Behavioral)
# ---------------------------------------------------------------------------

# --- Feature 1: Multi-Vector Semantic Layer ---
# Open-source transformer model (auto-falls back to TF-IDF if unavailable).
SEMANTIC_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

# Canonical Job Description, expressed as three facets that the semantic layer
# embeds and matches each candidate's three text "views" against.
# JD = "Founding Senior AI Engineer" (the role the 498 labels were judged for).
JD_FACETS = {
    # matched against the candidate SKILLS view
    'required_skills': (
        "Expert in large language models, retrieval-augmented generation (RAG), "
        "vector databases and embeddings (FAISS, Pinecone, Milvus, Weaviate, Qdrant), "
        "semantic search, transformers, prompt engineering, fine-tuning. "
        "Strong Python, PyTorch/TensorFlow, and production ML engineering: "
        "model serving, inference optimization, scalable data pipelines, MLOps."
    ),
    # matched against the candidate EXPERIENCE-TRAJECTORY view
    'ideal_experience': (
        "Five to nine years building and shipping machine learning systems in "
        "production at product companies and startups. Senior or lead engineer who "
        "has owned end-to-end ML/LLM systems from design to deployment, taken "
        "projects from zero to one, and grown in seniority and technical scope."
    ),
    # matched against the candidate FULL-PROFILE view
    'role_mission': (
        "Founding Senior AI Engineer to architect and build the core retrieval and "
        "LLM ranking platform: design multi-vector semantic search, deploy scalable "
        "low-latency RAG and ranking systems, and own the applied AI roadmap as an "
        "early, high-ownership member of the team."
    ),
}

# Weighted fusion of the three per-view cosine similarities -> fusion score.
SEMANTIC_FUSION_WEIGHTS = {
    'skills': 0.40,      # skills view vs required_skills
    'trajectory': 0.35,  # trajectory view vs ideal_experience
    'jd_match': 0.25,    # full-profile view vs role_mission
}

# --- Feature 3: Enhanced Anomaly / Honeypot Detection ---
ANOMALY_THRESHOLDS = {
    # a skill claimed longer than tenure (+ grace months) is impossible
    'skill_overflow_grace_months': 3,
    'skill_overflow_ratio': 1.05,
    # summed job tenure far exceeding career length => overlapping/fake timeline
    'timeline_overflow_ratio': 1.6,
    # months of overlap between consecutive jobs tolerated (advisory/part-time)
    'overlap_tolerance_months': 3,
    # endorsements wildly exceeding the candidate's network size
    'endorsement_to_connection_ratio': 3.0,
    # a profile inactive for this many days is "stale"
    'stale_days': 365,
    # anomaly_score at/above which a candidate is treated as synthetic & dropped
    'synthetic_score_cutoff': 0.6,
}

# --- Feature 2: Learning-to-Rank (LambdaMART / XGBoost rank:ndcg) ---
LTR_PARAMS = {
    'objective': 'rank:ndcg',
    'n_estimators': 200,
    'learning_rate': 0.03,
    'max_depth': 4,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'reg_lambda': 1.5,
    'min_child_weight': 2,
    'tree_method': 'hist',
    'random_state': 42,
}
# Final ranking = rank-fusion of the (near-ceiling) ensemble ordering and the
# list-optimized LTR score. Ensemble-dominant so we never regress below it on
# the labeled set, while the LTR refines ordering on the dense real pool.
LTR_FUSION_ENSEMBLE_WEIGHT = 0.6

# --- Feature 4: Behavioral Scoring (normalization references) ---
BEHAVIORAL_NORMS = {
    'profile_views_30d': 50.0,
    'search_appearance_30d': 80.0,
    'saved_by_recruiters_30d': 15.0,
    'applications_30d': 20.0,
    'connection_count': 1000.0,
    'endorsements_received': 100.0,
    'recent_active_days': 90.0,   # active within this window => full recency
}
