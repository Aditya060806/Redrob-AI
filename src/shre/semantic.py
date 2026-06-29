"""
Feature 1: Multi-Vector Semantic Layer.

Goes beyond keyword matching by embedding three *separate* views of each
candidate and matching them against three JD facets:

    candidate SKILLS view        <->  JD required_skills
    candidate TRAJECTORY view    <->  JD ideal_experience
    candidate FULL-PROFILE view  <->  JD role_mission

The three per-view cosine similarities are combined into a weighted fusion
score (config.SEMANTIC_FUSION_WEIGHTS). A FAISS index over the candidates'
full-profile vectors is queried with the JD role-mission anchor to produce a
fast retrieval ranking (-> semantic_rank_percentile).

The encoder is pluggable and degrades gracefully:
    1. sentence-transformers (all-MiniLM-L6-v2)   [preferred, true semantics]
    2. sklearn TfidfVectorizer                     [fallback, fit-once + saved]

So the layer always returns features; quality scales with what's installed.
"""

import os
import pickle

import numpy as np

from src.common.config import (
    SEMANTIC_MODEL_NAME,
    JD_FACETS,
    SEMANTIC_FUSION_WEIGHTS,
)

_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'models',
)
ENCODER_PATH = os.path.join(_MODELS_DIR, 'semantic_encoder.pkl')

# Ordered feature keys this layer contributes (kept fixed for vector alignment).
SEMANTIC_FEATURE_KEYS = (
    'semantic_skills_sim',
    'semantic_traj_sim',
    'semantic_jd_sim',
    'semantic_fusion_score',
    'semantic_rank_percentile',
)


def _l2_normalize(mat):
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


# ---------------------------------------------------------------------------
# Pluggable encoder
# ---------------------------------------------------------------------------
class SemanticEncoder:
    """Encodes text -> L2-normalized vectors. Transformer with TF-IDF fallback."""

    def __init__(self, backend=None, model=None):
        self.backend = backend          # 'transformer' | 'tfidf'
        self._model = model             # SentenceTransformer or TfidfVectorizer
        self._fitted = backend == 'transformer'

    # -- construction -------------------------------------------------------
    @classmethod
    def build(cls, corpus=None):
        """Build the best available encoder, fitting TF-IDF on `corpus` if used."""
        # Try the real transformer first.
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(SEMANTIC_MODEL_NAME)
            print(f"[semantic] Using transformer encoder: {SEMANTIC_MODEL_NAME}")
            return cls(backend='transformer', model=model)
        except Exception as e:  # noqa: BLE001 - any failure -> fallback
            print(f"[semantic] Transformer unavailable ({type(e).__name__}); "
                  f"falling back to TF-IDF encoder.")

        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(
            max_features=4096, ngram_range=(1, 2),
            stop_words='english', sublinear_tf=True,
        )
        enc = cls(backend='tfidf', model=vec)
        if corpus:
            enc.fit(corpus)
        return enc

    def fit(self, corpus):
        """Fit the TF-IDF vocabulary (no-op for the transformer backend)."""
        if self.backend == 'tfidf' and not self._fitted:
            safe = [t if isinstance(t, str) and t.strip() else 'na' for t in corpus]
            self._model.fit(safe)
            self._fitted = True
        return self

    # -- encoding -----------------------------------------------------------
    def encode(self, texts):
        """Encode a list of strings -> (n, dim) L2-normalized float32 array."""
        texts = [t if isinstance(t, str) and t.strip() else 'na' for t in texts]
        if self.backend == 'transformer':
            vecs = self._model.encode(texts, show_progress_bar=False,
                                      convert_to_numpy=True, normalize_embeddings=False)
        else:
            if not self._fitted:
                raise RuntimeError("TF-IDF encoder used before fit().")
            vecs = self._model.transform(texts).toarray()
        return _l2_normalize(vecs)

    # -- persistence --------------------------------------------------------
    def save(self, path=ENCODER_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {'backend': self.backend}
        # Transformer weights are reloaded by name; only TF-IDF vocab is pickled.
        payload['model'] = self._model if self.backend == 'tfidf' else None
        with open(path, 'wb') as f:
            pickle.dump(payload, f)
        return path

    @classmethod
    def load(cls, path=ENCODER_PATH):
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'rb') as f:
                payload = pickle.load(f)
        except Exception:
            return None
        if payload['backend'] == 'transformer':
            try:
                from sentence_transformers import SentenceTransformer
                return cls(backend='transformer',
                           model=SentenceTransformer(SEMANTIC_MODEL_NAME))
            except Exception:
                return None
        enc = cls(backend='tfidf', model=payload['model'])
        enc._fitted = True
        return enc


# ---------------------------------------------------------------------------
# Multi-vector scorer
# ---------------------------------------------------------------------------
class MultiVectorSemanticScorer:
    """Builds candidate views, scores semantic fit, and FAISS-ranks the pool."""

    def __init__(self, encoder=None):
        self.encoder = encoder
        self._anchors = None  # (skills_anchor, traj_anchor, mission_anchor)

    # -- text views ---------------------------------------------------------
    @staticmethod
    def skills_view(cand):
        skills = cand.get('skills', []) or []
        parts = []
        for s in skills:
            name = s.get('name', '')
            prof = s.get('proficiency', '')
            if name:
                parts.append(f"{name} ({prof})" if prof else name)
        return ', '.join(parts) or 'na'

    @staticmethod
    def trajectory_view(cand):
        profile = cand.get('profile', {})
        career = cand.get('career_history', []) or []
        parts = [profile.get('summary', '') or '']
        for j in career:
            title = j.get('title', '')
            desc = j.get('description', '')
            parts.append(f"{title}. {desc}".strip())
        return ' '.join(p for p in parts if p) or 'na'

    @classmethod
    def full_view(cls, cand):
        profile = cand.get('profile', {})
        head = ' '.join(str(profile.get(k, '')) for k in
                        ('headline', 'current_title', 'current_industry'))
        return f"{head} {cls.trajectory_view(cand)} {cls.skills_view(cand)}".strip()

    # -- main entry ---------------------------------------------------------
    def ensure_encoder(self, candidates):
        """Make sure we have a fitted encoder (load, else build on this corpus)."""
        if self.encoder is None:
            self.encoder = SemanticEncoder.load()
        if self.encoder is None:
            corpus = [self.full_view(c) for c in candidates]
            corpus += list(JD_FACETS.values())
            self.encoder = SemanticEncoder.build(corpus=corpus)
            self.encoder.save()
        return self.encoder

    def _anchor_vecs(self):
        if self._anchors is None:
            anchors = self.encoder.encode([
                JD_FACETS['required_skills'],
                JD_FACETS['ideal_experience'],
                JD_FACETS['role_mission'],
            ])
            self._anchors = anchors
        return self._anchors

    def score_batch(self, candidates):
        """Return a list (aligned with `candidates`) of semantic-feature dicts."""
        if not candidates:
            return []
        self.ensure_encoder(candidates)

        skills_anchor, traj_anchor, mission_anchor = self._anchor_vecs()

        skills_vecs = self.encoder.encode([self.skills_view(c) for c in candidates])
        traj_vecs = self.encoder.encode([self.trajectory_view(c) for c in candidates])
        full_vecs = self.encoder.encode([self.full_view(c) for c in candidates])

        # Cosine == dot product on L2-normalized vectors. Map [-1,1] -> [0,1].
        skills_sim = (skills_vecs @ skills_anchor.T).ravel()
        traj_sim = (traj_vecs @ traj_anchor.T).ravel()
        jd_sim = (full_vecs @ mission_anchor.T).ravel()
        skills_sim = (skills_sim + 1.0) / 2.0
        traj_sim = (traj_sim + 1.0) / 2.0
        jd_sim = (jd_sim + 1.0) / 2.0

        w = SEMANTIC_FUSION_WEIGHTS
        fusion = w['skills'] * skills_sim + w['trajectory'] * traj_sim + w['jd_match'] * jd_sim

        rank_pct = self._faiss_rank_percentile(full_vecs, mission_anchor)

        out = []
        for i in range(len(candidates)):
            out.append({
                'semantic_skills_sim': float(skills_sim[i]),
                'semantic_traj_sim': float(traj_sim[i]),
                'semantic_jd_sim': float(jd_sim[i]),
                'semantic_fusion_score': float(fusion[i]),
                'semantic_rank_percentile': float(rank_pct[i]),
            })
        return out

    # -- FAISS retrieval ----------------------------------------------------
    def _faiss_rank_percentile(self, full_vecs, mission_anchor):
        """Rank candidates by similarity to the JD mission; return percentile."""
        n = full_vecs.shape[0]
        sims = (full_vecs @ mission_anchor.T).ravel()
        try:
            import faiss
            index = faiss.IndexFlatIP(full_vecs.shape[1])
            index.add(np.ascontiguousarray(full_vecs, dtype=np.float32))
            query = np.ascontiguousarray(mission_anchor, dtype=np.float32)
            _, idx = index.search(query, n)          # ranked candidate indices
            order = idx.ravel()
        except Exception:
            order = np.argsort(-sims)                  # fallback: plain argsort
        # rank 0 = most similar; convert to percentile (1.0 = best).
        pct = np.empty(n, dtype=np.float32)
        for rank, cand_idx in enumerate(order):
            pct[cand_idx] = 1.0 - (rank / max(1, n - 1))
        return pct
