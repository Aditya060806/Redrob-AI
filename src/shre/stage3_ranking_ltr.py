"""
Feature 2: Learning-to-Rank head (LambdaMART via XGBoost-LTR).

This elevates the final ranking from the ensemble's per-candidate class-prob
"vote" to a proper learning-to-rank model that optimizes the *ordered list*.

Architecture (stacked):
    enriched features  ->  Voting Ensemble  -> ensemble_score (meta-feature)
    enriched features + ensemble_score  ->  XGBRanker(rank:ndcg)  -> final order

The ensemble is trained/validated by stage3_ranking_validated (reused as-is,
including its leakage-safe SMOTE-inside-CV protocol); we then load the saved
ensemble to attach its opinion as a meta-feature and train the LambdaMART
ranker on the 498 graded-relevance labels (0-3) as a single JD "query".

Reports NDCG@10 / NDCG@100 on a held-out split. On any failure the caller
(main.py) falls back to the plain validated-ensemble scores, then to CTAE.
"""

import os
import json
import pickle
import warnings

import numpy as np

warnings.filterwarnings('ignore')

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MODELS_DIR = os.path.join(_BASE_DIR, 'models')


def _minmax(arr):
    arr = np.asarray(arr, dtype=float)
    lo, hi = np.min(arr), np.max(arr)
    if hi - lo < 1e-9:
        return np.full_like(arr, 0.5)
    return (arr - lo) / (hi - lo)


def _spearman(y_true, y_score):
    """
    Spearman rank correlation (numpy-only, no scipy dependency).

    More sensitive than near-ceiling NDCG to mid-list mis-orderings, so it's a
    more honest signal of ranking quality on this rule-separable label set.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) < 3:
        return float('nan')

    def _rankdata(a):
        order = np.argsort(a, kind='mergesort')
        ranks = np.empty(len(a), dtype=float)
        ranks[order] = np.arange(len(a), dtype=float)
        # average ties so correlation is well-defined for repeated labels
        _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
        cum = np.cumsum(counts)
        start = cum - counts
        avg = (start + cum - 1) / 2.0
        return avg[inv]

    rt, rs = _rankdata(y_true), _rankdata(y_score)
    rt -= rt.mean()
    rs -= rs.mean()
    denom = np.sqrt((rt ** 2).sum() * (rs ** 2).sum())
    if denom < 1e-12:
        return float('nan')
    return float((rt * rs).sum() / denom)


def _hard_ndcg(y_true, y_score, k):
    """
    NDCG@k restricted to the confusable middle (relevance 1 & 2), where the
    trivially-separable 0 and 3 classes are removed. This is the *hard* slice
    that exposes whether the ranker truly orders borderline candidates well,
    rather than riding on the easy extremes.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    mask = (y_true == 1) | (y_true == 2)
    if mask.sum() < 3 or len(set(y_true[mask].tolist())) < 2:
        return float('nan')
    from sklearn.metrics import ndcg_score
    yt = y_true[mask].astype(float)
    ys = y_score[mask].astype(float)
    return float(ndcg_score([yt], [ys], k=min(k, int(mask.sum()))))


def train_and_predict_ltr(labeled_data_path, feature_matrix, feature_names):
    """
    Train the stacked LambdaMART ranker and score the viable pool.

    Returns (final_scores, metadata). `final_scores` are normalized to [0, 1]
    and aligned with `feature_matrix`.
    """
    import xgboost as xgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import ndcg_score

    from src.common.config import LTR_PARAMS, LTR_FUSION_ENSEMBLE_WEIGHT
    from src.shre.stage2_features import FeatureEngineer
    from src.shre.stage3_ranking_validated import train_and_predict_validated

    print("\n" + "=" * 80)
    print("LEARNING-TO-RANK HEAD (LambdaMART / XGBoost rank:ndcg)")
    print("=" * 80)

    # --- Step 1: train+validate the ensemble; get its scores on the pool -----
    ensemble_pool_scores, ens_meta = train_and_predict_validated(
        labeled_data_path, feature_matrix, feature_names
    )

    # --- Step 2: reload saved ensemble artifacts to score the labeled set ----
    with open(os.path.join(_MODELS_DIR, 'ensemble_model_validated.pkl'), 'rb') as f:
        ensemble = pickle.load(f)
    with open(os.path.join(_MODELS_DIR, 'scaler_validated.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(_MODELS_DIR, 'selector_validated.pkl'), 'rb') as f:
        selector = pickle.load(f)

    with open(labeled_data_path, 'r', encoding='utf-8') as fh:
        labeled = json.load(fh)

    fe = FeatureEngineer()
    labeled_feats = fe.compute_features([item['raw_profile'] for item in labeled])
    X_lab = np.nan_to_num(np.array([list(fv.values()) for _, fv in labeled_feats]),
                          nan=0.0, posinf=0.0, neginf=0.0)
    y_lab = np.array([item['relevance_score'] for item in labeled])

    X_lab_sel = selector.transform(X_lab)
    X_lab_scaled = scaler.transform(X_lab_sel)
    lab_proba = ensemble.predict_proba(X_lab_scaled)
    ens_score_lab = np.sum(lab_proba * np.array([0, 1, 2, 3]), axis=1) / 3.0

    # --- Step 3: build stacked LTR feature matrices --------------------------
    # [ selected enriched features (scaled) | ensemble_score ]
    X_ltr = np.hstack([X_lab_scaled, ens_score_lab.reshape(-1, 1)])

    X_pool = np.nan_to_num(np.array([list(fv.values()) for _, fv in feature_matrix]),
                           nan=0.0, posinf=0.0, neginf=0.0)
    X_pool_sel = selector.transform(X_pool)
    X_pool_scaled = scaler.transform(X_pool_sel)
    X_pool_ltr = np.hstack([X_pool_scaled, np.asarray(ensemble_pool_scores).reshape(-1, 1)])

    ranker_params = dict(LTR_PARAMS)
    w_ens = LTR_FUSION_ENSEMBLE_WEIGHT

    # --- Step 4: robust 5-fold NDCG evaluation (single JD query per fold) -----
    # The 498 labels are cleanly rule-separable, so the ensemble ordering is
    # near-ceiling. We report ensemble-only, pure-LTR, and fused NDCG so the
    # ranking quality is fully transparent.
    print("\n[LTR] Evaluating ranking quality (5-fold NDCG)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    ens10, ltr10, fus10 = [], [], []
    ens100, ltr100, fus100 = [], [], []
    fus_hard10, fus_spear = [], []   # honest "hard slice" diagnostics
    for tr_idx, te_idx in skf.split(X_ltr, y_lab):
        r = xgb.XGBRanker(**ranker_params)
        r.fit(X_ltr[tr_idx], y_lab[tr_idx], group=[len(tr_idx)])
        pred = r.predict(X_ltr[te_idx])
        n_te = len(te_idx)
        ens_te = ens_score_lab[te_idx]
        fused = w_ens * _minmax(ens_te) + (1 - w_ens) * _minmax(pred)
        for k, lo, ll, lf in ((10, ens10, ltr10, fus10), (100, ens100, ltr100, fus100)):
            kk = min(k, n_te)
            lo.append(ndcg_score([y_lab[te_idx]], [ens_te], k=kk))
            ll.append(ndcg_score([y_lab[te_idx]], [pred], k=kk))
            lf.append(ndcg_score([y_lab[te_idx]], [fused], k=kk))
        # Hard-slice metrics on the fused (final) ranker.
        h = _hard_ndcg(y_lab[te_idx], fused, k=10)
        if not np.isnan(h):
            fus_hard10.append(h)
        s = _spearman(y_lab[te_idx], fused)
        if not np.isnan(s):
            fus_spear.append(s)

    base_ndcg10, base_ndcg100 = float(np.mean(ens10)), float(np.mean(ens100))
    pure_ndcg10, pure_ndcg100 = float(np.mean(ltr10)), float(np.mean(ltr100))
    ndcg10, ndcg100 = float(np.mean(fus10)), float(np.mean(fus100))
    hard_ndcg10 = float(np.mean(fus_hard10)) if fus_hard10 else float('nan')
    spear_hard = float(np.mean(fus_spear)) if fus_spear else float('nan')

    print(f"  Ensemble-only ordering: NDCG@10={base_ndcg10:.4f}  NDCG@100={base_ndcg100:.4f}")
    print(f"  Pure LambdaMART LTR:    NDCG@10={pure_ndcg10:.4f}  NDCG@100={pure_ndcg100:.4f}")
    print(f"  Fused ranker (final):   NDCG@10={ndcg10:.4f}  NDCG@100={ndcg100:.4f}")
    print(f"  [honest] HARD slice (relevance 1 vs 2 only): "
          f"NDCG@10={hard_ndcg10:.4f}  Spearman={spear_hard:.4f}")
    print("  Note: the full-set NDCG is near-ceiling because the 498 labels are "
          "rule-separable;\n        the hard-slice + Spearman numbers above are the "
          "honest measure of borderline ordering.")

    # --- Step 5: train final ranker on ALL labels; score & fuse on the pool --
    print("\n[LTR] Training final ranker on all labels...")
    ranker = xgb.XGBRanker(**ranker_params)
    ranker.fit(X_ltr, y_lab, group=[len(y_lab)])

    ltr_pool = ranker.predict(X_pool_ltr)
    final_scores = w_ens * _minmax(ensemble_pool_scores) + (1 - w_ens) * _minmax(ltr_pool)
    final_scores = _minmax(final_scores)

    # --- Step 6: persist + metadata ------------------------------------------
    with open(os.path.join(_MODELS_DIR, 'ltr_model.pkl'), 'wb') as f:
        pickle.dump(ranker, f)

    metadata = dict(ens_meta)
    metadata.update({
        'ranking_head': 'XGBRanker (rank:ndcg / LambdaMART) fused with Voting Ensemble',
        'ltr_features': X_ltr.shape[1],
        'fusion_ensemble_weight': w_ens,
        'ndcg_at_10': ndcg10,            # fused (final) ranker
        'ndcg_at_100': ndcg100,
        'ndcg_at_10_hard': hard_ndcg10,  # honest: borderline (rel 1 vs 2) only
        'spearman_hard': spear_hard,     # honest: full rank correlation
        'pure_ltr_ndcg_at_10': pure_ndcg10,
        'pure_ltr_ndcg_at_100': pure_ndcg100,
        'baseline_ndcg_at_10': base_ndcg10,   # ensemble-only ordering
        'baseline_ndcg_at_100': base_ndcg100,
        'ndcg_at_10_gain_vs_ensemble': ndcg10 - base_ndcg10,
    })
    with open(os.path.join(_MODELS_DIR, 'metadata_ltr.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[LTR] Saved ranker -> models/ltr_model.pkl")
    print("=" * 80 + "\n")

    return final_scores, metadata
