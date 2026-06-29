"""
Inference-only scoring (PS requirement #4: lightning-fast at scale).

The training path (stage3_ranking_validated / _ltr) re-fits the full
XGB+LGB+CatBoost ensemble *and* the LambdaMART head on every call. That is fine
for refreshing the model, but it is the opposite of "lightning-fast" on a
100k-candidate pool.

This module loads the already-trained artifacts and scores an arbitrary pool
*without any retraining*:

    selected features (selector) -> scaled (scaler) -> ensemble.predict_proba
        -> ensemble_score meta-feature -> [scaled | ensemble_score]
        -> LambdaMART ranker -> fuse with ensemble ordering -> final scores

If the LTR head is missing it degrades to the ensemble ordering; if the
artifacts are missing or the feature width doesn't match it raises, and the
caller (main.run_shre) falls back to the full training path, then to CTAE.
"""

import os
import json
import pickle

import numpy as np

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MODELS_DIR = os.path.join(_BASE_DIR, 'models')

# Artifacts required for an inference-only run.
_ENSEMBLE = 'ensemble_model_validated.pkl'
_SCALER = 'scaler_validated.pkl'
_SELECTOR = 'selector_validated.pkl'
_LTR = 'ltr_model.pkl'
_META = 'metadata_validated.json'
_META_LTR = 'metadata_ltr.json'


def _minmax(arr):
    arr = np.asarray(arr, dtype=float)
    lo, hi = np.min(arr), np.max(arr)
    if hi - lo < 1e-9:
        return np.full_like(arr, 0.5)
    return (arr - lo) / (hi - lo)


def inference_artifacts_exist(models_dir=_MODELS_DIR):
    """True iff the minimum ensemble artifacts for inference are present."""
    return all(os.path.exists(os.path.join(models_dir, f))
               for f in (_ENSEMBLE, _SCALER, _SELECTOR))


def _load(models_dir, name):
    with open(os.path.join(models_dir, name), 'rb') as f:
        return pickle.load(f)


def predict_pool(feature_matrix, feature_names, models_dir=_MODELS_DIR):
    """
    Score `feature_matrix` with the saved models (no retraining).

    Returns (final_scores, metadata). Raises on any artifact / shape problem so
    the caller can fall back to the training path.
    """
    from src.common.config import LTR_FUSION_ENSEMBLE_WEIGHT

    if not inference_artifacts_exist(models_dir):
        raise FileNotFoundError("Saved ensemble artifacts not found for inference.")

    ensemble = _load(models_dir, _ENSEMBLE)
    scaler = _load(models_dir, _SCALER)
    selector = _load(models_dir, _SELECTOR)

    X_pool = np.nan_to_num(
        np.array([list(fv.values()) for _, fv in feature_matrix], dtype=float),
        nan=0.0, posinf=0.0, neginf=0.0,
    )

    # Guard against a feature-width mismatch (e.g. enrichment toggled off).
    expected = getattr(selector, 'n_features_in_', X_pool.shape[1])
    if X_pool.shape[1] != expected:
        raise ValueError(
            f"Feature width mismatch: pool has {X_pool.shape[1]} features, "
            f"saved selector expects {expected}. Retrain required."
        )

    X_sel = selector.transform(X_pool)
    X_scaled = scaler.transform(X_sel)

    proba = ensemble.predict_proba(X_scaled)
    n_classes = proba.shape[1]
    gain = np.arange(n_classes, dtype=float)
    ensemble_scores = np.sum(proba * gain, axis=1) / max(1.0, (n_classes - 1))

    # Load base metadata if available (for reporting parity with training).
    metadata = {}
    meta_path = os.path.join(models_dir, _META)
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    metadata['inference_mode'] = True
    metadata['scored_candidates'] = int(X_pool.shape[0])

    # Try to fuse with the LambdaMART head.
    ltr_path = os.path.join(models_dir, _LTR)
    if os.path.exists(ltr_path):
        try:
            ranker = _load(models_dir, _LTR)
            X_ltr = np.hstack([X_scaled, ensemble_scores.reshape(-1, 1)])
            ltr_pred = ranker.predict(X_ltr)
            w = LTR_FUSION_ENSEMBLE_WEIGHT
            final = w * _minmax(ensemble_scores) + (1 - w) * _minmax(ltr_pred)
            final = _minmax(final)
            metadata['ranking_head'] = 'LambdaMART fused with ensemble (inference)'
            # Surface saved NDCG from the last training run for transparency.
            ltr_meta_path = os.path.join(models_dir, _META_LTR)
            if os.path.exists(ltr_meta_path):
                with open(ltr_meta_path, 'r', encoding='utf-8') as f:
                    ltr_meta = json.load(f)
                for k in ('ndcg_at_10', 'ndcg_at_100',
                          'ndcg_at_10_hard', 'spearman_hard'):
                    if k in ltr_meta:
                        metadata[k] = ltr_meta[k]
            return final, metadata
        except Exception as e:  # noqa: BLE001 - degrade to ensemble ordering
            print(f"[inference] LTR head unavailable ({type(e).__name__}: {e}); "
                  f"using ensemble ordering.")

    metadata['ranking_head'] = 'Voting ensemble ordering (inference, no LTR)'
    return ensemble_scores, metadata
