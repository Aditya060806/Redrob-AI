"""
Ablation study for the "Opus 4.8" enhancements.

Isolates the contribution of each added feature block on the 498 labeled
profiles:

  * Classification quality  (5-fold macro-F1 / accuracy with an XGBoost proxy
    classifier) for feature configs:
        base (78)  ->  +anomaly+behavioral  ->  +semantic (full)
  * Ranking quality (5-fold NDCG@10/@100): ensemble-only ordering vs the fused
    LambdaMART ranking head.

Run:  py -3.13 analysis/ablation_enhanced.py
"""

import os
import sys
import json
import warnings

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, ndcg_score

from src.shre.stage2_features import FeatureEngineer
from src.common.config import LTR_PARAMS, LTR_FUSION_ENSEMBLE_WEIGHT

LABELED = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'labeling', 'combined_labels.json')


def _matrix(profiles, blocks):
    fe = FeatureEngineer()
    feats = fe.compute_features(profiles, enrich=bool(blocks), blocks=tuple(blocks))
    X = np.array([list(fv.values()) for _, fv in feats])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def classification_ablation(profiles, y):
    import xgboost as xgb

    configs = [
        ('base (78)', ()),
        ('+anomaly+behavioral', ('anomaly', 'behavioral')),
        ('+semantic (full)', ('anomaly', 'behavioral', 'semantic')),
    ]
    print("\n--- Classification ablation (5-fold, XGBoost proxy) ---")
    print(f"{'config':<24}{'features':>10}{'accuracy':>11}{'macro_f1':>11}")
    results = {}
    for name, blocks in configs:
        X = _matrix(profiles, blocks)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
        accs, f1s = [], []
        for tr, te in skf.split(X, y):
            clf = xgb.XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.85, colsample_bytree=0.85, objective='multi:softprob',
                num_class=4, tree_method='hist', random_state=42, eval_metric='mlogloss',
            )
            clf.fit(X[tr], y[tr])
            pred = clf.predict(X[te])
            accs.append(accuracy_score(y[te], pred))
            f1s.append(f1_score(y[te], pred, average='macro', zero_division=0))
        acc, f1 = float(np.mean(accs)), float(np.mean(f1s))
        results[name] = {'features': X.shape[1], 'accuracy': acc, 'macro_f1': f1}
        print(f"{name:<24}{X.shape[1]:>10}{acc:>11.4f}{f1:>11.4f}")
    return results


def ranking_ablation(profiles, y):
    import xgboost as xgb
    import pickle

    print("\n--- Ranking ablation (5-fold NDCG) ---")
    X = _matrix(profiles, ('anomaly', 'behavioral', 'semantic'))

    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    try:
        ensemble = pickle.load(open(os.path.join(models_dir, 'ensemble_model_validated.pkl'), 'rb'))
        scaler = pickle.load(open(os.path.join(models_dir, 'scaler_validated.pkl'), 'rb'))
        selector = pickle.load(open(os.path.join(models_dir, 'selector_validated.pkl'), 'rb'))
    except FileNotFoundError:
        print("  [skip] validated ensemble artifacts not found; run the pipeline first.")
        return {}

    Xs = scaler.transform(selector.transform(X))
    ens = np.sum(ensemble.predict_proba(Xs) * np.array([0, 1, 2, 3]), axis=1) / 3.0
    Xl = np.hstack([Xs, ens.reshape(-1, 1)])

    w = LTR_FUSION_ENSEMBLE_WEIGHT

    def mm(a):
        a = np.asarray(a, float)
        return (a - a.min()) / (a.max() - a.min() + 1e-9)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    rows = {'ensemble_only': ([], []), 'pure_ltr': ([], []), 'fused_final': ([], [])}
    for tr, te in skf.split(Xl, y):
        r = xgb.XGBRanker(**LTR_PARAMS)
        r.fit(Xl[tr], y[tr], group=[len(tr)])
        pred = r.predict(Xl[te])
        fused = w * mm(ens[te]) + (1 - w) * mm(pred)
        n = len(te)
        for key, sc in (('ensemble_only', ens[te]), ('pure_ltr', pred), ('fused_final', fused)):
            rows[key][0].append(ndcg_score([y[te]], [sc], k=min(10, n)))
            rows[key][1].append(ndcg_score([y[te]], [sc], k=min(100, n)))

    print(f"{'ranker':<18}{'NDCG@10':>10}{'NDCG@100':>11}")
    results = {}
    for key, (a10, a100) in rows.items():
        results[key] = {'ndcg_at_10': float(np.mean(a10)), 'ndcg_at_100': float(np.mean(a100))}
        print(f"{key:<18}{np.mean(a10):>10.4f}{np.mean(a100):>11.4f}")
    return results


def main():
    labeled = json.load(open(LABELED, encoding='utf-8'))
    profiles = [item['raw_profile'] for item in labeled]
    y = np.array([item['relevance_score'] for item in labeled])
    print(f"Loaded {len(y)} labeled profiles. Class dist: "
          f"{dict(zip(*np.unique(y, return_counts=True)))}")

    cls = classification_ablation(profiles, y)
    rnk = ranking_ablation(profiles, y)

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'analysis_results', 'ablation_enhanced.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({'classification': cls, 'ranking': rnk}, open(out, 'w'), indent=2)
    print(f"\nSaved ablation results -> {out}")


if __name__ == '__main__':
    main()
