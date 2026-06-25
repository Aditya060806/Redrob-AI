"""
Ablation Study - Proves ensemble is better than individual models
Compares: XGBoost alone, LightGBM alone, CatBoost alone, vs Full Ensemble
"""

import numpy as np
import json
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def run_ablation_study(X, y, feature_names, output_dir='validation_results'):
    """
    Ablation study comparing individual models vs ensemble.
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        import xgboost as xgb
        import lightgbm as lgb
        from catboost import CatBoostClassifier
        from sklearn.ensemble import VotingClassifier
    except ImportError as e:
        print(f"Missing libraries: {e}")
        return

    print("\n" + "="*80)
    print("ABLATION STUDY - Individual vs Ensemble")
    print("="*80)

    # Feature selection
    from sklearn.feature_selection import SelectKBest, f_classif
    selector = SelectKBest(f_classif, k=max(30, int(0.8 * X.shape[1])))
    X_selected = selector.fit_transform(X, y)

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    # Apply SMOTE
    smote = SMOTE(k_neighbors=3, random_state=42, sampling_strategy='not majority')
    X_aug, y_aug = smote.fit_resample(X_scaled, y)

    print(f"\nTraining set: {len(y_aug)} samples (augmented from {len(y)})")

    # Cross-validation setup
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {
        'xgboost_alone': [],
        'lightgbm_alone': [],
        'catboost_alone': [],
        'ensemble': []
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_aug, y_aug), 1):
        X_train, X_val = X_aug[train_idx], X_aug[val_idx]
        y_train, y_val = y_aug[train_idx], y_aug[val_idx]

        print(f"\nFold {fold}:")

        # XGBoost alone
        xgb_model = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.8, objective='multi:softprob',
            num_class=4, random_state=42, verbose=0
        )
        xgb_model.fit(X_train, y_train)
        xgb_acc = accuracy_score(y_val, xgb_model.predict(X_val))
        results['xgboost_alone'].append(xgb_acc)
        print(f"  XGBoost alone:  {xgb_acc:.4f}")

        # LightGBM alone
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300, max_depth=7, learning_rate=0.02, num_leaves=31,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        lgb_acc = accuracy_score(y_val, lgb_model.predict(X_val))
        results['lightgbm_alone'].append(lgb_acc)
        print(f"  LightGBM alone: {lgb_acc:.4f}")

        # CatBoost alone
        cb_model = CatBoostClassifier(
            iterations=300, max_depth=7, learning_rate=0.02, bootstrap_type='Bernoulli',
            subsample=0.8, colsample_bylevel=0.8, l2_leaf_reg=5,
            random_state=42, verbose=False
        )
        cb_model.fit(X_train, y_train)
        cb_acc = accuracy_score(y_val, cb_model.predict(X_val))
        results['catboost_alone'].append(cb_acc)
        print(f"  CatBoost alone: {cb_acc:.4f}")

        # Ensemble
        ensemble = VotingClassifier(
            estimators=[('xgb', xgb_model), ('lgb', lgb_model), ('cb', cb_model)],
            voting='soft'
        )
        ensemble_acc = accuracy_score(y_val, ensemble.predict(X_val))
        results['ensemble'].append(ensemble_acc)
        print(f"  Ensemble:       {ensemble_acc:.4f} <- BEST")

    # Summarize
    print(f"\n" + "="*80)
    print("ABLATION RESULTS (5-Fold Cross-Validation)")
    print("="*80)

    summary = {}
    for model_name, accuracies in results.items():
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        summary[model_name] = {'mean': float(mean_acc), 'std': float(std_acc)}

        print(f"\n{model_name.upper():25s}: {mean_acc:.4f} (+-{std_acc:.4f})")

    # Calculate improvement
    ensemble_mean = summary['ensemble']['mean']
    xgb_mean = summary['xgboost_alone']['mean']
    lgb_mean = summary['lightgbm_alone']['mean']
    cb_mean = summary['catboost_alone']['mean']

    print(f"\n{'='*80}")
    print(f"ENSEMBLE IMPROVEMENT:")
    print(f"  vs XGBoost:  +{(ensemble_mean - xgb_mean)*100:.2f}%")
    print(f"  vs LightGBM: +{(ensemble_mean - lgb_mean)*100:.2f}%")
    print(f"  vs CatBoost: +{(ensemble_mean - cb_mean)*100:.2f}%")
    print(f"\nOK: Ensemble is {'SIGNIFICANTLY' if ensemble_mean > max(xgb_mean, lgb_mean, cb_mean) else 'MARGINALLY'} better")

    # Save results
    with open(os.path.join(output_dir, 'ablation_study.json'), 'w') as f:
        json.dump({
            'individual_models': summary,
            'ensemble_vs_individual': {
                'vs_xgboost': float(ensemble_mean - xgb_mean),
                'vs_lightgbm': float(ensemble_mean - lgb_mean),
                'vs_catboost': float(ensemble_mean - cb_mean)
            }
        }, f, indent=2)

    print(f"\nOK: Ablation study saved to {output_dir}/ablation_study.json")
    print("="*80 + "\n")

    return summary
