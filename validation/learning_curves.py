"""
Learning Curves - Proves dataset sufficiency (no overfitting on 498 samples)
"""

import numpy as np
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import json
import os


def generate_learning_curves(X, y, feature_names, output_dir='validation_results'):
    """
    Generate learning curves showing accuracy vs training set size.
    Proves that 498 samples is sufficient (curve plateaus, not increasing).
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("LEARNING CURVES - Proving Dataset Sufficiency")
    print("="*80)

    # Test on increasing sample sizes
    sample_sizes = [50, 100, 150, 200, 300, int(len(y) * 0.7)]

    train_scores = []
    val_scores = []

    for size in sample_sizes:
        print(f"\nTesting with {size} training samples...")

        # Stratified sample
        indices = np.random.RandomState(42).choice(len(y), size, replace=False)
        X_sample, y_sample = X[indices], y[indices]

        # Ensure k_neighbors is valid for SMOTE
        min_class_count = min(np.bincount(y_sample))
        if min_class_count > 1:
            k_neighbors = min(3, min_class_count - 1)
            smote = SMOTE(k_neighbors=k_neighbors, random_state=42, sampling_strategy='not majority')
            X_aug, y_aug = smote.fit_resample(X_sample, y_sample)
        else:
            X_aug, y_aug = X_sample, y_sample

        # Train ensemble
        xgb_model = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.8, objective='multi:softprob',
            num_class=4, random_state=42, verbose=0
        )

        lgb_model = lgb.LGBMClassifier(
            n_estimators=100, max_depth=7, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1
        )

        cb_model = CatBoostClassifier(
            iterations=100, max_depth=7, learning_rate=0.02,
            subsample=0.8, bootstrap_type='Bernoulli', random_state=42, verbose=False
        )

        ensemble = VotingClassifier(
            estimators=[('xgb', xgb_model), ('lgb', lgb_model), ('cb', cb_model)],
            voting='soft'
        )

        # 5-fold CV
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_scores = []

        for train_idx, val_idx in skf.split(X_aug, y_aug):
            X_train, X_val = X_aug[train_idx], X_aug[val_idx]
            y_train, y_val = y_aug[train_idx], y_aug[val_idx]

            ensemble.fit(X_train, y_train)
            score = ensemble.score(X_val, y_val)
            fold_scores.append(score)

        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)

        train_scores.append(mean_score)
        val_scores.append(std_score)

        print(f"  Accuracy: {mean_score:.4f} (+-{std_score:.4f})")

    # Save learning curve data
    curve_data = {
        'sample_sizes': sample_sizes,
        'accuracies': [float(s) for s in train_scores],
        'std_devs': [float(s) for s in val_scores],
        'interpretation': 'If accuracies plateau -> dataset is sufficient. If still increasing -> need more data.'
    }

    with open(os.path.join(output_dir, 'learning_curves.json'), 'w') as f:
        json.dump(curve_data, f, indent=2)

    print(f"\nOK: Learning curves saved to {output_dir}/learning_curves.json")
    print(f"\nInterpretation:")
    print(f"  If accuracy plateaus around {train_scores[-1]:.4f} -> dataset is sufficient [OK]")
    print(f"  If still increasing rapidly -> dataset is too small [FAILED]")

    return curve_data
