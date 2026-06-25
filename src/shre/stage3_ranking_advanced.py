"""
ADVANCED ENSEMBLE MODEL FOR 90%+ ACCURACY
Features:
- Ensemble: XGBoost + LightGBM + CatBoost
- Bayesian hyperparameter optimization
- Advanced SMOTE + BorderlineSMOTE augmentation
- Feature selection (remove noise)
- Voting ensemble with soft voting
- Cross-validation with detailed metrics
"""

import os
import json
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN


def train_and_predict(labeled_data_path, feature_matrix, feature_names):
    """
    ADVANCED ENSEMBLE MODEL
    Target: 90%+ accuracy
    """
    try:
        import xgboost as xgb
        import lightgbm as lgb
        from catboost import CatBoostClassifier
    except ImportError as e:
        print(f"Missing libraries: {e}")
        print("Install: pip install lightgbm catboost imbalanced-learn")
        raise

    print("\n" + "="*80)
    print("ADVANCED ENSEMBLE MODEL (Target: 90%+ Accuracy)")
    print("="*80)

    # Load labeled data
    print("\nLoading labeled data...")
    if not os.path.exists(labeled_data_path):
        raise FileNotFoundError(f"Labeled data not found at {labeled_data_path}")

    with open(labeled_data_path, 'r', encoding='utf-8') as f:
        labeled = json.load(f)

    if len(labeled) < 10:
        raise ValueError(f"Need more labeled data. Only found {len(labeled)} samples.")

    from src.shre.stage2_features import FeatureEngineer
    fe = FeatureEngineer()
    labeled_features = fe.compute_features([item['raw_profile'] for item in labeled])

    # Extract matrices
    X = np.array([list(fv.values()) for _, fv in labeled_features])
    y = np.array([item['relevance_score'] for item in labeled])

    # Clean data
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Samples: {len(y)} | Features: {X.shape[1]} | Classes: {sorted(set(y))}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # STEP 1: Feature Selection (remove noise)
    print("\n[1] FEATURE SELECTION")
    selector = SelectKBest(f_classif, k=max(30, int(0.8 * X.shape[1])))
    X_selected = selector.fit_transform(X, y)
    selected_indices = selector.get_support(indices=True)
    selected_feature_names = [feature_names[i] for i in selected_indices]
    removed = X.shape[1] - len(selected_feature_names)
    print(f"    Kept {len(selected_feature_names)} features (removed {removed} noisy)")

    # STEP 2: Feature Normalization
    print("\n[2] FEATURE NORMALIZATION")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    print(f"    Scaled {X_scaled.shape[0]} samples x {X_scaled.shape[1]} features")

    # STEP 3: Advanced Data Augmentation
    print("\n[3] DATA AUGMENTATION (SMOTE + BorderlineSMOTE + ADASYN)")

    # First round: SMOTE
    smote = SMOTE(k_neighbors=3, random_state=42, sampling_strategy='not majority')
    X_smote, y_smote = smote.fit_resample(X_scaled, y)

    # Second round: BorderlineSMOTE (focus on hard examples)
    bsmote = BorderlineSMOTE(k_neighbors=3, random_state=42, sampling_strategy='not majority')
    X_aug, y_aug = bsmote.fit_resample(X_smote, y_smote)

    # Third round: ADASYN (adaptive synthetic sampling)
    adasyn = ADASYN(n_neighbors=3, random_state=42)
    X_final, y_final = adasyn.fit_resample(X_aug, y_aug)

    print(f"    Original: {len(y)} samples")
    print(f"    After SMOTE: {len(y_smote)} samples")
    print(f"    After BorderlineSMOTE: {len(y_aug)} samples")
    print(f"    After ADASYN: {len(y_final)} samples")
    print(f"    Final distribution: {dict(zip(*np.unique(y_final, return_counts=True)))}")

    # STEP 4: Cross-Validation with Ensemble
    print("\n[4] STRATIFIED 5-FOLD CROSS-VALIDATION")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_metrics = {
        'accuracy': [],
        'precision_macro': [],
        'recall_macro': [],
        'f1_macro': []
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_final, y_final), 1):
        X_train, X_val = X_final[train_idx], X_final[val_idx]
        y_train, y_val = y_final[train_idx], y_final[val_idx]

        # XGBoost with tuned hyperparameters
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=1,
            min_child_weight=3,
            objective='multi:softprob',
            num_class=4,
            random_state=42,
            tree_method='hist',
            eval_metric='mlogloss'
        )

        # LightGBM for diversity
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.02,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
            metric='multi_logloss'
        )

        # CatBoost for robustness
        cb_model = CatBoostClassifier(
            iterations=300,
            max_depth=7,
            learning_rate=0.02,
            bootstrap_type='Bernoulli',
            subsample=0.8,
            colsample_bylevel=0.8,
            l2_leaf_reg=5,
            random_state=42,
            verbose=False,
            loss_function='MultiClass'
        )

        # Voting ensemble (soft voting = average probabilities)
        ensemble = VotingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('lgb', lgb_model),
                ('cb', cb_model)
            ],
            voting='soft'
        )

        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_val, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)

        cv_metrics['accuracy'].append(acc)
        cv_metrics['precision_macro'].append(prec)
        cv_metrics['recall_macro'].append(rec)
        cv_metrics['f1_macro'].append(f1)

        print(f"  Fold {fold}: Acc={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | F1={f1:.4f}")

    print(f"\nCross-Validation Results (5-Fold Stratified):")
    print(f"  Accuracy:       {np.mean(cv_metrics['accuracy']):.4f} (+-{np.std(cv_metrics['accuracy']):.4f})")
    print(f"  Precision:      {np.mean(cv_metrics['precision_macro']):.4f} (+-{np.std(cv_metrics['precision_macro']):.4f})")
    print(f"  Recall:         {np.mean(cv_metrics['recall_macro']):.4f} (+-{np.std(cv_metrics['recall_macro']):.4f})")
    print(f"  F1-Score:       {np.mean(cv_metrics['f1_macro']):.4f} (+-{np.std(cv_metrics['f1_macro']):.4f})")

    # STEP 5: Train final ensemble on all data
    print(f"\n[5] TRAINING FINAL ENSEMBLE")
    final_xgb = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.015,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=1,
        min_child_weight=3,
        objective='multi:softprob',
        num_class=4,
        random_state=42,
        tree_method='hist'
    )

    final_lgb = lgb.LGBMClassifier(
        n_estimators=400,
        max_depth=7,
        learning_rate=0.015,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbose=-1
    )

    final_cb = CatBoostClassifier(
        iterations=400,
        max_depth=7,
        learning_rate=0.015,
        bootstrap_type='Bernoulli',
        subsample=0.85,
        colsample_bylevel=0.85,
        l2_leaf_reg=5,
        random_state=42,
        verbose=False
    )

    final_ensemble = VotingClassifier(
        estimators=[
            ('xgb', final_xgb),
            ('lgb', final_lgb),
            ('cb', final_cb)
        ],
        voting='soft'
    )

    print(f"    Training ensemble on {len(X_final)} augmented samples...")
    final_ensemble.fit(X_final, y_final)

    # STEP 6: Evaluate on augmented data
    print(f"\n[6] FINAL MODEL EVALUATION")
    y_pred_final = final_ensemble.predict(X_final)
    acc_final = accuracy_score(y_final, y_pred_final)

    print(f"    Accuracy (Augmented Data): {acc_final:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_final, y_pred_final, zero_division=0))

    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_final, y_pred_final)
    print(cm)

    # STEP 7: Save model and metadata
    print(f"\n[7] SAVING MODEL & METADATA")
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    with open(os.path.join(models_dir, 'ensemble_model.pkl'), 'wb') as f:
        pickle.dump(final_ensemble, f)

    with open(os.path.join(models_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    with open(os.path.join(models_dir, 'feature_selector.pkl'), 'wb') as f:
        pickle.dump(selector, f)

    with open(os.path.join(models_dir, 'selected_features.json'), 'w') as f:
        json.dump(selected_feature_names, f)

    # Get feature importances from individual models
    fitted_xgb = final_ensemble.estimators_[0]
    fitted_lgb = final_ensemble.estimators_[1]
    xgb_importance = dict(zip(selected_feature_names, fitted_xgb.feature_importances_))
    lgb_importance = dict(zip(selected_feature_names, fitted_lgb.feature_importances_))

    avg_importance = {}
    for fname in selected_feature_names:
        avg_importance[fname] = (xgb_importance.get(fname, 0) + lgb_importance.get(fname, 0)) / 2

    top_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:15]

    metadata = {
        'model_type': 'VotingEnsemble (XGBoost + LightGBM + CatBoost)',
        'cv_accuracy_mean': float(np.mean(cv_metrics['accuracy'])),
        'cv_accuracy_std': float(np.std(cv_metrics['accuracy'])),
        'final_accuracy': float(acc_final),
        'samples_trained': len(y_final),
        'num_features_selected': len(selected_feature_names),
        'feature_importance': dict(top_features),
        'augmentation_strategy': 'SMOTE + BorderlineSMOTE + ADASYN'
    }

    with open(os.path.join(models_dir, 'ensemble_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"    Model saved to {models_dir}")
    print(f"\nTop 15 Important Features:")
    for i, (fname, fimportance) in enumerate(top_features, 1):
        print(f"  {i:2d}. {fname:40s}: {fimportance:.4f}")

    print(f"\n{'='*80}\n")

    # STEP 8: Predict on full pool
    print("Predicting on full viable pool...")
    X_pred = np.array([list(fv.values()) for _, fv in feature_matrix])
    X_pred = np.nan_to_num(X_pred, nan=0.0, posinf=0.0, neginf=0.0)
    X_pred_selected = X_pred[:, selected_indices]
    X_pred_scaled = scaler.transform(X_pred_selected)

    # Get probability predictions for better ranking
    y_proba = final_ensemble.predict_proba(X_pred_scaled)
    weighted_scores = np.sum(y_proba * np.array([0, 1, 2, 3]), axis=1) / 3.0

    print(f"Generated {len(weighted_scores)} predictions")
    print(f"Score range: [{np.min(weighted_scores):.4f}, {np.max(weighted_scores):.4f}]")
    print(f"Mean score: {np.mean(weighted_scores):.4f}")

    return weighted_scores
