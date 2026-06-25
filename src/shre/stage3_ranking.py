import os
import json
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif

def train_and_predict(labeled_data_path, feature_matrix, feature_names):
    """
    Trains an XGBoost classifier using labeled data and predicts on the full feature matrix.
    FIXED: Uses XGBClassifier (not Ranker) for proper multi-class classification.
    Includes cross-validation, class balancing, and comprehensive evaluation.
    """
    try:
        import xgboost as xgb
    except ImportError as e:
        print(f"Failed to import XGBoost. {e}")
        raise

    print("Loading labeled data...")
    if not os.path.exists(labeled_data_path):
        raise FileNotFoundError(f"Labeled data not found at {labeled_data_path}")

    with open(labeled_data_path, 'r', encoding='utf-8') as f:
        labeled = json.load(f)

    if len(labeled) < 10:
        raise ValueError(f"Need more labeled data. Only found {len(labeled)} samples.")

    from src.shre.stage2_features import FeatureEngineer
    fe = FeatureEngineer()
    labeled_features = fe.compute_features([item['raw_profile'] for item in labeled])

    # Extract training matrices
    X = np.array([list(fv.values()) for _, fv in labeled_features])
    y = np.array([item['relevance_score'] for item in labeled])

    # Clean data
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # SMOTE oversampling for minority classes
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    print(f"SMOTE: {len(y)} -> {len(y_resampled)} samples (balanced)")

    print(f"\n{'='*80}")
    print(f"TRAINING XGBoost CLASSIFIER")
    print(f"{'='*80}")
    print(f"Samples: {len(y)}")
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {sorted(set(y))}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Calculate class weights for reporting
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = {int(i): float(w) for i, w in enumerate(class_weights)}
    print(f"Class weights: {class_weight_dict}\n")

    # Cross-validation with stratification
    print("Running 5-Fold Stratified Cross-Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_metrics = {
        'accuracy': [],
        'precision_macro': [],
        'recall_macro': [],
        'f1_macro': []
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y), 1):
        X_fold_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_fold_train, y_val = y[train_idx], y[val_idx]

        # SMOTE per fold to avoid data leakage
        smote_fold = SMOTE(random_state=42, k_neighbors=3)
        X_train_sm, y_train_sm = smote_fold.fit_resample(X_fold_train, y_fold_train)

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.7,
            objective='multi:softprob',
            num_class=4,
            random_state=42,
            tree_method='hist',
            eval_metric='mlogloss'
        )

        model.fit(X_train_sm, y_train_sm, verbose=False)
        y_pred = model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_val, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)

        cv_metrics['accuracy'].append(acc)
        cv_metrics['precision_macro'].append(prec)
        cv_metrics['recall_macro'].append(rec)
        cv_metrics['f1_macro'].append(f1)

        print(f"  Fold {fold}: Acc={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | F1={f1:.4f}")

    print(f"\nCross-Validation Results (5-Fold):")
    for metric, scores in cv_metrics.items():
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"  {metric:20s}: {mean_score:.4f} (+-{std_score:.4f})")

    # Train final model on SMOTE-resampled data
    print(f"\nTraining final model on {len(X_resampled)} SMOTE-resampled samples...")
    final_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.7,
        objective='multi:softprob',
        num_class=4,
        random_state=42,
        tree_method='hist',
        eval_metric='mlogloss'
    )

    final_model.fit(X_resampled, y_resampled, verbose=False)

    # Get feature importance
    feature_importance = [float(f) for f in final_model.feature_importances_]
    top_features = sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True)[:15]

    print(f"\nTop 15 Feature Importances:")
    for i, (fname, fimportance) in enumerate(top_features, 1):
        print(f"  {i:2d}. {fname:35s}: {fimportance:.4f}")

    # Full dataset evaluation
    y_pred_full = final_model.predict(X_scaled)
    acc_full = accuracy_score(y, y_pred_full)

    print(f"\nFinal Model Performance (Full Dataset):")
    print(f"  Accuracy: {acc_full:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y, y_pred_full, zero_division=0))
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y, y_pred_full)
    print(cm)

    # Save the model and metadata
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    with open(os.path.join(models_dir, 'xgboost_model.pkl'), 'wb') as f:
        pickle.dump(final_model, f)

    with open(os.path.join(models_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    with open(os.path.join(models_dir, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f)

    # Save comprehensive metadata
    metadata = {
        'n_estimators': 200,
        'max_depth': 5,
        'learning_rate': 0.03,
        'samples_trained': len(y),
        'num_features': len(feature_names),
        'classes': [0, 1, 2, 3],
        'cv_accuracy_mean': float(np.mean(cv_metrics['accuracy'])),
        'cv_accuracy_std': float(np.std(cv_metrics['accuracy'])),
        'final_accuracy': float(acc_full),
        'feature_importance': dict(top_features),
        'class_weights': class_weight_dict
    }

    with open(os.path.join(models_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel saved to {models_dir}")
    print(f"{'='*80}\n")

    # Predict on full viable pool
    print("Predicting on full viable pool...")
    X_pred = np.array([list(fv.values()) for _, fv in feature_matrix])
    X_pred = np.nan_to_num(X_pred, nan=0.0, posinf=0.0, neginf=0.0)
    X_pred_scaled = scaler.transform(X_pred)

    # Get prediction probabilities for better ranking
    scores = final_model.predict_proba(X_pred_scaled)
    # Use weighted score: higher class * probability
    weighted_scores = np.sum(scores * np.array([0, 1, 2, 3]), axis=1)

    return weighted_scores
