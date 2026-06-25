import json
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import xgboost as xgb

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from src.shre.stage2_features import FeatureEngineer

def test_ensemble():
    labeled_data_path = os.path.join(base_dir, 'labeling', 'combined_labels.json')
    with open(labeled_data_path, 'r', encoding='utf-8') as f:
        labeled = json.load(f)
        
    fe = FeatureEngineer()
    labeled_features = fe.compute_features([item['raw_profile'] for item in labeled])
    
    X = np.array([list(fv.values()) for _, fv in labeled_features])
    y = np.array([item['relevance_score'] for item in labeled])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_metrics = {
        'xgb': {'acc': [], 'f1': []},
        'rf': {'acc': [], 'f1': []},
        'hgb': {'acc': [], 'f1': []},
        'ensemble': {'acc': [], 'f1': []}
    }
    
    # Class weights for RF and HGB
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight('balanced', y)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y), 1):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        sw_train = sample_weights[train_idx]
        
        # 1. XGBoost
        clf_xgb = xgb.XGBClassifier(
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
        clf_xgb.fit(X_train, y_train, sample_weight=sw_train, verbose=False)
        probs_xgb = clf_xgb.predict_proba(X_val)
        
        # 2. Random Forest
        clf_rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        clf_rf.fit(X_train, y_train)
        probs_rf = clf_rf.predict_proba(X_val)
        
        # 3. HistGradientBoosting
        clf_hgb = HistGradientBoostingClassifier(
            max_iter=150,
            max_depth=5,
            learning_rate=0.05,
            class_weight='balanced',
            random_state=42
        )
        clf_hgb.fit(X_train, y_train)
        probs_hgb = clf_hgb.predict_proba(X_val)
        
        # Soft Voting Ensemble
        probs_ensemble = (probs_xgb + probs_rf + probs_hgb) / 3
        y_pred_ensemble = np.argmax(probs_ensemble, axis=1)
        
        # Individual Predictions
        y_pred_xgb = np.argmax(probs_xgb, axis=1)
        y_pred_rf = np.argmax(probs_rf, axis=1)
        y_pred_hgb = np.argmax(probs_hgb, axis=1)
        
        # Metrics
        for name, pred in zip(['xgb', 'rf', 'hgb', 'ensemble'], [y_pred_xgb, y_pred_rf, y_pred_hgb, y_pred_ensemble]):
            cv_metrics[name]['acc'].append(accuracy_score(y_val, pred))
            cv_metrics[name]['f1'].append(f1_score(y_val, pred, average='macro', zero_division=0))
            
    print("--- 5-Fold Cross-Validation Scores ---")
    for name in ['xgb', 'rf', 'hgb', 'ensemble']:
        mean_acc = np.mean(cv_metrics[name]['acc'])
        mean_f1 = np.mean(cv_metrics[name]['f1'])
        print(f"{name.upper():10s}: Mean Acc = {mean_acc:.4f} | Mean F1 Macro = {mean_f1:.4f}")

if __name__ == "__main__":
    test_ensemble()
