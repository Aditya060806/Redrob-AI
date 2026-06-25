import json
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import xgboost as xgb

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from src.shre.stage2_features import FeatureEngineer

def tune():
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
    
    # Sample weights for balancing
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight('balanced', y)
    
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=4,
        random_state=42,
        tree_method='hist',
        eval_metric='mlogloss'
    )
    
    # Massive parameter grid to test every combination
    param_grid = {
        'max_depth': [4, 5, 6],
        'learning_rate': [0.03, 0.05, 0.08],
        'n_estimators': [100, 200, 300],
        'subsample': [0.7, 0.85],
        'colsample_bytree': [0.7, 0.85]
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(model, param_grid, cv=skf, scoring='f1_macro', n_jobs=-1, verbose=1)
    
    grid.fit(X_scaled, y, sample_weight=sample_weights)
    
    print("FINE TUNING COMPLETE")
    print("Best F1 Macro Score:", grid.best_score_)
    print("Best Parameters:")
    for k, v in grid.best_params_.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    tune()
