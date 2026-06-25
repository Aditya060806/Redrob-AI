"""
PHASE 2: FEATURE IMPORTANCE ANALYSIS
Analyzes which features contribute most to model predictions for all 4 models.
Includes gain importance, permutation importance, correlation analysis, and redundancy detection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import VotingClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import json
import os
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")


def analyze_feature_importance(X, y, feature_names, output_dir='analysis_results'):
    """
    Comprehensive feature importance analysis for all models.

    Outputs:
    1. Top 20 Feature Importance Table (Gain-based)
    2. Gain Importance Plot
    3. Permutation Importance Plot
    4. Feature Correlation Heatmap
    5. Redundancy Detection Report
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*100)
    print("PHASE 2: FEATURE IMPORTANCE ANALYSIS")
    print("="*100)

    # Feature selection
    print("\n[2.1] Feature Selection...")
    selector = SelectKBest(f_classif, k=max(30, int(0.8 * X.shape[1])))
    X_selected = selector.fit_transform(X, y)
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    # Apply SMOTE
    print("[2.2] Applying SMOTE augmentation...")
    smote = SMOTE(k_neighbors=3, random_state=42, sampling_strategy='not majority')
    X_aug, y_aug = smote.fit_resample(X_scaled, y)

    # Train individual models
    print("[2.3] Training individual models...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.8, objective='multi:softprob',
        num_class=4, random_state=42, verbose=0, eval_metric='mlogloss'
    )

    lgb_model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=7, learning_rate=0.02,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1
    )

    cb_model = CatBoostClassifier(
        iterations=200, max_depth=7, learning_rate=0.02,
        subsample=0.8, bootstrap_type='Bernoulli', random_state=42, verbose=False
    )

    xgb_model.fit(X_aug, y_aug)
    lgb_model.fit(X_aug, y_aug)
    cb_model.fit(X_aug, y_aug)

    print("[2.4] Extracting feature importances...")

    # Extract gain-based importance
    xgb_importance = dict(zip(selected_features, xgb_model.feature_importances_))
    lgb_importance = dict(zip(selected_features, lgb_model.feature_importances_))
    cb_importance = dict(zip(selected_features, cb_model.feature_importances_))

    # Average importance across models
    avg_importance = {}
    for feat in selected_features:
        avg_importance[feat] = (xgb_importance.get(feat, 0) +
                               lgb_importance.get(feat, 0) +
                               cb_importance.get(feat, 0)) / 3

    # Top 20 features
    top_20 = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:20]

    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': [f[0] for f in top_20],
        'Avg_Importance': [f[1] for f in top_20],
        'XGB_Importance': [xgb_importance.get(f[0], 0) for f in top_20],
        'LGB_Importance': [lgb_importance.get(f[0], 0) for f in top_20],
        'CB_Importance': [cb_importance.get(f[0], 0) for f in top_20],
    })

    # Normalize importance scores
    importance_df['Normalized_Importance'] = (importance_df['Avg_Importance'] /
                                             importance_df['Avg_Importance'].sum())
    importance_df['Cumulative_Importance'] = importance_df['Normalized_Importance'].cumsum()

    print("\n[2.5] Top 20 Features by Average Importance:")
    print(importance_df[['Feature', 'Avg_Importance', 'Normalized_Importance',
                        'Cumulative_Importance']].to_string(index=False))

    # Save importance table
    importance_df.to_csv(os.path.join(output_dir, 'feature_importance_table.csv'), index=False)

    # ===== VISUALIZATION 1: Gain Importance Plot =====
    fig, ax = plt.subplots(figsize=(12, 8))

    y_pos = np.arange(len(importance_df))
    ax.barh(y_pos, importance_df['Avg_Importance'].values, color='#2E86AB', alpha=0.8)

    # Color gradient by cumulative importance
    for i, (idx, row) in enumerate(importance_df.iterrows()):
        if row['Cumulative_Importance'] <= 0.8:
            ax.get_children()[i].set_color('#2E86AB')
        else:
            ax.get_children()[i].set_color('#A9A9A9')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(importance_df['Feature'].values, fontsize=10)
    ax.set_xlabel('Average Importance Score', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 Features by Importance (Gain-Based)\nBlue: Top 80% cumulative importance',
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase2_gain_importance.png'), dpi=300, bbox_inches='tight')
    print(f"\nOK: Saved: phase2_gain_importance.png")
    plt.close()

    # ===== VISUALIZATION 2: Model Comparison =====
    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.arange(len(importance_df))
    width = 0.25

    ax.bar(x - width, importance_df['XGB_Importance'], width, label='XGBoost', color='#2E86AB')
    ax.bar(x, importance_df['LGB_Importance'], width, label='LightGBM', color='#A23B72')
    ax.bar(x + width, importance_df['CB_Importance'], width, label='CatBoost', color='#F18F01')

    ax.set_ylabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance Across Models (Top 20)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(importance_df['Feature'].values, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase2_model_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"OK: Saved: phase2_model_comparison.png")
    plt.close()

    # ===== PERMUTATION IMPORTANCE =====
    print("\n[2.6] Computing permutation importance (this may take a moment)...")

    # Use ensemble for permutation importance
    ensemble = VotingClassifier(
        estimators=[('xgb', xgb_model), ('lgb', lgb_model), ('cb', cb_model)],
        voting='soft'
    )

    # Compute on validation set. A VotingClassifier wrapping already-fit
    # estimators is itself NOT fitted, so it must be fit before use.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(skf.split(X_aug, y_aug))
    X_tr, y_tr = X_aug[train_idx], y_aug[train_idx]
    X_val, y_val = X_aug[val_idx], y_aug[val_idx]
    ensemble.fit(X_tr, y_tr)

    perm_importance = permutation_importance(
        ensemble, X_val, y_val, n_repeats=10, random_state=42, n_jobs=1
    )

    perm_df = pd.DataFrame({
        'Feature': selected_features,
        'Importance_Mean': perm_importance.importances_mean,
        'Importance_Std': perm_importance.importances_std,
    }).sort_values('Importance_Mean', ascending=False)

    perm_top20 = perm_df.head(20)

    print("\nTop 20 Features by Permutation Importance:")
    print(perm_top20[['Feature', 'Importance_Mean', 'Importance_Std']].to_string(index=False))

    perm_df.to_csv(os.path.join(output_dir, 'permutation_importance.csv'), index=False)

    # ===== VISUALIZATION 3: Permutation Importance Plot =====
    fig, ax = plt.subplots(figsize=(12, 8))

    y_pos = np.arange(len(perm_top20))
    ax.barh(y_pos, perm_top20['Importance_Mean'].values,
           xerr=perm_top20['Importance_Std'].values,
           color='#A23B72', alpha=0.8, capsize=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(perm_top20['Feature'].values, fontsize=10)
    ax.set_xlabel('Permutation Importance', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 Features by Permutation Importance (with std)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase2_permutation_importance.png'),
               dpi=300, bbox_inches='tight')
    print(f"\nOK: Saved: phase2_permutation_importance.png")
    plt.close()

    # ===== FEATURE CORRELATION HEATMAP =====
    print("\n[2.7] Computing feature correlations...")

    # Use top 20 features for correlation
    top_20_features = [f[0] for f in top_20]
    top_20_indices = [selected_features.index(f) for f in top_20_features]
    X_top20 = X_scaled[:, top_20_indices]

    correlation_matrix = np.corrcoef(X_top20.T)
    corr_df = pd.DataFrame(correlation_matrix,
                          index=top_20_features,
                          columns=top_20_features)

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    sns.heatmap(corr_df, mask=mask, annot=False, cmap='coolwarm', center=0,
               square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax,
               vmin=-1, vmax=1)
    ax.set_title('Feature Correlation Heatmap (Top 20 Features)',
                fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase2_correlation_heatmap.png'),
               dpi=300, bbox_inches='tight')
    print(f"OK: Saved: phase2_correlation_heatmap.png")
    plt.close()

    # ===== REDUNDANCY DETECTION =====
    print("\n[2.8] Analyzing feature redundancy...")

    redundancy_report = {
        'highly_correlated_pairs': [],
        'potential_redundant_features': [],
    }

    # Find highly correlated pairs (> 0.95)
    for i in range(len(corr_df)):
        for j in range(i+1, len(corr_df)):
            if abs(corr_df.iloc[i, j]) > 0.95:
                redundancy_report['highly_correlated_pairs'].append({
                    'feature1': corr_df.index[i],
                    'feature2': corr_df.columns[j],
                    'correlation': float(corr_df.iloc[i, j]),
                })

    # Identify features with low variance
    feature_var = np.var(X_scaled, axis=0)
    low_var_threshold = np.percentile(feature_var, 25)

    low_var_features = [selected_features[i] for i in range(len(selected_features))
                       if feature_var[i] < low_var_threshold]

    print(f"\nHighly Correlated Pairs (|corr| > 0.95): {len(redundancy_report['highly_correlated_pairs'])}")
    for pair in redundancy_report['highly_correlated_pairs']:
        print(f"  {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.4f}")

    print(f"\nLow Variance Features (bottom 25%): {len(low_var_features)}")
    for feat in low_var_features[:5]:
        print(f"  {feat}")

    # Save redundancy report
    with open(os.path.join(output_dir, 'redundancy_analysis.json'), 'w') as f:
        json.dump({
            'highly_correlated_pairs': redundancy_report['highly_correlated_pairs'],
            'low_variance_features_count': len(low_var_features),
            'low_variance_threshold': float(low_var_threshold),
        }, f, indent=2)

    # ===== SUMMARY STATISTICS =====
    print("\n[2.9] Feature Importance Summary:")
    print(f"\n  Top 5 Features by Gain-Based Importance:")
    for i, row in importance_df.head(5).iterrows():
        print(f"    {i+1}. {row['Feature']:40s} - {row['Normalized_Importance']:.4f}")

    print(f"\n  Top 5 Features by Permutation Importance:")
    for i, row in perm_top20.head(5).iterrows():
        print(f"    {i+1}. {row['Feature']:40s} - {row['Importance_Mean']:.6f}")

    print(f"\n  Cumulative Importance (80% threshold):")
    n_features_80 = (importance_df['Cumulative_Importance'] <= 0.80).sum()
    print(f"    {n_features_80} features account for 80% of importance")
    print(f"    That's {n_features_80/len(importance_df)*100:.1f}% of top 20 features")

    # Save analysis summary
    summary = {
        'total_selected_features': len(selected_features),
        'top_20_features': [f[0] for f in top_20],
        'top_5_features': list(importance_df['Feature'].head(5).values),
        'num_features_for_80_importance': int(n_features_80),
        'correlation_pairs_high': len(redundancy_report['highly_correlated_pairs']),
    }

    with open(os.path.join(output_dir, 'phase2_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*100)

    return importance_df, perm_top20, corr_df
