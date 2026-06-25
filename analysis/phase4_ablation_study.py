"""
PHASE 4: ABLATION STUDY
Evaluates individual models vs ensemble and different feature groups.
Quantifies the contribution of each component.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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


def run_ablation_study(X, y, feature_names, output_dir='analysis_results'):
    """
    Ablation Study with two components:
    A) Model Comparison: XGBoost, LightGBM, CatBoost, Ensemble
    B) Feature Group Analysis: Different feature combinations
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*100)
    print("PHASE 4: ABLATION STUDY")
    print("="*100)

    # Feature selection
    print("\n[4.1] Feature selection and preprocessing...")
    selector = SelectKBest(f_classif, k=max(30, int(0.8 * X.shape[1])))
    X_selected = selector.fit_transform(X, y)
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    # ===== PART A: MODEL COMPARISON =====
    print("\n[4.2] Part A: Model Comparison")
    print("Comparing individual models vs ensemble...\n")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Accumulate per-fold scores for every model, then average across folds.
    model_names = ['XGBoost', 'LightGBM', 'CatBoost', 'Ensemble']
    fold_scores = {m: {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1_Score': []}
                   for m in model_names}

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Apply SMOTE
        smote = SMOTE(k_neighbors=3, random_state=42, sampling_strategy='not majority')
        X_train_aug, y_train_aug = smote.fit_resample(X_train, y_train)

        # Train models
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

        xgb_model.fit(X_train_aug, y_train_aug)
        lgb_model.fit(X_train_aug, y_train_aug)
        cb_model.fit(X_train_aug, y_train_aug)

        # Predictions
        models = {
            'XGBoost': xgb_model,
            'LightGBM': lgb_model,
            'CatBoost': cb_model,
        }

        # Ensemble must be fit explicitly (wrapping pre-fit estimators is not enough)
        ensemble = VotingClassifier(
            estimators=[('xgb', xgb_model), ('lgb', lgb_model), ('cb', cb_model)],
            voting='soft'
        )
        ensemble.fit(X_train_aug, y_train_aug)
        models['Ensemble'] = ensemble

        for model_name, model in models.items():
            y_pred = model.predict(X_val)
            fold_scores[model_name]['Accuracy'].append(accuracy_score(y_val, y_pred))
            fold_scores[model_name]['Precision'].append(
                precision_score(y_val, y_pred, average='macro', zero_division=0))
            fold_scores[model_name]['Recall'].append(
                recall_score(y_val, y_pred, average='macro', zero_division=0))
            fold_scores[model_name]['F1_Score'].append(
                f1_score(y_val, y_pred, average='macro', zero_division=0))

    # Average across folds
    model_summary = pd.DataFrame({
        'Model': model_names,
        'Accuracy': [np.mean(fold_scores[m]['Accuracy']) for m in model_names],
        'Precision': [np.mean(fold_scores[m]['Precision']) for m in model_names],
        'Recall': [np.mean(fold_scores[m]['Recall']) for m in model_names],
        'F1_Score': [np.mean(fold_scores[m]['F1_Score']) for m in model_names],
    })
    model_summary.to_csv(os.path.join(output_dir, 'ablation_model_comparison.csv'), index=False)

    print("\n[4.3] Model Comparison Results:")
    print(model_summary.to_string(index=False))

    # ===== VISUALIZATION 1: Model Comparison =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        bars = ax.bar(model_summary['Model'], model_summary[metric], color=colors)

        # Highlight ensemble
        bars[-1].set_color('#06A77D')
        bars[-1].set_edgecolor('black')
        bars[-1].set_linewidth(2)

        ax.set_ylabel(metric, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        ax.set_ylim([0.7, 1.0])
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('Part A: Model Comparison (5-Fold CV)', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase4_model_comparison.png'),
               dpi=300, bbox_inches='tight')
    print(f"\nOK: Saved: phase4_model_comparison.png")
    plt.close()

    # ===== PART B: FEATURE GROUP ANALYSIS =====
    print("\n[4.4] Part B: Feature Group Analysis")
    print("Analyzing contribution of different feature groups...\n")

    # Define feature groups aligned with the four engineered families in the
    # FeatureEngineer (experience/trajectory, technical/skill+domain,
    # engagement/recruiter, and the explicit interaction "*_x_*" features).
    def _match(keys):
        return [f for f in selected_features if any(k in f.lower() for k in keys)]

    interaction = [f for f in selected_features if '_x_' in f.lower()]
    experience = [f for f in _match(['year', 'job', 'seniority', 'role', 'stint',
                                      'duration', 'trajectory', 'consulting', 'stable',
                                      'tenure', 'ideal_years']) if '_x_' not in f.lower()]
    technical = [f for f in _match(['skill', 'endorse', 'ml', 'backend', 'vector',
                                    'domain', 'github', 'degree', 'institution',
                                    'cs_background', 'assessment', 'production',
                                    'foundation', 'deploy', 'eval', 'specialized'])
                 if '_x_' not in f.lower()]
    engagement = [f for f in _match(['response', 'interview', 'open_to_work',
                                     'notice', 'completeness', 'recruiter',
                                     'is_open']) if '_x_' not in f.lower()]

    feature_groups = {
        'experience': experience,
        'technical': technical,
        'engagement': engagement,
        'interaction': interaction,
    }

    # Ensure all features are accounted for (anything unmatched -> 'other')
    all_in_groups = set()
    for group_features in feature_groups.values():
        all_in_groups.update(group_features)

    remaining_features = [f for f in selected_features if f not in all_in_groups]
    if remaining_features:
        feature_groups['other'] = remaining_features

    feature_group_results = {
        'Feature_Group': [],
        'Num_Features': [],
        'Accuracy': [],
        'F1_Score': [],
    }

    print("Feature Group Breakdown:")
    for group_name, features in feature_groups.items():
        print(f"  {group_name:20s}: {len(features):3d} features")
        feature_group_results['Feature_Group'].append(group_name)
        feature_group_results['Num_Features'].append(len(features))

    # Test each feature group + all features
    test_groups = list(feature_groups.keys()) + ['all']

    for group_to_test in test_groups:
        accs = []
        f1s = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
            if group_to_test == 'all':
                X_train_group = X_scaled[train_idx]
                X_val_group = X_scaled[val_idx]
            else:
                group_indices = [i for i, f in enumerate(selected_features)
                               if f in feature_groups[group_to_test]]
                if not group_indices:
                    continue
                X_train_group = X_scaled[train_idx][:, group_indices]
                X_val_group = X_scaled[val_idx][:, group_indices]

            y_train, y_val = y[train_idx], y[val_idx]

            # Apply SMOTE
            smote = SMOTE(k_neighbors=3, random_state=42, sampling_strategy='not majority')
            X_train_aug, y_train_aug = smote.fit_resample(X_train_group, y_train)

            # Train ensemble with this feature group
            xgb_m = xgb.XGBClassifier(
                n_estimators=150, max_depth=6, learning_rate=0.02,
                subsample=0.8, colsample_bytree=0.8, objective='multi:softprob',
                num_class=4, random_state=42, verbose=0, eval_metric='mlogloss'
            )

            lgb_m = lgb.LGBMClassifier(
                n_estimators=150, max_depth=7, learning_rate=0.02,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbose=-1
            )

            cb_m = CatBoostClassifier(
                iterations=150, max_depth=7, learning_rate=0.02,
                subsample=0.8, bootstrap_type='Bernoulli', random_state=42, verbose=False
            )

            xgb_m.fit(X_train_aug, y_train_aug)
            lgb_m.fit(X_train_aug, y_train_aug)
            cb_m.fit(X_train_aug, y_train_aug)

            ensemble_group = VotingClassifier(
                estimators=[('xgb', xgb_m), ('lgb', lgb_m), ('cb', cb_m)],
                voting='soft'
            )
            ensemble_group.fit(X_train_aug, y_train_aug)

            y_pred = ensemble_group.predict(X_val_group)
            accs.append(accuracy_score(y_val, y_pred))
            f1s.append(f1_score(y_val, y_pred, average='macro', zero_division=0))

        if accs:  # Only if we had valid folds
            if group_to_test != 'all':
                feature_group_results['Accuracy'].append(np.mean(accs))
                feature_group_results['F1_Score'].append(np.mean(f1s))
            else:
                # Insert at beginning for "all" features
                feature_group_results['Feature_Group'].insert(0, 'All Features')
                feature_group_results['Num_Features'].insert(0, len(selected_features))
                feature_group_results['Accuracy'].insert(0, np.mean(accs))
                feature_group_results['F1_Score'].insert(0, np.mean(f1s))

    feature_group_df = pd.DataFrame(feature_group_results)

    print("\n[4.5] Feature Group Analysis Results:")
    print(feature_group_df.to_string(index=False))

    feature_group_df.to_csv(os.path.join(output_dir, 'ablation_feature_groups.csv'), index=False)

    # ===== VISUALIZATION 2: Feature Group Impact =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].barh(feature_group_df['Feature_Group'], feature_group_df['Accuracy'],
                color='#2E86AB', alpha=0.8)
    axes[0].set_xlabel('Accuracy', fontsize=11, fontweight='bold')
    axes[0].set_title('Feature Group Impact on Accuracy', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    axes[0].set_xlim([0.7, 0.95])

    # F1 Score
    axes[1].barh(feature_group_df['Feature_Group'], feature_group_df['F1_Score'],
                color='#A23B72', alpha=0.8)
    axes[1].set_xlabel('Macro F1 Score', fontsize=11, fontweight='bold')
    axes[1].set_title('Feature Group Impact on F1 Score', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    axes[1].set_xlim([0.6, 0.95])

    plt.suptitle('Part B: Feature Group Analysis', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase4_feature_groups.png'),
               dpi=300, bbox_inches='tight')
    print(f"\nOK: Saved: phase4_feature_groups.png")
    plt.close()

    # ===== ABLATION SUMMARY =====
    print("\n[4.6] Ablation Study Summary:")

    ensemble_acc = model_summary[model_summary['Model'] == 'Ensemble']['Accuracy'].values[0]
    best_individual = model_summary[model_summary['Model'] != 'Ensemble']['Accuracy'].max()
    ensemble_gain = ensemble_acc - best_individual

    print(f"\n  Ensemble Accuracy:          {ensemble_acc:.4f}")
    print(f"  Best Individual Model:      {best_individual:.4f}")
    print(f"  Ensemble Gain:              +{ensemble_gain:.4f} ({ensemble_gain*100:.2f}%)")

    best_group = feature_group_df.iloc[0]
    print(f"\n  Best Feature Group:         {best_group['Feature_Group']} (Acc: {best_group['Accuracy']:.4f})")

    worst_group = feature_group_df.iloc[-1] if len(feature_group_df) > 1 else best_group
    print(f"  Worst Individual Group:     {worst_group['Feature_Group']} (Acc: {worst_group['Accuracy']:.4f})")

    # Save ablation summary
    _indiv = model_summary[model_summary['Model'] != 'Ensemble']
    best_individual_name = _indiv.loc[_indiv['Accuracy'].idxmax(), 'Model']
    summary = {
        'ensemble_improvement': float(ensemble_gain),
        'ensemble_accuracy': float(ensemble_acc),
        'best_individual_model': str(best_individual_name),
        'best_individual_accuracy': float(best_individual),
        'best_feature_group': str(best_group['Feature_Group']),
        'model_comparison': model_summary.to_dict(orient='records'),
        'feature_groups_analysis': feature_group_df.to_dict(orient='records'),
    }

    with open(os.path.join(output_dir, 'phase4_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*100 + "\n")

    return model_summary, feature_group_df
