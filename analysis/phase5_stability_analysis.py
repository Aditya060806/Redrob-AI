"""
PHASE 5: STABILITY ANALYSIS
Tests model robustness using repeated stratified k-fold cross-validation.
10 repetitions x 5 folds = 50 independent evaluations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import json
import os
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")


def stability_analysis(X, y, feature_names, output_dir='analysis_results'):
    """
    Repeated Stratified K-Fold Cross-Validation: 10 repetitions x 5 folds.

    Outputs:
    1. Distribution statistics (mean, std, min, max)
    2. Distribution plots
    3. Box plots
    4. Statistical summary
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*100)
    print("PHASE 5: STABILITY ANALYSIS")
    print("="*100)

    # Feature selection and preprocessing
    print("\n[5.1] Preprocessing data...")
    selector = SelectKBest(f_classif, k=max(30, int(0.8 * X.shape[1])))
    X_selected = selector.fit_transform(X, y)
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    # Repeated Stratified K-Fold
    print("[5.2] Running 10 repetitions x 5 folds (50 total evaluations)...\n")

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

    metrics_history = {
        'fold': [],
        'repetition': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_macro': [],
    }

    fold_count = 0

    for rep_idx, (train_idx, val_idx) in enumerate(rskf.split(X_scaled, y)):
        fold_in_rep = rep_idx % 5  # Which fold within this repetition
        rep_num = rep_idx // 5  # Which repetition

        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Apply SMOTE
        smote = SMOTE(k_neighbors=3, random_state=42, sampling_strategy='not majority')
        X_train_aug, y_train_aug = smote.fit_resample(X_train, y_train)

        # Train ensemble
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

        # Fit the ensemble directly (it fits the three base estimators itself);
        # no need to fit them separately first.
        ensemble = VotingClassifier(
            estimators=[('xgb', xgb_model), ('lgb', lgb_model), ('cb', cb_model)],
            voting='soft'
        )
        ensemble.fit(X_train_aug, y_train_aug)

        y_pred = ensemble.predict(X_val)

        # Metrics
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_val, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)

        metrics_history['fold'].append(fold_in_rep + 1)
        metrics_history['repetition'].append(rep_num + 1)
        metrics_history['accuracy'].append(acc)
        metrics_history['precision'].append(prec)
        metrics_history['recall'].append(rec)
        metrics_history['f1_macro'].append(f1)

        if (rep_idx + 1) % 10 == 0:
            print(f"  Completed {rep_idx + 1}/50 evaluations...")

    print("\n[5.3] Computing statistics...\n")

    df_metrics = pd.DataFrame(metrics_history)

    # Summary statistics
    summary_stats = {
        'accuracy': {
            'mean': df_metrics['accuracy'].mean(),
            'std': df_metrics['accuracy'].std(),
            'min': df_metrics['accuracy'].min(),
            'max': df_metrics['accuracy'].max(),
            'median': df_metrics['accuracy'].median(),
            'q25': df_metrics['accuracy'].quantile(0.25),
            'q75': df_metrics['accuracy'].quantile(0.75),
        },
        'precision': {
            'mean': df_metrics['precision'].mean(),
            'std': df_metrics['precision'].std(),
            'min': df_metrics['precision'].min(),
            'max': df_metrics['precision'].max(),
            'median': df_metrics['precision'].median(),
            'q25': df_metrics['precision'].quantile(0.25),
            'q75': df_metrics['precision'].quantile(0.75),
        },
        'recall': {
            'mean': df_metrics['recall'].mean(),
            'std': df_metrics['recall'].std(),
            'min': df_metrics['recall'].min(),
            'max': df_metrics['recall'].max(),
            'median': df_metrics['recall'].median(),
            'q25': df_metrics['recall'].quantile(0.25),
            'q75': df_metrics['recall'].quantile(0.75),
        },
        'f1_macro': {
            'mean': df_metrics['f1_macro'].mean(),
            'std': df_metrics['f1_macro'].std(),
            'min': df_metrics['f1_macro'].min(),
            'max': df_metrics['f1_macro'].max(),
            'median': df_metrics['f1_macro'].median(),
            'q25': df_metrics['f1_macro'].quantile(0.25),
            'q75': df_metrics['f1_macro'].quantile(0.75),
        }
    }

    # Print summary
    print("STABILITY SUMMARY (50 Evaluations: 10 reps x 5 folds):")
    print("="*80)

    for metric_name, stats in summary_stats.items():
        print(f"\n{metric_name.upper()}:")
        print(f"  Mean:        {stats['mean']:.4f}")
        print(f"  Std Dev:     {stats['std']:.4f}")
        print(f"  Range:       [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"  Median:      {stats['median']:.4f}")
        print(f"  IQR:         [{stats['q25']:.4f}, {stats['q75']:.4f}]")
        print(f"  CV (%)       {(stats['std']/stats['mean']*100):.2f}%")

    # ===== VISUALIZATION 1: Distribution Plots =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = ['accuracy', 'precision', 'recall', 'f1_macro']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']

    for idx, (metric, color) in enumerate(zip(metrics, colors)):
        ax = axes[idx // 2, idx % 2]

        # Histogram
        ax.hist(df_metrics[metric], bins=15, alpha=0.7, color=color, edgecolor='black')

        # Add mean and std lines
        mean = df_metrics[metric].mean()
        std = df_metrics[metric].std()

        ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.4f}')
        ax.axvline(mean - std, color='orange', linestyle=':', linewidth=2, label=f'+-1 Std: {std:.4f}')
        ax.axvline(mean + std, color='orange', linestyle=':', linewidth=2)

        ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title(f'{metric.replace("_", " ").title()} Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Stability Analysis: 50 Independent Evaluations', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase5_distributions.png'), dpi=300, bbox_inches='tight')
    print(f"\nOK: Saved: phase5_distributions.png")
    plt.close()

    # ===== VISUALIZATION 2: Box Plots =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, (metric, color) in enumerate(zip(metrics, colors)):
        ax = axes[idx // 2, idx % 2]

        bp = ax.boxplot([df_metrics[metric]], widths=0.5, patch_artist=True,
                        boxprops=dict(facecolor=color, alpha=0.7),
                        medianprops=dict(color='red', linewidth=2),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))

        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.set_title(f'{metric.replace("_", " ").title()} Box Plot', fontsize=12, fontweight='bold')
        ax.set_xticklabels([''])
        ax.grid(True, alpha=0.3, axis='y')

        # Add statistics text
        mean = df_metrics[metric].mean()
        std = df_metrics[metric].std()
        ax.text(1.3, mean, f'mean={mean:.4f}\nstd={std:.4f}',
               fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Stability Analysis: Box Plot View', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase5_boxplots.png'), dpi=300, bbox_inches='tight')
    print(f"OK: Saved: phase5_boxplots.png")
    plt.close()

    # ===== VISUALIZATION 3: Time Series View =====
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(df_metrics['accuracy'], marker='o', markersize=4, linestyle='-',
           alpha=0.6, color='#2E86AB', label='Accuracy')
    ax.plot(df_metrics['f1_macro'], marker='s', markersize=4, linestyle='-',
           alpha=0.6, color='#A23B72', label='Macro F1')

    # Add rolling average
    window = 5
    ax.plot(df_metrics['accuracy'].rolling(window).mean(), linestyle='--',
           linewidth=2, color='#2E86AB', label=f'Accuracy (rolling {window})', alpha=0.9)
    ax.plot(df_metrics['f1_macro'].rolling(window).mean(), linestyle='--',
           linewidth=2, color='#A23B72', label=f'F1 (rolling {window})', alpha=0.9)

    ax.set_xlabel('Fold Index (1-50)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Stability Analysis: Metric Progression Across 50 Evaluations',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.75, 0.95])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase5_timeseries.png'), dpi=300, bbox_inches='tight')
    print(f"OK: Saved: phase5_timeseries.png")
    plt.close()

    # ===== STABILITY ASSESSMENT =====
    print("\n[5.4] Stability Assessment:")

    cv_accuracy = (summary_stats['accuracy']['std'] / summary_stats['accuracy']['mean']) * 100
    cv_f1 = (summary_stats['f1_macro']['std'] / summary_stats['f1_macro']['mean']) * 100

    print(f"\n  Coefficient of Variation (CV):")
    print(f"    Accuracy: {cv_accuracy:.2f}%")
    print(f"    Macro F1: {cv_f1:.2f}%")

    if cv_accuracy < 2.0:
        stability = "EXCELLENT"
    elif cv_accuracy < 3.5:
        stability = "GOOD"
    elif cv_accuracy < 5.0:
        stability = "ACCEPTABLE"
    else:
        stability = "VARIABLE"

    print(f"\n  Model Stability: {stability}")

    # Save CSV
    df_metrics.to_csv(os.path.join(output_dir, 'stability_detailed_results.csv'), index=False)

    # Save summary as JSON
    summary_json = {
        'num_evaluations': 50,
        'num_repetitions': 10,
        'num_folds': 5,
        'accuracy': {k: float(v) for k, v in summary_stats['accuracy'].items()},
        'precision': {k: float(v) for k, v in summary_stats['precision'].items()},
        'recall': {k: float(v) for k, v in summary_stats['recall'].items()},
        'f1_macro': {k: float(v) for k, v in summary_stats['f1_macro'].items()},
        'stability_assessment': stability,
        'coefficient_of_variation_accuracy': cv_accuracy,
        'coefficient_of_variation_f1': cv_f1,
    }

    with open(os.path.join(output_dir, 'phase5_summary.json'), 'w') as f:
        json.dump(summary_json, f, indent=2)

    print("\n" + "="*100 + "\n")

    return df_metrics, summary_stats
