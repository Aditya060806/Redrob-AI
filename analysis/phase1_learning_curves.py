"""
PHASE 1: LEARNING CURVE ANALYSIS
Demonstrates whether 498 labeled samples is sufficient and identifies
whether the model is underfitting or overfitting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import json
import os

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)


def create_learning_curves(X, y, feature_names, output_dir='analysis_results'):
    """
    Train models on increasing amounts of training data.

    Measures:
    - Training set accuracy
    - Validation set accuracy
    - Training set Macro F1
    - Validation set Macro F1

    Sample sizes: 20%, 40%, 60%, 80%, 100% of training data
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*100)
    print("PHASE 1: LEARNING CURVE ANALYSIS")
    print("="*100)

    # Feature selection (done once on full data)
    print("\n[1.1] Feature Selection...")
    selector = SelectKBest(f_classif, k=max(30, int(0.8 * X.shape[1])))
    X_selected = selector.fit_transform(X, y)
    selected_indices = selector.get_support(indices=True)

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    # Define sample sizes (percentages of training data)
    sample_percentages = [20, 40, 60, 80, 100]
    results = {
        'sample_size_percent': [],
        'num_samples': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'train_f1': [],
        'val_f1': [],
    }

    print("[1.2] Training models on increasing dataset sizes...\n")

    # Use stratified k-fold for validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for pct in sample_percentages:
        print(f"Training on {pct}% of data...")

        # Sample data
        n_samples = max(10, int(len(y) * pct / 100))
        indices = np.random.RandomState(42).choice(len(y), n_samples, replace=False)
        X_sample = X_scaled[indices]
        y_sample = y[indices]

        # Metrics across folds
        train_accs = []
        val_accs = []
        train_f1s = []
        val_f1s = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_sample, y_sample)):
            X_train, X_val = X_sample[train_idx], X_sample[val_idx]
            y_train, y_val = y_sample[train_idx], y_sample[val_idx]

            # Apply SMOTE only to training set
            smote = SMOTE(k_neighbors=min(3, len(np.unique(y_train))-1),
                         random_state=42, sampling_strategy='not majority')
            try:
                X_train_aug, y_train_aug = smote.fit_resample(X_train, y_train)
            except:
                X_train_aug, y_train_aug = X_train, y_train

            # Train ensemble
            xgb_model = xgb.XGBClassifier(
                n_estimators=150, max_depth=6, learning_rate=0.02,
                subsample=0.8, colsample_bytree=0.8, objective='multi:softprob',
                num_class=4, random_state=42, verbose=0, eval_metric='mlogloss'
            )

            lgb_model = lgb.LGBMClassifier(
                n_estimators=150, max_depth=7, learning_rate=0.02,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbose=-1
            )

            cb_model = CatBoostClassifier(
                iterations=150, max_depth=7, learning_rate=0.02,
                subsample=0.8, bootstrap_type='Bernoulli', random_state=42, verbose=False
            )

            ensemble = VotingClassifier(
                estimators=[('xgb', xgb_model), ('lgb', lgb_model), ('cb', cb_model)],
                voting='soft'
            )

            ensemble.fit(X_train_aug, y_train_aug)

            # Metrics on training set
            y_train_pred = ensemble.predict(X_train_aug)
            train_acc = accuracy_score(y_train_aug, y_train_pred)
            train_f1 = f1_score(y_train_aug, y_train_pred, average='macro', zero_division=0)

            # Metrics on validation set
            y_val_pred = ensemble.predict(X_val)
            val_acc = accuracy_score(y_val, y_val_pred)
            val_f1 = f1_score(y_val, y_val_pred, average='macro', zero_division=0)

            train_accs.append(train_acc)
            val_accs.append(val_acc)
            train_f1s.append(train_f1)
            val_f1s.append(val_f1)

        # Average across folds
        results['sample_size_percent'].append(pct)
        results['num_samples'].append(n_samples)
        results['train_accuracy'].append(np.mean(train_accs))
        results['val_accuracy'].append(np.mean(val_accs))
        results['train_f1'].append(np.mean(train_f1s))
        results['val_f1'].append(np.mean(val_f1s))

        print(f"  Train Acc: {np.mean(train_accs):.4f} | Val Acc: {np.mean(val_accs):.4f} | "
              f"Train F1: {np.mean(train_f1s):.4f} | Val F1: {np.mean(val_f1s):.4f}\n")

    # Create DataFrame
    df_results = pd.DataFrame(results)

    # Save table
    print("\n[1.3] Learning Curve Results Table:")
    print(df_results.to_string(index=False))

    df_results.to_csv(os.path.join(output_dir, 'learning_curves_table.csv'), index=False)

    # Plot learning curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    axes[0].plot(df_results['sample_size_percent'], df_results['train_accuracy'],
                marker='o', linewidth=2, markersize=8, label='Training Accuracy', color='#2E86AB')
    axes[0].plot(df_results['sample_size_percent'], df_results['val_accuracy'],
                marker='s', linewidth=2, markersize=8, label='Validation Accuracy', color='#A23B72')
    axes[0].axhline(y=df_results['val_accuracy'].iloc[-1], color='red', linestyle='--',
                   alpha=0.5, label='Plateau Level')
    axes[0].set_xlabel('Training Data (%)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_title('Learning Curve: Accuracy', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0.7, 0.95])

    # F1 plot
    axes[1].plot(df_results['sample_size_percent'], df_results['train_f1'],
                marker='o', linewidth=2, markersize=8, label='Training Macro F1', color='#2E86AB')
    axes[1].plot(df_results['sample_size_percent'], df_results['val_f1'],
                marker='s', linewidth=2, markersize=8, label='Validation Macro F1', color='#A23B72')
    axes[1].axhline(y=df_results['val_f1'].iloc[-1], color='red', linestyle='--',
                   alpha=0.5, label='Plateau Level')
    axes[1].set_xlabel('Training Data (%)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Macro F1 Score', fontsize=12, fontweight='bold')
    axes[1].set_title('Learning Curve: Macro F1', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0.65, 0.95])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase1_learning_curves.png'), dpi=300, bbox_inches='tight')
    print(f"\nOK: Plot saved: phase1_learning_curves.png")
    plt.close()

    # Analysis
    print("\n[1.4] Learning Curve Analysis:")

    # Calculate gaps and trends
    final_val_acc = df_results['val_accuracy'].iloc[-1]
    first_val_acc = df_results['val_accuracy'].iloc[0]
    gap = final_val_acc - first_val_acc

    final_train_acc = df_results['train_accuracy'].iloc[-1]
    overfit_gap = final_train_acc - final_val_acc

    # Check if curve is plateauing
    last_improvements = [
        df_results['val_accuracy'].iloc[-1] - df_results['val_accuracy'].iloc[-2],
        df_results['val_accuracy'].iloc[-2] - df_results['val_accuracy'].iloc[-3],
    ]
    avg_last_improvement = np.mean(last_improvements)

    analysis = {
        'total_improvement': gap,
        'overfit_gap': overfit_gap,
        'final_validation_accuracy': final_val_acc,
        'final_macro_f1': df_results['val_f1'].iloc[-1],
        'avg_last_improvement': avg_last_improvement,
        'is_plateauing': avg_last_improvement < 0.01,
        'is_underfitting': overfit_gap < 0.02,
        'is_overfitting': overfit_gap > 0.05,
    }

    print(f"\n  Total improvement (20% -> 100%):     {analysis['total_improvement']:.4f}")
    print(f"  Overfit gap (Train - Val):          {analysis['overfit_gap']:.4f}")
    print(f"  Final Validation Accuracy:          {analysis['final_validation_accuracy']:.4f}")
    print(f"  Final Macro F1:                     {analysis['final_macro_f1']:.4f}")
    print(f"  Avg improvement (80% -> 100%):       {analysis['avg_last_improvement']:.4f}")

    # Interpretation
    print("\n[1.5] Interpretation:")
    if analysis['is_plateauing']:
        print("  OK: CURVE IS PLATEAUING")
        print("    -> Additional data likely to provide minimal gains")
        print("    -> 498 labeled samples appears SUFFICIENT")
    else:
        print("  FAILED: CURVE IS STILL INCREASING")
        print("    -> More data would likely improve performance")
        print("    -> Current sample size may be limiting")

    if analysis['is_underfitting']:
        print("  OK: Model is NOT OVERFITTING (Train-Val gap < 2%)")
        print("    -> Model generalizes well")
        print("    -> Complexity level appears appropriate")
    elif analysis['is_overfitting']:
        print("  WARNING: Model shows OVERFITTING signs (Train-Val gap > 5%)")
        print("    -> Regularization could be increased")
        print("    -> Consider simpler features")
    else:
        print("  OK: Model shows BALANCED FIT (2% < Train-Val gap < 5%)")
        print("    -> Healthy balance between bias and variance")

    # Save analysis
    with open(os.path.join(output_dir, 'phase1_analysis.json'), 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"\nOK: Analysis saved to phase1_analysis.json")
    print("="*100 + "\n")

    return df_results, analysis
