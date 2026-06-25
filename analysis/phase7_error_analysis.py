"""
PHASE 7: ERROR ANALYSIS
Analyzes all misclassifications to understand model weaknesses.
Categorizes errors and identifies patterns in failures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import json
import os
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")


def error_analysis(X, y, feature_names, output_dir='analysis_results'):
    """
    Comprehensive error analysis on test set.

    Outputs:
    1. Misclassification confusion matrix
    2. Error categorization
    3. Per-class error rates
    4. Feature analysis of errors
    5. Difficulty assessment
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*100)
    print("PHASE 7: ERROR ANALYSIS")
    print("="*100)

    # Preprocessing and train/test split
    print("\n[7.1] Preprocessing and splitting data...")
    selector = SelectKBest(f_classif, k=max(30, int(0.8 * X.shape[1])))
    X_selected = selector.fit_transform(X, y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    # Use last 15% as test set
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.15, random_state=42, stratify=y
    )

    # Train on remaining data
    smote = SMOTE(k_neighbors=3, random_state=42, sampling_strategy='not majority')
    X_train_aug, y_train_aug = smote.fit_resample(X_train, y_train)

    # Train ensemble
    print("[7.2] Training ensemble on 85% of data...")

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

    ensemble = VotingClassifier(
        estimators=[('xgb', xgb_model), ('lgb', lgb_model), ('cb', cb_model)],
        voting='soft'
    )
    ensemble.fit(X_train_aug, y_train_aug)

    # Get predictions on test set
    print("[7.3] Evaluating on test set...")

    y_pred = ensemble.predict(X_test)
    y_proba = ensemble.predict_proba(X_test)
    confidence = np.max(y_proba, axis=1)

    # ===== ERROR CATEGORIZATION =====
    print("\n[7.4] Analyzing errors...")

    errors = y_test != y_pred
    num_errors = np.sum(errors)
    error_rate = num_errors / len(y_test)

    print(f"  Total Test Samples: {len(y_test)}")
    print(f"  Errors: {num_errors} ({error_rate:.2%})")

    # Build error dataframe
    error_df = pd.DataFrame({
        'true_class': y_test,
        'pred_class': y_pred,
        'confidence': confidence,
        'error': errors,
        'feature_sum': np.sum(np.abs(X_test), axis=1),  # Feature richness
    })

    # ===== CONFUSION DETAILS =====
    print("\n[7.5] Error Breakdown by Class Transition:")

    confusion_details = []

    for true_class in range(4):
        for pred_class in range(4):
            if true_class == pred_class:
                continue

            mask = (error_df['true_class'] == true_class) & (error_df['pred_class'] == pred_class)
            count = np.sum(mask)

            if count > 0:
                avg_confidence = error_df[mask]['confidence'].mean()
                avg_feature_sum = error_df[mask]['feature_sum'].mean()

                print(f"  Class {true_class} -> Class {pred_class}: {count} errors " +
                     f"(avg conf: {avg_confidence:.4f}, avg richness: {avg_feature_sum:.2f})")

                confusion_details.append({
                    'from_class': true_class,
                    'to_class': pred_class,
                    'num_errors': count,
                    'avg_confidence': avg_confidence,
                    'avg_feature_richness': avg_feature_sum,
                })

    confusion_df = pd.DataFrame(confusion_details)

    # ===== ERROR ANALYSIS BY CLASS =====
    print("\n[7.6] Per-Class Error Analysis:")

    class_stats = []

    for cls in range(4):
        mask = error_df['true_class'] == cls
        class_total = np.sum(mask)
        class_errors = np.sum((error_df['true_class'] == cls) & errors)
        class_error_rate = class_errors / class_total if class_total > 0 else 0

        avg_confidence_correct = error_df[(error_df['true_class'] == cls) & ~errors]['confidence'].mean()
        avg_confidence_incorrect = error_df[(error_df['true_class'] == cls) & errors]['confidence'].mean()

        print(f"  Class {cls}:")
        print(f"    Total Samples: {class_total}")
        print(f"    Errors: {class_errors} ({class_error_rate:.2%})")
        print(f"    Avg Confidence (Correct): {avg_confidence_correct:.4f}")
        print(f"    Avg Confidence (Incorrect): {avg_confidence_incorrect if not np.isnan(avg_confidence_incorrect) else 'N/A'}")

        class_stats.append({
            'class': cls,
            'total': class_total,
            'errors': class_errors,
            'error_rate': class_error_rate,
            'avg_confidence_correct': avg_confidence_correct,
        })

    class_stats_df = pd.DataFrame(class_stats)

    # ===== FEATURE ANALYSIS OF ERRORS =====
    print("\n[7.7] Feature Analysis of Misclassified Samples...")

    error_feature_richness = error_df[errors]['feature_sum'].mean()
    correct_feature_richness = error_df[~errors]['feature_sum'].mean()

    print(f"  Average Feature Richness (Correct): {correct_feature_richness:.2f}")
    print(f"  Average Feature Richness (Errors):  {error_feature_richness:.2f}")
    print(f"  Difference: {correct_feature_richness - error_feature_richness:.2f}")

    # ===== VISUALIZATION 1: Confusion Matrix (Errors Only) =====
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
               xticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'],
               yticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'],
               cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Class', fontsize=12, fontweight='bold')
    ax.set_title(f'Test Set Confusion Matrix (n={len(y_test)})', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase7_confusion_matrix.png'),
               dpi=300, bbox_inches='tight')
    print(f"\nOK: Saved: phase7_confusion_matrix.png")
    plt.close()

    # ===== VISUALIZATION 2: Error Rate by Class =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Error rates
    axes[0].bar(class_stats_df['class'], class_stats_df['error_rate'],
               color='#A23B72', alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0].set_xlabel('Class', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Error Rate', fontsize=11, fontweight='bold')
    axes[0].set_title('Error Rate by Class', fontsize=12, fontweight='bold')
    axes[0].set_ylim([0, max(class_stats_df['error_rate']) * 1.2])
    axes[0].grid(True, alpha=0.3, axis='y')

    # Confidence comparison
    x_pos = np.arange(len(class_stats_df))
    width = 0.35

    axes[1].bar(x_pos - width/2, class_stats_df['avg_confidence_correct'],
               width, label='Correct Predictions', color='#2E86AB', alpha=0.8)

    incorrect_conf = []
    for cls in range(4):
        mask = (error_df['true_class'] == cls) & errors
        if np.sum(mask) > 0:
            incorrect_conf.append(error_df[mask]['confidence'].mean())
        else:
            incorrect_conf.append(0)

    axes[1].bar(x_pos + width/2, incorrect_conf,
               width, label='Incorrect Predictions', color='#F18F01', alpha=0.8)

    axes[1].set_xlabel('Class', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Average Confidence', fontsize=11, fontweight='bold')
    axes[1].set_title('Prediction Confidence by Correctness', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(['Class 0', 'Class 1', 'Class 2', 'Class 3'])
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase7_error_rates.png'),
               dpi=300, bbox_inches='tight')
    print(f"OK: Saved: phase7_error_rates.png")
    plt.close()

    # ===== VISUALIZATION 3: Error Flow Sankey (simplified as grouped bar) =====
    if not confusion_df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))

        x_pos = np.arange(len(confusion_df))
        colors_map = {
            (0, 1): '#FF6B6B', (0, 2): '#FF8C8C', (0, 3): '#FFADAD',
            (1, 0): '#FFC300', (1, 2): '#FFD700', (1, 3): '#FFED4E',
            (2, 0): '#95E1D3', (2, 1): '#38A169', (2, 3): '#48BB78',
            (3, 0): '#6C5CE7', (3, 1): '#A29BFE', (3, 2): '#DFE6E9',
        }

        bars = ax.bar(x_pos, confusion_df['num_errors'],
                     color=[colors_map.get((row['from_class'], row['to_class']), '#999')
                           for _, row in confusion_df.iterrows()],
                     edgecolor='black', linewidth=1.5)

        # Labels
        labels = [f"C{row['from_class']}->C{row['to_class']}" for _, row in confusion_df.iterrows()]
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')

        ax.set_ylabel('Number of Errors', fontsize=11, fontweight='bold')
        ax.set_title('Misclassification Flow: From -> To Class', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'phase7_error_flow.png'),
                   dpi=300, bbox_inches='tight')
        print(f"OK: Saved: phase7_error_flow.png")
        plt.close()

    # ===== SAVE SUMMARY =====
    summary = {
        'test_set_size': len(y_test),
        'total_errors': int(num_errors),
        'overall_error_rate': float(error_rate),
        'overall_accuracy': float(1 - error_rate),
        'class_statistics': class_stats_df.to_dict(orient='records'),
        'error_transitions': confusion_df.to_dict(orient='records'),
        'feature_richness_analysis': {
            'correct_avg': float(correct_feature_richness),
            'incorrect_avg': float(error_feature_richness),
        },
    }

    with open(os.path.join(output_dir, 'phase7_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*100 + "\n")

    return error_df, class_stats_df
