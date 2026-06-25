"""
PHASE 6: HONEYPOT VALIDATION
Tests model robustness against synthetic fake/low-quality profiles.
Validates honeypot detection by the filtering stage.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import json
import os
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")


def honeypot_validation(X, y, feature_names, output_dir='analysis_results'):
    """
    Evaluates honeypot detection robustness.
    Creates synthetic "fake" profiles and tests if model correctly rejects them.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*100)
    print("PHASE 6: HONEYPOT VALIDATION")
    print("="*100)

    # Preprocessing
    print("\n[6.1] Preprocessing and model training...")
    selector = SelectKBest(f_classif, k=max(30, int(0.8 * X.shape[1])))
    X_selected = selector.fit_transform(X, y)
    selected_indices = selector.get_support(indices=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    smote = SMOTE(k_neighbors=3, random_state=42, sampling_strategy='not majority')
    X_aug, y_aug = smote.fit_resample(X_scaled, y)

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

    ensemble = VotingClassifier(
        estimators=[('xgb', xgb_model), ('lgb', lgb_model), ('cb', cb_model)],
        voting='soft'
    )
    ensemble.fit(X_aug, y_aug)

    # ===== CREATE SYNTHETIC HONEYPOTS =====
    print("\n[6.2] Generating synthetic honeypot profiles...")

    honeypot_samples = []

    # Type 1: Keyword Stuffing (all features maxed out)
    keyword_stuff = np.percentile(X_scaled, 95, axis=0)
    honeypot_samples.append(('Keyword_Stuffing', keyword_stuff))

    # Type 2: Flat Profile (all features same value)
    flat_profile = np.ones_like(X_scaled[0]) * np.median(X_scaled)
    honeypot_samples.append(('Flat_Profile', flat_profile))

    # Type 3: Random Noise (extreme values)
    noise_profile = np.random.uniform(-3, 3, X_scaled.shape[1])
    honeypot_samples.append(('Random_Noise', noise_profile))

    # Type 4: Minimal Profile (all zeros)
    minimal = np.zeros_like(X_scaled[0])
    honeypot_samples.append(('Minimal_Profile', minimal))

    # Type 5: Impossible Skills (skill duration > career duration)
    impossible = X_scaled[0].copy()
    impossible[::2] = np.percentile(X_scaled, 90, axis=0)[::2]  # High skills
    impossible[1::2] = np.percentile(X_scaled, 10, axis=0)[1::2]  # Low experience
    honeypot_samples.append(('Impossible_Skills', impossible))

    # Generate multiple copies for statistical significance
    num_copies_per_type = 50
    X_honeypot = []
    y_honeypot = []
    honeypot_types = []

    for honey_type, profile in honeypot_samples:
        for _ in range(num_copies_per_type):
            # Add small noise
            noisy_profile = profile + np.random.normal(0, 0.05, len(profile))
            X_honeypot.append(noisy_profile)
            y_honeypot.append(0)  # All honeypots should be Class 0 (not relevant)
            honeypot_types.append(honey_type)

    X_honeypot = np.array(X_honeypot)
    y_honeypot = np.array(y_honeypot)

    print(f"  Generated {len(X_honeypot)} synthetic honeypot profiles")
    print(f"  Types: {len(set(honeypot_types))}")

    # ===== EVALUATE ON HONEYPOTS =====
    print("\n[6.3] Evaluating model on synthetic honeypots...")

    y_pred_honeypot = ensemble.predict(X_honeypot)
    y_proba_honeypot = ensemble.predict_proba(X_honeypot)
    confidence_honeypot = np.max(y_proba_honeypot, axis=1)

    # Honeypot detection rate (correctly identified as Class 0)
    honeypot_correct = np.sum(y_pred_honeypot == 0)
    honeypot_detection_rate = honeypot_correct / len(y_honeypot)

    print(f"  Honeypot Detection Rate: {honeypot_detection_rate:.2%}")
    print(f"  Correctly Rejected: {honeypot_correct}/{len(y_honeypot)}")

    # ===== ANALYSIS BY HONEYPOT TYPE =====
    print("\n[6.4] Analysis by Honeypot Type:")

    honeypot_df = pd.DataFrame({
        'type': honeypot_types,
        'predicted_class': y_pred_honeypot,
        'confidence': confidence_honeypot,
    })

    type_stats = []

    for honey_type in set(honeypot_types):
        mask = honeypot_df['type'] == honey_type
        correct = np.sum(honeypot_df[mask]['predicted_class'] == 0)
        total = np.sum(mask)
        detection_rate = correct / total

        avg_confidence = honeypot_df[mask]['confidence'].mean()

        print(f"  {honey_type:25s}: {detection_rate:6.1%} detected (avg conf: {avg_confidence:.4f})")

        type_stats.append({
            'type': honey_type,
            'detection_rate': detection_rate,
            'num_samples': total,
            'avg_confidence': avg_confidence,
        })

    type_stats_df = pd.DataFrame(type_stats)

    # ===== VISUALIZATION 1: Honeypot Detection by Type =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Detection rates
    colors = plt.cm.RdYlGn(type_stats_df['detection_rate'].values)
    axes[0].barh(type_stats_df['type'], type_stats_df['detection_rate'],
                color=colors, edgecolor='black', linewidth=1.5)
    axes[0].set_xlabel('Detection Rate', fontsize=11, fontweight='bold')
    axes[0].set_title('Honeypot Detection Rate by Type', fontsize=12, fontweight='bold')
    axes[0].set_xlim([0, 1.0])
    axes[0].grid(True, alpha=0.3, axis='x')

    # Confidence distribution
    axes[1].bar(type_stats_df['type'], type_stats_df['avg_confidence'],
               color='#A23B72', alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('Average Confidence', fontsize=11, fontweight='bold')
    axes[1].set_title('Model Confidence on Honeypots', fontsize=12, fontweight='bold')
    axes[1].set_ylim([0, 1.0])
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Honeypot Validation Results', fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase6_honeypot_analysis.png'),
               dpi=300, bbox_inches='tight')
    print(f"\nOK: Saved: phase6_honeypot_analysis.png")
    plt.close()

    # ===== CONFUSION MATRIX =====
    # Force all four classes so the 4x4 matrix always matches the tick labels,
    # even when the model never predicts some classes for the honeypots.
    cm = confusion_matrix(y_honeypot, y_pred_honeypot, labels=[0, 1, 2, 3])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax,
               xticklabels=['Not Relevant', 'Somewhat', 'Relevant', 'Highly Relevant'],
               yticklabels=['Not Relevant', 'Somewhat', 'Relevant', 'Highly Relevant'])
    ax.set_xlabel('Predicted Class', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Class (All Honeypots = 0)', fontsize=11, fontweight='bold')
    ax.set_title('Confusion Matrix: Honeypot Detection', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase6_confusion_matrix.png'),
               dpi=300, bbox_inches='tight')
    print(f"OK: Saved: phase6_confusion_matrix.png")
    plt.close()

    # ===== SUMMARY REPORT =====
    print("\n[6.5] Honeypot Validation Summary:")

    # Calculate metrics
    tn = np.sum((y_honeypot == 0) & (y_pred_honeypot == 0))
    fp = np.sum((y_honeypot == 0) & (y_pred_honeypot != 0))

    honeypot_precision = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"\n  Overall Detection Rate:     {honeypot_detection_rate:.2%}")
    print(f"  Honeypot Precision:         {honeypot_precision:.2%}")
    print(f"  Worst Case Type:            {type_stats_df.loc[type_stats_df['detection_rate'].idxmin(), 'type']}")
    print(f"  Best Case Type:             {type_stats_df.loc[type_stats_df['detection_rate'].idxmax(), 'type']}")

    summary = {
        'num_honeypot_samples': len(X_honeypot),
        'num_honeypot_types': len(set(honeypot_types)),
        'overall_detection_rate': float(honeypot_detection_rate),
        'honeypot_precision': float(honeypot_precision),
        'type_statistics': type_stats_df.to_dict(orient='records'),
        'confusion_matrix': cm.tolist(),
    }

    with open(os.path.join(output_dir, 'phase6_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*100 + "\n")

    return honeypot_df, type_stats_df
