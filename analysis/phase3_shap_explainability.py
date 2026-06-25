"""
PHASE 3: SHAP EXPLAINABILITY ANALYSIS
Provides human-interpretable explanations for model predictions using SHAP values.
Includes global summaries and individual candidate explanations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import shap
import json
import os
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")


def explain_with_shap(X, y, feature_names, output_dir='analysis_results'):
    """
    SHAP-based explainability analysis.

    Outputs:
    1. Global SHAP Summary Plot
    2. SHAP Feature Ranking
    3. SHAP Dependence Plots (top 5 features)
    4. Waterfall Plot for Top-Ranked Candidate
    5. Waterfall Plot for Lowest-Ranked Candidate
    6. Human-readable explanations
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*100)
    print("PHASE 3: SHAP EXPLAINABILITY")
    print("="*100)

    # Feature selection and preprocessing
    print("\n[3.1] Preprocessing data...")
    selector = SelectKBest(f_classif, k=max(30, int(0.8 * X.shape[1])))
    X_selected = selector.fit_transform(X, y)
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    # Apply SMOTE
    smote = SMOTE(k_neighbors=3, random_state=42, sampling_strategy='not majority')
    X_aug, y_aug = smote.fit_resample(X_scaled, y)

    # Train ensemble
    print("[3.2] Training ensemble model...")
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

    ensemble = VotingClassifier(
        estimators=[('xgb', xgb_model), ('lgb', lgb_model), ('cb', cb_model)],
        voting='soft'
    )
    # VotingClassifier over pre-fit estimators is not itself fitted.
    ensemble.fit(X_aug, y_aug)

    # ===== SHAP ANALYSIS =====
    print("[3.3] Computing SHAP values (this may take several minutes)...")

    # Use TreeExplainer for XGBoost
    explainer_xgb = shap.TreeExplainer(xgb_model)
    shap_values_xgb = explainer_xgb.shap_values(X_aug)

    # Get class 3 (highest relevance) SHAP values. Depending on the
    # shap/xgboost version this is either a list[n_classes] of (n,f) arrays or a
    # single (n, f, n_classes) ndarray. Handle both robustly.
    if isinstance(shap_values_xgb, list):
        shap_class3 = np.asarray(shap_values_xgb[-1])
    else:
        sv = np.asarray(shap_values_xgb)
        shap_class3 = sv[:, :, -1] if sv.ndim == 3 else sv

    print("[3.4] Creating SHAP summary plots...")

    # ===== VISUALIZATION 1: SHAP Summary Plot (Bar) =====
    fig, ax = plt.subplots(figsize=(12, 8))

    # Mean absolute SHAP values
    mean_abs_shap = np.abs(shap_class3).mean(axis=0)
    feature_importance_shap = pd.DataFrame({
        'Feature': selected_features,
        'Mean_Abs_SHAP': mean_abs_shap
    }).sort_values('Mean_Abs_SHAP', ascending=True).tail(15)

    feature_importance_shap.plot(kind='barh', x='Feature', y='Mean_Abs_SHAP',
                                ax=ax, legend=False, color='#2E86AB')
    ax.set_xlabel('Mean |SHAP value|', fontsize=12, fontweight='bold')
    ax.set_title('SHAP Feature Importance (Top 15 Features)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase3_shap_summary.png'), dpi=300, bbox_inches='tight')
    print(f"OK: Saved: phase3_shap_summary.png")
    plt.close()

    # ===== VISUALIZATION 2: SHAP Summary Plot (Density) =====
    # Create custom summary plot
    shap.summary_plot(shap_class3, X_aug, feature_names=selected_features,
                     plot_type="violin", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase3_shap_density.png'), dpi=300, bbox_inches='tight')
    print(f"OK: Saved: phase3_shap_density.png")
    plt.close()

    # ===== VISUALIZATION 3: SHAP Dependence Plots (Top 5 Features) =====
    top_5_indices = np.argsort(mean_abs_shap)[-5:]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, feat_idx in enumerate(top_5_indices):
        ax = axes[i]
        scatter = ax.scatter(X_aug[:, feat_idx], shap_class3[:, feat_idx],
                           c=y_aug, cmap='viridis', alpha=0.5, s=20)
        ax.set_xlabel(selected_features[feat_idx], fontsize=10, fontweight='bold')
        ax.set_ylabel('SHAP value', fontsize=10, fontweight='bold')
        ax.set_title(f'{selected_features[feat_idx]} vs SHAP', fontsize=10)
        ax.grid(True, alpha=0.3)

    # Hide unused subplot
    axes[-1].set_visible(False)

    plt.suptitle('SHAP Dependence Plots (Top 5 Features)', fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase3_shap_dependence.png'), dpi=300, bbox_inches='tight')
    print(f"OK: Saved: phase3_shap_dependence.png")
    plt.close()

    # ===== INDIVIDUAL CANDIDATE EXPLANATIONS =====
    print("\n[3.5] Generating individual candidate explanations...")

    # Get predictions
    y_pred = ensemble.predict(X_aug)
    y_proba = ensemble.predict_proba(X_aug)
    confidence = np.max(y_proba, axis=1)

    # Find best and worst ranked candidates
    best_idx = np.argmax(confidence)
    worst_idx = np.argmin(confidence)

    explanations = []

    for idx, label in [(best_idx, "BEST_RANKED"), (worst_idx, "WORST_RANKED")]:
        explanation = generate_candidate_explanation(
            idx, X_aug[idx], y_pred[idx], y_proba[idx],
            shap_class3[idx], selected_features, label
        )
        explanations.append(explanation)

        # Save waterfall plot
        shap_val = shap_class3[idx]
        # expected_value may be a scalar, list, or ndarray (one per class).
        _ev = explainer_xgb.expected_value
        base_value = float(np.atleast_1d(_ev)[-1])
        shap_explanation = shap.Explanation(
            values=shap_val,
            base_values=base_value,
            data=X_aug[idx],
            feature_names=selected_features
        )

        fig = plt.figure(figsize=(12, 6))
        shap.plots.waterfall(shap_explanation, show=False)
        plt.title(f'SHAP Waterfall: {label} Candidate (Predicted Class {y_pred[idx]})',
                 fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'phase3_waterfall_{label}.png'),
                   dpi=300, bbox_inches='tight')
        print(f"OK: Saved: phase3_waterfall_{label}.png")
        plt.close()

    # Save explanations as JSON
    with open(os.path.join(output_dir, 'phase3_candidate_explanations.json'), 'w') as f:
        json.dump(explanations, f, indent=2)

    # Print explanations
    print("\n[3.6] Individual Candidate Explanations:\n")
    for exp in explanations:
        print_candidate_explanation(exp)

    # ===== FEATURE INTERACTION ANALYSIS =====
    print("\n[3.7] Analyzing feature interactions...")

    # Get top contributing features for best candidate
    best_shap = shap_class3[best_idx]
    top_features_idx = np.argsort(np.abs(best_shap))[-5:]

    interaction_summary = {
        'best_candidate': {
            'predicted_class': int(y_pred[best_idx]),
            'confidence': float(confidence[best_idx]),
            'top_features': [
                {
                    'feature': selected_features[i],
                    'value': float(X_aug[best_idx, i]),
                    'shap_value': float(best_shap[i]),
                    'contribution': 'positive' if best_shap[i] > 0 else 'negative'
                }
                for i in top_features_idx[::-1]
            ]
        },
        'worst_candidate': {
            'predicted_class': int(y_pred[worst_idx]),
            'confidence': float(confidence[worst_idx]),
            'top_features': [
                {
                    'feature': selected_features[i],
                    'value': float(X_aug[worst_idx, i]),
                    'shap_value': float(shap_class3[worst_idx, i]),
                    'contribution': 'positive' if shap_class3[worst_idx, i] > 0 else 'negative'
                }
                for i in np.argsort(np.abs(shap_class3[worst_idx]))[-5:][::-1]
            ]
        }
    }

    with open(os.path.join(output_dir, 'phase3_feature_interactions.json'), 'w') as f:
        json.dump(interaction_summary, f, indent=2)

    print("="*100 + "\n")

    return explanations, interaction_summary


def generate_candidate_explanation(idx, features, pred_class, proba, shap_vals, feature_names, label):
    """Generate human-readable explanation for a candidate."""

    # Get top contributing features (both positive and negative)
    top_indices = np.argsort(np.abs(shap_vals))[-5:]

    positive_features = []
    negative_features = []

    for feat_idx in top_indices[::-1]:
        shap_val = shap_vals[feat_idx]
        if shap_val > 0:
            positive_features.append({
                'feature': feature_names[feat_idx],
                'shap_value': float(shap_val),
                'feature_value': float(features[feat_idx])
            })
        else:
            negative_features.append({
                'feature': feature_names[feat_idx],
                'shap_value': float(shap_val),
                'feature_value': float(features[feat_idx])
            })

    return {
        'label': label,
        'predicted_class': int(pred_class),
        'class_probabilities': [float(p) for p in proba],
        'confidence': float(np.max(proba)),
        'positive_factors': positive_features,
        'negative_factors': negative_features,
    }


def print_candidate_explanation(explanation):
    """Print human-readable candidate explanation."""

    label = explanation['label']
    pred_class = explanation['predicted_class']
    confidence = explanation['confidence']

    print(f"\n{'='*80}")
    print(f"CANDIDATE: {label} (Predicted Class {pred_class}, Confidence {confidence:.2%})")
    print(f"{'='*80}")

    if explanation['positive_factors']:
        print("\nOK: POSITIVE FACTORS (Increase Relevance Score):")
        for i, factor in enumerate(explanation['positive_factors'], 1):
            print(f"  {i}. {factor['feature']:50s} SHAP: +{factor['shap_value']:.4f}")
            print(f"     (Feature value: {factor['feature_value']:.4f})")

    if explanation['negative_factors']:
        print("\nFAILED: NEGATIVE FACTORS (Decrease Relevance Score):")
        for i, factor in enumerate(explanation['negative_factors'], 1):
            print(f"  {i}. {factor['feature']:50s} SHAP: {factor['shap_value']:.4f}")
            print(f"     (Feature value: {factor['feature_value']:.4f})")

    print()
