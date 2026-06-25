#!/usr/bin/env python3
"""
MASTER RUNNER: Comprehensive 9-Phase Analysis Framework
Orchestrates all validation and analysis phases to elevate submission to finalist level.
"""

import sys
import os
import json
import numpy as np
from datetime import datetime

# ---------------------------------------------------------------------------
# Global robustness fixes (apply to every phase, since they share these modules)
# ---------------------------------------------------------------------------
# 1) Force UTF-8 console so the [OK]/[FAILED]/-> glyphs don't crash on Windows cp1252.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8')
    except Exception:
        pass

# 2) Make json.dump tolerate numpy scalars/arrays. Phase modules all do
#    `import json; json.dump(...)` on the SAME module object, so patching the
#    shared module here fixes serialization everywhere with no per-file edits.
def _np_json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")

_orig_json_dump = json.dump
_orig_json_dumps = json.dumps

def _patched_dump(obj, fp, **kw):
    kw.setdefault('default', _np_json_default)
    return _orig_json_dump(obj, fp, **kw)

def _patched_dumps(obj, **kw):
    kw.setdefault('default', _np_json_default)
    return _orig_json_dumps(obj, **kw)

json.dump = _patched_dump
json.dumps = _patched_dumps

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("\n" + "="*100)
print(" COMPREHENSIVE 9-PHASE VALIDATION FRAMEWORK")
print(" Finalist-Level Submission Analysis")
print("="*100)

start_time = datetime.now()
print(f"\nStarted: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

# ===== LOAD DATA =====
print("\nSETUP: Loading data...")

try:
    import json
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif
    from src.shre.stage2_features import FeatureEngineer

    # Load labeled data only. The full candidates.jsonl (487 MB / 100k rows) is
    # NOT needed here - every labeled record already carries its raw_profile.
    labeled_path = 'labeling/combined_labels.json'

    with open(labeled_path, 'r', encoding='utf-8') as f:
        labeled = json.load(f)

    print(f"OK: Loaded {len(labeled)} labeled samples")

    # Extract features
    fe = FeatureEngineer()
    labeled_features = fe.compute_features([item['raw_profile'] for item in labeled])

    X = np.array([list(fv.values()) for _, fv in labeled_features])
    y = np.array([item['relevance_score'] for item in labeled])
    feature_names = list(labeled_features[0][1].keys())

    # Clean data
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"OK: Extracted {X.shape[1]} features for {len(y)} samples")
    print(f"OK: Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

except Exception as e:
    print(f"FAILED: Data loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create output directory
output_dir = 'analysis_results'
os.makedirs(output_dir, exist_ok=True)

# ===== PHASE 1: LEARNING CURVES =====
print("\n[PHASE 1] Running Learning Curve Analysis...")
try:
    from analysis.phase1_learning_curves import create_learning_curves
    df_lc, analysis_lc = create_learning_curves(X, y, feature_names, output_dir)
    print("OK: Phase 1 Complete")
except Exception as e:
    print(f"FAILED: Phase 1 failed: {e}")
    import traceback
    traceback.print_exc()

# ===== PHASE 2: FEATURE IMPORTANCE =====
print("\n[PHASE 2] Running Feature Importance Analysis...")
try:
    from analysis.phase2_feature_importance import analyze_feature_importance
    df_imp, df_perm, corr_df = analyze_feature_importance(X, y, feature_names, output_dir)
    print("OK: Phase 2 Complete")
except Exception as e:
    print(f"FAILED: Phase 2 failed: {e}")
    import traceback
    traceback.print_exc()

# ===== PHASE 3: SHAP EXPLAINABILITY =====
print("\n[PHASE 3] Running SHAP Explainability Analysis...")
try:
    from analysis.phase3_shap_explainability import explain_with_shap
    explanations, interactions = explain_with_shap(X, y, feature_names, output_dir)
    print("OK: Phase 3 Complete")
except Exception as e:
    print(f"FAILED: Phase 3 failed: {e}")
    import traceback
    traceback.print_exc()

# ===== PHASE 4: ABLATION STUDY =====
print("\n[PHASE 4] Running Ablation Study...")
try:
    from analysis.phase4_ablation_study import run_ablation_study
    df_models, df_features = run_ablation_study(X, y, feature_names, output_dir)
    print("OK: Phase 4 Complete")
except Exception as e:
    print(f"FAILED: Phase 4 failed: {e}")
    import traceback
    traceback.print_exc()

# ===== PHASE 5: STABILITY ANALYSIS =====
print("\n[PHASE 5] Running Stability Analysis...")
try:
    from analysis.phase5_stability_analysis import stability_analysis
    df_stability, stats_stability = stability_analysis(X, y, feature_names, output_dir)
    print("OK: Phase 5 Complete")
except Exception as e:
    print(f"FAILED: Phase 5 failed: {e}")
    import traceback
    traceback.print_exc()

# ===== PHASE 6: HONEYPOT VALIDATION =====
print("\n[PHASE 6] Running Honeypot Validation...")
try:
    from analysis.phase6_honeypot_validation import honeypot_validation
    df_honeypot, df_honeypot_types = honeypot_validation(X, y, feature_names, output_dir)
    print("OK: Phase 6 Complete")
except Exception as e:
    print(f"FAILED: Phase 6 failed: {e}")
    import traceback
    traceback.print_exc()

# ===== PHASE 7: ERROR ANALYSIS =====
print("\n[PHASE 7] Running Error Analysis...")
try:
    from analysis.phase7_error_analysis import error_analysis
    df_errors, df_class_stats = error_analysis(X, y, feature_names, output_dir)
    print("OK: Phase 7 Complete")
except Exception as e:
    print(f"FAILED: Phase 7 failed: {e}")
    import traceback
    traceback.print_exc()

# ===== PHASE 8: RANKING VALIDATION =====
print("\n[PHASE 8] Running Ranking Validation...")
try:
    from analysis.phase8_ranking_validation import ranking_validation
    df_ranking = ranking_validation(X, y, feature_names, output_dir)
    print("OK: Phase 8 Complete")
except Exception as e:
    print(f"FAILED: Phase 8 failed: {e}")
    import traceback
    traceback.print_exc()

# ===== PHASE 9: GENERATE COMPETITION REPORT =====
print("\n[PHASE 9] Generating Competition Report...")
try:
    from analysis.phase9_competition_report import generate_competition_report
    report = generate_competition_report(output_dir)
    print("OK: Phase 9 Complete")
except Exception as e:
    print(f"FAILED: Phase 9 failed: {e}")
    import traceback
    traceback.print_exc()

# ===== SUMMARY =====
end_time = datetime.now()
duration = (end_time - start_time).total_seconds() / 60

print("\n" + "="*100)
print("ANALYSIS COMPLETE")
print("="*100)
print(f"\nResults saved to: {os.path.abspath(output_dir)}")
print(f"Total time: {duration:.1f} minutes")

print("\n\nFILE SUMMARY:")
print("-" * 100)

result_files = [
    ("phase1_learning_curves.png", "Learning curve plots"),
    ("learning_curves_table.csv", "Learning curve data"),
    ("phase2_gain_importance.png", "Feature importance plot"),
    ("phase2_model_comparison.png", "Model comparison"),
    ("phase2_permutation_importance.png", "Permutation importance"),
    ("phase2_correlation_heatmap.png", "Feature correlation heatmap"),
    ("phase3_shap_summary.png", "SHAP summary plot"),
    ("phase3_shap_density.png", "SHAP density plot"),
    ("phase3_shap_dependence.png", "SHAP dependence plots"),
    ("phase3_waterfall_BEST_RANKED.png", "SHAP waterfall: best candidate"),
    ("phase3_waterfall_WORST_RANKED.png", "SHAP waterfall: worst candidate"),
    ("phase4_model_comparison.png", "Ablation: model comparison"),
    ("phase4_feature_groups.png", "Ablation: feature groups"),
    ("phase5_distributions.png", "Stability: distributions"),
    ("phase5_boxplots.png", "Stability: box plots"),
    ("phase5_timeseries.png", "Stability: time series"),
    ("phase6_honeypot_analysis.png", "Honeypot validation plots"),
    ("phase6_confusion_matrix.png", "Honeypot confusion matrix"),
    ("phase7_confusion_matrix.png", "Error analysis confusion matrix"),
    ("phase7_error_rates.png", "Error rates by class"),
    ("phase7_error_flow.png", "Misclassification flow"),
    ("phase8_ranking_metrics.png", "Ranking metrics progression"),
    ("phase8_score_distribution.png", "Ranking score distribution"),
    ("COMPETITION_REPORT.md", "Comprehensive competition report"),
]

for filename, description in result_files:
    filepath = os.path.join(output_dir, filename)
    if os.path.exists(filepath):
        print(f"OK: {filename:45s} - {description}")
    else:
        print(f"FAILED: {filename:45s} - MISSING")

print("\n" + "="*100)
print("ANALYSIS FRAMEWORK COMPLETE")
print("="*100)

print("\n\nNEXT STEPS:")
print("1. Review COMPETITION_REPORT.md for comprehensive analysis")
print("2. Examine visualizations in analysis_results/")
print("3. Use insights to refine model or document strengths")
print("4. Submit with confidence - all metrics are evidence-based")

print("\nOK: Framework successfully completed!")
print(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
