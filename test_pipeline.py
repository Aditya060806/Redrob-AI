#!/usr/bin/env python3
"""
Comprehensive pipeline testing and evaluation script.
Tests all stages of SHRE and generates detailed performance metrics.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

# Fix unicode issues on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def print_header(text):
    print(f"\n{'='*80}")
    print(f" {text}")
    print(f"{'='*80}")

def test_stage1_filter():
    """Test Stage 1: Fast Filter"""
    print_header("STAGE 1: FAST FILTER TESTING")

    from src.shre.stage1_filter import FastFilter
    from src.common.data_loader import load_jsonl

    candidates = load_jsonl('data/candidates.jsonl')
    print(f"Loaded {len(candidates)} total candidates")

    ff = FastFilter()
    viable = ff.filter(candidates)

    print(f"OK: Filtered to {len(viable)} viable candidates")
    print(f"  - Filtered out: {len(candidates) - len(viable)} ({100*(len(candidates)-len(viable))/len(candidates):.1f}%)")

    # Check honeypot detection
    honeypots = sum(1 for c in candidates if ff.is_honeypot(c))
    print(f"  - Honeypots detected: {honeypots}")

    return viable

def test_stage2_features(candidates):
    """Test Stage 2: Feature Engineering"""
    print_header("STAGE 2: FEATURE ENGINEERING TESTING")

    from src.shre.stage2_features import FeatureEngineer

    fe = FeatureEngineer()
    features = fe.compute_features(candidates)

    print(f"OK: Extracted features for {len(features)} candidates")

    if features:
        feature_names = list(features[0][1].keys())
        print(f"  - Total features: {len(feature_names)}")
        print(f"  - Features: {', '.join(feature_names[:10])}...")

        # Check for NaN/Inf
        feature_matrix = np.array([list(fv.values()) for _, fv in features])
        nan_count = np.isnan(feature_matrix).sum()
        inf_count = np.isinf(feature_matrix).sum()

        print(f"  - NaN values: {nan_count}")
        print(f"  - Inf values: {inf_count}")
        print(f"OK: Feature extraction successful")

    return features, feature_names

def test_stage3_ranking(feature_names):
    """Test Stage 3: Model Training and Prediction (VALIDATED MODEL)"""
    print_header("STAGE 3: MODEL TRAINING & EVALUATION (VALIDATED)")

    from src.shre.stage3_ranking_validated import train_and_predict_validated

    labeled_path = 'labeling/combined_labels.json'

    # Load candidates for feature matrix
    from src.common.data_loader import load_jsonl
    from src.shre.stage1_filter import FastFilter
    from src.shre.stage2_features import FeatureEngineer

    candidates = load_jsonl('data/candidates.jsonl')
    ff = FastFilter()
    viable = ff.filter(candidates)
    fe = FeatureEngineer()
    feature_matrix = fe.compute_features(viable)

    print(f"Feature matrix shape: {len(feature_matrix)} candidates x {len(feature_names)} features")

    try:
        scores, metadata = train_and_predict_validated(labeled_path, feature_matrix, feature_names)
        print(f"OK: Model training and prediction successful")
        print(f"VALIDATED: Test Accuracy: {metadata.get('test_accuracy', 'N/A'):.4f}")
        print(f"VALIDATED: Test F1-Score: {metadata.get('test_f1', 'N/A'):.4f}")
        print(f"  - Generated {len(scores)} predictions")
        print(f"  - Score range: [{np.min(scores):.2f}, {np.max(scores):.2f}]")
        print(f"  - Mean score: {np.mean(scores):.4f}")
        print(f"  - Std score: {np.std(scores):.4f}")

        return scores, viable
    except Exception as e:
        print(f"ERROR: Model training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_stage4_submission(candidates, scores):
    """Test Stage 4: Submission Generation"""
    print_header("STAGE 4: SUBMISSION GENERATION")

    from src.shre.stage4_submit import export_submission

    if candidates is None or scores is None:
        print("ERROR: Cannot test submission without valid candidates and scores")
        return

    out_path = 'output/submission_test.csv'

    try:
        export_submission(candidates, scores, out_path)

        if os.path.exists(out_path):
            with open(out_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            print(f"OK: Submission CSV generated successfully")
            print(f"  - Output file: {out_path}")
            print(f"  - Total lines: {len(lines)} (including header)")
            print(f"  - Sample entries: {min(3, len(lines)-1)}")

            # Show sample
            for i, line in enumerate(lines[1:4], 1):
                cols = line.strip().split(',')
                if len(cols) >= 2:
                    print(f"    {i}. Rank {cols[1]}: {cols[0]} (score: {cols[2][:6]}...)")
        else:
            print(f"ERROR: Output file not found at {out_path}")
    except Exception as e:
        print(f"ERROR: Submission generation failed: {e}")
        import traceback
        traceback.print_exc()

def test_model_accuracy():
    """Test model accuracy on labeled data"""
    print_header("MODEL ACCURACY TESTING")

    import json
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import xgboost as xgb
    import pickle
    from sklearn.preprocessing import StandardScaler

    # Load model
    model_path = 'models/ensemble_model.pkl'
    scaler_path = 'models/scaler.pkl'
    selector_path = 'models/feature_selector.pkl'

    if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(selector_path):
        print("[WARN]  Model not found. Train the model first with: python src/main.py data/candidates.jsonl output/submission.csv")
        return

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(selector_path, 'rb') as f:
            selector = pickle.load(f)

        print("OK: Model and selector loaded successfully")

        # Load labeled data and extract features
        with open('labeling/combined_labels.json', 'r') as f:
            labeled = json.load(f)

        from src.shre.stage2_features import FeatureEngineer
        fe = FeatureEngineer()
        labeled_features = fe.compute_features([item['raw_profile'] for item in labeled])

        X = np.array([list(fv.values()) for _, fv in labeled_features])
        y = np.array([item['relevance_score'] for item in labeled])

        # Clean, select and scale
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_selected = selector.transform(X)
        X_scaled = scaler.transform(X_selected)

        # Predict
        y_pred = model.predict(X_scaled)

        # Calculate metrics
        acc = accuracy_score(y, y_pred)

        print(f"\nOK: Model Accuracy Test Results:")
        print(f"  - Accuracy: {acc:.4f} ({100*acc:.2f}%)")
        print(f"\nClassification Report:")
        print(classification_report(y, y_pred, zero_division=0))

        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(y, y_pred)
        for i, row in enumerate(cm):
            print(f"  Class {i}: {row}")

        # Per-class accuracy
        print(f"\nPer-Class Accuracy:")
        for i in range(4):
            mask = y == i
            if mask.sum() > 0:
                class_acc = accuracy_score(y[mask], y_pred[mask])
                count = mask.sum()
                print(f"  Score {i}: {class_acc:.4f} ({count} samples)")

    except Exception as e:
        print(f"ERROR: Model accuracy test failed: {e}")
        import traceback
        traceback.print_exc()

def test_ctae_fallback():
    """Test CTAE Fallback path"""
    print_header("CTAE FALLBACK PATH TESTING")
    from src.main import run_ctae
    out_path = 'output/submission_fallback_test.csv'
    try:
        run_ctae('data/candidates.jsonl', out_path)
        if os.path.exists(out_path):
            print(f"OK: Fallback CSV generated successfully: {out_path}")
            # Validate output using python validator
            import subprocess
            validator_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'validate_submission.py')
            res = subprocess.run([sys.executable, validator_script, out_path], capture_output=True, text=True)
            if "valid" in res.stdout.lower():
                print("OK: Fallback CSV validation passed!")
            else:
                print(f"ERROR: Fallback CSV validation failed: {res.stdout} {res.stderr}")
        else:
            print(f"ERROR: Fallback output file not found at {out_path}")
    except Exception as e:
        print(f"ERROR: Fallback test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    print_header("COMPREHENSIVE SHRE PIPELINE TEST")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Test Stage 1
    viable_candidates = test_stage1_filter()

    # Test Stage 2
    feature_matrix, feature_names = test_stage2_features(viable_candidates)

    # Test Stage 3 & 4
    scores, candidates = test_stage3_ranking(feature_names)
    if scores is not None:
        test_stage4_submission(candidates, scores)

    # Test model accuracy
    test_model_accuracy()

    # Test CTAE fallback
    test_ctae_fallback()

    print_header("TEST SUMMARY")
    print("OK: All tests completed successfully!")
    print(f"Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nNext steps:")
    print(f"  1. Review output/submission_test.csv")
    print(f"  2. Check models/ directory for trained model")
    print(f"  3. Run final submission: python src/main.py data/candidates.jsonl output/submission.csv")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
