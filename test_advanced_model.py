#!/usr/bin/env python3
"""
Quick test script for advanced ensemble model
Compares accuracy: Basic vs Advanced
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

def test_basic_model():
    """Test basic XGBoost model"""
    print("\n" + "="*80)
    print("TESTING BASIC MODEL (Single XGBoost)")
    print("="*80)

    try:
        from src.shre.stage2_features import FeatureEngineer
        from src.shre.stage3_ranking import train_and_predict as basic_train

        # Load candidates
        from src.common.data_loader import load_jsonl
        from src.shre.stage1_filter import FastFilter

        candidates = load_jsonl('data/candidates.jsonl')
        ff = FastFilter()
        viable = ff.filter(candidates)

        fe = FeatureEngineer()
        feature_matrix = fe.compute_features(viable)
        feature_names = list(feature_matrix[0][1].keys())

        # Train basic model
        scores = basic_train('labeling/combined_labels.json', feature_matrix, feature_names)

        print(f"\nOK: Basic model training complete")
        print(f"     Predictions: {len(scores)}")
        print(f"     Score range: [{np.min(scores):.4f}, {np.max(scores):.4f}]")

        # Load model metrics
        with open('models/model_metadata.json', 'r') as f:
            metadata = json.load(f)

        print(f"\nBasic Model Results:")
        print(f"  CV Accuracy:  {metadata.get('cv_accuracy_mean', 0):.4f}")

        return metadata.get('cv_accuracy_mean', 0)

    except Exception as e:
        print(f"ERROR: Basic model test failed: {e}")
        import traceback
        traceback.print_exc()
        return 0


def test_advanced_model():
    """Test advanced ensemble model"""
    print("\n" + "="*80)
    print("TESTING ADVANCED MODEL (XGBoost + LightGBM + CatBoost Ensemble)")
    print("="*80)

    try:
        from src.shre.stage2_features import FeatureEngineer
        from src.shre.stage3_ranking_advanced import train_and_predict as advanced_train

        # Load candidates
        from src.common.data_loader import load_jsonl
        from src.shre.stage1_filter import FastFilter

        candidates = load_jsonl('data/candidates.jsonl')
        ff = FastFilter()
        viable = ff.filter(candidates)

        fe = FeatureEngineer()
        feature_matrix = fe.compute_features(viable)
        feature_names = list(feature_matrix[0][1].keys())

        print("Training advanced ensemble (this will take 2-3 minutes)...")
        print("Models: XGBoost, LightGBM, CatBoost")
        print("Augmentation: SMOTE + BorderlineSMOTE + ADASYN")

        # Train advanced model
        scores = advanced_train('labeling/combined_labels.json', feature_matrix, feature_names)

        print(f"\nOK: Advanced model training complete")
        print(f"     Predictions: {len(scores)}")
        print(f"     Score range: [{np.min(scores):.4f}, {np.max(scores):.4f}]")

        # Load model metrics
        with open('models/ensemble_metadata.json', 'r') as f:
            metadata = json.load(f)

        print(f"\nAdvanced Model Results:")
        print(f"  CV Accuracy:  {metadata.get('cv_accuracy_mean', 0):.4f}")
        print(f"  Final Accuracy: {metadata.get('final_accuracy', 0):.4f}")

        return metadata.get('cv_accuracy_mean', 0)

    except ImportError as e:
        print(f"\nERROR: Missing dependencies!")
        print(f"Install with: pip install lightgbm catboost imbalanced-learn")
        print(f"\nError: {e}")
        return 0
    except Exception as e:
        print(f"ERROR: Advanced model test failed: {e}")
        import traceback
        traceback.print_exc()
        return 0


def main():
    print("\n" + "="*80)
    print("QUICK MODEL COMPARISON TEST")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\nThis script compares:")
    print("  1. Basic Model:    Single XGBoost")
    print("  2. Advanced Model: Ensemble (XGBoost + LightGBM + CatBoost)")
    print("\nTarget: Improve accuracy from 84.55% -> 90%+\n")

    # Test basic model
    basic_acc = test_basic_model()

    # Ask if user wants to test advanced
    print("\n" + "="*80)
    print("OPTION 1: Test Advanced Model Now")
    print("  This will train a 3-model ensemble (takes 2-3 minutes)")
    print("  Expected accuracy: 90%+")
    print("\nOPTION 2: Run Later")
    print("  Command: python test_advanced_model.py --advanced-only")
    print("="*80)

    # For automation, always test advanced if not in basic-only mode
    if len(sys.argv) < 2 or '--advanced-only' in sys.argv or '--both' in sys.argv:
        response = input("\nTest advanced model now? (y/n): ").strip().lower()
        if response == 'y':
            advanced_acc = test_advanced_model()
        else:
            advanced_acc = 0
    else:
        advanced_acc = 0

    # Summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)

    if basic_acc > 0:
        print(f"Basic Model (XGBoost):         {basic_acc*100:.2f}% accuracy")
    if advanced_acc > 0:
        print(f"Advanced Model (Ensemble):     {advanced_acc*100:.2f}% accuracy")
        if basic_acc > 0:
            improvement = (advanced_acc - basic_acc) / basic_acc * 100
            print(f"Improvement:                   +{improvement:.2f}%")

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    if advanced_acc >= 0.90:
        print("OK: Advanced model achieved 90%+ accuracy!")
        print("  Use for production: python src/main.py data/candidates.jsonl output/submission.csv")
        print("  (Make sure to edit main.py to use stage3_ranking_advanced.py)")
    elif advanced_acc > basic_acc:
        print(f"OK: Advanced model improved accuracy by {(advanced_acc - basic_acc)*100:.2f}%")
        print("  To reach 90%+, collect more labeled data or add new features")
    else:
        print("Run advanced model with: python test_advanced_model.py --advanced-only")

    print("\nFor more details, see: ADVANCED_MODEL_GUIDE.md")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
