import os
import sys
import json
import traceback

def load_jsonl(filepath):
    print(f"Loading {filepath}...")
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def run_shre(candidates_path, labeled_path, out_path):
    print("=== RUNNING SHRE (Validated ML Pipeline - Competition Grade) ===")
    from src.shre.stage1_filter import FastFilter
    from src.shre.stage2_features import FeatureEngineer
    from src.shre.stage3_ranking_validated import train_and_predict_validated
    from src.shre.stage4_submit import export_submission

    candidates = load_jsonl(candidates_path)

    ff = FastFilter()
    viable = ff.filter(candidates)
    print(f"Stage 1: Filtered {len(candidates)} down to {len(viable)} viable candidates.")

    fe = FeatureEngineer()
    feature_matrix = fe.compute_features(viable)
    feature_names = list(feature_matrix[0][1].keys())
    print(f"Stage 2: Extracted {len(feature_names)} features.")

    scores, metadata = train_and_predict_validated(labeled_path, feature_matrix, feature_names)
    print("Stage 3: Validated Ensemble prediction complete.")
    print(f"  - Test Accuracy: {metadata.get('test_accuracy', 'N/A'):.4f}")
    print(f"  - Test F1-Score: {metadata.get('test_f1', 'N/A'):.4f}")

    export_submission(viable, scores, out_path)

def run_ctae(candidates_path, out_path):
    print("=== RUNNING CTAE (Pure Python Fallback) ===")
    from src.ctae.ranker import run_ctae_ranking
    run_ctae_ranking(candidates_path, out_path)

def main():
    if len(sys.argv) < 3:
        print("Usage: python src/main.py <input.jsonl> <output.csv>")
        sys.exit(1)
        
    candidates_path = sys.argv[1]
    out_path = sys.argv[2]
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    labeled_path = os.path.join(base_dir, 'labeling', 'combined_labels.json')
    
    try:
        run_shre(candidates_path, labeled_path, out_path)
    except Exception as e:
        print("\n!!! SHRE FAILED !!!")
        traceback.print_exc()
        print("\nAttempting CTAE Fallback...")
        try:
            run_ctae(candidates_path, out_path)
        except Exception as e2:
            print("\n!!! CTAE FALLBACK ALSO FAILED !!!")
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main()
