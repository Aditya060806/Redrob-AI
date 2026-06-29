import os
import sys
import json
import argparse
import traceback


def load_jsonl(filepath):
    print(f"Loading {filepath}...")
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def _fmt(value):
    """Format a metric for logging whether it's a number or 'N/A'."""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def run_shre(candidates_path, labeled_path, out_path, jd=None, force_train=False):
    """
    Enhanced ML ranking pipeline.

    By default this runs in *inference mode* when trained artifacts exist
    (lightning-fast: no retraining), and falls back to the full training path
    when models are missing or `force_train=True`. A JobDescription (`jd`)
    re-targets the Stage-1 experience gate and the semantic-fit signal so the
    engine understands the specific role being hired for.
    """
    from src.shre.job_description import JobDescription
    from src.shre.stage1_filter import FastFilter
    from src.shre.stage2_features import FeatureEngineer
    from src.shre.stage4_submit import export_submission
    from src.shre.inference import inference_artifacts_exist, predict_pool

    if jd is None:
        jd = JobDescription.default()

    print("=== RUNNING SHRE (Enhanced ML Pipeline - 'Opus 4.8' Grade) ===")
    print(f"    {jd.describe()}")

    candidates = load_jsonl(candidates_path)

    # Stage 1: Enhanced anomaly/honeypot pre-filter + (JD-driven) experience gate.
    ff = FastFilter(jd=jd)
    viable = ff.filter(candidates)
    print(f"Stage 1: Filtered {len(candidates)} down to {len(viable)} viable candidates.")

    # Stage 2: 78 base features + anomaly + behavioral + JD-aware semantic.
    fe = FeatureEngineer(jd=jd)
    feature_matrix = fe.compute_features(viable)
    feature_names = list(feature_matrix[0][1].keys())
    print(f"Stage 2: Extracted {len(feature_names)} enriched features.")

    # Stage 3: inference (fast) when models exist, else train.
    use_inference = (not force_train) and inference_artifacts_exist()
    scores = metadata = None

    if use_inference:
        try:
            print("Stage 3: Inference mode (scoring with saved models, no retraining).")
            scores, metadata = predict_pool(feature_matrix, feature_names)
            print(f"  - Ranking head: {metadata.get('ranking_head', 'ensemble')}")
            print(f"  - Scored {metadata.get('scored_candidates', len(viable))} candidates.")
        except Exception as e:
            print(f"\n[Stage 3] Inference failed ({type(e).__name__}: {e}); "
                  f"falling back to full training path.")
            scores = None

    if scores is None:
        scores, metadata = _train_path(labeled_path, feature_matrix, feature_names)

    print(f"  - Test Accuracy: {_fmt(metadata.get('test_accuracy', 'N/A'))}")
    print(f"  - Test F1-Score: {_fmt(metadata.get('test_f1', 'N/A'))}")

    export_submission(viable, scores, out_path)


def _train_path(labeled_path, feature_matrix, feature_names):
    """Full training path: LambdaMART head, falling back to validated ensemble."""
    from src.shre.stage3_ranking_validated import train_and_predict_validated
    try:
        from src.shre.stage3_ranking_ltr import train_and_predict_ltr
        scores, metadata = train_and_predict_ltr(labeled_path, feature_matrix, feature_names)
        print("Stage 3: Learning-to-Rank (LambdaMART) prediction complete.")
        print(f"  - NDCG@10:  {_fmt(metadata.get('ndcg_at_10', 'N/A'))}")
        print(f"  - NDCG@100: {_fmt(metadata.get('ndcg_at_100', 'N/A'))}")
        print(f"  - HARD NDCG@10 (rel 1 vs 2): {_fmt(metadata.get('ndcg_at_10_hard', 'N/A'))}"
              f"  Spearman: {_fmt(metadata.get('spearman_hard', 'N/A'))}")
    except Exception as e:
        print(f"\n[Stage 3] LTR head failed ({type(e).__name__}: {e}); "
              f"falling back to validated ensemble ranking.")
        scores, metadata = train_and_predict_validated(labeled_path, feature_matrix, feature_names)
        print("Stage 3: Validated Ensemble prediction complete.")
    return scores, metadata


def run_ctae(candidates_path, out_path):
    print("=== RUNNING CTAE (Pure Python Fallback) ===")
    from src.ctae.ranker import run_ctae_ranking
    run_ctae_ranking(candidates_path, out_path)


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="SHRE candidate ranking pipeline (Opus 4.8).")
    parser.add_argument('input', help="Input candidates .jsonl path")
    parser.add_argument('output', help="Output submission .csv path")
    parser.add_argument('--jd', default=None,
                        help="Job description as raw text or a path to a .txt "
                             "file. Re-targets the experience gate and semantic "
                             "fit. Defaults to the founding-engineer role.")
    parser.add_argument('--train', '--retrain', dest='train', action='store_true',
                        help="Force full retraining instead of fast inference.")
    return parser.parse_args(argv)


def main():
    # Back-compat: support the old positional `input output` invocation.
    if len(sys.argv) < 3:
        print("Usage: python -m src.main <input.jsonl> <output.csv> "
              "[--jd <text|path>] [--train]")
        sys.exit(1)

    args = _parse_args(sys.argv[1:])

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    labeled_path = os.path.join(base_dir, 'labeling', 'combined_labels.json')

    from src.shre.job_description import JobDescription
    jd = JobDescription.from_source(args.jd)

    try:
        run_shre(args.input, labeled_path, args.output,
                 jd=jd, force_train=args.train)
    except Exception:
        print("\n!!! SHRE FAILED !!!")
        traceback.print_exc()
        print("\nAttempting CTAE Fallback...")
        try:
            run_ctae(args.input, args.output)
        except Exception:
            print("\n!!! CTAE FALLBACK ALSO FAILED !!!")
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
