"""
Ranking Metrics - NDCG, MAP, Precision@K, Recall@K
Proves model quality for Top 100 ranking
"""

import numpy as np
import json
import os


def dcg(relevances, k=100):
    """Discounted Cumulative Gain"""
    relevances = np.array(relevances)[:k]
    gains = 2 ** (relevances) - 1
    discounts = np.log2(np.arange(len(gains)) + 2)
    return np.sum(gains / discounts)


def ndcg(y_true, y_pred_proba, k=100):
    """Normalized DCG@K"""
    ideal_relevance = np.sort(y_true)[::-1]
    actual_relevance = y_true[np.argsort(y_pred_proba)[::-1]]

    ideal_dcg = dcg(ideal_relevance, k)
    actual_dcg = dcg(actual_relevance, k)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def average_precision(y_true, y_pred_proba, k=100):
    """Mean Average Precision@K"""
    sorted_indices = np.argsort(y_pred_proba)[::-1][:k]
    sorted_relevance = y_true[sorted_indices]

    precisions = []
    num_relevant = 0

    for i, rel in enumerate(sorted_relevance):
        if rel > 0:
            num_relevant += 1
            precision_at_i = num_relevant / (i + 1)
            precisions.append(precision_at_i)

    if len(precisions) == 0:
        return 0.0
    return np.mean(precisions)


def precision_at_k(y_true, y_pred_proba, k=100):
    """Precision@K"""
    sorted_indices = np.argsort(y_pred_proba)[::-1][:k]
    sorted_relevance = y_true[sorted_indices]
    return np.mean(sorted_relevance > 0)


def recall_at_k(y_true, y_pred_proba, k=100):
    """Recall@K"""
    sorted_indices = np.argsort(y_pred_proba)[::-1][:k]
    sorted_relevance = y_true[sorted_indices]

    num_relevant_in_k = np.sum(sorted_relevance > 0)
    total_relevant = np.sum(y_true > 0)

    if total_relevant == 0:
        return 0.0
    return num_relevant_in_k / total_relevant


def evaluate_ranking_metrics(y_true, y_pred_proba, output_dir='validation_results'):
    """
    Evaluate ranking quality using multiple metrics.
    Proves model quality for Top 100 ranking.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("RANKING METRICS - Top 100 Quality Evaluation")
    print("="*80)

    metrics_k = [10, 50, 100]
    results = {}

    for k in metrics_k:
        print(f"\n@ Top {k}:")

        ndcg_k = ndcg(y_true, y_pred_proba, k)
        map_k = average_precision(y_true, y_pred_proba, k)
        prec_k = precision_at_k(y_true, y_pred_proba, k)
        rec_k = recall_at_k(y_true, y_pred_proba, k)

        print(f"  NDCG@{k}:      {ndcg_k:.4f}")
        print(f"  MAP@{k}:       {map_k:.4f}")
        print(f"  Precision@{k}: {prec_k:.4f}")
        print(f"  Recall@{k}:    {rec_k:.4f}")

        results[f'ndcg_{k}'] = float(ndcg_k)
        results[f'map_{k}'] = float(map_k)
        results[f'precision_{k}'] = float(prec_k)
        results[f'recall_{k}'] = float(rec_k)

    # Save results
    with open(os.path.join(output_dir, 'ranking_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nOK: Ranking metrics saved to {output_dir}/ranking_metrics.json")
    print(f"\nInterpretation:")
    print(f"  NDCG > 0.7 -> Good ranking quality [OK]")
    print(f"  NDCG < 0.5 -> Poor ranking quality [FAILED]")

    return results
