#!/usr/bin/env python3
"""
End-to-end test for the enhanced "Opus 4.8" pipeline.

Validates the four new features and that the full SHRE -> CTAE fallback chain
still works, running on the bundled 10-candidate sample.

Run:  py -3.13 test_enhanced.py
"""

import os
import sys
import csv
import json

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SAMPLE = 'data/sample_candidates.jsonl'
LABELED = 'labeling/combined_labels.json'

_passed, _failed = 0, 0


def check(name, cond, detail=""):
    global _passed, _failed
    mark = "OK " if cond else "XX "
    print(f"  [{mark}] {name}" + (f" - {detail}" if detail else ""))
    if cond:
        _passed += 1
    else:
        _failed += 1


def load(path):
    return [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]


def test_features():
    print("\n[1] Anomaly / Behavioral / Semantic modules")
    from src.shre.anomaly import AnomalyDetector
    from src.shre.behavioral import BehavioralScorer
    from src.shre.semantic import MultiVectorSemanticScorer, SEMANTIC_FEATURE_KEYS

    cands = load(SAMPLE)
    a = AnomalyDetector().analyze(cands[0])
    check("anomaly returns score in [0,1]", 0.0 <= a['anomaly_score'] <= 1.0, str(a['anomaly_score']))
    check("anomaly returns flags list + synthetic bool",
          isinstance(a['flags'], list) and isinstance(a['is_synthetic'], bool))

    b = BehavioralScorer().score(cands[0])
    check("behavioral has 5 sub-scores", len(b) == 5, str(list(b)))
    check("behavioral_composite in [0,1]", 0.0 <= b['behavioral_composite'] <= 1.0)

    sem = MultiVectorSemanticScorer().score_batch(cands)
    check("semantic returns one dict per candidate", len(sem) == len(cands))
    check("semantic has all 5 keys", set(SEMANTIC_FEATURE_KEYS).issubset(sem[0]))
    check("semantic_fusion_score in [0,1]", 0.0 <= sem[0]['semantic_fusion_score'] <= 1.0,
          f"{sem[0]['semantic_fusion_score']:.3f}")


def test_pipeline_stages():
    print("\n[2] Stage 1 + Stage 2 enrichment")
    import numpy as np
    from src.shre.stage1_filter import FastFilter
    from src.shre.stage2_features import FeatureEngineer

    cands = load(SAMPLE)
    viable = FastFilter().filter(cands)
    check("stage1 keeps a non-empty subset", 0 < len(viable) <= len(cands),
          f"{len(viable)}/{len(cands)}")

    feats = FeatureEngineer().compute_features(viable)
    n_feat = len(feats[0][1])
    check("enriched feature count > 78 base", n_feat > 78, f"{n_feat} features")
    X = np.array([list(fv.values()) for _, fv in feats])
    check("no NaN/Inf in feature matrix", np.isnan(X).sum() == 0 and np.isinf(X).sum() == 0)
    enriched_keys = {'anomaly_score', 'behavioral_composite', 'semantic_fusion_score'}
    check("enrichment keys present in vector", enriched_keys.issubset(feats[0][1].keys()))


def test_end_to_end():
    print("\n[3] Full SHRE pipeline (LTR head)")
    from src.main import run_shre
    out = 'output/submission_enhanced_test.csv'
    run_shre(SAMPLE, LABELED, out)
    check("submission CSV written", os.path.exists(out))
    if os.path.exists(out):
        rows = list(csv.DictReader(open(out, encoding='utf-8')))
        check("CSV has expected columns",
              {'candidate_id', 'rank', 'score', 'reasoning'}.issubset(rows[0].keys()))
        check("reasoning is populated", all(r['reasoning'].strip() for r in rows))
        scores = [float(r['score']) for r in rows]
        check("scores within [0,1]", all(0.0 <= s <= 1.0 for s in scores))
        check("ranking is monotonically non-increasing",
              all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)))


def test_ctae_fallback():
    print("\n[4] CTAE fallback still works")
    from src.main import run_ctae
    out = 'output/submission_ctae_test.csv'
    run_ctae(SAMPLE, out)
    check("CTAE CSV written", os.path.exists(out))
    if os.path.exists(out):
        rows = list(csv.DictReader(open(out, encoding='utf-8')))
        check("CTAE produced rows", len(rows) > 0, f"{len(rows)} rows")


def main():
    print("=" * 70)
    print(" ENHANCED PIPELINE TEST ('Opus 4.8')")
    print("=" * 70)
    test_features()
    test_pipeline_stages()
    test_end_to_end()
    test_ctae_fallback()
    print("\n" + "=" * 70)
    print(f" RESULT: {_passed} passed, {_failed} failed")
    print("=" * 70)
    sys.exit(1 if _failed else 0)


if __name__ == '__main__':
    main()
