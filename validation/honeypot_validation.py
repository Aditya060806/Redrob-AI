"""
Honeypot Detection Validation
Proves honeypot detection is working correctly with metrics:
- Detection rate (true positives)
- False positive rate
- False negative rate
"""

import json
import os
import numpy as np


def validate_honeypot_detection(candidates, output_dir='validation_results'):
    """
    Validate honeypot detection performance.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("HONEYPOT DETECTION VALIDATION")
    print("="*80)

    from src.shre.stage1_filter import FastFilter
    ff = FastFilter()

    # Test on labeled honeypots and real candidates
    honeypot_detections = 0
    real_detections = 0
    false_positives = 0
    false_negatives = 0

    honeypot_rules_triggered = {
        'skill_overflow': 0,
        'timeline_overlap': 0
    }

    for candidate in candidates[:500]:  # Test on larger sample
        is_detected = ff.is_honeypot(candidate)

        # Ground truth: check mathematically if there is skill overflow or extreme overlap
        years = candidate.get('profile', {}).get('years_of_experience', 0)
        skills = candidate.get('skills', [])
        history = candidate.get('career_history', [])
        
        max_possible_months = int(years * 12) + 3
        
        has_skill_overflow = False
        for skill in skills:
            if skill.get('duration_months', 0) > max_possible_months * 1.05:
                has_skill_overflow = True
                break
                
        total_months = sum(job.get('duration_months', 0) for job in history)
        has_timeline_overlap = total_months > max_possible_months * 1.5
        
        is_actual_honeypot = has_skill_overflow or has_timeline_overlap

        if is_detected and is_actual_honeypot:
            honeypot_detections += 1
            if has_skill_overflow:
                honeypot_rules_triggered['skill_overflow'] += 1
            if has_timeline_overlap:
                honeypot_rules_triggered['timeline_overlap'] += 1
        elif not is_detected and not is_actual_honeypot:
            real_detections += 1
        elif is_detected and not is_actual_honeypot:
            false_positives += 1
        elif not is_detected and is_actual_honeypot:
            false_negatives += 1

    total_tested = 500

    # Calculate metrics
    honeypot_count = honeypot_detections + false_negatives
    real_count = real_detections + false_positives

    detection_rate = honeypot_detections / honeypot_count if honeypot_count > 0 else 0
    false_pos_rate = false_positives / real_count if real_count > 0 else 0
    false_neg_rate = false_negatives / honeypot_count if honeypot_count > 0 else 0
    accuracy = (honeypot_detections + real_detections) / total_tested

    print(f"\nTest Sample: {total_tested} candidates")
    print(f"  - Honeypots: {honeypot_count}")
    print(f"  - Real candidates: {real_count}")

    print(f"\nDetection Metrics:")
    print(f"  - True Positives (honeypots detected):   {honeypot_detections}")
    print(f"  - True Negatives (real candidates OK):   {real_detections}")
    print(f"  - False Positives (real marked honeypot): {false_positives}")
    print(f"  - False Negatives (honeypot missed):      {false_negatives}")

    print(f"\nPerformance Rates:")
    print(f"  - Detection Rate (TPR):    {detection_rate:.4f} ({detection_rate*100:.2f}%)")
    print(f"  - False Positive Rate:     {false_pos_rate:.4f} ({false_pos_rate*100:.2f}%)")
    print(f"  - False Negative Rate:     {false_neg_rate:.4f} ({false_neg_rate*100:.2f}%)")
    print(f"  - Overall Accuracy:        {accuracy:.4f} ({accuracy*100:.2f}%)")

    print(f"\nRules Triggered:")
    for rule, count in sorted(honeypot_rules_triggered.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {rule}: {count} times")

    # Save validation results
    validation_report = {
        'test_sample_size': total_tested,
        'honeypot_count': int(honeypot_count),
        'real_count': int(real_count),
        'true_positives': int(honeypot_detections),
        'true_negatives': int(real_detections),
        'false_positives': int(false_positives),
        'false_negatives': int(false_negatives),
        'detection_rate': float(detection_rate),
        'false_positive_rate': float(false_pos_rate),
        'false_negative_rate': float(false_neg_rate),
        'overall_accuracy': float(accuracy),
        'rules_triggered': honeypot_rules_triggered
    }

    with open(os.path.join(output_dir, 'honeypot_validation.json'), 'w') as f:
        json.dump(validation_report, f, indent=2)

    print(f"\nOK: Honeypot validation saved to {output_dir}/honeypot_validation.json")
    print(f"\nValidation Status:")
    if detection_rate > 0.8:
        print(f"  OK: Honeypot detection is RELIABLE (detection rate > 80%)")
    else:
        print(f"  WARNING: Honeypot detection needs improvement (detection rate < 80%)")

    if false_pos_rate < 0.2:
        print(f"  OK: False positive rate is LOW (< 20%)")
    else:
        print(f"  WARNING: False positive rate is HIGH (> 20%)")

    print("="*80 + "\n")

    return validation_report
