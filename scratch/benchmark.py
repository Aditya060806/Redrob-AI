import os
import csv
import sys

def load_csv(filepath):
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

def main():
    print("=== SHRE vs CTAE Benchmark Tool ===")
    print("This tool compares the final output of the two models.")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    shre_path = os.path.join(base_dir, 'output', 'submission.csv')
    ctae_path = os.path.join(base_dir, 'output', 'ctae_submission.csv') # Placeholder if we run CTAE separately
    
    shre_res = load_csv(shre_path)
    ctae_res = load_csv(ctae_path)
    
    if not shre_res:
        print(f"Could not find SHRE output at {shre_path}. Run the pipeline first.")
        return
        
    if not ctae_res:
        print(f"Could not find CTAE output at {ctae_path}. (You may need to force a CTAE run to benchmark).")
        print("For now, here are stats on your current submission:")
        
        # Just analyze the current submission
        scores = [float(row['score']) for row in shre_res]
        print(f"Top Score: {max(scores):.4f}")
        print(f"Bottom Score: {min(scores):.4f}")
        
        # Check reasoning quality
        reasonings = [row['reasoning'] for row in shre_res]
        avg_len = sum(len(r) for r in reasonings) / len(reasonings) if reasonings else 0
        print(f"Average Reasoning Length: {avg_len:.0f} chars")
        return
        
    # If both exist, do an overlap comparison
    shre_ids = {row['candidate_id'] for row in shre_res}
    ctae_ids = {row['candidate_id'] for row in ctae_res}
    
    overlap = len(shre_ids.intersection(ctae_ids))
    print(f"\nOverlap in Top 100: {overlap}%")
    print("A lower overlap means the ML model (SHRE) is finding candidates the rule-based model (CTAE) missed.")

if __name__ == '__main__':
    main()
