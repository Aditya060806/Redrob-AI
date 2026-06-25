import json
import random
import os
import sys

def load_candidates(filepath, sample_size=250):
    candidates = []
    if not os.path.exists(filepath):
        print(f"Error: Could not find candidates file at {filepath}")
        print("Please ensure you have placed candidates.jsonl in the data/ directory.")
        sys.exit(1)
        
    print(f"Loading candidates from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))
                
    random.seed(42)  # Fixed seed for reproducibility
    random.shuffle(candidates)
    return candidates[:sample_size]

def load_progress(output_file):
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_progress(labeled_data, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(labeled_data, f, indent=2, ensure_ascii=False)

def print_candidate(cand, current, total):
    print("\n" + "="*80)
    print(f" CANDIDATE [{current}/{total}] - ID: {cand.get('candidate_id', 'UNKNOWN')} ")
    print("="*80)
    
    profile = cand.get('profile', {})
    print(f"\n[PROFILE]")
    print(f"Name/ID: {profile.get('anonymized_name', 'N/A')}")
    print(f"Title:   {profile.get('current_title', 'N/A')} @ {profile.get('current_company', 'N/A')}")
    print(f"Exp:     {profile.get('years_of_experience', 0)} years")
    
    summary = profile.get('summary', '')
    if summary:
        print(f"Summary: {summary[:300]}..." if len(summary) > 300 else f"Summary: {summary}")

    print(f"\n[TOP SKILLS]")
    skills = cand.get('skills', [])
    skill_names = [f"{s.get('name', '')} ({s.get('proficiency', '')})" for s in skills[:10]]
    print(", ".join(skill_names) if skill_names else "None listed")

    print(f"\n[RECENT EXPERIENCE]")
    history = cand.get('career_history', [])
    for job in history[:3]:  # Show last 3 jobs
        duration = job.get('duration_months', 0)
        years = duration // 12
        months = duration % 12
        dur_str = f"{years}y {months}m" if years > 0 else f"{months}m"
        print(f"- {job.get('title', 'N/A')} @ {job.get('company', 'N/A')} ({dur_str})")
        desc = job.get('description', '')
        if desc:
            print(f"  > {desc[:150]}..." if len(desc) > 150 else f"  > {desc}")

    print(f"\n[SIGNALS]")
    signals = cand.get('redrob_signals', {})
    print(f"GitHub Score: {signals.get('github_activity_score', 'N/A')}")
    print(f"Response Rate: {signals.get('recruiter_response_rate', 0)*100:.0f}%")
    print(f"Notice Period: {signals.get('notice_period_days', 'N/A')} days")
    print("="*80)

def main():
    # Setup paths relative to the project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'candidates.jsonl')
    out_path = os.path.join(base_dir, 'labeling', 'labeled_candidates.json')
    
    candidates = load_candidates(data_path, sample_size=250)
    labeled_data = load_progress(out_path)
    
    # Track which IDs have already been labeled
    labeled_ids = {item['candidate_id'] for item in labeled_data}
    
    print(f"\nFound {len(labeled_data)} already labeled candidates.")
    print(f"Resuming labeling session. Target: 250 candidates.\n")
    
    count = len(labeled_data)
    
    for cand in candidates:
        cand_id = cand.get('candidate_id')
        if cand_id in labeled_ids:
            continue
            
        print_candidate(cand, count + 1, len(candidates))
        
        while True:
            print("\nRATE THIS CANDIDATE FOR 'Founding Senior AI Engineer'")
            print("  0 = Not qualified (Missing core skills, too junior, consulting only)")
            print("  1 = Borderline (Some gaps, but might be worth a look)")
            print("  2 = Strong (Good fit, 4-5 yrs applied ML, solid skills)")
            print("  3 = Perfect (Built RAG/vector db at scale, product company, ready to hire)")
            print("  q = Quit and save progress")
            
            choice = input("\nYour rating (0, 1, 2, 3, or q): ").strip().lower()
            
            if choice == 'q':
                print(f"\nSaving progress ({count} labeled). Exiting.")
                sys.exit(0)
                
            if choice in ['0', '1', '2', '3']:
                score = int(choice)
                
                labeled_item = {
                    'candidate_id': cand_id,
                    'relevance_score': score,
                    'raw_profile': cand 
                }
                
                labeled_data.append(labeled_item)
                labeled_ids.add(cand_id)
                save_progress(labeled_data, out_path)
                count += 1
                break
            else:
                print("Invalid input. Please enter 0, 1, 2, 3, or q.")
                
    print(f"\n Congratulations! You have labeled {count} candidates.")

if __name__ == "__main__":
    main()
