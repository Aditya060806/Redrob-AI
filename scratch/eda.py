import json
import os
from collections import Counter

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'candidates.jsonl')
    
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}. Please place the dataset there.")
        return
        
    print("Running Exploratory Data Analysis (EDA)...")
    companies = Counter()
    skills = Counter()
    titles = Counter()
    total = 0
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            cand = json.loads(line)
            total += 1
            
            profile = cand.get('profile', {})
            titles[profile.get('current_title', '').lower()] += 1
            
            for job in cand.get('career_history', []):
                comp_name = job.get('company', '').lower()
                if comp_name:
                    companies[comp_name] += 1
                
            for skill in cand.get('skills', []):
                skill_name = skill.get('name', '').lower()
                if skill_name:
                    skills[skill_name] += 1
                
    print(f"\n=== TOTAL CANDIDATES: {total} ===")
    
    print("\n=== TOP 15 COMPANIES (Used for PFAW Classification) ===")
    for comp, count in companies.most_common(15):
        print(f"  {comp}: {count}")
        
    print("\n=== TOP 15 SKILLS ===")
    for skill, count in skills.most_common(15):
        print(f"  {skill}: {count}")
        
    print("\n=== TOP 10 CURRENT TITLES ===")
    for title, count in titles.most_common(10):
        print(f"  {title}: {count}")

if __name__ == '__main__':
    main()
