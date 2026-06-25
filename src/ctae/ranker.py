import os
import json
import csv
import re
from src.common.config import (
    MIN_YEARS_EXP,
    MAX_YEARS_EXP,
    TARGET_YEARS_MIN,
    TARGET_YEARS_MAX,
    SKILL_PILLARS,
    CONSULTING_COMPANIES
)

def is_honeypot(cand):
    profile = cand.get('profile', {})
    years = profile.get('years_of_experience', 0)
    skills = cand.get('skills', [])
    history = cand.get('career_history', [])
    
    max_possible_months = int(years * 12) + 3
    for skill in skills:
        if skill.get('duration_months', 0) > max_possible_months * 1.05:
            return True
            
    total_months = sum(job.get('duration_months', 0) for job in history)
    if total_months > max_possible_months * 1.5:
        return True
        
    return False

def count_pillar_hits(cand):
    profile = cand.get('profile', {})
    combined_text = (
        profile.get('summary', '') + ' ' +
        ' '.join(s.get('name', '') for s in cand.get('skills', [])) + ' ' +
        ' '.join(j.get('description', '') for j in cand.get('career_history', []))
    ).lower()
    
    hits = 0
    for pillar, keywords in SKILL_PILLARS.items():
        if any(kw in combined_text for kw in keywords):
            hits += 1
    return hits

def evaluate_candidate(cand):
    profile = cand.get('profile', {})
    signals = cand.get('redrob_signals', {})
    
    # 1. Experience score
    years = profile.get('years_of_experience', 0)
    if not (MIN_YEARS_EXP <= years <= MAX_YEARS_EXP):
        return 0.0
        
    if TARGET_YEARS_MIN <= years <= TARGET_YEARS_MAX:
        exp_score = 1.0
    else:
        exp_score = 0.5
        
    # 2. Skill score
    hits = count_pillar_hits(cand)
    if hits < 2:
        return 0.0 # Must have at least 2 pillars to be viable
        
    skill_score = hits / 4.0 # Scale to [0.5, 1.0]
    
    # 3. Company type score
    current_company = profile.get('current_company', '').lower()
    current_title = profile.get('current_title', '').lower()
    
    is_consulting = any(c in current_company for c in CONSULTING_COMPANIES) or "consultant" in current_title
    company_score = 0.5 if is_consulting else 1.0
    
    # 4. Behavioral score
    resp_rate = signals.get('recruiter_response_rate', 0.5)
    github_score = signals.get('github_activity_score', 0) / 100.0
    behavioral_score = (resp_rate + github_score) / 2.0
    
    # 5. Notice Period Penalty
    notice_days = signals.get('notice_period_days', 0)
    notice_penalty = 1.0
    if notice_days > 90:
        notice_penalty = 0.7
    elif notice_days > 60:
        notice_penalty = 0.85
        
    # Final composite score scaled to [0.0, 1.0] range
    raw_score = (exp_score * 1.2 + skill_score * 1.0 + company_score * 0.5 + behavioral_score * 0.3) * notice_penalty
    return round(raw_score / 3.0, 4)

def generate_reasoning(cand):
    """
    Generates specific, detailed, non-hallucinated reasoning from actual candidate data,
    phrased as a realistic and professional evaluation sentence for the CTAE fallback.
    """
    profile = cand.get('profile', {})
    signals = cand.get('redrob_signals', {})
    career = cand.get('career_history', [])

    title = profile.get('current_title', 'Engineer')
    years = profile.get('years_of_experience', 0)
    company = profile.get('current_company', 'N/A')

    # Start sentence
    intro = f"[CTAE Fallback] {title} with {years:.1f} years of experience at {company}"

    # Skills
    skills = cand.get('skills', [])
    key_skills = [s.get('name', '') for s in skills if s.get('name')][:3]
    
    focus = ""
    if key_skills:
        focus = f", with skills in {', '.join(key_skills)}"

    # Behavioral
    resp_rate = signals.get('recruiter_response_rate', 0)
    behavior = ""
    if resp_rate > 0.7:
        behavior = "; candidate has shown high responsiveness to recruiters"

    # Assemble
    reasoning = f"{intro}{focus}{behavior}."
    reasoning = reasoning.replace("..", ".").replace(" .", ".").replace("  ", " ").strip()
    
    return reasoning[:220]

def run_ctae_ranking(candidates_path, out_path):
    print(f"Loading candidates from {candidates_path}...")
    candidates = []
    with open(candidates_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))
                
    print(f"Filtering and scoring {len(candidates)} candidates...")
    scored_candidates = []
    for cand in candidates:
        if is_honeypot(cand):
            continue
            
        score = evaluate_candidate(cand)
        if score > 0:
            scored_candidates.append({
                'candidate_id': cand.get('candidate_id'),
                'score': score,
                'raw': cand
            })
            
    print(f"Scored {len(scored_candidates)} viable candidates.")
    
    # Sort: score descending, then candidate_id ascending
    scored_candidates.sort(key=lambda x: (-x['score'], x['candidate_id']))
    
    top_100 = scored_candidates[:100]
    
    print(f"Writing top 100 fallback results to {out_path}...")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
        for rank, item in enumerate(top_100, 1):
            writer.writerow([
                item['candidate_id'],
                rank,
                item['score'],
                generate_reasoning(item['raw'])
            ])
            
    # Also write full rankings
    full_out = os.path.join(os.path.dirname(out_path), 'rankings_full.csv')
    with open(full_out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
        for rank, item in enumerate(scored_candidates, 1):
            writer.writerow([
                item['candidate_id'],
                rank,
                item['score'],
                generate_reasoning(item['raw'])
            ])
            
    print("CTAE Fallback complete!")
