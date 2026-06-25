import sys
from src.common.config import MIN_YEARS_EXP, MAX_YEARS_EXP, SKILL_PILLARS

class FastFilter:
    def filter(self, candidates):
        viable = []
        for cand in candidates:
            # Check honeypot
            if self.is_honeypot(cand):
                continue
                
            years = cand.get('profile', {}).get('years_of_experience', 0)
            if not (MIN_YEARS_EXP <= years <= MAX_YEARS_EXP):
                continue
                
            hits = self._count_pillar_hits(cand)
            if hits < 2:
                continue
                
            viable.append(cand)
        return viable
        
    def is_honeypot(self, candidate):
        years = candidate.get('profile', {}).get('years_of_experience', 0)
        skills = candidate.get('skills', [])
        history = candidate.get('career_history', [])
        
        # Skill overflow
        max_possible_months = int(years * 12) + 3
        for skill in skills:
            if skill.get('duration_months', 0) > max_possible_months * 1.05:
                return True
                
        # Timeline overlap (basic check)
        # Using a conservative 1.5x multiplier to avoid false positives with part-time/advisory
        total_months = sum(job.get('duration_months', 0) for job in history)
        if total_months > max_possible_months * 1.5:
            return True
            
        return False
        
    def _count_pillar_hits(self, candidate):
        combined_text = (
            candidate.get('profile', {}).get('summary', '') + ' ' +
            ' '.join(s.get('name', '') for s in candidate.get('skills', [])) + ' ' +
            ' '.join(j.get('description', '') for j in candidate.get('career_history', []))
        ).lower()
        
        hits = 0
        for pillar, keywords in SKILL_PILLARS.items():
            if any(kw in combined_text for kw in keywords):
                hits += 1
        return hits
