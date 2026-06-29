import sys
from src.common.config import MIN_YEARS_EXP, MAX_YEARS_EXP, SKILL_PILLARS
from src.shre.anomaly import AnomalyDetector


class FastFilter:
    """
    Stage 1 pre-filter. Uses the Enhanced Anomaly Detector (Feature 3) to drop
    synthetic / honeypot profiles, then gates on experience band and a minimum
    number of skill pillars. Anomaly results are cached and stashed on each
    surviving candidate (`_anomaly`) so later stages / reasoning can reuse them
    without recomputing.

    The experience band is JD-driven (deep job understanding): if a parsed
    JobDescription is supplied its min/max years gate candidates; otherwise we
    fall back to the canonical 3-15 year band from config.
    """

    def __init__(self, jd=None):
        self.detector = AnomalyDetector()
        if jd is None:
            self.min_years = MIN_YEARS_EXP
            self.max_years = MAX_YEARS_EXP
        else:
            self.min_years = jd.min_years
            self.max_years = jd.max_years

    def filter(self, candidates):
        viable = []
        for cand in candidates:
            result = self.detector.analyze(cand)

            # Enhanced honeypot/synthetic pre-filter.
            if result['is_synthetic']:
                continue

            years = cand.get('profile', {}).get('years_of_experience', 0)
            if not (self.min_years <= years <= self.max_years):
                continue

            hits = self._count_pillar_hits(cand)
            if hits < 2:
                continue

            cand['_anomaly'] = result  # reuse downstream (features + reasoning)
            viable.append(cand)
        return viable

    def is_honeypot(self, candidate):
        """Back-compat shim: now backed by the enhanced anomaly detector."""
        return self.detector.is_synthetic(candidate)

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
