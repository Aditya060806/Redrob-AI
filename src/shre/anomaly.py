"""
Feature 3: Enhanced Honeypot / Anomaly Detection (pre-filter + signal).

Replaces the brittle 1.05x / 1.5x heuristics in stage1_filter with a richer,
multi-signal detector that catches:
  * timeline anomalies  - overlapping jobs, end-before-start, future dates,
                          summed tenure far exceeding career length
  * skill anomalies     - skills claimed longer than the whole career,
                          endorsements wildly exceeding network size
  * synthetic/fake flags - inverted salary ranges, OSS claims with no GitHub,
                          degenerate assessment scores, last-active before
                          signup, stale/abandoned profiles, zero-trust profiles

It returns a continuous anomaly_score in [0, 1], a list of human-readable
flags (used in reasoning), and an `is_synthetic` decision used as a hard
pre-filter. The score and flag count are also fed into the feature vector so
the ensemble / LTR head can learn from them.
"""

from datetime import datetime

from src.common.config import ANOMALY_THRESHOLDS as T


def _parse_date(value):
    """Parse an ISO 'YYYY-MM-DD' date; return None on any failure/None."""
    if not value or not isinstance(value, str):
        return None
    for fmt in ('%Y-%m-%d', '%Y-%m', '%Y'):
        try:
            return datetime.strptime(value[:10], fmt)
        except ValueError:
            continue
    return None


# Phrases that imply meaningful open-source / GitHub activity.
_OSS_PHRASES = (
    'open source', 'open-source', 'oss', 'github', 'maintainer',
    'contributor to', 'published a library', 'popular repo', 'stars on',
)


class AnomalyDetector:
    """Detects timeline / skill / synthetic anomalies in a raw candidate dict."""

    # Reference "now" for recency checks. Pinned to the dataset era so behaviour
    # is deterministic and reproducible regardless of wall-clock at run time.
    REFERENCE_NOW = datetime(2026, 6, 1)

    def analyze(self, candidate):
        """Return {anomaly_score, flags, is_synthetic, components}."""
        flags = []
        # Each check contributes a weighted penalty in [0, 1].
        penalties = {}

        penalties['timeline'] = self._timeline_penalty(candidate, flags)
        penalties['skills'] = self._skill_penalty(candidate, flags)
        penalties['synthetic'] = self._synthetic_penalty(candidate, flags)

        # Weighted blend -> overall anomaly score.
        score = min(1.0, (
            0.40 * penalties['timeline'] +
            0.30 * penalties['skills'] +
            0.45 * penalties['synthetic']
        ))

        is_synthetic = (
            score >= T['synthetic_score_cutoff'] or
            penalties['timeline'] >= 0.99 or
            penalties['skills'] >= 0.99
        )

        return {
            'anomaly_score': round(score, 4),
            'flags': flags,
            'is_synthetic': bool(is_synthetic),
            'components': penalties,
        }

    # --- convenience boolean used by the fast pre-filter -------------------
    def is_synthetic(self, candidate):
        return self.analyze(candidate)['is_synthetic']

    # --- individual penalty checks ----------------------------------------
    def _timeline_penalty(self, candidate, flags):
        profile = candidate.get('profile', {})
        history = candidate.get('career_history', []) or []
        years = profile.get('years_of_experience', 0) or 0
        max_months = int(years * 12) + T['skill_overflow_grace_months']

        penalty = 0.0

        # 1. Summed tenure far exceeding career length => impossible/overlap.
        total_months = sum(j.get('duration_months', 0) or 0 for j in history)
        if max_months > 0 and total_months > max_months * T['timeline_overflow_ratio']:
            penalty = max(penalty, 1.0)
            flags.append('timeline: claimed tenure exceeds career length')

        # 2. Per-job date sanity: end < start, or future dates.
        bad_dates = 0
        intervals = []
        for j in history:
            start = _parse_date(j.get('start_date'))
            end = _parse_date(j.get('end_date'))
            if start and end and end < start:
                bad_dates += 1
            if start and start > self.REFERENCE_NOW:
                bad_dates += 1
            if end and end > self.REFERENCE_NOW:
                bad_dates += 1
            if start:
                intervals.append((start, end or self.REFERENCE_NOW))
        if bad_dates:
            penalty = max(penalty, 0.8)
            flags.append('timeline: invalid or future job dates')

        # 3. Overlapping employment windows beyond a small tolerance.
        intervals.sort()
        overlap_months = 0.0
        for (s1, e1), (s2, e2) in zip(intervals, intervals[1:]):
            if s2 < e1:
                overlap_months += (e1 - s2).days / 30.0
        if overlap_months > T['overlap_tolerance_months']:
            penalty = max(penalty, min(1.0, overlap_months / 24.0))
            flags.append('timeline: overlapping employment periods')

        return penalty

    def _skill_penalty(self, candidate, flags):
        profile = candidate.get('profile', {})
        skills = candidate.get('skills', []) or []
        signals = candidate.get('redrob_signals', {})
        years = profile.get('years_of_experience', 0) or 0
        max_months = (int(years * 12) + T['skill_overflow_grace_months'])

        penalty = 0.0

        # 1. A skill claimed for longer than the whole career.
        if max_months > 0:
            for s in skills:
                dur = s.get('duration_months', 0) or 0
                if dur > max_months * T['skill_overflow_ratio']:
                    penalty = max(penalty, 1.0)
                    flags.append('skills: skill duration exceeds total experience')
                    break

        # 2. Endorsements wildly exceeding the network size (bought/fake).
        connections = signals.get('connection_count', 0) or 0
        endorsements = signals.get('endorsements_received', 0) or 0
        if connections >= 10 and endorsements > connections * T['endorsement_to_connection_ratio']:
            penalty = max(penalty, 0.5)
            flags.append('skills: endorsements disproportionate to network')

        return penalty

    def _synthetic_penalty(self, candidate, flags):
        signals = candidate.get('redrob_signals', {})
        profile = candidate.get('profile', {})
        penalty = 0.0

        # 1. Inverted salary range (min > max) => corrupt/synthetic record.
        sal = signals.get('expected_salary_range_inr_lpa', {}) or {}
        smin, smax = sal.get('min'), sal.get('max')
        if smin is not None and smax is not None and smin > smax > 0:
            penalty = max(penalty, 0.7)
            flags.append('synthetic: inverted salary range')

        # 2. OSS claims in the summary but no linked GitHub.
        summary = (profile.get('summary', '') or '').lower()
        github = signals.get('github_activity_score', -1)
        if github == -1 and any(p in summary for p in _OSS_PHRASES):
            penalty = max(penalty, 0.35)
            flags.append('synthetic: open-source claims without GitHub link')

        # 3. Degenerate assessment scores (all identical across >=3 tests).
        scores = list((signals.get('skill_assessment_scores', {}) or {}).values())
        if len(scores) >= 3 and len(set(round(float(x)) for x in scores)) == 1:
            penalty = max(penalty, 0.3)
            flags.append('synthetic: identical assessment scores')

        # 4. last_active before signup => impossible activity record.
        signup = _parse_date(signals.get('signup_date'))
        last_active = _parse_date(signals.get('last_active_date'))
        if signup and last_active and last_active < signup:
            penalty = max(penalty, 0.5)
            flags.append('synthetic: active before signup')

        # 5. Stale / abandoned profile.
        if last_active:
            stale_days = (self.REFERENCE_NOW - last_active).days
            if stale_days > T['stale_days']:
                penalty = max(penalty, 0.25)
                flags.append('synthetic: profile inactive for over a year')

        # 6. Zero-trust profile: nothing verified AND no platform activity.
        nothing_verified = not (
            signals.get('verified_email') or signals.get('verified_phone') or
            signals.get('linkedin_connected')
        )
        no_activity = (
            (signals.get('profile_views_received_30d', 0) or 0) == 0 and
            (signals.get('search_appearance_30d', 0) or 0) == 0 and
            (signals.get('connection_count', 0) or 0) == 0
        )
        if nothing_verified and no_activity:
            penalty = max(penalty, 0.4)
            flags.append('synthetic: unverified with no platform activity')

        return penalty
