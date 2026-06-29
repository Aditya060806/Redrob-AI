"""
Feature 4: Behavioral Scoring Module.

RETRO's stage-2 features touch only a few engagement signals (response rate,
github, completeness). The Redrob schema exposes a much richer behavioral
surface that goes under-utilized: recruiter demand (search appearances, saves,
profile views), candidate activity & recency (last active, applications),
open-source / community signals (github, endorsements, connections), and
reliability (interview completion, offer acceptance, identity verification).

This module distills those into four interpretable sub-scores plus a composite,
all in [0, 1], which are emitted as features for the ensemble / LTR head and
surfaced in the reasoning output.
"""

from datetime import datetime

from src.common.config import BEHAVIORAL_NORMS as N


def _parse_date(value):
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value[:10], '%Y-%m-%d')
    except ValueError:
        return None


def _sat(value, ceiling):
    """Saturating normalization: value/ceiling clamped to [0, 1]."""
    if ceiling <= 0:
        return 0.0
    return max(0.0, min(1.0, float(value) / ceiling))


class BehavioralScorer:
    """Computes activity / demand / OSS / reliability behavioral sub-scores."""

    REFERENCE_NOW = datetime(2026, 6, 1)

    def score(self, candidate):
        """Return dict of sub-scores + composite (all in [0, 1])."""
        signals = candidate.get('redrob_signals', {}) or {}

        activity = self._activity_score(signals)
        demand = self._demand_score(signals)
        oss = self._oss_score(signals)
        reliability = self._reliability_score(signals)

        composite = (
            0.25 * activity +
            0.30 * demand +
            0.25 * oss +
            0.20 * reliability
        )

        return {
            'activity_score': round(activity, 4),
            'demand_score': round(demand, 4),
            'oss_score': round(oss, 4),
            'reliability_score': round(reliability, 4),
            'behavioral_composite': round(composite, 4),
        }

    # --- sub-scores --------------------------------------------------------
    def _activity_score(self, signals):
        """Recency + self-driven engagement (applications, response speed)."""
        last_active = _parse_date(signals.get('last_active_date'))
        if last_active:
            days = max(0, (self.REFERENCE_NOW - last_active).days)
            recency = max(0.0, 1.0 - days / max(1.0, N['recent_active_days'] * 4))
        else:
            recency = 0.3

        applications = _sat(signals.get('applications_submitted_30d', 0) or 0,
                            N['applications_30d'])

        resp_time = signals.get('avg_response_time_hours', 48) or 48
        responsiveness = max(0.0, 1.0 - min(1.0, resp_time / 96.0))

        open_to_work = 1.0 if signals.get('open_to_work_flag') else 0.0

        return (0.40 * recency + 0.20 * applications +
                0.20 * responsiveness + 0.20 * open_to_work)

    def _demand_score(self, signals):
        """How much the market (recruiters) pulls for this candidate."""
        views = _sat(signals.get('profile_views_received_30d', 0) or 0,
                     N['profile_views_30d'])
        appearances = _sat(signals.get('search_appearance_30d', 0) or 0,
                           N['search_appearance_30d'])
        saved = _sat(signals.get('saved_by_recruiters_30d', 0) or 0,
                     N['saved_by_recruiters_30d'])
        response = signals.get('recruiter_response_rate', 0.5)
        response = 0.5 if response is None else max(0.0, min(1.0, response))
        return (0.30 * views + 0.30 * appearances +
                0.25 * saved + 0.15 * response)

    def _oss_score(self, signals):
        """Open-source / community footprint."""
        github = signals.get('github_activity_score', -1)
        github = 0.25 if github is None or github < 0 else github / 100.0
        endorsements = _sat(signals.get('endorsements_received', 0) or 0,
                            N['endorsements_received'])
        connections = _sat(signals.get('connection_count', 0) or 0,
                           N['connection_count'])
        return 0.55 * github + 0.25 * endorsements + 0.20 * connections

    def _reliability_score(self, signals):
        """Follow-through + identity trust."""
        interview = signals.get('interview_completion_rate', 0.5)
        interview = 0.5 if interview is None else max(0.0, min(1.0, interview))

        offer = signals.get('offer_acceptance_rate', -1)
        offer = 0.5 if offer is None or offer < 0 else max(0.0, min(1.0, offer))

        verified = sum(1 for k in ('verified_email', 'verified_phone', 'linkedin_connected')
                       if signals.get(k)) / 3.0

        completeness = (signals.get('profile_completeness_score', 50) or 50) / 100.0

        return (0.30 * interview + 0.25 * offer +
                0.25 * verified + 0.20 * completeness)
