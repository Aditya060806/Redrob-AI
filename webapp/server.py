"""
Redrob SHRE — interactive web frontend (no Streamlit, no new deps).

A self-contained Python standard-library HTTP server. It does NOT retrain or
modify anything: it loads the already-trained model artifacts and runs
READ-ONLY inference, reusing the existing src/ pipeline modules untouched.

Run:  py -3.13 webapp/server.py      (then open http://localhost:8000)
"""

import os
import re
import sys
import copy
import json
import pickle
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

# --- make the project importable (read-only) -------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
WEBDIR = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(ROOT, 'models')

from src.shre.stage1_filter import FastFilter
from src.shre.stage2_features import FeatureEngineer
from src.shre.stage4_submit import _generate_reasoning
from src.shre.anomaly import AnomalyDetector
from src.common.config import (
    LTR_FUSION_ENSEMBLE_WEIGHT as FUSION_W,
    MIN_YEARS_EXP, MAX_YEARS_EXP, SKILL_PILLARS,
)

# --- lazy-loaded singletons (loaded once, reused) --------------------------
_ARTIFACTS = {}


def _artifacts():
    if not _ARTIFACTS:
        for key, fname in (('ensemble', 'ensemble_model_validated.pkl'),
                           ('scaler', 'scaler_validated.pkl'),
                           ('selector', 'selector_validated.pkl'),
                           ('ltr', 'ltr_model.pkl')):
            with open(os.path.join(MODELS, fname), 'rb') as f:
                _ARTIFACTS[key] = pickle.load(f)
    return _ARTIFACTS


def _minmax(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return a
    lo, hi = a.min(), a.max()
    return np.full_like(a, 0.5) if hi - lo < 1e-9 else (a - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# READ-ONLY inference — mirrors stage3 LTR fusion without retraining
# ---------------------------------------------------------------------------
def _pillar_hits(cand):
    text = (
        cand.get('profile', {}).get('summary', '') + ' ' +
        ' '.join(s.get('name', '') for s in cand.get('skills', [])) + ' ' +
        ' '.join(j.get('description', '') for j in cand.get('career_history', []))
    ).lower()
    return sum(1 for kws in SKILL_PILLARS.values() if any(k in text for k in kws))


def _name_of(cand):
    p = cand.get('profile', {})
    return p.get('anonymized_name') or p.get('current_title') or cand.get('candidate_id', 'Candidate')


def _gate(candidates):
    """Mirror FastFilter's gate but capture a human reason for each drop."""
    det = AnomalyDetector()
    viable, dropped = [], []
    for cand in candidates:
        res = det.analyze(cand)
        cand['_anomaly'] = res
        base = {'name': _name_of(cand), 'candidate_id': cand.get('candidate_id', '—'),
                'is_user': bool(cand.get('_is_user'))}
        if res['is_synthetic']:
            why = res['flags'][0] if res['flags'] else 'failed synthetic-profile checks'
            dropped.append({**base, 'reason': f'flagged as synthetic — {why}'})
            continue
        years = cand.get('profile', {}).get('years_of_experience', 0) or 0
        if not (MIN_YEARS_EXP <= years <= MAX_YEARS_EXP):
            dropped.append({**base, 'reason':
                            f'{years} yrs experience is outside the {MIN_YEARS_EXP:.0f}–{MAX_YEARS_EXP:.0f} yr range'})
            continue
        if _pillar_hits(cand) < 2:
            dropped.append({**base, 'reason':
                            'fewer than 2 of the required skill pillars (ML / vector-RAG / engineering / eval)'})
            continue
        viable.append(cand)
    return viable, dropped


def rank_candidates(candidates):
    """Score & rank raw candidate dicts. Returns (results, dropped, info)."""
    art = _artifacts()
    viable, dropped = _gate(candidates)
    info = {'received': len(candidates), 'viable': len(viable), 'dropped': len(dropped)}
    if not viable:
        return [], dropped, info

    feats = FeatureEngineer().compute_features(viable)  # 93 features + stashes
    X = np.nan_to_num(np.array([list(fv.values()) for _, fv in feats]),
                      nan=0.0, posinf=0.0, neginf=0.0)

    Xs = art['scaler'].transform(art['selector'].transform(X))
    proba = art['ensemble'].predict_proba(Xs)
    fit = np.sum(proba * np.array([0, 1, 2, 3]), axis=1) / 3.0  # absolute fit [0,1]
    Xl = np.hstack([Xs, fit.reshape(-1, 1)])
    ltr_raw = art['ltr'].predict(Xl)
    final = _minmax(FUSION_W * _minmax(fit) + (1 - FUSION_W) * _minmax(ltr_raw))

    order = np.argsort(-final)
    results = []
    for rank, i in enumerate(order, 1):
        cand = viable[i]
        prof = cand.get('profile', {})
        sem = cand.get('_semantic', {}) or {}
        beh = cand.get('_behavioral', {}) or {}
        anom = cand.get('_anomaly', {}) or {}
        results.append({
            'rank': rank,
            'candidate_id': cand.get('candidate_id', '—'),
            'name': prof.get('anonymized_name') or prof.get('current_title', 'Candidate'),
            'title': prof.get('current_title', ''),
            'company': prof.get('current_company', ''),
            'years': prof.get('years_of_experience', 0),
            'is_user': bool(cand.get('_is_user')),
            'fit_score': round(float(fit[i]) * 100, 1),
            'final_score': round(float(final[i]), 4),
            'semantic_fit': round(float(sem.get('semantic_fusion_score', 0)) * 100, 1),
            'behavioral': round(float(beh.get('behavioral_composite', 0)) * 100, 1),
            'anomaly_score': round(float(anom.get('anomaly_score', 0)) * 100, 1),
            'anomaly_flags': anom.get('flags', []),
            'reasoning': _generate_reasoning(cand),
        })
    return results, dropped, info


# ---------------------------------------------------------------------------
# Build a schema-compatible candidate dict from the interactive form
# ---------------------------------------------------------------------------
_PROF = {'expert', 'advanced', 'intermediate', 'beginner'}


def build_candidate(form, idx):
    years = float(form.get('years') or 0)
    tenure_m = max(1, int(years * 12))
    skill_dur = min(tenure_m, 30)  # keep skill durations <= tenure (no false honeypot)

    skills = []
    for name in [s.strip() for s in (form.get('skills') or '').split(',') if s.strip()]:
        skills.append({'name': name, 'proficiency': 'advanced',
                       'endorsements': 12, 'duration_months': skill_dur})

    title = (form.get('title') or 'Engineer').strip()
    company = (form.get('company') or 'N/A').strip()
    summary = (form.get('summary') or '').strip()
    desc = (form.get('description') or summary).strip()

    gh = form.get('github')
    gh = float(gh) if gh not in (None, '') else 60.0
    resp = form.get('response_rate')
    resp = float(resp) if resp not in (None, '') else 0.7
    notice = int(form.get('notice') or 60)
    open_w = bool(form.get('open_to_work', True))

    return {
        'candidate_id': f"USER_{idx:03d}_{secrets.token_hex(2)}",
        '_is_user': True,
        'profile': {
            'anonymized_name': (form.get('name') or title).strip(),
            'headline': f"{title} | {company}",
            'summary': summary,
            'location': form.get('location', 'India'),
            'country': 'India',
            'years_of_experience': years,
            'current_title': title,
            'current_company': company,
            'current_company_size': '201-500',
            'current_industry': form.get('industry', 'Technology'),
        },
        'career_history': [{
            'company': company, 'title': title,
            'start_date': '2021-01-01', 'end_date': None,
            'duration_months': tenure_m, 'is_current': True,
            'industry': form.get('industry', 'Technology'),
            'company_size': '201-500', 'description': desc,
        }],
        'education': [],
        'skills': skills,
        'redrob_signals': {
            'profile_completeness_score': 90, 'signup_date': '2021-02-01',
            'last_active_date': '2026-05-15', 'open_to_work_flag': open_w,
            'profile_views_received_30d': 30, 'applications_submitted_30d': 6,
            'recruiter_response_rate': resp, 'avg_response_time_hours': 12,
            'skill_assessment_scores': {s['name']: 80 for s in skills[:4]},
            'connection_count': 600, 'endorsements_received': 120,
            'notice_period_days': notice,
            'expected_salary_range_inr_lpa': {'min': 30, 'max': 50},
            'preferred_work_mode': 'hybrid', 'willing_to_relocate': True,
            'github_activity_score': gh, 'search_appearance_30d': 40,
            'saved_by_recruiters_30d': 8, 'interview_completion_rate': 0.9,
            'offer_acceptance_rate': 0.7, 'verified_email': True,
            'verified_phone': True, 'linkedin_connected': True,
        },
    }


# ---------------------------------------------------------------------------
# Forgiving ingestion — map common alternative field names onto the strict
# schema (data/candidate_schema.json) and coerce types, so a reasonable JSON
# "just works" instead of silently producing generic text or 500-crashing.
# ---------------------------------------------------------------------------
_TITLE_KEYS = ('current_title', 'title', 'role', 'job_title', 'jobTitle',
               'position', 'designation', 'headline')
_COMPANY_KEYS = ('current_company', 'company', 'employer', 'organization',
                 'organisation', 'org', 'current_employer')
_YEARS_KEYS = ('years_of_experience', 'years', 'experience_years', 'yoe',
               'total_experience', 'experience', 'years_experience',
               'yearsOfExperience')
_NAME_KEYS = ('anonymized_name', 'name', 'full_name', 'fullName', 'candidate_name')
_SUMMARY_KEYS = ('summary', 'bio', 'about', 'overview', 'profile_summary')
_INDUSTRY_KEYS = ('current_industry', 'industry', 'sector', 'domain')
_LOCATION_KEYS = ('location', 'city', 'region', 'place')
_SKILLNAME_KEYS = ('name', 'skill', 'skill_name', 'skillName', 'title')


def _pluck(dicts, keys):
    """First present, non-empty value for any of `keys` across `dicts` (in order)."""
    for d in dicts:
        if not isinstance(d, dict):
            continue
        for k in keys:
            v = d.get(k)
            if v not in (None, '', [], {}):
                return v
    return None


def _coerce_years(v):
    """Pull a numeric year count out of a number or a string like '8 years'."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = re.search(r'\d+(?:\.\d+)?', v)
        return float(m.group()) if m else None
    return None


def _normalize_skills(raw):
    """Coerce skills into the required list-of-objects shape (each with a name)."""
    if isinstance(raw, str):
        raw = [s for s in (p.strip() for p in raw.split(',')) if s]
    if isinstance(raw, dict):
        # e.g. {"Python": "expert", "RAG": "advanced"} -> name/proficiency objects
        raw = [{'name': k, 'proficiency': str(v)} for k, v in raw.items()]
    out = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                nm = item.strip()
                if nm:
                    out.append({'name': nm})
            elif isinstance(item, dict):
                s = dict(item)
                if 'name' not in s:
                    nm = _pluck((s,), _SKILLNAME_KEYS)
                    if nm:
                        s['name'] = nm
                if s.get('name'):
                    out.append(s)
    return out


def normalize_candidate(obj, idx=0):
    """
    Map a loosely-structured candidate dict onto the strict schema.

    Returns (candidate, error). `error` is a human-readable string when the
    object can't be recognized as a candidate at all (so the caller can reject
    it with a clear message instead of ranking generic defaults); otherwise it
    is None and `candidate` is schema-shaped enough for the pipeline.
    """
    label = f"Candidate {idx + 1}"
    if not isinstance(obj, dict):
        return None, f"{label}: expected a JSON object, got {type(obj).__name__}."

    cand = copy.deepcopy(obj)
    prof = dict(cand['profile']) if isinstance(cand.get('profile'), dict) else {}
    sources = (prof, cand)  # prefer values already under profile.*

    title = _pluck(sources, _TITLE_KEYS)
    company = _pluck(sources, _COMPANY_KEYS)
    years = _coerce_years(_pluck(sources, _YEARS_KEYS))
    name = _pluck(sources, _NAME_KEYS)
    summary = _pluck(sources, _SUMMARY_KEYS)
    industry = _pluck(sources, _INDUSTRY_KEYS)
    location = _pluck(sources, _LOCATION_KEYS)
    skills = _normalize_skills(cand.get('skills'))

    # Reject only when NOTHING identifying was found — otherwise the engine would
    # rank an all-defaults profile and emit the "hardcoded-looking" generic text.
    if not (title or (years is not None and years > 0) or skills):
        return None, (
            f"{label}: could not recognize any candidate fields (no title, "
            f"years of experience, or skills). Expected the shape in "
            f"data/candidate_schema.json, e.g. "
            f'{{"profile":{{"current_title":"...","years_of_experience":7}},'
            f'"skills":[{{"name":"RAG"}}]}}.')

    # Fill the canonical profile, preserving any already-correct values.
    prof.setdefault('current_title', title or 'Engineer')
    prof.setdefault('current_company', company or 'N/A')
    prof.setdefault('years_of_experience', years if years is not None else 0)
    prof.setdefault('anonymized_name', name or prof['current_title'])
    prof.setdefault('summary', summary or '')
    prof.setdefault('current_industry', industry or 'Technology')
    prof.setdefault('location', location or '')
    prof.setdefault('headline', f"{prof['current_title']} | {prof['current_company']}")
    cand['profile'] = prof
    cand['skills'] = skills

    # career_history / education must be lists of objects; synthesize a current
    # role from the profile when none was supplied (drives trajectory features).
    history = [j for j in (cand.get('career_history') or []) if isinstance(j, dict)]
    if not history:
        yrs = prof.get('years_of_experience', 0) or 0
        history = [{
            'company': prof['current_company'], 'title': prof['current_title'],
            'start_date': None, 'end_date': None,
            'duration_months': int(float(yrs) * 12) if yrs else 0,
            'is_current': True, 'industry': prof['current_industry'],
            'company_size': prof.get('current_company_size', ''),
            'description': prof.get('summary', ''),
        }]
    cand['career_history'] = history
    cand['education'] = [e for e in (cand.get('education') or []) if isinstance(e, dict)]

    sig = cand.get('redrob_signals')
    cand['redrob_signals'] = sig if isinstance(sig, dict) else {}
    cand.setdefault('candidate_id', f"USER_{idx + 1:03d}")
    return cand, None


def _strip_fences(text):
    """Remove markdown code fences (```json ... ```) if present."""
    text = text.strip()
    if text.startswith('```'):
        text = re.sub(r'^```[a-zA-Z]*\s*', '', text)
        text = re.sub(r'\s*```$', '', text.strip())
    return text.strip()


def _extract_json_objects(text):
    """Brace-match every top-level {...} object, ignoring braces inside strings."""
    objs, depth, start, in_str, esc = [], 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    objs.append(text[start:i + 1])
                    start = None
    return objs


def parse_candidates_text(text):
    """
    Parse candidate text in ANY common shape and return (candidates, errors):
    a JSON array, a single object, pretty-printed JSON, markdown-fenced JSON,
    or one-object-per-line JSONL.
    """
    text = _strip_fences(text or '')
    if not text.strip():
        return [], []

    # 1) Whole text is valid JSON (array or single object)?
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)], []
        if isinstance(data, dict):
            return [data], []
    except json.JSONDecodeError:
        pass

    # 2) Strict JSONL (one object per line)?
    cands, errors, jsonl_ok = [], [], True
    for ln, line in enumerate(text.splitlines(), 1):
        line = line.strip().rstrip(',')
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                cands.append(obj)
        except json.JSONDecodeError:
            jsonl_ok = False
            break
    if jsonl_ok and cands:
        return cands, []

    # 3) Last resort: brace-match every top-level object in the blob.
    cands = []
    for i, chunk in enumerate(_extract_json_objects(text), 1):
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict):
                cands.append(obj)
        except json.JSONDecodeError as e:
            errors.append(f"Object {i}: invalid JSON ({e.msg}).")
    if not cands and not errors:
        errors.append("Could not find any JSON candidate objects in the input.")
    return cands, errors


def load_sample_pool():
    path = os.path.join(ROOT, 'data', 'sample_candidates.jsonl')
    pool = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                pool.append(json.loads(line))
    return pool


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quieter console
        pass

    def _send(self, code, body, ctype='application/json'):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode('utf-8')
        elif isinstance(body, str):
            body = body.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            with open(os.path.join(WEBDIR, 'index.html'), 'rb') as f:
                self._send(200, f.read().decode('utf-8'), 'text/html; charset=utf-8')
        elif self.path == '/api/sample':
            self._send(200, load_sample_pool())
        else:
            self._send(404, {'error': 'not found'})

    def do_POST(self):
        if self.path != '/api/rank':
            return self._send(404, {'error': 'not found'})
        try:
            length = int(self.headers.get('Content-Length', 0))
            payload = json.loads(self.rfile.read(length) or b'{}')
        except Exception as e:
            return self._send(400, {'error': f'bad request: {e}'})

        candidates, errors, warnings = self._collect_candidates(payload)
        if errors:
            return self._send(400, {'error': '; '.join(errors)})

        try:
            results, dropped, info = rank_candidates(candidates)
            if warnings:
                info['warnings'] = warnings
            self._send(200, {'results': results, 'dropped': dropped, 'info': info})
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send(500, {'error': f'ranking failed: {e}'})

    def _collect_candidates(self, payload):
        """
        Assemble the candidate pool from the three input modes + options.

        Returns (candidates, errors, warnings). `errors` -> hard 400; `warnings`
        are non-fatal notes (e.g. some uploaded objects were unrecognizable but
        others ranked) surfaced alongside the results.
        """
        candidates, errors, warnings = [], [], []
        mode = payload.get('mode', 'form')

        if mode == 'form':
            for i, form in enumerate(payload.get('forms', [])):
                if not (form.get('title') and form.get('years') and form.get('skills')):
                    errors.append(f"Candidate {i + 1}: title, years and at least one skill are required.")
                    continue
                candidates.append(build_candidate(form, i))
        elif mode in ('paste', 'upload'):
            text = (payload.get('text') or '').strip()
            if not text:
                src = 'file' if mode == 'upload' else 'box'
                return [], [f"No {mode} content received — please choose/paste a {src} first."], []
            parsed, errors = parse_candidates_text(text)
            # Normalize every parsed object onto the schema; collect per-object
            # rejection reasons rather than silently ranking generic defaults.
            for i, obj in enumerate(parsed):
                cand, err = normalize_candidate(obj, i)
                if err:
                    warnings.append(err)
                    continue
                cand['_is_user'] = True
                candidates.append(cand)

        # Guard: if the user supplied input but nothing usable came out, surface
        # WHY (the normalization reasons) instead of silently ranking nothing or
        # falling back to the sample pool (that looked "hardcoded").
        if not candidates and not errors:
            errors = warnings or ["No candidates were found in your input."]
            warnings = []
        if errors:
            return [], errors, []

        if payload.get('include_pool'):
            candidates.extend(load_sample_pool())  # opt-in benchmark context
        return candidates, errors, warnings


def main():
    port = int(os.environ.get('PORT', '8000'))
    print("Loading model artifacts (read-only)...")
    _artifacts()
    print(f"\n  Redrob SHRE frontend running:")
    print(f"    ->  http://localhost:{port}\n  (Ctrl+C to stop)")
    ThreadingHTTPServer(('0.0.0.0', port), Handler).serve_forever()


if __name__ == '__main__':
    main()
