"""
Deep Job Understanding (PS requirement #1).

Turns a *raw, free-text job description* into the structured artifacts the rest
of the pipeline consumes:

  * three semantic FACETS  (required_skills / ideal_experience / role_mission)
    that the Multi-Vector Semantic Layer matches candidate views against, and
  * a parsed EXPERIENCE BAND (min / max / target years) that Stage-1 gating uses.

Design goals
------------
* **Zero network, deterministic.** Parsing is pure heuristics (section + regex
  detection), so it runs offline and reproducibly during ranking.
* **Backwards compatible.** `JobDescription.default()` reproduces the hardcoded
  "Founding Senior AI Engineer" role exactly, so behaviour is unchanged when no
  JD is supplied (and it stays aligned with the 498 labels, which were judged
  for that role).
* **Graceful.** If a section can't be found we fall back to the full JD text so
  the encoder always has signal; if a band can't be parsed we keep the default.

The honest limitation: the supervised ensemble is still trained on labels for
the founding-engineer role. A custom JD re-targets the *semantic* fit signal
(which is JD-relative) and the *hard experience gate*, but the learned
"higher fit -> higher relevance" relationship is what transfers across roles.
"""

import os
import re

from src.common.config import (
    JD_FACETS,
    MIN_YEARS_EXP,
    MAX_YEARS_EXP,
    TARGET_YEARS_MIN,
    TARGET_YEARS_MAX,
)

# Header cues used to slice a JD into facets. Matched case-insensitively as a
# line that *looks like* a heading (short, often ending in ':' or a bare label).
_SKILL_HEADERS = (
    'required skills', 'requirements', 'qualifications', 'must have',
    'must-have', 'skills', 'tech stack', 'technical requirements',
    'what you bring', 'what we are looking for', 'what we’re looking for',
    'who you are', 'your profile', 'minimum qualifications',
)
_MISSION_HEADERS = (
    'responsibilities', 'role', 'about the role', 'about the job',
    'what you will do', 'what you’ll do', 'what you will be doing',
    'the role', 'job description', 'overview', 'the mission', 'mission',
    'your impact', 'day to day', 'day-to-day',
)
_EXPERIENCE_HEADERS = (
    'experience', 'who we want', 'seniority', 'ideal candidate',
)

# Skill keywords used to harvest a skills facet when there is no clear section.
_SKILL_HINTS = (
    'python', 'pytorch', 'tensorflow', 'sql', 'rag', 'llm', 'llms', 'vector',
    'embedding', 'embeddings', 'faiss', 'pinecone', 'milvus', 'weaviate',
    'qdrant', 'transformer', 'transformers', 'nlp', 'mlops', 'kubernetes',
    'docker', 'spark', 'airflow', 'kafka', 'fine-tune', 'fine-tuning',
    'prompt', 'semantic search', 'retrieval', 'inference', 'deployment',
    'machine learning', 'deep learning', 'recommendation', 'ranking',
)


def _looks_like_header(line):
    """Cheap heuristic: a heading is a short line, optionally ending in ':'."""
    s = line.strip().rstrip(':').strip()
    if not s:
        return None
    if len(s.split()) > 7:
        return None
    low = s.lower()
    for grp, headers in (('skills', _SKILL_HEADERS),
                         ('mission', _MISSION_HEADERS),
                         ('experience', _EXPERIENCE_HEADERS)):
        if any(low == h or low.startswith(h) for h in headers):
            return grp
    return None


def _split_sections(text):
    """Group JD lines into {'skills','mission','experience','body'} buckets."""
    buckets = {'skills': [], 'mission': [], 'experience': [], 'body': []}
    current = 'body'
    for raw_line in text.splitlines():
        header = _looks_like_header(raw_line)
        if header:
            current = header
            continue
        if raw_line.strip():
            buckets[current].append(raw_line.strip())
    return {k: ' '.join(v).strip() for k, v in buckets.items()}


def _parse_experience_band(text):
    """
    Extract (min, max, target_min, target_max) years from the JD text.

    Returns None if nothing parseable is found.
    """
    low = text.lower()

    # "5-9 years", "5 to 9 years", "5–9 yrs"
    m = re.search(r'(\d{1,2})\s*(?:-|–|to)\s*(\d{1,2})\s*(?:\+)?\s*(?:years|yrs|yr)', low)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        if 0 < lo <= hi <= 50:
            # Hard gate widens around the stated target band by a couple years.
            return (max(0.0, lo - 2.0), float(hi + 6), float(lo), float(hi))

    # "5+ years", "at least 5 years", "minimum 5 years"
    m = re.search(r'(?:at least|minimum of|minimum|min\.?|over)?\s*(\d{1,2})\s*\+?\s*(?:years|yrs|yr)', low)
    if m:
        lo = int(m.group(1))
        if 0 < lo <= 50:
            return (max(0.0, lo - 1.0), float(lo + 10), float(lo), float(lo + 4))

    return None


class JobDescription:
    """Structured, semantic-ready view of a target role."""

    def __init__(self, facets, min_years, max_years,
                 target_min, target_max, raw='', source='default'):
        self.facets = facets                # {required_skills, ideal_experience, role_mission}
        self.min_years = float(min_years)
        self.max_years = float(max_years)
        self.target_min = float(target_min)
        self.target_max = float(target_max)
        self.raw = raw
        self.source = source                # 'default' | 'text' | 'file'

    # -- constructors -------------------------------------------------------
    @classmethod
    def default(cls):
        """The original hardcoded 'Founding Senior AI Engineer' role."""
        return cls(
            facets=dict(JD_FACETS),
            min_years=MIN_YEARS_EXP, max_years=MAX_YEARS_EXP,
            target_min=TARGET_YEARS_MIN, target_max=TARGET_YEARS_MAX,
            raw='', source='default',
        )

    @classmethod
    def from_text(cls, text):
        """Parse a raw JD string into facets + experience band."""
        text = (text or '').strip()
        if not text:
            return cls.default()

        sections = _split_sections(text)
        default_facets = dict(JD_FACETS)

        # required_skills facet: explicit skills section, else harvest skill
        # sentences from the body, else the default.
        skills_text = sections['skills']
        if not skills_text:
            skills_text = _harvest_skill_text(text)
        required_skills = skills_text or default_facets['required_skills']

        # role_mission facet: the responsibilities/about section, else the
        # first chunk of the JD body, else the default.
        mission_text = sections['mission'] or sections['body']
        role_mission = (mission_text[:1200].strip() if mission_text
                        else default_facets['role_mission'])

        # ideal_experience facet: experience section + any stated band, else
        # the body, else the default.
        band = _parse_experience_band(text)
        exp_text = sections['experience']
        if band:
            lo, hi, tlo, thi = band
            band_sentence = (f"{int(tlo)} to {int(thi)} years of relevant "
                             f"professional experience.")
            exp_text = (band_sentence + ' ' + exp_text).strip()
        if not exp_text:
            exp_text = sections['body'][:600].strip()
        ideal_experience = exp_text or default_facets['ideal_experience']

        facets = {
            'required_skills': required_skills,
            'ideal_experience': ideal_experience,
            'role_mission': role_mission,
        }

        if band:
            lo, hi, tlo, thi = band
        else:
            lo, hi = MIN_YEARS_EXP, MAX_YEARS_EXP
            tlo, thi = TARGET_YEARS_MIN, TARGET_YEARS_MAX

        return cls(facets=facets, min_years=lo, max_years=hi,
                   target_min=tlo, target_max=thi, raw=text, source='text')

    @classmethod
    def from_source(cls, jd_arg):
        """
        Build from a CLI/UI argument that may be a file path, raw text, or None.
        """
        if not jd_arg:
            return cls.default()
        if isinstance(jd_arg, str) and os.path.isfile(jd_arg):
            with open(jd_arg, 'r', encoding='utf-8') as f:
                obj = cls.from_text(f.read())
            obj.source = 'file'
            return obj
        return cls.from_text(str(jd_arg))

    def describe(self):
        """Short human-readable summary for logs / UI."""
        return (f"JD[{self.source}] exp {self.min_years:g}-{self.max_years:g}y "
                f"(target {self.target_min:g}-{self.target_max:g}y); "
                f"facets: skills={len(self.facets['required_skills'])} chars, "
                f"mission={len(self.facets['role_mission'])} chars")


def _harvest_skill_text(text):
    """Collect sentences that mention concrete skills, as a skills-facet proxy."""
    sentences = re.split(r'(?<=[.!?])\s+|\n', text)
    hits = [s.strip() for s in sentences
            if any(k in s.lower() for k in _SKILL_HINTS)]
    return ' '.join(hits).strip()
