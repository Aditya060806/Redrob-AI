import numpy as np
from src.common.config import DOMAINS, CONSULTING_COMPANIES

class FeatureEngineer:
    """
    Transforms raw candidate profile into 150+ robust features.
    Handles dirty JSON schema gracefully with .get() and defaults.
    """
    
    def compute_features(self, candidates):
        """Returns (candidate_id, feature_vector) for each candidate"""
        features = []
        
        for cand in candidates:
            cand_id = cand.get('candidate_id', 'UNKNOWN')
            feature_vec = {}
            
            feature_vec.update(self._trajectory_features(cand))
            feature_vec.update(self._skill_features(cand))
            feature_vec.update(self._education_features(cand))
            feature_vec.update(self._engagement_features(cand))
            feature_vec.update(self._domain_features(cand))
            feature_vec.update(self._text_features(cand))
            feature_vec.update(self._company_features(cand))
            feature_vec.update(self._assessment_features(cand))
            feature_vec.update(self._interaction_features(feature_vec))
            
            features.append((cand_id, feature_vec))
            
        return features

    def _trajectory_features(self, candidate):
        features = {}
        profile = candidate.get('profile', {})
        career = candidate.get('career_history', [])
        years = profile.get('years_of_experience', 0)
        
        features['years_total'] = years
        features['years_squared'] = years ** 2
        features['num_jobs'] = len(career)
        features['avg_job_duration'] = sum(j.get('duration_months', 0) for j in career) / max(1, len(career))
        
        seniorities = [self._infer_seniority(j.get('title', '')) for j in career]
        features['current_seniority'] = seniorities[-1] if seniorities else 0
        features['max_seniority'] = max(seniorities) if seniorities else 0
        features['seniority_progression'] = (max(seniorities) - min(seniorities) if len(seniorities) > 1 else 0)
        features['is_upward_trajectory'] = float(all(seniorities[i] <= seniorities[i+1] for i in range(len(seniorities)-1))) if len(seniorities) > 1 else 0.5
        
        months_in_current = career[-1].get('duration_months', 0) if career else 0
        features['months_in_current_role'] = months_in_current
        features['is_current_role_stable'] = float(months_in_current >= 24)
        features['job_stability_ratio'] = features['avg_job_duration'] / max(12, months_in_current)
        
        # PFAW consulting check - penalize if currently at consulting company
        companies = [j.get('company', '').lower() for j in career]
        consulting_ratio = sum(1 for c in companies if any(cons in c for cons in CONSULTING_COMPANIES)) / max(1, len(companies))
        features['consulting_role_ratio'] = consulting_ratio

        # Heavy penalty if current role is consulting
        current_company = companies[-1] if companies else ''
        features['is_currently_consulting'] = float(any(cons in current_company for cons in CONSULTING_COMPANIES))
        features['consulting_penalty'] = 0.1 if features['is_currently_consulting'] else 1.0
        
        return features

    def _skill_features(self, candidate):
        features = {}
        skills = candidate.get('skills', [])
        assessments = candidate.get('redrob_signals', {}).get('skill_assessment_scores', {})
        
        features['num_skills'] = len(skills)
        features['num_expert_skills'] = sum(1 for s in skills if s.get('proficiency') == 'expert')
        features['num_advanced_skills'] = sum(1 for s in skills if s.get('proficiency') in ['expert', 'advanced'])
        
        total_endorsements = sum(s.get('endorsements', 0) for s in skills)
        features['total_endorsements'] = total_endorsements
        features['avg_endorsements_per_skill'] = total_endorsements / max(1, len(skills))
        
        skill_durations = [s.get('duration_months', 0) for s in skills]
        features['avg_skill_duration_months'] = sum(skill_durations) / max(1, len(skills))
        features['max_skill_duration_months'] = max(skill_durations) if skill_durations else 0
        
        # ML skills
        ml_keywords = ['ml', 'machine learning', 'deep learning', 'nlp', 'llm', 'neural']
        core_ml_skills = [s for s in skills if any(kw in s.get('name', '').lower() for kw in ml_keywords)]
        features['core_ml_skill_count'] = len(core_ml_skills)

        # Backend/infrastructure skills
        backend_keywords = ['python', 'java', 'c++', 'go', 'rust', 'spark', 'airflow', 'kafka', 'sql', 'postgres']
        backend_skills = [s for s in skills if any(kw in s.get('name', '').lower() for kw in backend_keywords)]
        features['backend_skill_count'] = len(backend_skills)

        # Vector/RAG skills
        vector_keywords = ['vector', 'milvus', 'pinecone', 'weaviate', 'faiss', 'rag', 'retrieval', 'embedding']
        vector_skills = [s for s in skills if any(kw in s.get('name', '').lower() for kw in vector_keywords)]
        features['vector_db_skill_count'] = len(vector_skills)

        # Has critical skills combo
        features['has_ml_and_backend'] = float(len(core_ml_skills) > 0 and len(backend_skills) > 0)
        features['has_ml_and_vector_db'] = float(len(core_ml_skills) > 0 and len(vector_skills) > 0)
        
        return features

    def _education_features(self, candidate):
        features = {}
        education = candidate.get('education', [])
        
        features['has_education_data'] = 1.0 if education else 0.0
        features['num_degrees'] = len(education)
        
        degree_ranking = {'phd': 4, 'masters': 3, 'bachelors': 2, 'diploma': 1, 'certificate': 0}
        degrees = [degree_ranking.get(e.get('degree', '').lower(), 0) for e in education]
        features['highest_degree_level'] = max(degrees) if degrees else 0
        
        tier_ranking = {'tier_1': 1.0, 'tier_2': 0.75, 'tier_3': 0.5, 'tier_4': 0.25, 'unknown': 0.5}
        tiers = [tier_ranking.get(e.get('tier', 'unknown'), 0.5) for e in education]
        features['highest_institution_tier'] = max(tiers) if tiers else 0
        features['avg_institution_tier'] = sum(tiers) / max(1, len(tiers))
        
        fields = [e.get('field_of_study', '').lower() for e in education]
        features['is_cs_background'] = float(any('computer' in f or 'cs' in f or 'it' in f for f in fields))
        
        return features

    def _engagement_features(self, candidate):
        features = {}
        signals = candidate.get('redrob_signals', {})
        
        features['profile_completeness'] = signals.get('profile_completeness_score', 50) / 100.0
        features['is_open_to_work'] = float(signals.get('open_to_work_flag', False))
        features['notice_period_days'] = signals.get('notice_period_days', 90)
        
        features['recruiter_response_rate'] = signals.get('recruiter_response_rate', 0.5)
        features['avg_response_time_hours'] = signals.get('avg_response_time_hours', 48)
        
        features['interview_completion_rate'] = signals.get('interview_completion_rate', 0.5)
        
        github_score = signals.get('github_activity_score', -1)
        features['github_activity'] = max(0.0, github_score / 100.0) if github_score >= 0 else 0.3
        
        return features

    def _domain_features(self, candidate):
        """Extract domain-specific expertise features."""
        features = {}
        combined_text = (
            candidate.get('profile', {}).get('summary', '') + ' ' +
            ' '.join(j.get('title', '') + ' ' + j.get('description', '') for j in candidate.get('career_history', []))
        ).lower()

        domain_scores = []
        for domain, keywords in DOMAINS.items():
            # Count keyword hits (normalized)
            hits = sum(1 for kw in keywords if kw in combined_text)
            # Use min to cap at 1.0, but accumulate hits properly
            score = min(1.0, hits / max(1, len(keywords)))
            features[f'domain_{domain}_score'] = score
            domain_scores.append(score)

        features['num_specialized_domains'] = sum(1 for s in domain_scores if s > 0.3)
        features['max_domain_depth'] = max(domain_scores) if domain_scores else 0
        features['avg_domain_depth'] = np.mean(domain_scores) if domain_scores else 0
        features['total_domain_hits'] = sum(1 for s in domain_scores if s > 0)

        # Critical for this role: RAG + Vector DB + LLM
        rag_score = features.get('domain_rag_score', 0)
        vector_db_score = features.get('domain_vector_db_score', 0)
        llm_score = features.get('domain_llm_score', 0)
        features['critical_domain_combo'] = rag_score * vector_db_score * llm_score

        return features

    def _text_features(self, candidate):
        """Extract text-based statistical features from summary and descriptions."""
        features = {}
        summary = candidate.get('profile', {}).get('summary', '')
        descriptions = ' '.join(j.get('description', '') for j in candidate.get('career_history', []))
        combined = summary + ' ' + descriptions

        features['summary_length'] = len(summary)
        features['summary_word_count'] = len(summary.split())
        features['desc_total_length'] = len(descriptions)

        # Keyword density for critical terms
        combined_lower = combined.lower()
        critical_terms = ['production', 'scale', 'deploy', 'optimize', 'latency', 'throughput', 'pipeline', 'architecture']
        features['production_keyword_density'] = sum(1 for t in critical_terms if t in combined_lower) / len(critical_terms)

        # Startup vs big-company language
        startup_terms = ['founding', 'built from scratch', '0 to 1', 'early stage', 'startup', 'mvp']
        features['startup_language_score'] = sum(1 for t in startup_terms if t in combined_lower) / len(startup_terms)

        return features

    def _company_features(self, candidate):
        """Extract company size trajectory and industry diversity."""
        features = {}
        career = candidate.get('career_history', [])
        profile = candidate.get('profile', {})

        # Company size mapping
        size_map = {'1-10': 1, '11-50': 2, '51-200': 3, '201-500': 4, '501-1000': 5, '1001-5000': 6, '5001-10000': 7, '10001+': 8}
        sizes = [size_map.get(j.get('company_size', ''), 3) for j in career]
        features['avg_company_size'] = np.mean(sizes) if sizes else 3
        features['min_company_size'] = min(sizes) if sizes else 3
        features['has_startup_exp'] = float(any(s <= 2 for s in sizes))
        features['has_bigco_exp'] = float(any(s >= 6 for s in sizes))
        features['company_size_variance'] = float(np.var(sizes)) if len(sizes) > 1 else 0

        # Current company size
        curr_size = size_map.get(profile.get('current_company_size', ''), 3)
        features['current_company_size_num'] = curr_size

        # Industry diversity
        industries = set(j.get('industry', '') for j in career if j.get('industry'))
        features['num_unique_industries'] = len(industries)

        # Product-company detection
        product_industries = ['technology', 'software', 'saas', 'fintech', 'e-commerce', 'ai', 'data']
        industry_list = [j.get('industry', '').lower() for j in career]
        features['product_company_ratio'] = sum(1 for ind in industry_list if any(p in ind for p in product_industries)) / max(1, len(industry_list))

        # Job-hopping penalty
        durations = [j.get('duration_months', 0) for j in career]
        features['short_stint_count'] = sum(1 for d in durations if 0 < d < 12)
        features['short_stint_ratio'] = features['short_stint_count'] / max(1, len(durations))

        return features

    def _assessment_features(self, candidate):
        """Extract skill assessment scores from redrob_signals."""
        features = {}
        assessments = candidate.get('redrob_signals', {}).get('skill_assessment_scores', {})

        if assessments:
            scores = list(assessments.values())
            features['avg_assessment_score'] = np.mean(scores)
            features['max_assessment_score'] = max(scores)
            features['min_assessment_score'] = min(scores)
            features['num_assessments'] = len(scores)
        else:
            features['avg_assessment_score'] = 0
            features['max_assessment_score'] = 0
            features['min_assessment_score'] = 0
            features['num_assessments'] = 0

        return features

    def _interaction_features(self, fv):
        """Create interaction features that capture combined signals."""
        features = {}
        features['exp_x_progression'] = fv.get('years_total', 0) * max(0, fv.get('seniority_progression', 0))

        advanced_skills_norm = min(1.0, fv.get('num_advanced_skills', 0) / 10.0)
        features['skills_x_engagement'] = advanced_skills_norm * fv.get('recruiter_response_rate', 0.5)

        features['github_x_ml_skills'] = fv.get('github_activity', 0) * min(1.0, fv.get('core_ml_skill_count', 0) / 5.0)

        seniority_norm = min(1.0, fv.get('current_seniority', 0) / 4.0)
        features['seniority_x_specialization'] = seniority_norm * fv.get('max_domain_depth', 0)

        features['stability_x_engagement'] = fv.get('is_current_role_stable', 0) * fv.get('recruiter_response_rate', 0.5)

        # NEW: Domain combo x experience (seasoned specialist)
        features['domain_x_years'] = fv.get('avg_domain_depth', 0) * min(1.0, fv.get('years_total', 0) / 10.0)

        # NEW: Product company x ML skills (ideal candidate signal)
        features['product_x_ml'] = fv.get('product_company_ratio', 0) * min(1.0, fv.get('core_ml_skill_count', 0) / 5.0)

        # NEW: Assessment x endorsement (validated skills)
        features['assessment_x_endorsement'] = fv.get('avg_assessment_score', 0) / 100.0 * min(1.0, fv.get('total_endorsements', 0) / 50.0)

        # NEW: Ideal years band (bell curve centered on 5-9 years)
        years = fv.get('years_total', 0)
        if 5 <= years <= 9:
            features['ideal_years_score'] = 1.0
        elif 4 <= years < 5 or 9 < years <= 12:
            features['ideal_years_score'] = 0.7
        else:
            features['ideal_years_score'] = max(0, 1.0 - abs(years - 7) * 0.1)

        return features

    def _infer_seniority(self, title):
        """Infer seniority level from job title using regex patterns."""
        import re
        t = title.lower()

        # Use word boundaries to avoid false matches (e.g., "lead" in "leads")
        if re.search(r'\b(intern|fresher|trainee)\b', t):
            return 0
        if re.search(r'\b(junior|jr|grad)\b', t):
            return 1
        if re.search(r'\b(director|vp|chief|head|founder|cto|ceo)\b', t):
            return 4
        if re.search(r'\b(principal|staff|architect)\b', t):
            return 3.5
        if re.search(r'\b(senior|sr|lead)\b', t):
            return 3
        if re.search(r'\b(engineer|scientist|specialist|manager|analyst)\b', t):
            return 2
        return 2
