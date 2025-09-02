import autogen
from typing import List, Dict, Any
import json
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
import re

logger = logging.getLogger(__name__)

class AutoGenMatchingSystem:
    def __init__(self, openai_api_key: str, model: SentenceTransformer):
        self.model = model
        self.api_key = openai_api_key
        
        # Configure AutoGen
        self.config_list = [{
            "model": "gpt-4",
            "api_key": openai_api_key,
            "max_tokens": 1000,
            "temperature": 0.7
        }]
        
        self._initialize_agents()
    
    def _initialize_agents(self):
        try:
            # Candidate Analyzer Agent
            self.candidate_analyzer = autogen.AssistantAgent(
                name="CandidateAnalyzer",
                system_message="""You are a Candidate Analysis Expert specializing in resume evaluation and skills assessment.

Your responsibilities:
- Analyze candidate profiles for technical competencies
- Assess experience levels and career progression
- Identify strengths, specializations, and growth areas
- Evaluate education and certification relevance
- Provide structured insights about candidate capabilities

Always provide clear, objective analysis based on the candidate data provided.
Format responses as structured insights that help with matching decisions.""",
                llm_config={"config_list": self.config_list, "timeout": 30}
            )
            
            # Job Matching Agent
            self.job_matcher = autogen.AssistantAgent(
                name="JobMatcher",
                system_message="""You are a Job Matching Specialist expert in candidate-role compatibility analysis.

Your responsibilities:
- Analyze alignment between candidate skills and job requirements
- Identify skill gaps and transferable skills
- Assess experience level fit for the role
- Evaluate potential for success in the position
- Provide detailed matching rationale

Focus on both technical fit and potential for growth. Consider not just current skills but learning ability and adaptability.
Provide specific, actionable insights about the match quality.""",
                llm_config={"config_list": self.config_list, "timeout": 30}
            )
            
            # Screening Question Generator
            self.screening_agent = autogen.AssistantAgent(
                name="ScreeningAgent", 
                system_message="""You are a Technical Screening Expert who creates targeted interview questions.

Your responsibilities:
- Generate role-specific technical questions
- Create behavioral questions that assess cultural fit
- Design situational questions to evaluate problem-solving
- Ensure questions are unbiased and inclusive
- Tailor difficulty to candidate experience level

Generate 4 high-quality questions: 2 technical, 1 behavioral, 1 situational.
Make questions specific to the role and candidate background.""",
                llm_config={"config_list": self.config_list, "timeout": 30}
            )
            
            # User proxy for coordinating interactions
            self.user_proxy = autogen.UserProxyAgent(
                name="MatchingCoordinator",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=0,
                code_execution_config=False,
                is_termination_msg=lambda x: x.get("content", "").strip().endswith("TERMINATE")
            )
            
        except Exception as e:
            logger.error(f"AutoGen agent initialization failed: {e}")
            raise RuntimeError(f"Cannot initialize AutoGen agents: {e}")
    
    def analyze_candidate_match(self, candidate: Any, job: Any) -> Dict[str, Any]:
        """Perform comprehensive candidate matching with AutoGen enhancement"""
        try:
            start_time = time.time()
            
            # Core vector similarity computation (fast baseline)
            similarity_data = self._compute_vector_similarity(candidate, job)
            
            # AutoGen enhanced analysis
            try:
                enhanced_analysis = self._run_autogen_analysis(candidate, job, similarity_data)
                processing_time = (time.time() - start_time) * 1000
                
                return {
                    **similarity_data,
                    **enhanced_analysis,
                    "processing_time_ms": processing_time,
                    "autogen_enhanced": True
                }
                
            except Exception as autogen_error:
                logger.warning(f"AutoGen enhancement failed, using baseline: {autogen_error}")
                processing_time = (time.time() - start_time) * 1000
                
                return {
                    **similarity_data,
                    "processing_time_ms": processing_time,
                    "autogen_enhanced": False,
                    "fallback_reason": str(autogen_error)
                }
                
        except Exception as e:
            logger.error(f"Candidate matching failed: {e}")
            return self._create_error_result(candidate, job)
    
    def _compute_vector_similarity(self, candidate: Any, job: Any) -> Dict[str, Any]:
        """Compute baseline vector similarity and skill analysis"""
        try:
            # Create text representations
            job_text = f"{job.title} {job.description} {' '.join(job.skills_required)}"
            candidate_text = f"{candidate.resume_text} {' '.join(candidate.skills)} {' '.join(candidate.education)}"
            
            # Compute embeddings and similarity
            job_embedding = self.model.encode([job_text])
            candidate_embedding = self.model.encode([candidate_text])
            
            similarity_score = cosine_similarity(job_embedding, candidate_embedding)[0][0]
            match_percentage = int(similarity_score * 100)
            
            # Detailed skill analysis
            skill_analysis = self._analyze_skills(candidate.skills, job.skills_required)
            
            # Experience fit assessment
            experience_fit = self._assess_experience_fit(candidate.experience_years, job.title)
            
            return {
                "similarity_score": float(similarity_score),
                "match_percentage": match_percentage,
                "skill_analysis": skill_analysis,
                "experience_fit": experience_fit,
                "baseline_explanation": f"Vector similarity: {similarity_score:.3f}. Skills match: {skill_analysis['overlap_count']}/{skill_analysis['total_required']}."
            }
            
        except Exception as e:
            logger.error(f"Vector similarity computation failed: {e}")
            raise
    
    def _run_autogen_analysis(self, candidate: Any, job: Any, similarity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run AutoGen multi-agent analysis"""
        
        # Candidate analysis
        candidate_prompt = f"""Analyze this candidate profile for the position:

Candidate: {candidate.name}
Experience: {candidate.experience_years} years
Skills: {', '.join(candidate.skills)}
Education: {', '.join(candidate.education) if candidate.education else 'Not specified'}
Certifications: {', '.join(candidate.certifications) if candidate.certifications else 'None listed'}
Previous Companies: {', '.join(candidate.previous_companies) if candidate.previous_companies else 'Not specified'}

Provide a structured analysis of this candidate's:
1. Technical competency level
2. Experience depth and breadth  
3. Educational background relevance
4. Career progression indicators
5. Key strengths and specializations

Keep analysis concise but insightful."""

        try:
            candidate_analysis_msg = [{"role": "user", "content": candidate_prompt}]
            candidate_response = self.candidate_analyzer.generate_reply(
                messages=candidate_analysis_msg,
                sender=self.user_proxy
            )
        except Exception as e:
            logger.warning(f"Candidate analysis failed: {e}")
            candidate_response = f"Analysis unavailable. Candidate has {candidate.experience_years} years experience in {', '.join(candidate.skills[:3])}."
        
        # Job matching analysis
        match_prompt = f"""Evaluate the match between this candidate and job requirements:

Job: {job.title} at {job.company}
Location: {job.location} 
Required Skills: {', '.join(job.skills_required)}
Salary Range: ${job.salary_min:,} - ${job.salary_max:,}

Candidate Profile:
- Experience: {candidate.experience_years} years
- Skills: {', '.join(candidate.skills)}
- Vector Similarity: {similarity_data['similarity_score']:.3f}
- Skills Match: {similarity_data['skill_analysis']['overlap_count']}/{similarity_data['skill_analysis']['total_required']}

Provide detailed matching analysis including:
1. Technical skill alignment and gaps
2. Experience level appropriateness  
3. Growth potential in this role
4. Potential concerns or risks
5. Overall match assessment

Be specific about why this is or isn't a good match."""

        try:
            match_analysis_msg = [{"role": "user", "content": match_prompt}]
            match_response = self.job_matcher.generate_reply(
                messages=match_analysis_msg,
                sender=self.user_proxy
            )
        except Exception as e:
            logger.warning(f"Match analysis failed: {e}")
            match_response = f"Match analysis unavailable. {similarity_data['match_percentage']}% compatibility based on skills and experience."
        
        # Generate screening questions
        screening_prompt = f"""Generate 4 targeted screening questions for this candidate-job combination:

Position: {job.title}
Key Skills: {', '.join(job.skills_required[:5])}
Company: {job.company}

Candidate Background:
- Experience Level: {candidate.experience_years} years
- Key Skills: {', '.join(candidate.skills[:5])}
- Previous Experience: {', '.join(candidate.previous_companies[:2]) if candidate.previous_companies else 'Various companies'}

Create exactly 4 questions:
1. Technical question about core job requirements
2. Experience-based question about their background
3. Behavioral question about work style/collaboration  
4. Situational question about problem-solving

Make questions specific, relevant, and appropriate for the experience level."""

        try:
            screening_msg = [{"role": "user", "content": screening_prompt}]
            screening_response = self.screening_agent.generate_reply(
                messages=screening_msg,
                sender=self.user_proxy
            )
            screening_questions = self._extract_questions_from_response(screening_response)
        except Exception as e:
            logger.warning(f"Screening question generation failed: {e}")
            screening_questions = self._generate_fallback_questions(job, candidate)
        
        return {
            "candidate_analysis": candidate_response if isinstance(candidate_response, str) else str(candidate_response),
            "match_analysis": match_response if isinstance(match_response, str) else str(match_response),
            "screening_questions": screening_questions
        }
    
    def _extract_questions_from_response(self, response: str) -> List[str]:
        """Extract individual questions from agent response"""
        try:
            response_text = response if isinstance(response, str) else str(response)
            
            # Look for numbered questions
            patterns = [
                r'\d+\.\s+(.+?)(?=\d+\.|$)',  # "1. Question text"
                r'(?:Question \d+:|Q\d+:)\s*(.+?)(?=(?:Question \d+:|Q\d+:)|$)',  # "Question 1: text"
                r'[•\-]\s*(.+?)(?=[•\-]|$)',  # Bulleted questions
            ]
            
            questions = []
            for pattern in patterns:
                matches = re.findall(pattern, response_text, re.DOTALL | re.MULTILINE)
                if matches:
                    questions = [q.strip().replace('\n', ' ') for q in matches if len(q.strip()) > 15]
                    break
            
            # Clean up questions
            cleaned_questions = []
            for q in questions:
                # Remove extra whitespace and ensure it ends with a question mark
                clean_q = re.sub(r'\s+', ' ', q.strip())
                if clean_q and not clean_q.endswith('?'):
                    clean_q += '?'
                if len(clean_q) > 10:
                    cleaned_questions.append(clean_q)
            
            return cleaned_questions[:4] if cleaned_questions else self._generate_fallback_questions(None, None)
            
        except Exception as e:
            logger.warning(f"Question extraction failed: {e}")
            return self._generate_fallback_questions(None, None)
    
    def _generate_fallback_questions(self, job: Any, candidate: Any) -> List[str]:
        """Generate fallback screening questions when AutoGen fails"""
        generic_questions = [
            "What interests you most about this role and our company?",
            "Can you describe a challenging technical problem you've solved recently?",
            "How do you approach learning new technologies or skills?",
            "Tell me about a time when you had to collaborate with a difficult team member."
        ]
        
        if job and job.skills_required:
            skill = job.skills_required[0]
            generic_questions[1] = f"Can you describe your experience with {skill} and how you've used it in previous projects?"
        
        return generic_questions
    
    def _analyze_skills(self, candidate_skills: List[str], job_skills: List[str]) -> Dict[str, Any]:
        """Detailed skill overlap analysis"""
        if not job_skills:
            return {"overlap_count": 0, "total_required": 0, "overlap_percentage": 0}
        
        candidate_skills_lower = {skill.lower() for skill in candidate_skills}
        job_skills_lower = {skill.lower() for skill in job_skills}
        
        overlapping = [skill for skill in job_skills if skill.lower() in candidate_skills_lower]
        missing = [skill for skill in job_skills if skill.lower() not in candidate_skills_lower]
        
        return {
            "overlap_count": len(overlapping),
            "total_required": len(job_skills),
            "overlap_percentage": int((len(overlapping) / len(job_skills)) * 100),
            "overlapping_skills": overlapping,
            "missing_skills": missing
        }
    
    def _assess_experience_fit(self, candidate_years: int, job_title: str) -> Dict[str, Any]:
        """Assess experience level appropriateness"""
        title_lower = job_title.lower()
        
        # Determine expected experience range
        if any(word in title_lower for word in ["senior", "lead", "principal", "staff"]):
            expected_min, expected_max = 5, 15
            level = "Senior"
        elif any(word in title_lower for word in ["junior", "entry", "associate"]):
            expected_min, expected_max = 0, 3
            level = "Junior"
        else:
            expected_min, expected_max = 2, 7
            level = "Mid-level"
        
        # Assess fit
        if candidate_years < expected_min:
            fit = "Under-qualified"
            gap = expected_min - candidate_years
        elif candidate_years > expected_max:
            fit = "Over-qualified"
            gap = candidate_years - expected_max
        else:
            fit = "Good match"
            gap = 0
        
        return {
            "candidate_years": candidate_years,
            "expected_level": level,
            "expected_range": f"{expected_min}-{expected_max} years",
            "fit_assessment": fit,
            "experience_gap": gap
        }
    
    def _create_error_result(self, candidate: Any, job: Any) -> Dict[str, Any]:
        """Create error result when matching completely fails"""
        return {
            "similarity_score": 0.0,
            "match_percentage": 0,
            "processing_time_ms": 0,
            "candidate_analysis": "Analysis failed due to system error",
            "match_analysis": "Matching analysis unavailable",
            "screening_questions": ["What interests you about this position?"],
            "skill_analysis": {"overlap_count": 0, "overlap_percentage": 0},
            "experience_fit": {"fit_assessment": "Unknown"},
            "autogen_enhanced": False,
            "error": True,
            "error_message": "Complete matching failure"
        }