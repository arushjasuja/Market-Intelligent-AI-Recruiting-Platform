from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool
from typing import Dict, Any, List
import json
import re
import logging

logger = logging.getLogger(__name__)

class BiasDetectionTool(BaseTool):
    name: str = "bias_detector"
    description: str = "Detects and analyzes potential bias in job descriptions"
    
    def _run(self, content: str) -> str:
        bias_categories = {
            'gender': {
                'words': ['guys', 'rockstar', 'ninja', 'guru', 'superman', 'hero', 'brotherhood'],
                'weight': 3
            },
            'age': {
                'words': ['young', 'energetic', 'fresh', 'millennial', 'digital native', 'recent grad'],
                'weight': 2
            },
            'culture': {
                'words': ['culture fit', 'beer', 'ping pong', 'family', 'work hard play hard', 'startup mentality'],
                'weight': 2
            },
            'aggressive': {
                'words': ['aggressive', 'competitive', 'hustler', 'killer', 'dominant', 'crush', 'destroy'],
                'weight': 3
            },
            'exclusionary': {
                'words': ['native speaker', 'no visa', 'local candidates only', 'perfect english'],
                'weight': 4
            }
        }
        
        content_lower = content.lower()
        detected_issues = {}
        total_score = 0
        
        for category, config in bias_categories.items():
            found_words = [word for word in config['words'] if word in content_lower]
            if found_words:
                detected_issues[category] = found_words
                total_score += len(found_words) * config['weight']
        
        # Cap score at 10
        bias_score = min(total_score, 10)
        
        suggestions = self._generate_suggestions(detected_issues)
        
        return json.dumps({
            "bias_score": bias_score,
            "detected_issues": detected_issues,
            "suggestions": suggestions,
            "assessment": "Low" if bias_score <= 3 else "Medium" if bias_score <= 6 else "High"
        })
    
    def _generate_suggestions(self, issues: Dict) -> List[str]:
        suggestions = []
        
        if 'gender' in issues:
            suggestions.append("Replace gendered language with inclusive terms like 'team member' or 'professional'")
        
        if 'age' in issues:
            suggestions.append("Focus on skills and experience rather than age-related descriptors")
        
        if 'culture' in issues:
            suggestions.append("Emphasize professional development and meaningful work over cultural perks")
        
        if 'aggressive' in issues:
            suggestions.append("Use collaborative language that emphasizes teamwork and growth")
        
        if 'exclusionary' in issues:
            suggestions.append("Ensure language welcomes diverse candidates and avoids exclusionary requirements")
        
        return suggestions

class ContentStructureTool(BaseTool):
    name: str = "content_structurer"
    description: str = "Creates well-structured job description templates"
    
    def _run(self, job_data: str) -> str:
        try:
            data = json.loads(job_data)
            
            structure = {
                "sections": [
                    "company_overview",
                    "role_summary", 
                    "key_responsibilities",
                    "required_qualifications",
                    "preferred_qualifications",
                    "compensation_benefits"
                ],
                "content_guidelines": {
                    "company_overview": f"Brief introduction to {data.get('company', 'the company')} and its mission",
                    "role_summary": f"Clear description of the {data.get('title', 'position')} and its impact",
                    "key_responsibilities": "3-5 specific, actionable responsibilities",
                    "required_qualifications": "Must-have skills and experience",
                    "preferred_qualifications": "Nice-to-have skills that would be beneficial",
                    "compensation_benefits": "Salary range and key benefits"
                },
                "best_practices": [
                    "Use clear, concise language",
                    "Focus on impact and growth opportunities",
                    "Include specific technologies and skills",
                    "Mention team structure and collaboration"
                ]
            }
            
            return json.dumps(structure)
            
        except Exception as e:
            return json.dumps({"error": f"Content structuring failed: {e}"})

class LangChainContentGenerator:
    def __init__(self, openai_api_key: str):
        try:
            self.llm = ChatOpenAI(
                api_key=openai_api_key,
                model="gpt-4",
                temperature=0.7,
                max_tokens=2000
            )
            
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            self.tools = [BiasDetectionTool(), ContentStructureTool()]
            
        except Exception as e:
            logger.error(f"LangChain initialization failed: {e}")
            raise RuntimeError(f"Cannot initialize LangChain: {e}")
    
    def generate_job_description(self, job_data: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Phase 1: Strategy Development
            strategy_context = self._develop_strategy(job_data, market_data)
            
            # Phase 2: Content Generation
            content = self._generate_content(job_data, market_data, strategy_context)
            
            # Phase 3: Brand Voice Application
            branded_content = self._apply_brand_voice(content, job_data)
            
            # Phase 4: Quality Assurance and Bias Check
            qa_result = self._quality_assurance(branded_content)
            
            return {
                "content": qa_result["final_content"],
                "bias_score": qa_result["bias_score"],
                "bias_analysis": qa_result["bias_analysis"],
                "market_integration": bool(market_data),
                "processing_stages": ["Strategy", "Content Generation", "Brand Application", "Quality Assurance"],
                "improvements_made": qa_result.get("improvements", [])
            }
            
        except Exception as e:
            logger.error(f"Content generation failed: {e}")
            return self._create_fallback_content(job_data, market_data)
    
    def _develop_strategy(self, job_data: Dict[str, Any], market_data: Dict[str, Any]) -> str:
        """Phase 1: Strategy Agent - Develop content strategy"""
        try:
            strategy_prompt = f"""
            As a Content Strategy Agent, develop a comprehensive content strategy for this job posting:
            
            Job Details:
            - Title: {job_data.get('title')}
            - Company: {job_data.get('company')}
            - Location: {job_data.get('location')}
            - Skills: {', '.join(job_data.get('skills_required', []))}
            
            Market Context:
            - Market Demand: {market_data.get('market_demand', 'Unknown')}
            - Top Skills: {[skill['skill'] for skill in market_data.get('top_skills', [])[:5]]}
            - Salary Range: ${market_data.get('salary_range', {}).get('min', 'N/A')} - ${market_data.get('salary_range', {}).get('max', 'N/A')}
            
            Develop a strategy that:
            1. Positions the role competitively in the market
            2. Highlights unique value propositions
            3. Attracts top talent while filtering for fit
            4. Uses market insights to strengthen the appeal
            
            Provide strategic recommendations for tone, key messages, and positioning.
            """
            
            response = self.llm.invoke(strategy_prompt)
            return response.content
            
        except Exception as e:
            logger.warning(f"Strategy development failed: {e}")
            return "Focus on clear role description with competitive positioning"
    
    def _generate_content(self, job_data: Dict[str, Any], market_data: Dict[str, Any], strategy: str) -> str:
        """Phase 2: Writer Agent - Generate core content"""
        try:
            # Get content structure
            structure_result = self.tools[1]._run(json.dumps(job_data))
            structure = json.loads(structure_result)
            
            content_prompt = f"""
            As a Job Description Writer Agent, create compelling content based on this strategy:
            
            Strategy: {strategy}
            
            Job Requirements:
            - Title: {job_data.get('title')}
            - Company: {job_data.get('company')}
            - Location: {job_data.get('location')}
            - Required Skills: {', '.join(job_data.get('skills_required', []))}
            - Salary: ${job_data.get('salary_min', 0):,} - ${job_data.get('salary_max', 0):,}
            
            Market Intelligence:
            - High-demand skills in market: {', '.join([skill['skill'] for skill in market_data.get('top_skills', [])[:3]])}
            - Market demand level: {market_data.get('market_demand', 'Medium')}
            
            Create a comprehensive job description with these sections:
            1. Company Overview (2-3 sentences)
            2. Role Summary (2-3 sentences highlighting impact)
            3. Key Responsibilities (4-5 bullet points)
            4. Required Qualifications (technical skills and experience)
            5. Preferred Qualifications (nice-to-have skills)
            6. What We Offer (compensation, benefits, growth)
            
            Make it engaging, specific, and market-competitive. Use active language and focus on growth opportunities.
            """
            
            response = self.llm.invoke(content_prompt)
            return response.content
            
        except Exception as e:
            logger.warning(f"Content generation failed: {e}")
            return self._create_basic_content(job_data)
    
    def _apply_brand_voice(self, content: str, job_data: Dict[str, Any]) -> str:
        """Phase 3: Brand Agent - Apply consistent brand voice"""
        try:
            brand_prompt = f"""
            As a Brand Voice Agent, refine this job description for {job_data.get('company')}:
            
            Current Content:
            {content}
            
            Apply these brand voice principles:
            1. Professional yet approachable tone
            2. Clear, concise communication
            3. Emphasis on innovation and growth
            4. Inclusive language that welcomes diverse candidates
            5. Authentic representation of company culture
            
            Ensure the content:
            - Reflects the company's values and culture
            - Uses consistent terminology
            - Maintains professional standards
            - Avoids jargon and buzzwords
            - Is accessible to a broad audience
            
            Return the refined job description.
            """
            
            response = self.llm.invoke(brand_prompt)
            return response.content
            
        except Exception as e:
            logger.warning(f"Brand voice application failed: {e}")
            return content
    
    def _quality_assurance(self, content: str) -> Dict[str, Any]:
        """Phase 4: QA Agent - Comprehensive quality check"""
        try:
            # Run bias detection
            bias_result = json.loads(self.tools[0]._run(content))
            
            # If bias score is high, attempt improvement
            if bias_result["bias_score"] > 3:
                improvement_prompt = f"""
                As a Quality Assurance Agent, improve this job description to reduce bias:
                
                Current Content:
                {content}
                
                Bias Issues Detected:
                {json.dumps(bias_result["detected_issues"], indent=2)}
                
                Suggestions:
                {chr(10).join(bias_result["suggestions"])}
                
                Rewrite the job description to:
                1. Remove or replace biased language
                2. Use inclusive terminology
                3. Focus on skills and qualifications
                4. Welcome diverse candidates
                5. Maintain professional tone
                
                Return the improved job description.
                """
                
                try:
                    response = self.llm.invoke(improvement_prompt)
                    improved_content = response.content
                    
                    # Re-check bias score
                    final_bias_result = json.loads(self.tools[0]._run(improved_content))
                    
                    return {
                        "final_content": improved_content,
                        "bias_score": final_bias_result["bias_score"],
                        "bias_analysis": final_bias_result,
                        "improvements": ["Bias reduction applied", "Inclusive language enhanced"]
                    }
                    
                except Exception as improvement_error:
                    logger.warning(f"Content improvement failed: {improvement_error}")
                    return {
                        "final_content": content,
                        "bias_score": bias_result["bias_score"],
                        "bias_analysis": bias_result,
                        "improvements": ["Improvement attempted but failed"]
                    }
            else:
                return {
                    "final_content": content,
                    "bias_score": bias_result["bias_score"],
                    "bias_analysis": bias_result,
                    "improvements": ["Content passed initial bias check"]
                }
                
        except Exception as e:
            logger.error(f"Quality assurance failed: {e}")
            return {
                "final_content": content,
                "bias_score": 2,
                "bias_analysis": {"bias_score": 2, "assessment": "Unknown"},
                "improvements": ["QA process failed"]
            }
    
    def _create_basic_content(self, job_data: Dict[str, Any]) -> str:
        """Create basic content template when AI generation fails"""
        title = job_data.get('title', 'Position')
        company = job_data.get('company', 'Our Company')
        location = job_data.get('location', 'Location')
        skills = job_data.get('skills_required', [])
        
        return f"""# {title} - {company}

## About the Role
We are seeking a talented {title} to join our innovative team in {location}. This position offers an exciting opportunity to work with cutting-edge technology and contribute to meaningful projects.

## Key Responsibilities
• Develop and implement technical solutions using modern technologies
• Collaborate effectively with cross-functional teams
• Drive technical excellence and best practices
• Contribute to product development and strategic initiatives
• Mentor team members and share knowledge

## Required Qualifications
• Strong experience with {', '.join(skills[:3]) if skills else 'relevant technologies'}
• Excellent problem-solving and analytical skills
• Strong communication and collaboration abilities
• Bachelor's degree in relevant field or equivalent experience
• Proven track record of delivering high-quality solutions

## What We Offer
• Competitive compensation package
• Comprehensive benefits including health, dental, and vision
• Professional development opportunities
• Flexible work arrangements
• Collaborative and inclusive work environment

Ready to make an impact? We'd love to hear from you!"""
    
    def _create_fallback_content(self, job_data: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback result when all generation fails"""
        basic_content = self._create_basic_content(job_data)
        
        return {
            "content": basic_content,
            "bias_score": 2,
            "bias_analysis": {"bias_score": 2, "assessment": "Not analyzed"},
            "market_integration": bool(market_data),
            "processing_stages": ["Fallback Generation"],
            "improvements_made": ["Used template due to generation failure"]
        }