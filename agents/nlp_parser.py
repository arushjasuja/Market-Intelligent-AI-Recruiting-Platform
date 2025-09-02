import spacy
import re
from typing import List, Dict, Any
from datetime import datetime
from dateutil.parser import parse as date_parse
import logging

logger = logging.getLogger(__name__)

class EnhancedResumeParser:
    def __init__(self):
        self.nlp = self._load_spacy_model()
        self.skills_database = self._build_skills_database()
        
    def _load_spacy_model(self):
        """Load spaCy model with error handling"""
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            return None
    
    def _build_skills_database(self) -> Dict[str, List[str]]:
        """Build comprehensive skills database"""
        return {
            "programming_languages": [
                "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Rust",
                "R", "Scala", "Kotlin", "Swift", "PHP", "Ruby", "Perl", "MATLAB", "Julia"
            ],
            "ai_ml_frameworks": [
                "TensorFlow", "PyTorch", "scikit-learn", "Keras", "OpenCV", "YOLO", "BERT",
                "GPT", "LLaMA", "Transformers", "HuggingFace", "spaCy", "NLTK", "Pandas",
                "NumPy", "SciPy", "Matplotlib", "Seaborn", "Plotly", "MLflow", "Kubeflow"
            ],
            "cloud_platforms": [
                "AWS", "Amazon Web Services", "GCP", "Google Cloud Platform", "Google Cloud",
                "Azure", "Microsoft Azure", "Docker", "Kubernetes", "Terraform", "Helm",
                "Jenkins", "GitLab CI", "GitHub Actions", "CircleCI", "Ansible", "Chef"
            ],
            "databases": [
                "SQL", "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch",
                "Apache Cassandra", "Neo4j", "InfluxDB", "TimescaleDB", "DynamoDB",
                "BigQuery", "Snowflake", "Redshift", "ClickHouse", "Apache Spark"
            ],
            "web_technologies": [
                "React", "Angular", "Vue.js", "Node.js", "Express.js", "Django", "Flask",
                "FastAPI", "Spring Boot", "Next.js", "Nuxt.js", "Gatsby", "HTML", "CSS",
                "SASS", "Less", "Webpack", "Babel", "GraphQL", "REST API"
            ],
            "mobile_development": [
                "React Native", "Flutter", "iOS", "Android", "Swift", "Kotlin",
                "Xamarin", "Ionic", "Cordova", "React Native"
            ],
            "devops_tools": [
                "Docker", "Kubernetes", "Jenkins", "GitLab", "GitHub", "Terraform",
                "Ansible", "Chef", "Puppet", "Prometheus", "Grafana", "ELK Stack",
                "Datadog", "New Relic", "Splunk"
            ],
            "data_tools": [
                "Apache Spark", "Hadoop", "Apache Kafka", "Apache Airflow", "Databricks",
                "Jupyter", "Apache Beam", "Tableau", "Power BI", "Looker", "D3.js"
            ]
        }
    
    def parse_resume(self, text: str, filename: str = "") -> Dict[str, Any]:
        """Enhanced resume parsing with comprehensive NLP analysis"""
        try:
            parsed_data = {
                "name": self._extract_name(text, filename),
                "skills": self._extract_skills(text),
                "experience_years": self._extract_experience_years(text),
                "education": self._extract_education(text),
                "certifications": self._extract_certifications(text),
                "previous_companies": self._extract_companies(text),
                "resume_text": text
            }
            
            # Validate parsed data
            parsed_data = self._validate_parsed_data(parsed_data)
            
            return parsed_data
            
        except Exception as e:
            logger.error(f"Resume parsing failed: {e}")
            return self._create_fallback_result(text, filename)
    
    def _extract_name(self, text: str, filename: str) -> str:
        """Multi-method name extraction"""
        
        # Method 1: Clean filename
        if filename and len(filename) > 3:
            name_from_file = re.sub(r'[^\w\s]', ' ', filename.replace('.pdf', ''))
            name_from_file = re.sub(r'\d+', '', name_from_file).strip()
            if len(name_from_file.split()) >= 2:
                return name_from_file.title()
        
        # Method 2: NLP entity extraction
        if self.nlp:
            try:
                doc = self.nlp(text[:1500])  # First 1500 chars
                for ent in doc.ents:
                    if ent.label_ == "PERSON" and len(ent.text.split()) >= 2:
                        # Validate it's likely a full name
                        if re.match(r'^[A-Za-z\s\-\.]+$', ent.text) and len(ent.text) > 4:
                            return ent.text.title()
            except Exception as e:
                logger.warning(f"NLP name extraction failed: {e}")
        
        # Method 3: Pattern matching
        name_patterns = [
            r'^([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',  # Start of document
            r'Name\s*:?\s*([A-Z][a-z]+\s+[A-Z][a-z]+)',  # "Name:" field
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s*\n.*?(?:@|email|phone)',  # Before contact info
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                name = match.group(1).strip()
                if 2 <= len(name.split()) <= 3:  # 2-3 words
                    return name.title()
        
        return "Unknown Candidate"
    
    def _extract_skills(self, text: str) -> List[str]:
        """Comprehensive skill extraction using multiple methods"""
        found_skills = set()
        text_lower = text.lower()
        
        # Method 1: Direct matching with all skill variations
        all_skills = []
        for category_skills in self.skills_database.values():
            all_skills.extend(category_skills)
        
        # Add common variations
        skill_variations = {
            "JavaScript": ["JS", "Javascript", "ECMAScript"],
            "TypeScript": ["TS", "Typescript"],
            "React": ["ReactJS", "React.js"],
            "Node.js": ["NodeJS", "Node"],
            "PostgreSQL": ["Postgres", "psql"],
            "MongoDB": ["Mongo", "mongo"],
            "Machine Learning": ["ML"],
            "Deep Learning": ["DL"],
            "Natural Language Processing": ["NLP"],
            "Computer Vision": ["CV"],
            "Amazon Web Services": ["AWS"],
            "Google Cloud Platform": ["GCP", "Google Cloud"],
        }
        
        # Expand skills list with variations
        expanded_skills = set(all_skills)
        for main_skill, variations in skill_variations.items():
            expanded_skills.update(variations)
        
        # Match skills with word boundaries
        for skill in expanded_skills:
            # Create pattern with word boundaries
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(pattern, text_lower):
                # Add the canonical form
                canonical = skill
                for main, vars in skill_variations.items():
                    if skill in vars:
                        canonical = main
                        break
                found_skills.add(canonical)
        
        # Method 2: NLP-based extraction for additional technical terms
        if self.nlp:
            try:
                doc = self.nlp(text)
                for ent in doc.ents:
                    if ent.label_ in ["PRODUCT", "ORG"] and len(ent.text) > 2:
                        # Check if it's a technology term
                        ent_lower = ent.text.lower()
                        tech_indicators = ['tech', 'soft', 'data', 'web', 'cloud', 'api', 'db', 'sql']
                        if any(indicator in ent_lower for indicator in tech_indicators):
                            found_skills.add(ent.text)
            except Exception as e:
                logger.warning(f"NLP skill extraction failed: {e}")
        
        # Method 3: Section-based extraction
        skill_section_patterns = [
            r'(?:technical\s+)?skills\s*:?\s*\n(.{0,1000}?)(?:\n\n|\n[A-Z][a-z]+:)',
            r'technologies\s*:?\s*(.{0,500}?)(?:\n[A-Z]|\n\n)',
            r'programming\s+languages\s*:?\s*(.{0,300}?)(?:\n[A-Z]|\n\n)',
        ]
        
        for pattern in skill_section_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                # Extract skills from matched section
                for skill in expanded_skills:
                    if skill.lower() in match.lower():
                        found_skills.add(skill)
        
        return sorted(list(found_skills))[:25]  # Limit to top 25 skills
    
    def _extract_experience_years(self, text: str) -> int:
        """Multi-method experience extraction"""
        
        # Method 1: Explicit experience statements
        experience_patterns = [
            r'(\d{1,2})\+?\s*years?\s*(?:of\s+)?(?:professional\s+)?experience',
            r'(\d{1,2})\+?\s*yrs?\s*(?:of\s+)?experience',
            r'experience\s*:?\s*(\d{1,2})\+?\s*years?',
            r'(\d{1,2})\+?\s*years?\s*(?:of\s+)?(?:professional\s+)?(?:work\s+)?experience',
            r'over\s+(\d{1,2})\s+years?\s+(?:of\s+)?experience',
        ]
        
        for pattern in experience_patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                years_list = [int(m) for m in matches if m.isdigit() and 0 <= int(m) <= 50]
                if years_list:
                    return max(years_list)
        
        # Method 2: Calculate from employment history
        calculated_years = self._calculate_employment_duration(text)
        if calculated_years > 0:
            return calculated_years
        
        # Method 3: Estimate from graduation date
        grad_estimate = self._estimate_from_education_dates(text)
        if grad_estimate > 0:
            return grad_estimate
        
        return 0
    
    def _calculate_employment_duration(self, text: str) -> int:
        """Calculate total employment duration from date ranges"""
        try:
            current_year = datetime.now().year
            total_months = 0
            
            # Enhanced date patterns
            date_patterns = [
                # Month Year - Month Year
                r'(\w{3,9})\s+(\d{4})\s*[–\-−−]\s*(\w{3,9})\s+(\d{4})',
                # MM/YYYY - MM/YYYY
                r'(\d{1,2})/(\d{4})\s*[–\-−−]\s*(\d{1,2})/(\d{4})',
                # YYYY - YYYY
                r'(\d{4})\s*[–\-−−]\s*(\d{4})',
                # Month Year - Present/Current
                r'(\w{3,9})\s+(\d{4})\s*[–\-−−]\s*(?:present|current)',
                # MM/YYYY - Present/Current
                r'(\d{1,2})/(\d{4})\s*[–\-−−]\s*(?:present|current)',
                # YYYY - Present/Current
                r'(\d{4})\s*[–\-−−]\s*(?:present|current)',
            ]
            
            month_map = {
                'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
                'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
                'aug': 8, 'august': 8, 'sep': 9, 'september': 9, 'oct': 10, 'october': 10,
                'nov': 11, 'november': 11, 'dec': 12, 'december': 12
            }
            
            processed_ranges = set()  # Avoid double-counting
            
            for pattern in date_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    try:
                        if len(match) == 4:  # Month Year - Month Year
                            start_month = month_map.get(match[0].lower()[:3], 1)
                            start_year = int(match[1])
                            end_month = month_map.get(match[2].lower()[:3], 12)
                            end_year = int(match[3])
                        elif len(match) == 2 and match[1].lower() in ['present', 'current']:
                            # Year - Present
                            start_year = int(match[0])
                            start_month = 1
                            end_year = current_year
                            end_month = datetime.now().month
                        elif len(match) == 2:  # YYYY - YYYY
                            start_year = int(match[0])
                            start_month = 1
                            end_year = int(match[1])
                            end_month = 12
                        else:
                            continue
                        
                        # Validate years
                        if not (1980 <= start_year <= current_year and start_year <= end_year <= current_year + 1):
                            continue
                        
                        # Create range identifier to avoid duplicates
                        range_id = f"{start_year}{start_month:02d}-{end_year}{end_month:02d}"
                        if range_id in processed_ranges:
                            continue
                        processed_ranges.add(range_id)
                        
                        # Calculate duration in months
                        duration_months = (end_year - start_year) * 12 + (end_month - start_month)
                        if duration_months > 0:
                            total_months += duration_months
                    
                    except (ValueError, KeyError):
                        continue
            
            # Convert to years, cap at reasonable maximum
            total_years = total_months // 12
            return min(total_years, 40)
            
        except Exception as e:
            logger.warning(f"Employment duration calculation failed: {e}")
            return 0
    
    def _estimate_from_education_dates(self, text: str) -> int:
        """Estimate experience from education completion dates"""
        try:
            current_year = datetime.now().year
            
            # Look for graduation/completion years
            graduation_patterns = [
                r'graduated?\s*(?:in\s*)?(\d{4})',
                r'graduation\s*(?:year\s*)?:?\s*(\d{4})',
                r'(?:bachelor|master|phd|degree).*?(\d{4})',
                r'(?:b\.?s\.?|m\.?s\.?|b\.?a\.?|m\.?a\.?).*?(\d{4})',
                r'class\s+of\s+(\d{4})',
            ]
            
            graduation_years = []
            for pattern in graduation_patterns:
                matches = re.findall(pattern, text.lower())
                for match in matches:
                    year = int(match)
                    if 1980 <= year <= current_year:
                        graduation_years.append(year)
            
            if graduation_years:
                latest_graduation = max(graduation_years)
                estimated_experience = max(0, current_year - latest_graduation)
                return min(estimated_experience, 35)  # Cap at 35 years
            
            return 0
            
        except Exception as e:
            logger.warning(f"Education date estimation failed: {e}")
            return 0
    
    def _extract_education(self, text: str) -> List[str]:
        """Extract education information with comprehensive patterns"""
        education_items = []
        text_lower = text.lower()
        
        # Degree patterns
        degree_patterns = [
            r'(bachelor[\'s]*\s+(?:of\s+)?(?:science|arts|engineering|business)?\s*(?:in\s+)?[\w\s]+)',
            r'(master[\'s]*\s+(?:of\s+)?(?:science|arts|engineering|business)?\s*(?:in\s+)?[\w\s]+)',
            r'(ph\.?d\.?\s*(?:in\s+)?[\w\s]+)',
            r'(doctorate\s*(?:in\s+)?[\w\s]+)',
            r'(b\.?s\.?\s*(?:in\s+)?[\w\s]+)',
            r'(m\.?s\.?\s*(?:in\s+)?[\w\s]+)',
            r'(b\.?a\.?\s*(?:in\s+)?[\w\s]+)',
            r'(m\.?a\.?\s*(?:in\s+)?[\w\s]+)',
            r'(m\.?b\.?a\.?\s*(?:in\s+)?[\w\s]*)',
        ]
        
        for pattern in degree_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                degree = match.strip()
                if 5 < len(degree) < 100:  # Reasonable length
                    education_items.append(degree.title())
        
        # Institution patterns using NLP
        if self.nlp:
            try:
                doc = self.nlp(text)
                for ent in doc.ents:
                    if ent.label_ == "ORG":
                        org_lower = ent.text.lower()
                        if any(keyword in org_lower for keyword in ['university', 'college', 'institute', 'school']):
                            if ent.text not in education_items:
                                education_items.append(ent.text)
            except Exception as e:
                logger.warning(f"NLP education extraction failed: {e}")
        
        return education_items[:5]  # Limit to 5 items
    
    def _extract_certifications(self, text: str) -> List[str]:
        """Extract professional certifications"""
        certifications = []
        text_lower = text.lower()
        
        # Certification patterns
        cert_patterns = [
            r'(aws certified [\w\s]+)',
            r'(google cloud [\w\s]+ certified)',
            r'(microsoft certified [\w\s]+)',
            r'(certified [\w\s]+ (?:professional|associate|expert))',
            r'(pmp certified)',
            r'(cissp)',
            r'(comptia [\w\s]+)',
            r'(cisco certified [\w\s]+)',
            r'(certified kubernetes [\w\s]+)',
            r'(tensorflow certified)',
            r'(scrum master certified)',
            r'(agile certified [\w\s]+)',
        ]
        
        for pattern in cert_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                cert = match.strip()
                if len(cert) > 3:
                    certifications.append(cert.title())
        
        # Section-based extraction
        cert_section_patterns = [
            r'certifications?\s*:?\s*\n(.{0,800}?)(?:\n\n|\n[A-Z][a-z]+:)',
            r'(?:professional\s+)?certifications?\s*:?\s*(.{0,500}?)(?:\n[A-Z]|\n\n)',
        ]
        
        for pattern in cert_section_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                # Split by common delimiters and clean
                items = re.split(r'[,\n•\-\*]', match)
                for item in items:
                    item = item.strip()
                    if 3 < len(item) < 100 and item not in certifications:
                        certifications.append(item.title())
        
        return certifications[:10]  # Limit to 10 certifications
    
    def _extract_companies(self, text: str) -> List[str]:
        """Extract previous companies using NLP and patterns"""
        companies = set()
        
        # NLP-based extraction
        if self.nlp:
            try:
                doc = self.nlp(text)
                for ent in doc.ents:
                    if ent.label_ == "ORG":
                        org_name = ent.text.strip()
                        # Filter out likely non-companies
                        if self._is_likely_company(org_name):
                            companies.add(org_name)
            except Exception as e:
                logger.warning(f"NLP company extraction failed: {e}")
        
        # Pattern-based extraction
        company_patterns = [
            r'(?:at\s+|@\s*)([A-Z][a-zA-Z\s&\.,]+?)(?:\s*[,\n]|\s*-|\s*\(|\s*\d{4})',
            r'([A-Z][a-zA-Z\s&\.,]+?)\s+(?:\d{4}\s*[–\-−]\s*(?:\d{4}|present|current))',
            r'(?:worked\s+at\s+|employed\s+at\s+)([A-Z][a-zA-Z\s&\.,]+)',
        ]
        
        for pattern in company_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                company = match.strip()
                if self._is_likely_company(company):
                    companies.add(company)
        
        return list(companies)[:8]  # Limit to 8 companies
    
    def _is_likely_company(self, text: str) -> bool:
        """Check if text is likely a company name"""
        if not text or len(text) < 2:
            return False
        
        text_lower = text.lower()
        
        # Exclude common non-company terms
        exclusions = [
            'university', 'college', 'school', 'institute', 'education',
            'email', 'phone', 'address', 'linkedin', 'github',
            'january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december'
        ]
        
        if any(exclusion in text_lower for exclusion in exclusions):
            return False
        
        # Must be reasonable length and format
        if not (2 <= len(text) <= 50):
            return False
        
        # Should start with capital letter
        if not text[0].isupper():
            return False
        
        return True
    
    def _validate_parsed_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean parsed data"""
        # Ensure all required fields exist with proper types
        validated = {
            "name": str(data.get("name", "Unknown Candidate")),
            "skills": list(set(data.get("skills", [])))[:20],  # Remove duplicates, limit to 20
            "experience_years": max(0, min(50, int(data.get("experience_years", 0)))),  # 0-50 years
            "education": list(set(data.get("education", [])))[:5],  # Remove duplicates, limit to 5
            "certifications": list(set(data.get("certifications", [])))[:10],  # Remove duplicates, limit to 10
            "previous_companies": list(set(data.get("previous_companies", [])))[:8],  # Remove duplicates, limit to 8
            "resume_text": str(data.get("resume_text", ""))
        }
        
        return validated
    
    def _create_fallback_result(self, text: str, filename: str) -> Dict[str, Any]:
        """Create fallback result when parsing completely fails"""
        name = "Unknown Candidate"
        if filename:
            name = re.sub(r'[^\w\s]', ' ', filename.replace('.pdf', '')).strip().title()
        
        # Basic skill extraction as fallback
        basic_skills = []
        common_skills = ["Python", "Java", "JavaScript", "SQL", "AWS", "React", "Docker", "Git"]
        
        text_lower = text.lower()
        for skill in common_skills:
            if skill.lower() in text_lower:
                basic_skills.append(skill)
        
        return {
            "name": name,
            "skills": basic_skills,
            "experience_years": 0,
            "education": [],
            "certifications": [],
            "previous_companies": [],
            "resume_text": text
        }