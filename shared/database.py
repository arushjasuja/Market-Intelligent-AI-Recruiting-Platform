# Handle SQLite3 compatibility for ChromaDB
import sys
import logging

# Try to handle ChromaDB SQLite requirements
try:
    # First, try the standard approach for systems with pysqlite3
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    # Fallback to standard sqlite3 if pysqlite3 is not available
    try:
        import sqlite3
        # This is fine for most systems
        pass
    except ImportError:
        # Last resort fallback
        pass

import chromadb
import json
from typing import List
from datetime import datetime

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, path: str = "./chroma_db"):
        try:
            # Initialize ChromaDB with error handling
            self.client = chromadb.PersistentClient(path=path)
            self.jobs = self.client.get_or_create_collection("jobs")
            self.candidates = self.client.get_or_create_collection("candidates")
            self.reports = self.client.get_or_create_collection("reports")
            logger.info(f"Database initialized successfully at {path}")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            # Try alternative initialization
            try:
                logger.warning("Attempting alternative ChromaDB initialization...")
                import chromadb.config
                settings = chromadb.config.Settings(anonymized_telemetry=False)
                self.client = chromadb.PersistentClient(path=path, settings=settings)
                self.jobs = self.client.get_or_create_collection("jobs")
                self.candidates = self.client.get_or_create_collection("candidates")
                self.reports = self.client.get_or_create_collection("reports")
                logger.info("Alternative database initialization successful")
            except Exception as e2:
                logger.error(f"Alternative database initialization also failed: {e2}")
                raise RuntimeError(f"Cannot initialize database. Try installing ChromaDB dependencies: {e}")
    
    def _serialize_data(self, data: dict) -> dict:
        """Serialize complex data types for ChromaDB storage"""
        serialized = {}
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                serialized[key] = json.dumps(value)
            elif isinstance(value, datetime):
                serialized[key] = value.isoformat()
            else:
                serialized[key] = str(value)
        return serialized
    
    def _deserialize_data(self, data: dict) -> dict:
        """Deserialize data from ChromaDB storage"""
        json_fields = {
            'salary_range', 'top_skills', 'skills_required', 'skills', 
            'education', 'certifications', 'previous_companies', 'trends',
            'detailed_analysis', 'screening_questions'
        }
        
        deserialized = {}
        for key, value in data.items():
            if key in json_fields:
                try:
                    deserialized[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    deserialized[key] = [] if key.endswith('s') else {}
            else:
                deserialized[key] = value
        return deserialized
    
    def store_job(self, job):
        try:
            data = self._serialize_data(job.model_dump())
            self.jobs.upsert(
                documents=[f"{job.title} {job.description}"],
                metadatas=[data],
                ids=[job.id]
            )
            logger.info(f"Job stored successfully: {job.id}")
        except Exception as e:
            logger.error(f"Failed to store job {job.id}: {e}")
            raise
    
    def store_candidate(self, candidate):
        try:
            data = self._serialize_data(candidate.model_dump())
            self.candidates.upsert(
                documents=[candidate.resume_text],
                metadatas=[data],
                ids=[candidate.id]
            )
            logger.info(f"Candidate stored successfully: {candidate.id}")
        except Exception as e:
            logger.error(f"Failed to store candidate {candidate.id}: {e}")
            raise
    
    def store_report(self, report):
        try:
            data = self._serialize_data(report.model_dump())
            self.reports.upsert(
                documents=[f"{report.job_title} {report.location}"],
                metadatas=[data],
                ids=[report.id]
            )
            logger.info(f"Report stored successfully: {report.id}")
        except Exception as e:
            logger.error(f"Failed to store report {report.id}: {e}")
            raise
    
    def get_all_jobs(self):
        from .models import Job
        try:
            results = self.jobs.get()
            if not results['metadatas']:
                return []
            
            jobs = []
            for metadata in results['metadatas']:
                try:
                    data = self._deserialize_data(metadata)
                    data['created_at'] = datetime.fromisoformat(data['created_at'])
                    jobs.append(Job(**data))
                except Exception as e:
                    logger.warning(f"Skipping invalid job record: {e}")
            
            logger.info(f"Retrieved {len(jobs)} jobs")
            return jobs
        except Exception as e:
            logger.error(f"Failed to retrieve jobs: {e}")
            return []
    
    def get_all_candidates(self):
        from .models import Candidate
        try:
            results = self.candidates.get()
            if not results['metadatas']:
                return []
            
            candidates = []
            for metadata in results['metadatas']:
                try:
                    data = self._deserialize_data(metadata)
                    data['created_at'] = datetime.fromisoformat(data['created_at'])
                    candidates.append(Candidate(**data))
                except Exception as e:
                    logger.warning(f"Skipping invalid candidate record: {e}")
            
            logger.info(f"Retrieved {len(candidates)} candidates")
            return candidates
        except Exception as e:
            logger.error(f"Failed to retrieve candidates: {e}")
            return []
    
    def get_reports(self):
        from .models import MarketReport
        try:
            results = self.reports.get()
            if not results['metadatas']:
                return []
            
            reports = []
            for metadata in results['metadatas']:
                try:
                    data = self._deserialize_data(metadata)
                    data['generated_at'] = datetime.fromisoformat(data['generated_at'])
                    reports.append(MarketReport(**data))
                except Exception as e:
                    logger.warning(f"Skipping invalid report record: {e}")
            
            logger.info(f"Retrieved {len(reports)} reports")
            return reports
        except Exception as e:
            logger.error(f"Failed to retrieve reports: {e}")
            return []