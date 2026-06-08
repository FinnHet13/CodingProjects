"""
Data Access Layer - reads job listings from a local CSV file.

CsvJobClient is the main class here. It:
    1. Loads jobs.csv into memory once at startup
    2. Builds a BM25 search index over job titles, companies, locations, and countries
    3. Expands search queries with synonyms (e.g. "Analytics" also matches "Data Science")
    4. Returns ranked results for any search term or location query
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
import pandas as pd
from rank_bm25 import BM25Okapi
import numpy as np
import re
# ==== Search Constants ====

# Synonym dictionary for search term matching
SYNONYMS = {
    # Variations of job titles and related terms
    "researcher": ["researcher", "research assistant", "research associate", "scientist"],

    "data scientist": ["data science", "statistical analysis"],

    "data analyst": ["analytics specialist", "data insights analyst", "data analysis", "data analytics", "analytics"],

    "bi analyst": ["business intelligence analyst", "data visualization specialist", "reporting analyst", "bi", "business intelligence", "data visualization", "reporting", "dashboards", "data insights"],
    
    "data engineer": ["data architect", "data pipeline engineer", "big data engineer", "data engineering", "etl", "extract transform load", "data pipeline", "data integration", "data architecture", "data modeling", "data warehousing"],

    "machine learning engineer": ["ml engineer", "ml", "ai engineer", "deep learning engineer", "generative ai engineer", "genAI engineer", "artificial intelligence", "ai", "machine learning", "deep learning", "neural networks", "natural language processing", "nlp", "computer vision", "predictive analytics", "descriptive analytics", "prescriptive analytics", "forecasting", "predictive modeling"],

    "business analyst": ["analytics consultant"],

    # Tools and Technologies
    "python": ["r", "sql", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch"],
    "sql": ["structured query language", "database query", "data query"],
    "power bi": ["tableau", "qlikview", "looker", "business intelligence tools", "bi tools"],
    "tableau": ["power bi", "qlikview", "looker", "data visualization tools"],
    "sap": ["erp", "sap hana", "sap bi", "enterprise resource planning"],
    "big data": ["hadoop", "spark", "kafka", "hive", "big data technologies"],
    "cloud": ["aws", "azure", "google cloud", "gcp", "cloud computing"],

    # Methodologies and Domains
    "forecasting": ["predictive analytics", "time series analysis", "predictive modeling"],
    "natural language processing": ["nlp", "text mining", "text analytics", "language processing"],
    "computer vision": ["image processing", "image recognition", "cv"],
    "statistical analysis": ["statistics", "data analysis", "quantitative analysis", "econometrics"],

    # Location Synonyms (German-English and Common Variations)
    "munich": ["münchen", "munchen"],
    "berlin": ["berlin, be, de", "berlin-kreuzberg"],
    "darmstadt": ["darmstadt, he, de"],
    "dresden": ["dresden, sn, de"],
    "hamburg": ["hamburg, hh, de"],
    "heidelberg": ["heidelberg, bw, de"],
    "cologne": ["köln", "koeln"],
    "frankfurt": ["frankfurt am main", "frankfurt/main"],
    "stuttgart": ["stuttgart, bw, de"],
    "düsseldorf": ["duesseldorf", "dusseldorf", "düsseldorf, nw, de"],
    "bonn": ["bonn, nw, de"],
    "aachen": ["aachen, nw, de"],
    "heilbronn": ["heilbronn, bw, de"],
    "leipzig": ["leipzig, sn, de"],
    "kassel": ["kassel, he, de"],
    "mainz": ["mainz, rp, de"],
    "bremen": ["bremen, hb, de"],
    "freiburg": ["freiburg, bw, de"],
    "grünwald": ["gruenwald"],
    "grünheide": ["gruenheide", "grünheide (mark)"],
    "eschborn": ["eschborn, he, de"],
    "metzingen": ["metzingen, bw, de"],
    "ottobeuren": ["ottobeuren, by, de"],
    "rietheim-weilheim": ["rietheim-weilheim, bw, de"],
    "sinzing": ["sinzing, by, de"],
    "bad homburg": ["bad homburg vor der höhe", "bad homburg, he, de"],
    "biberach": ["biberach an der riß", "biberach, bw, de"],
    "garching": ["garching bei münchen", "garching, by, de"],
    "weiden": ["weiden, by, de"],
    "potsdam": ["potsdam, bb, de"],
    "dortmund": ["dortmund, nw, de"],
    "neckarsulm": ["neckarsulm, bw, de"],
    "hasbergen": ["hasbergen, nw, de"],
    "lautzenhausen": ["lautzenhausen, rp, de"],
    "giessen": ["gießen", "giessen, he, de"],
    "oberursel": ["oberursel, he, de"],
    "nörvenich": ["noervenich", "nörvenich, nw, de"],
    "reutlingen": ["reutlingen, bw, de"],
    "kaiserslautern": ["kaiserslautern, rp, de"],
    "cunningham": ["cunningham, by, de"],
}

# ==== Description API Module Constants ====

# ==== Classes ====

class JobListing(BaseModel):
    """Pydantic model for job listing data"""
    id: str
    site: str
    job_url: str
    job_url_direct: Optional[str] = None
    title: str
    company: str
    location: str
    date_posted: Optional[str] = None
    job_type: Optional[str] = None
    is_remote: Optional[bool] = None
    job_level: Optional[str] = None
    job_function: Optional[str] = None
    listing_type: Optional[str] = None
    emails: Optional[str] = None
    description: Optional[str] = None
    company_industry: Optional[str] = None
    company_logo: Optional[str] = None
    search_term: str
    country: Optional[str] = None
    
    @validator("is_remote", pre=True)
    def empty_string_to_none_for_bool(cls, value):
        """
        Validates and converts the is_remote field value.
        
        Args:
            value: The input value to validate/convert
            
        Returns:
            bool or None: Converted boolean value or None if empty
        """
        if value == "":
            return None
        if isinstance(value, str):
            value_lower = value.lower()
            if value_lower in ("true", "1", "yes"):
                return True
            elif value_lower in ("false", "0", "no"):
                return False
        return value

    @validator("*", pre=True)
    def nan_to_none(cls, value):
        """Convert pandas NaN values to None for all optional fields."""
        if isinstance(value, float) and pd.isna(value):
            return None
        return value

class JobResponse(BaseModel):
    """Pydantic model for API response"""
    search_term: str
    count: int
    jobs: List[JobListing]


class CsvJobClient:
    """Client for retrieving job data from a local CSV file."""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._data: List[Dict[str, Any]] = []
        # BM25 index built once at load time
        self._bm25 = None
        self._load_data()

    def _load_data(self):
        try:
            df = pd.read_csv(self.csv_path, dtype={'id': str})
            df = df.fillna('')
            self._data = df.to_dict(orient='records')
            # Tokenize each job's combined fields; BM25 operates on token lists
            corpus = [
                self._tokenize(' '.join([
                    job.get('title', ''),
                    job.get('company', ''),
                    job.get('location', ''),
                    job.get('search_term', ''),
                    job.get('country', ''),
                ]))
                for job in self._data
            ]
            self._bm25 = BM25Okapi(corpus)
            print(f"Loaded {len(self._data)} jobs from {self.csv_path}")
        except Exception as e:
            print(f"Error loading CSV data from {self.csv_path}: {e}")
            self._data = []

    def get_all_data(self) -> List[Dict[str, Any]]:
        return self._data

    def _normalize_text(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text.lower().strip())

    def _tokenize(self, text: str) -> List[str]:
        return self._normalize_text(text).split()

    def _get_synonyms(self, term: str) -> List[str]:
        term = self._normalize_text(term)
        for key, synonyms in SYNONYMS.items():
            if term == key or term in synonyms:
                return [key] + synonyms
        return [term]

    def query_by_search_term(self, search_term: str, min_score: float = 0.5) -> JobResponse:
        """
        Query job listings using BM25 ranking.
        Synonyms are appended to the query to improve recall.
        Compound queries like "Germany Data Analyst" work natively — each
        token is scored independently and summed by BM25.

        Args:
            search_term: Term to search (title, company, location, country, or combination)
            min_score: Minimum BM25 score to include a result (scores are unbounded;
                       0.5 filters near-zero matches)

        Returns:
            JobResponse with ranked matching job listings
        """
        if self._bm25 is None:
            return JobResponse(search_term=search_term, count=0, jobs=[])

        normalized = self._normalize_text(search_term)
        # Expand query with synonyms so e.g. "analytics" also surfaces "data science" jobs
        synonyms = self._get_synonyms(normalized)
        query_tokens = self._tokenize(' '.join(set([normalized] + synonyms)))

        scores = self._bm25.get_scores(query_tokens)

        # Keep only results above threshold, ranked by score
        indices = np.where(scores >= min_score)[0]
        indices = indices[np.argsort(scores[indices])[::-1]]

        job_listings = []
        for idx in indices:
            try:
                job_listings.append(JobListing(**self._data[idx]))
            except Exception as e:
                print(f"Error parsing job data: {e}")

        return JobResponse(
            search_term=search_term,
            count=len(job_listings),
            jobs=job_listings
        )
