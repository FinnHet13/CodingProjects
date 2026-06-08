"""
Consolidated Data Access Layer:
1. CSV-based job data ingestion - For retrieving job data from a local CSV file
2. Job Description API functionality - For retrieving job descriptions from external API

CsvJobClient Module Explanation:
    query_by_search_term():
    1. Fuzzy Matching:
        * Uses fuzzywuzzy to handle typos and partial matches. For example, "Data Sciene" will match "Data Science".
        * fuzz.partial_ratio is used to calculate similarity scores, with a threshold (min_score) to filter out low-relevance matches.
    2. Synonym Handling:
        * A SYNONYMS dictionary maps terms to their synonyms. For example, searching for "Analytics" will also match "Data Science".
        * The _get_synonyms method retrieves synonyms for the search term.
    3. Flexible Search:
        * The _calculate_match_score method searches across all fields (title, company, location, search_term).
        * Scores are weighted: exact matches are highest, synonym matches are slightly lower, and fuzzy matches are weighted even lower.
    4. Text Normalization:
        * The _normalize_text method ensures consistent matching by lowercasing and removing extra whitespace.
    5. Efficiency:
        * Data is loaded once at startup from the CSV file and cached in memory.
        * Results are ranked by relevance using a scoring system.
"""
import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
import pandas as pd
from rank_bm25 import BM25Okapi
import numpy as np
import re
import requests
import sqlite3

# ==== Sheet API Module Constants ====

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

JOB_TITLES = [
    "Data Scientist",
    "Business Analyst",
    "Software Engineer",
    "Product Manager",
    "Data Engineer",
    "Machine Learning Engineer",
    "Data Analyst",
    "Business Intelligence Analyst",
    "Project Manager",
    "DevOps Engineer",
    "Big Data Engineer",
    "Business Systems Analyst",
    "Data Architect"
]

JOB_SYNONYMS = {
    # Software
    "Junior Software Developer": "Software Development",
    "Senior Software Developer": "Software Development",
    "Software Developer": "Software Development",
    "Software Development Engineer": "Software Development",
    
    # Data Science
    "Junior Data Scientist": "Data Science",
    "Senior Data Scientist": "Data Science",
    "Data Scientist": "Data Science",
    
    # Machine Learning
    "Junior Machine Learning Engineer": "Machine Learning",
    "Machine Learning Engineer": "Machine Learning",
    
    # Data Analysis
    "Junior Data Analyst": "Data Analysis",
    "Data Analyst": "Data Analysis",
    
    # Business Analyst terms added
    "Junior Business Analyst": "Business Analysis",
    "Senior Business Analyst": "Business Analysis",
    "Business Analyst": "Business Analysis",
    
    # Trainee positions
    "Trainee Software Developer": "Software Development",
    "Trainee Data Scientist": "Data Science",
    "Trainee Machine Learning Engineer": "Machine Learning",
    "Trainee Data Analyst": "Data Analysis",
    "Trainee Business Analyst": "Business Analysis",
    "Traineeship Software Developer": "Software Development",
    "Traineeship Data Scientist": "Data Science",
    "Traineeship Machine Learning Engineer": "Machine Learning",
    "Traineeship Data Analyst": "Data Analysis",
    "Traineeship Business Analyst": "Business Analysis",

    # Full-time positions
    "Full-time Software Developer": "Software Development",
    "Full-time Data Scientist": "Data Science",
    "Full-time Machine Learning Engineer": "Machine Learning",
    "Full-time Data Analyst": "Data Analysis",
    "Full-time Business Analyst": "Business Analysis",
}

# ==== Sheet API Module Classes ====

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

# ==== Skills API Module Functions ====

def get_access_token():
    """
    Get access token from EMSI Lightcast Skills API.
    
    Makes an HTTP request to the EMSI Lightcast auth endpoint to retrieve an OAuth access token
    for subsequent API calls.
    
    Returns:
        str or None: The access token if successful, None otherwise
    """
    url = "https://auth.emsicloud.com/connect/token"
    payload = "client_id=11upn5xpu4dikqe3&client_secret=FVBb8tWw&grant_type=client_credentials&scope=emsi_open"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    
    response = requests.request("POST", url, data=payload, headers=headers)
    
    if response.status_code == 200:
        return response.json().get('access_token')
    else:
        print(f"Error getting token: {response.status_code}")
        print(response.text)
        return None

def get_skills_for_job(job_title):
    """
    Get skills and description for a job title using EMSI Lightcast Skills API.
    
    This function uses the EMSI Lightcast Skills API to extract skills associated with a job title 
    and retrieves a description. If no specific skills are found, it returns a general
    message.
    
    Args:
        job_title (str): The job title to retrieve skills for
        
    Returns:
        str: A description of the job skills or an error message
    """
    # Check if the job title exists in the synonym dictionary
    job_title = JOB_SYNONYMS.get(job_title, job_title)  # Use synonym if available

    # Get access token
    token = get_access_token()
    if not token:
        return "Error: Unable to authenticate with API"
    
    # URL for the API endpoint
    url = "https://emsiservices.com/skills/versions/latest/extract"
    
    # Request payload with job title
    payload = {
        "text": job_title
    }

    # Header with API token
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    # Send API request
    response = requests.post(url, json=payload, headers=headers)

    # Check if request was successful
    if response.status_code == 200:
        data = response.json()
        # If no specific skills are found, return a general description of the job title
        if data.get("data"):
            # Return the domain description if skills are found
            description = data["data"][0]["skill"]["description"]
        else:
            # If no skills, return the job domain description
            description = f"No specific skills found for {job_title}"
        return description
    else:
        print(f"Error: {response.status_code}")
        return f"Error fetching description for {job_title}"

def save_job_skills_to_db(job_titles, db_file="backend/job_descriptions.db"):
    """
    Save job descriptions to SQLite database.
    
    Creates a SQLite database and table if they don't exist, then fetches and stores
    descriptions for each job title using the EMSI Lightcase Skills API.
    
    Args:
        job_titles (list): List of job titles to fetch descriptions for
        db_file (str): Path to the SQLite database file
        
    Returns:
        None
    """
    # Connect to SQLite database (or create it if it doesn't exist)
    os.makedirs(os.path.dirname(db_file), exist_ok=True)
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS job_descriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_title TEXT NOT NULL,
            description TEXT
        )
    ''')

    # Insert job descriptions into the table
    for job_title in job_titles:
        description = get_skills_for_job(job_title)
        cursor.execute('''
            INSERT INTO job_descriptions (job_title, description)
            VALUES (?, ?)
        ''', (job_title, description))

    # Commit the transaction and close the connection
    conn.commit()
    conn.close()

def get_job_description_from_db(job_title, db_file="backend/job_descriptions.db"):
    """
    Fetch a job description from the SQLite database.
    
    Searches the database for a job description matching the provided job title,
    considering synonyms from the JOB_SYNONYMS dictionary.
    
    Args:
        job_title (str): The job title to retrieve the description for
        db_file (str): Path to the SQLite database file
        
    Returns:
        str: The job description if found, or an error/not found message
    """
    job_title = JOB_SYNONYMS.get(job_title, job_title)  # Use synonym if available
    
    try:
        # Connect to SQLite database
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # Query the database for job description
        cursor.execute('''
            SELECT description FROM job_descriptions 
            WHERE job_title LIKE ?
        ''', (f"%{job_title}%",))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return result[0]
        else:
            return f"No description found for {job_title}"
            
    except Exception as e:
        print(f"Database error: {e}")
        return f"Error retrieving description for {job_title}"
