"""
Business Logic Module - Contains the business logic for the application.
"""
import os
import pandas as pd
from data import CsvJobClient

CSV_PATH = os.path.join(os.path.dirname(__file__), 'scripts', 'jobs.csv')
DESCRIPTIONS_PATH = os.path.join(os.path.dirname(__file__), 'backend', 'description.csv')

# Two variables initialised to None here so they exist at module scope before any function runs.
# Both are assigned their real values in the following functions.
# _csv_client will hold the loaded CsvJobClient instance (CSV in memory + BM25 search index).
# _csv_mtime will hold the jobs.csv modification timestamp from when that load happened,
# so get_csv_client() can detect that the "Daily Job Update" Task Scheduler task has
# overwritten jobs.csv with the new jobs and a reload can be triggered without
# needing to restart the Flask app.
_csv_client = None
_csv_mtime = None

# Load description.csv once at startup into a dict: {lowercase title -> description text}
try:
    _desc_df = pd.read_csv(DESCRIPTIONS_PATH)
    _descriptions = dict(zip(_desc_df['Job Title'].str.lower(), _desc_df['Description']))
except Exception as e:
    print(f"Warning: Could not load job descriptions from {DESCRIPTIONS_PATH}: {e}")
    _descriptions = {}

def get_csv_client():
    """
    Get or create the CsvJobClient instance.

    On first call, the full CSV is loaded into memory and a BM25 search index is built.
    That in-memory copy is reused for every subsequent request (no file reads per request).

    On each call, the file's modification time is checked against when it was last loaded.
    If jobs.csv has changed since the last load, the CSV and index are rebuilt in memory.
    This allows the app to automatically serve the latest job listings after the
    "Daily Job Update" Windows Task Scheduler task runs scrape_jobs.py each morning,
    without requiring a manual Flask app restart.

    Returns:
        CsvJobClient: The singleton client instance
    """
    global _csv_client, _csv_mtime
    current_mtime = os.path.getmtime(CSV_PATH)
    if _csv_client is None or current_mtime != _csv_mtime:
        _csv_client = CsvJobClient(csv_path=CSV_PATH)
        _csv_mtime = current_mtime
    return _csv_client

def search_jobs(search_term, job_levels=None):
    """
    Search for jobs by search term using CSV client.

    Performs a search for job listings matching the provided search term using
    Rank BM25 provided by the CsvJobClient.
    The results are converted from Pydantic models to standard dictionaries for
    easier processing and serialization.

    Args:
        search_term (str): The term to search for in job listings
        job_levels (List[str], optional): List of job levels to filter by (e.g. ['entry level', 'internship'])

    Returns:
        Tuple[int, List[Dict]]: A tuple containing the count of matching jobs and
                               a list of job dictionaries
    """
    client = get_csv_client()
    result = client.query_by_search_term(search_term)

    # Convert Pydantic models to dictionaries
    jobs = pd.DataFrame([job.__dict__ for job in result.jobs])

    if job_levels and len(jobs) > 0 and 'job_level' in jobs.columns:
        normalized = [l.lower() for l in job_levels]
        jobs = jobs[jobs['job_level'].fillna('').str.lower().isin(normalized)]

    total = len(jobs)
    jobs = jobs.to_dict(orient='records')

    return total, jobs

def get_job_description(search_term):
    """
    Look up a plain-English overview for a job title from description.csv.

    Args:
        search_term (str): The job title to look up (e.g. "Data Analyst")

    Returns:
        str: Description text, or empty string if not found
    """
    term = search_term.lower()
    for title, desc in _descriptions.items():
        if term in title or title in term:
            return desc
    return ""

def get_analytics_data():
    """
    Get data for analytics dashboard.
    
    Retrieves all job data and processes it to generate analytics charts including:
    1. Count of job listings by search term
    2. Distribution of remote vs. non-remote jobs
    3. Count of jobs by company industry
    4. Timeline of job listings over time
    
    The data is processed using pandas for efficient aggregation and formatting.
    
    Returns:
        Dict[str, Dict]: A dictionary containing data for various analytics charts,
                        with keys for each chart type and values as data dictionaries
    """
    # Fetch all job data from CSV using the client
    client = get_csv_client()
    data = client.get_all_data()
    
    # Convert data to a pandas DataFrame
    df = pd.DataFrame(data)

    # Chart 1: Count of each search_term (Top 5 only)
    search_term_count = df['search_term'].value_counts().head(5)

    # Chart 2: Count of 'is_remote'
    is_remote_count = df['is_remote'].value_counts()

    # Chart 3: Count of each company_industry
    company_industry_count = df['company_industry'].value_counts()

    # Chart 4: Job listings over time (based on date_posted)
    df['date_posted'] = pd.to_datetime(df['date_posted'], errors='coerce')
    job_posted_over_time = df.groupby(df['date_posted'].dt.date).size()

    # Convert date keys to strings (for JSON serialization)
    job_posted_over_time = {str(date): count for date, count in job_posted_over_time.items()}
    
    return {
        'search_term_count': search_term_count.to_dict(),
        'is_remote_count': is_remote_count.to_dict(),
        'company_industry_count': company_industry_count.to_dict(),
        'job_posted_over_time': job_posted_over_time
    }

def print_job_titles_and_companies():
    """
    Print all job titles and their companies.
    
    A utility function that fetches all jobs and prints each job's title and company
    to the console. This function is primarily used for debugging and data exploration.
    
    Returns:
        None: This function only prints to the console and doesn't return a value
    """
    client = get_csv_client()
    data = client.get_all_data()
    for job in data:
        print(f"{job['title']} at {job['company']}")
