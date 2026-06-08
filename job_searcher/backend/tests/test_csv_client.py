"""
Test script for CSV Job Client module
    Purpose:
        Test the functionalities of the CsvJobClient module.
        Validate data retrieval and query filtering from a CSV file.
    Key Components:
        CsvJobClient: The main client that reads job data from a CSV file.
        JobResponse & JobListing: Data structures used to encapsulate job listings.
    Testing Approach:
        Utilizes pytest as the testing framework.
        Uses a temporary CSV fixture file to mimic real-world job listings.
"""
import sys
import os
import csv
import tempfile
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data import CsvJobClient, JobResponse, JobListing

# Sample test data
SAMPLE_JOBS = [
    {
        "id": "gd-1009664941547",
        "site": "glassdoor",
        "job_url": "https://www.glassdoor.de/job-listing/j?jl=1009664941547",
        "job_url_direct": "",
        "title": "Internship Data Science and AI",
        "company": "Gen Re",
        "location": "Cunningham",
        "date_posted": "2025-03-08",
        "job_type": "",
        "is_remote": False,
        "job_level": "",
        "job_function": "",
        "listing_type": "organic",
        "emails": "",
        "description": "**Markdown..",
        "company_industry": "",
        "company_logo": "https://media.glassdoor.com/sql/932658/gen-re-squarelogo-1425376828542.png",
        "search_term": "Data Science",
        "country": "Germany",
    },
    {
        "id": "gd-1009664811576",
        "site": "glassdoor",
        "job_url": "https://www.glassdoor.de/job-listing/j?jl=1009664811576",
        "job_url_direct": "",
        "title": "Praktikum Data Analyst (m/f/d)",
        "company": "Telefonica",
        "location": "München",
        "date_posted": "2025-03-08",
        "job_type": "",
        "is_remote": False,
        "job_level": "",
        "job_function": "",
        "listing_type": "organic",
        "emails": "-recruiting@telefonica.com",
        "description": "**Markdown..",
        "company_industry": "",
        "company_logo": "https://media.glassdoor.com/sql/5905123/virgin-media-o2-squareLogo-1698935143602.png",
        "search_term": "Data Science",
        "country": "Germany",
    },
    {
        "id": "gd-1009664484826",
        "site": "glassdoor",
        "job_url": "https://www.glassdoor.de/job-listing/j?jl=1009664484826",
        "job_url_direct": "",
        "title": "Internship: Data Analysis and Visualization (f/m/div)",
        "company": "Infineon Technologies",
        "location": "München",
        "date_posted": "2025-03-07",
        "job_type": "",
        "is_remote": False,
        "job_level": "",
        "job_function": "",
        "listing_type": "organic",
        "emails": "",
        "description": "**Markdown..",
        "company_industry": "",
        "company_logo": "https://media.glassdoor.com/sql/8915/infineon-technologies-squarelogo.png",
        "search_term": "Data Science",
        "country": "Germany",
    }
]

# CSV column order matching the sample data
CSV_COLUMNS = [
    "id", "site", "job_url", "job_url_direct", "title", "company", "location",
    "date_posted", "job_type", "is_remote", "job_level", "job_function",
    "listing_type", "emails", "description", "company_industry", "company_logo",
    "search_term", "country"
]


@pytest.fixture
def csv_file(tmp_path):
    """Create a temporary CSV file from sample data."""
    csv_path = tmp_path / "test_jobs.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, quoting=csv.QUOTE_NONNUMERIC, escapechar="\\")
        writer.writeheader()
        writer.writerows(SAMPLE_JOBS)
    return str(csv_path)


class TestCsvJobClient:
    """Test cases for CsvJobClient"""

    def test_get_all_data(self, csv_file):
        """Test retrieving all data from CSV"""
        client = CsvJobClient(csv_path=csv_file)
        result = client.get_all_data()

        assert len(result) == 3
        assert result[0]['title'] == "Internship Data Science and AI"
        assert result[1]['company'] == "Telefonica"
        assert result[2]['company'] == "Infineon Technologies"

    def test_query_by_search_term(self, csv_file):
        """Test querying jobs by search term"""
        client = CsvJobClient(csv_path=csv_file)

        # Test search for "Data Science" (should match all 3 jobs)
        result = client.query_by_search_term('Data Science')
        assert isinstance(result, JobResponse)
        assert result.search_term == 'data science'
        assert result.count == 3

        # Test search for a non-existent term
        result = client.query_by_search_term('Non-existent Term')
        assert result.count == 0
        assert len(result.jobs) == 0

    def test_empty_csv(self, tmp_path):
        """Test handling of an empty CSV (just headers)"""
        csv_path = tmp_path / "empty.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)
        client = CsvJobClient(csv_path=str(csv_path))
        assert client.get_all_data() == []

    def test_missing_csv(self, tmp_path):
        """Test handling of a missing CSV file"""
        client = CsvJobClient(csv_path=str(tmp_path / "nonexistent.csv"))
        assert client.get_all_data() == []


# Allows direct execution of the tests
if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
