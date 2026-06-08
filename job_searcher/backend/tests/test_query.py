"""
Integration test for CsvJobClient using the real jobs.csv data file.

This test reads from scripts/jobs.csv to verify the client works end-to-end.
"""
import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data import CsvJobClient

# Path to the real CSV data file
CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'scripts',
    'jobs.csv'
)

# Create a test client directly
def create_test_client():
    print(f"Using CSV path: {CSV_PATH}")
    return CsvJobClient(csv_path=CSV_PATH)

# Create an instance of the client
test_client = create_test_client()

# Fetch all data from the CSV
print("Fetching all data...")
all_data = test_client.get_all_data()
print(f"Retrieved {len(all_data)} records")

# Print the first record to verify the data structure
if all_data:
    print("\nFirst record:")
    first_record = all_data[0]
    print(json.dumps({k: str(v)[:100] for k, v in first_record.items()}, indent=2))
else:
    print("No records found!")

# Try several search terms, a job title, a company name, a location, and a non-existent term
search_terms = ['Data Science', 'Infineon Technologies', 'München', 'Non-existent Term']

# Query for each search term and display the results
for search_term in search_terms:
    print(f"\nQuerying for: {search_term}")
    response = test_client.query_by_search_term(search_term)
    print(f"Found {response.count} matching jobs for '{search_term}'")
    
    if response.count > 0:
        print("\nMatching job titles:")
        for job in response.jobs[:5]:  # Show first 5 matches to avoid flooding console
            print(f"- {job.title} at {job.company}")
    else:
        print(f"No matches found for '{search_term}'")

# Final check if no matches were found for any term
if not any(test_client.query_by_search_term(term).count > 0 for term in search_terms):
    print("\nWARNING: No matches found for any test search terms. Check if the data source contains expected content.")