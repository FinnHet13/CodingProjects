import pandas as pd
import csv
import os
from jobspy import scrape_jobs
import time

# Search terms for Germany and Singapore (English)
english_search_terms = [
    "Data Science",
    "Analytics",
    "Data Analytics",
    "Data Engineering",
    "Business Intelligence",
    "IT-Consulting"
]

# Search terms for Spain (both English and Spanish)
spain_search_terms = [
    "Data Science",
    "Analytics",
    "Data Analytics",
    "Data Engineering",
    "Business Intelligence",
    "IT-Consulting",
    "Ciencia de Datos",
    "Analítica",
    "Análisis de Datos",
    "Ingeniería de Datos",
    "Inteligencia de Negocios",
    "Consultoría de TI"
]

# Create an empty DataFrame to store all results
all_jobs = pd.DataFrame()

# Helper function to safely concatenate DataFrames, handling empty cases
def safe_concat(df_list):
    # Filter out empty DataFrames
    non_empty_dfs = [df for df in df_list if not df.empty]

    # If all DataFrames are empty, return an empty DataFrame
    if not non_empty_dfs:
        return pd.DataFrame()

    # Concatenate non-empty DataFrames
    return pd.concat(non_empty_dfs, ignore_index=True)

# Loop through each search term for Germany
for term in english_search_terms:
    print(f"Scraping jobs for: {term} in Germany")

    # Format Google search term
    google_search = f"{term} jobs deutschland"

    try:
        # Scrape jobs for non-LinkedIn sites with location "Deutschland"
        # Excluding Glassdoor as it is giving error
        jobs_non_linkedin = scrape_jobs(
            site_name=["indeed", "google"],
            search_term=term,
            location="Deutschland",
            job_type="fulltime",
            google_search_term=google_search,
            results_wanted=50,
            hours_old=336,  # 14 days
            company_logo=True,
            country_indeed="Germany",
        )

        # Scrape LinkedIn jobs with location "Germany"
        jobs_linkedin = scrape_jobs(
            site_name=["linkedin"],
            search_term=term,
            location="Germany",  # Different location for LinkedIn
            job_type="fulltime",
            results_wanted=50,
            hours_old=336,  # 14 days
            linkedin_fetch_description=True,
            company_logo=True,
        )

        # Safely combine the results from both calls
        jobs = safe_concat([jobs_non_linkedin, jobs_linkedin])

        # Only proceed if we have results
        if not jobs.empty:
            jobs["search_term"] = term
            jobs["country"] = "Germany"

            # Append to the overall jobs DataFrame
            all_jobs = safe_concat([all_jobs, jobs])

            print(f"Found {len(jobs)} jobs for {term} in Germany")
        else:
            print(f"No jobs found for {term} in Germany")

        # Add delay to avoid rate limiting
        time.sleep(2)

    except Exception as e:
        print(f"Error scraping {term} in Germany: {e}")

# Loop through each search term for Spain
for term in spain_search_terms:
    print(f"Scraping jobs for: {term} in Spain")

    # Format Google search term for Spain
    google_search = f"{term} trabajo españa"

    try:
        # Scrape jobs for non-LinkedIn sites with location "Spain"
        # Excluding Glassdoor as it's not supported for Spain
        jobs_non_linkedin = scrape_jobs(
            site_name=["indeed", "google"],  # Removed "glassdoor"
            search_term=term,
            location="Spain",
            job_type="fulltime",
            google_search_term=google_search,
            results_wanted=50,
            hours_old=336,  # 14 days
            company_logo=True,
            country_indeed="Spain",
        )

        # Scrape LinkedIn jobs with location "Spain"
        jobs_linkedin = scrape_jobs(
            site_name=["linkedin"],
            search_term=term,
            location="Spain",
            job_type="fulltime",
            results_wanted=50,
            hours_old=336,  # 14 days
            linkedin_fetch_description=True,
            company_logo=True,
        )

        # Safely combine the results from both calls
        jobs = safe_concat([jobs_non_linkedin, jobs_linkedin])

        # Only proceed if we have results
        if not jobs.empty:
            jobs["search_term"] = term
            jobs["country"] = "Spain"

            # Append to the overall jobs DataFrame
            all_jobs = safe_concat([all_jobs, jobs])

            print(f"Found {len(jobs)} jobs for {term} in Spain")
        else:
            print(f"No jobs found for {term} in Spain")

        # Add delay to avoid rate limiting
        time.sleep(2)

    except Exception as e:
        print(f"Error scraping {term} in Spain: {e}")

# Loop through each search term for Singapore
for term in english_search_terms:
    print(f"Scraping jobs for: {term} in Singapore")

    # Format Google search term for Singapore
    google_search = f"{term} job singapore"

    try:
        # Scrape jobs for non-LinkedIn sites with location "Singapore"
        # Excluding Glassdoor as it is giving error
        jobs_non_linkedin = scrape_jobs(
            site_name=["indeed", "google"],
            search_term=term,
            location="Singapore",
            job_type="fulltime",
            google_search_term=google_search,
            results_wanted=50,
            hours_old=336,  # 14 days
            company_logo=True,
            country_indeed="Singapore",
        )

        # Scrape LinkedIn jobs with location "Singapore"
        jobs_linkedin = scrape_jobs(
            site_name=["linkedin"],
            search_term=term,
            location="Singapore",
            job_type="fulltime",
            results_wanted=50,
            hours_old=336,  # 14 days
            linkedin_fetch_description=True,
            company_logo=True,
        )

        # Safely combine the results from both calls
        jobs = safe_concat([jobs_non_linkedin, jobs_linkedin])

        # Only proceed if we have results
        if not jobs.empty:
            jobs["search_term"] = term
            jobs["country"] = "Singapore"

            # Append to the overall jobs DataFrame
            all_jobs = safe_concat([all_jobs, jobs])

            print(f"Found {len(jobs)} jobs for {term} in Singapore")
        else:
            print(f"No jobs found for {term} in Singapore")

        # Add delay to avoid rate limiting
        time.sleep(2)

    except Exception as e:
        print(f"Error scraping {term} in Singapore: {e}")

# Drop the specified columns
if not all_jobs.empty:
    columns_to_drop = [
        "company_url",
        "company_url_direct",
        "company_addresses",
        "company_num_employees",
        "company_revenue",
        "company_description",
        "min_amount",
        "max_amount",
        "salary_source",
        "interval",
        "currency"
    ]

    all_jobs = all_jobs.drop(columns=columns_to_drop, errors='ignore')

    # Remove duplicates based on job title and company
    all_jobs = all_jobs.drop_duplicates(subset=["title", "company"], keep="first")

    # Export to CSV
    csv_path = "scripts/jobs.csv"
    all_jobs.to_csv(csv_path, quoting=csv.QUOTE_NONNUMERIC, escapechar="\\", index=False)

    print(f"Total unique jobs found: {len(all_jobs)}")
    print(f"Data exported to {csv_path}")
else:
    print("No jobs found in any country.")