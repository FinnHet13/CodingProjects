import pandas as pd
import csv
from jobspy import scrape_jobs
import time

# ---- Scraping settings ----
RESULTS_WANTED = 50   # results per search term per site
HOURS_OLD      = 336  # only include jobs posted in the last 14 days
SLEEP_SECONDS  = 2    # pause between requests to avoid rate limiting

# Search terms used for Germany and Singapore
english_search_terms = [
    "Data Science",
    "Analytics",
    "Data Analytics",
    "Data Engineering",
    "Business Intelligence",
    "IT-Consulting"
]

# Spain uses the same English terms plus Spanish equivalents
spain_search_terms = english_search_terms + [
    "Ciencia de Datos",
    "Analítica",
    "Análisis de Datos",
    "Ingeniería de Datos",
    "Inteligencia de Negocios",
    "Consultoría de TI"
]


def safe_concat(df_list):
    """Concatenate a list of DataFrames, ignoring any empty ones."""
    non_empty = [df for df in df_list if not df.empty]
    return pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()


def scrape_country(country, search_terms, google_suffix, indeed_location=None):
    """
    Scrape job listings for every search term in a given country.

    Fetches from Indeed + Google, then separately from LinkedIn, and combines
    the results. Each job is tagged with its search term and country.

    Args:
        country (str): Country name used by LinkedIn and for tagging (e.g. "Germany")
        search_terms (list): Job titles/terms to search for
        google_suffix (str): Words added to each term for the Google search query
                             (e.g. "jobs deutschland" or "trabajo españa")
        indeed_location (str, optional): Location string for Indeed/Google.
                                         Defaults to country if not provided.
                                         Needed when the local name differs (e.g. "Deutschland").

    Returns:
        pd.DataFrame: All jobs found across every search term for this country
    """
    if indeed_location is None:
        indeed_location = country

    country_jobs = pd.DataFrame()

    for term in search_terms:
        print(f"Scraping jobs for: {term} in {country}")
        try:
            jobs_non_linkedin = scrape_jobs(
                site_name=["indeed", "google"],
                search_term=term,
                location=indeed_location,
                job_type="fulltime",
                google_search_term=f"{term} {google_suffix}",
                results_wanted=RESULTS_WANTED,
                hours_old=HOURS_OLD,
                company_logo=True,
                country_indeed=country,
            )

            jobs_linkedin = scrape_jobs(
                site_name=["linkedin"],
                search_term=term,
                location=country,
                job_type="fulltime",
                results_wanted=RESULTS_WANTED,
                hours_old=HOURS_OLD,
                linkedin_fetch_description=True,
                company_logo=True,
            )

            jobs = safe_concat([jobs_non_linkedin, jobs_linkedin])

            if not jobs.empty:
                jobs["search_term"] = term
                jobs["country"] = country
                country_jobs = safe_concat([country_jobs, jobs])
                print(f"Found {len(jobs)} jobs for {term} in {country}")
            else:
                print(f"No jobs found for {term} in {country}")

            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            print(f"Error scraping {term} in {country}: {e}")

    return country_jobs


# ---- Run scraping for all countries and combine ----
all_jobs = safe_concat([
    scrape_country("Germany",   english_search_terms, "jobs deutschland", indeed_location="Deutschland"),
    scrape_country("Spain",     spain_search_terms,   "trabajo españa"),
    scrape_country("Singapore", english_search_terms, "job singapore"),
])

# ---- Clean up and export ----
if not all_jobs.empty:
    columns_to_drop = [
        "company_url", "company_url_direct", "company_addresses",
        "company_num_employees", "company_revenue", "company_description",
        "min_amount", "max_amount", "salary_source", "interval", "currency"
    ]
    all_jobs = all_jobs.drop(columns=columns_to_drop, errors='ignore')
    all_jobs = all_jobs.drop_duplicates(subset=["title", "company"], keep="first")

    csv_path = "scripts/jobs.csv"
    all_jobs.to_csv(csv_path, quoting=csv.QUOTE_NONNUMERIC, escapechar="\\", index=False)

    print(f"Total unique jobs found: {len(all_jobs)}")
    print(f"Data exported to {csv_path}")
else:
    print("No jobs found in any country.")
