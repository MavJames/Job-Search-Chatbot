import logging

import pandas as pd
from fastmcp import FastMCP
from jobspy import scrape_jobs

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("jobspy-server")


def _safe_scrape(site_name, search_term, location, results_wanted):
    """Helper to safely scrape jobs with error handling"""
    try:
        jobs = scrape_jobs(
            site_name=site_name,
            search_term=search_term,
            location=location,
            results_wanted=results_wanted,
        )
        return jobs
    except Exception as e:
        logger.error(f"Error scraping jobs for '{search_term}' in '{location}': {e}")
        return pd.DataFrame()


@mcp.tool()
def search_jobs(query: str, location: str = "", limit: int = 10):
    """
    Search job listings using python-jobspy.
    Returns the most recent job postings that match a given title or keyword.
    """
    jobs = _safe_scrape(
        site_name=["indeed", "linkedin"],
        search_term=query,
        location=location,
        results_wanted=limit,
    )

    if jobs.empty:
        return []

    return jobs.to_dict(orient="records")


@mcp.tool()
def jobs_by_company(company: str, limit: int = 10):
    """
    Search for job listings posted by a specific company.
    """
    jobs = _safe_scrape(
        site_name=["linkedin", "indeed"],
        search_term=company,
        location="",
        results_wanted=limit,
    )

    if jobs.empty:
        return []

    # Filter by company name (case-insensitive)
    if "company" in jobs.columns:
        company_jobs = jobs[jobs["company"].str.contains(company, case=False, na=False)]
        return company_jobs.to_dict(orient="records")

    return []


@mcp.tool()
def top_companies_for_role(role: str, location: str = "", limit: int = 50):
    """
    Finds which companies post the most listings for a specific role.
    Returns a summary count by company.
    """
    jobs = _safe_scrape(
        site_name=["linkedin", "indeed"],
        search_term=role,
        location=location,
        results_wanted=limit,
    )

    if jobs.empty or "company" not in jobs.columns:
        return []

    summary = jobs["company"].value_counts().reset_index()
    summary.columns = ["company", "num_postings"]
    return summary.head(10).to_dict(orient="records")


@mcp.tool()
def summarize_salary_trends(role: str, location: str = "", limit: int = 100):
    """
    Scrape salary data for a role and summarize min, max, and average.
    """
    jobs = _safe_scrape(
        site_name=["indeed"], search_term=role, location=location, results_wanted=limit
    )

    if jobs.empty or "salary" not in jobs.columns:
        return {"message": "No salary data available for that role or location."}

    salary_data = jobs["salary"].dropna()
    if salary_data.empty:
        return {"message": "No salary data available for that role or location."}

    # Convert salary strings to numeric values (JobSpy standardizes some fields)
    numeric_salaries = pd.to_numeric(salary_data, errors="coerce").dropna()

    if numeric_salaries.empty:
        return {"message": "Could not parse salary data."}

    return {
        "count": len(numeric_salaries),
        "min_salary": float(numeric_salaries.min()),
        "max_salary": float(numeric_salaries.max()),
        "avg_salary": float(numeric_salaries.mean()),
    }


@mcp.tool()
def remote_jobs(role: str, limit: int = 10):
    """
    Quickly find remote jobs for a given role.
    """
    jobs = _safe_scrape(
        site_name=["linkedin", "indeed"],
        search_term=f"{role} remote",
        location="",
        results_wanted=limit,
    )

    if jobs.empty:
        return []

    return jobs.to_dict(orient="records")


if __name__ == "__main__":
    mcp.run()
