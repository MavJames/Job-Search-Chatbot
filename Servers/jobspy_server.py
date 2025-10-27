from fastmcp import FastMCP
from jobspy import scrape_jobs
import pandas as pd

mcp = FastMCP("jobspy-server")

@mcp.tool()
def search_jobs(query: str, location: str = "", limit: int = 10):
    """
    Search job listings using python-jobspy.
    Returns the most recent job postings that match a given title or keyword.
    """
    jobs = scrape_jobs(
        site_name=["indeed", "linkedin"],
        search_term=query,
        location=location,
        results_wanted=limit
    )
    return jobs.to_dict(orient="records")


@mcp.tool()
def jobs_by_company(company: str, limit: int = 10):
    """
    Search for job listings posted by a specific company.
    """
    jobs = scrape_jobs(
        site_name=["linkedin", "indeed"],
        search_term=company,
        results_wanted=limit
    )
    company_jobs = jobs[jobs["company"].str.contains(company, case=False, na=False)]
    return company_jobs.to_dict(orient="records")


@mcp.tool()
def top_companies_for_role(role: str, location: str = "", limit: int = 50):
    """
    Finds which companies post the most listings for a specific role.
    Returns a summary count by company.
    """
    jobs = scrape_jobs(
        site_name=["linkedin", "indeed"],
        search_term=role,
        location=location,
        results_wanted=limit
    )
    summary = jobs["company"].value_counts().reset_index()
    summary.columns = ["company", "num_postings"]
    return summary.head(10).to_dict(orient="records")


@mcp.tool()
def summarize_salary_trends(role: str, location: str = "", limit: int = 100):
    """
    Scrape salary data for a role and summarize min, max, and average.
    """
    jobs = scrape_jobs(
        site_name=["indeed"],
        search_term=role,
        location=location,
        results_wanted=limit
    )
    salary_data = jobs["salary"].dropna()
    if salary_data.empty:
        return {"message": "No salary data available for that role or location."}

    # Convert salary strings to numeric values (JobSpy standardizes some fields)
    numeric_salaries = pd.to_numeric(salary_data, errors="coerce").dropna()
    return {
        "count": len(numeric_salaries),
        "min_salary": float(numeric_salaries.min()),
        "max_salary": float(numeric_salaries.max()),
        "avg_salary": float(numeric_salaries.mean())
    }


@mcp.tool()
def remote_jobs(role: str, limit: int = 10):
    """
    Quickly find remote jobs for a given role.
    """
    jobs = scrape_jobs(
        site_name=["linkedin", "indeed"],
        search_term=f"{role} remote",
        results_wanted=limit
    )
    return jobs.to_dict(orient="records")


if __name__ == "__main__":
    mcp.run()
