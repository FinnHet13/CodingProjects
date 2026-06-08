from flask import Flask, render_template, request, make_response, g, redirect, url_for
import os
import json
from logic import search_jobs, get_job_description, get_analytics_data

FRONTEND_TEMPLATE_FOLDER = os.path.join(os.path.dirname(__file__), 'frontend', 'templates')

app = Flask(__name__,
            static_folder='frontend/static', 
            template_folder=FRONTEND_TEMPLATE_FOLDER)

@app.route('/')
def analytics():
    """
    Render the analytics dashboard homepage.
    
    Retrieves analytics data from the business logic layer and passes it to the
    analytics template for rendering charts and visualizations. The analytics include
    job counts by search term, remote work status, company industry, and job postings
    over time.
    
    Returns:
        HTML: Rendered analytics.html template with chart data
    """
    # Get analytics data from logic layer
    analytics_data = get_analytics_data()
    
    # Pass the processed data to the template
    return render_template(
        'analytics.html',
        search_term_count=analytics_data['search_term_count'],
        is_remote_count=analytics_data['is_remote_count'],
        company_industry_count=analytics_data['company_industry_count'],
        job_posted_over_time=analytics_data['job_posted_over_time']
    )

@app.route('/jobs', methods=['GET', 'POST'])
def index_post():
    """
    Handle job search requests and display search results.
    
    This function processes both GET and POST requests for job searches, with a 
    preference for GET parameters to maintain URL visibility of search terms.
    It performs the job search, handles pagination, retrieves relevant job descriptions,
    and returns either a rendered HTML template or JSON response based on the request's
    Accept header.
    
    Args (via request):
        search_term (str): The term to search for jobs
        page (int, optional): Current page for pagination, defaults to 1
        limit (int, optional): Number of results per page, defaults to 20
    
    Returns:
        HTML or JSON: Rendered index.html template with search results or
                     JSON response containing jobs data based on the Accept header
    """
    # Handle both GET and POST requests, but prioritize GET for visibility in the URL
    search_term = request.args.get('search_term', '')
    
    # Get pagination parameters
    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 20))
    
    # Fall back to POST data if no GET parameter and method is POST
    if not search_term and request.method == 'POST':
        search_term = request.form.get('search_term', '')
    
    # If no search term provided, render the empty search form
    if not search_term:
        return render_template('index.html')
    
    # Get optional job level filters (multi-value: ?job_level=entry+level&job_level=internship)
    job_levels = request.args.getlist('job_level') or None

    # Fetch jobs matching the search term
    total_count, all_jobs = search_jobs(search_term, job_levels=job_levels)

    # Apply pagination
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    paginated_jobs = all_jobs[start_idx:end_idx] if start_idx < len(all_jobs) else []
    
    # Get job description for the search term
    description = get_job_description(search_term)
    
    # Log some info for debugging
    print(f"\nQuerying for: {search_term}")
    print(f"Found {total_count} matching jobs")
    print(f"Returning page {page} with {len(paginated_jobs)} jobs")
    
    # Check if request wants JSON
    if request.headers.get('Accept') == 'application/json':
        response_data = {
            "jobs": paginated_jobs,
            "total_count": total_count,
            "db_description": description,
            "page": page,
            "limit": limit,
            "has_more": (start_idx + len(paginated_jobs)) < total_count
        }
        return json.dumps(response_data), 200, {'Content-Type': 'application/json'}
    
    # Create response with template and cookie
    response = make_response(render_template('index.html', 
                                           jobs=paginated_jobs,
                                           total_count=total_count, 
                                           search_term=search_term,
                                           description=description,
                                           page=page,
                                           limit=limit))
    response.set_cookie('search_term', json.dumps(search_term, indent=4))
    return response

@app.route('/jobs/<search_term>', methods=['GET'])
def index_by_city(search_term):
    """
    Redirect search_term URL path to the /jobs endpoint with a query parameter.
    
    This route handles the SEO-friendly URL pattern /jobs/{search_term} and
    redirects it to the standard /jobs?search_term={search_term} format to 
    maintain consistent routing while supporting both URL patterns.
    
    Args:
        search_term (str): The search term specified in the URL path
    
    Returns:
        Redirect: Redirects to index_post route with search_term as query parameter
    """
    # Redirect to the /jobs endpoint with the search term as a query parameter
    return redirect(url_for('index_post', search_term=search_term))

@app.route('/api/jobs', methods=['GET'])
def api_jobs():
    """
    API endpoint for retrieving job listings in JSON format.
    
    Provides a RESTful API interface for querying job data with search terms and
    pagination parameters. Returns JSON response with job listings, total count,
    and job description information.
    
    Args (via query parameters):
        search_term (str): Term to search for jobs
        page (int, optional): Page number for pagination, defaults to 1
        limit (int, optional): Number of results per page, defaults to 20
    
    Returns:
        JSON: JSON response containing jobs, count, and description data
    """
    # Get query parameters
    search_term = request.args.get('search_term', '')
    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 20))
    
    # Calculate offset for pagination
    offset = (page - 1) * limit

    job_levels = request.args.getlist('job_level') or None

    # Fetch jobs matching the search term
    total_count, all_jobs = search_jobs(search_term, job_levels=job_levels)
    
    # Apply pagination
    paginated_jobs = all_jobs[offset:offset + limit] if offset < len(all_jobs) else []
    
    # Get job description for the search term
    db_description = get_job_description(search_term) if search_term else ""
    
    # Prepare response
    response_data = {
        "jobs": paginated_jobs,
        "total_count": total_count,
        "db_description": db_description
    }
    
    return json.dumps(response_data), 200, {'Content-Type': 'application/json'}

if __name__ == '__main__':
    #app.run(debug=True, host = '0, 0, 0, 0', port=5002)
    app.run(debug=True, host='127.0.0.1', port=5002) # For local running