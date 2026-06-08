### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/WCC-NOVA-SBE/wcc-project-24-25-wcc-63907-65756-63855-64091.git
   cd wcc-project-24-25-wcc-63907-65756-63855-64091
   ```

2. Install the required dependencies:
   #### Using pip
    ```bash
    pip install -r requirements.txt
    ```

    #### Using Conda
    ```bash
    conda env create -f environment.yml
    conda activate my_project_env
    ```

### Project Architecture

This project follows a three-layer architecture pattern that separates concerns and promotes maintainability:

#### 1. Data Access Layer (`data.py`)

The data access layer is responsible for:
- Direct interaction with data sources (Google Sheets API, SQLite database)
- CRUD operations on job listings and descriptions
- API authentication and connection management
- Data retrieval with fuzzy matching and synonym handling
- Low-level data operations

Key components:
- `GoogleSheetsClient`: Handles all interactions with Google Sheets API
- `get_access_token()`: Generates access token for EMSI Lightcast Skills API
- `get_skills_for_job()`: Get skills and descriptions for a job title using EMSI Lightcast Skills API
- `save_job_skills_to_db()`: Save job descriptions from EMSI Lightcast Skills API to SQLite Database

#### 2. Business Logic Layer (`logic.py`)

The business logic layer handles:
- Application rules and business processes
- Data validation and transformation
- Intermediation between data access and presentation layers
- Query orchestration and result formatting
- Analytics data preparation

Key functions:
- `search_jobs()`: Performs job searching with fuzzy matching and relevance scoring using Google Sheets client
- `get_job_description()`: Retrieves and formats job descriptions from EMSI Lightcast Skills API
- `get_analytics_data()`: Prepares data for analytics dashboard/home page

#### 3. Presentation Layer (`app.py`)

The presentation layer is responsible for:
- HTTP request handling and routing
- User interface rendering via Flask templates
- Request parameter processing
- Response formatting (HTML or JSON)
- User session management

Key routes:
- `/`: Home page/Analytics dashboard
- `/jobs`: Search Page
- `/jobs?search_term=<search_term>`: Search endpoint
- `/api/jobs`: JSON API endpoint

#### Data Flow Example

1. User searches for a job term at the `/` endpoint
2. `app.py` (presentation) receives the request and calls `search_jobs()` from `logic.py`
3. `logic.py` (business logic) processes the request and calls methods from `data.py`
4. `data.py` (data access) queries Google Sheets and retrieves job listings
5. Results flow back through the layers, with each adding its own processing
6. `app.py` renders the final webpage to the user via the endpoint `/jobs?search_term=<search_term>`

#### File Structure and Explanations
[structure.txt](structure.txt): Provides a breakdown of the project's file structure and their respective purposes.