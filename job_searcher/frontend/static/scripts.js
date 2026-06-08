document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const header = document.getElementById('main-header');
    const searchButton = document.getElementById('search-button'); // Renamed from clear-search
    const searchInput = document.getElementById('secondary-search-input');
    const searchInfoText = document.getElementById('search-info');
    const jobsList = document.getElementById('jobs-list');
    const jobDetailPanel = document.getElementById('job-detail-panel');
    const jobDetail = document.getElementById('job-detail');
    const closeDetailBtn = document.getElementById('close-detail');
    const loadingIndicator = document.getElementById('loading-indicator');

    // Global state: central data strore for application's runtime information
    let currentState = {
        searchTerm: '',        // The current search query
        currentPage: 1,        // Current page number for pagination
        limit: 20,             // Maximum number of jobs to load per page
        totalJobs: 0,          // Total count of jobs matching the search
        loadedJobs: 0,         // Number of jobs currently loaded in the UI
        loading: false,        // Flag indicating if jobs are being fetched
        selectedJobId: null,   // ID of the currently selected job
        hasMoreJobs: true      // Flag indicating if more jobs can be loaded
    };

    // Initialize
    init();

    // Functions
    /**
     * Initializes the application by setting up event listeners and loading initial jobs.
     */
    function init() {
        // Set up event listeners: event listeners are programming pattern that wait for 
        // an event to occur and then executes code in response to that event. 
        // In JavaScript, event listeners are used to create interactive web applications 
        // by responding to user actions or browser events.
        setupEventListeners();
        
        // Load initial jobs
        fetchJobs();
    }

    /**
     * Sets up all event listeners for the application, including scroll behavior,
     * search functionality, detail panel, and infinite scrolling.
     */
    function setupEventListeners() {
        // Header shadow on scroll
        window.addEventListener('scroll', () => {
            if (window.scrollY > 10) {
                header.classList.add('scrolled');
            } else {
                header.classList.remove('scrolled');
            }
        });

        // Search button (renamed from Clear search button)
        if (searchButton) {
            searchButton.addEventListener('click', clearSearch);
        }

        // Show/hide search button based on input
        if (searchInput) {
            searchInput.addEventListener('input', () => {
                if (searchButton) {
                    searchButton.style.display = searchInput.value ? 'block' : 'none';
                }
            });
        }

        // Close detail panel
        if (closeDetailBtn) {
            closeDetailBtn.addEventListener('click', closeJobDetail);
        }

        // Infinite scroll for jobs list
        const jobsListContainer = document.querySelector('.jobs-list-container');
        if (jobsListContainer) {
            jobsListContainer.addEventListener('scroll', handleScroll);
        }

        // Job level filter chips — reset to page 1 and re-fetch on change
        document.querySelectorAll('.job-level-filter').forEach(cb => {
            cb.addEventListener('change', () => {
                currentState.currentPage = 1;
                currentState.hasMoreJobs = true;
                fetchJobs(false);
            });
        });
    }

    /**
     * Clears the search input field and hides the search button.
     */
    function clearSearch() {
        if (searchInput) {
            searchInput.value = '';
            if (searchButton) {
                searchButton.style.display = 'none';
            }
        }
    }

    /**
     * Handles scrolling in the jobs list container to implement infinite scrolling.
     * Triggers loading more jobs when the user scrolls close to the bottom.
     * @param {Event} e - The scroll event
     */
    function handleScroll(e) {
        const { scrollTop, scrollHeight, clientHeight } = e.target;
        
        // Check if scrolled to bottom (with 50px threshold)
        if (scrollHeight - scrollTop <= clientHeight + 50) {
            loadMoreJobs();
        }
    }

    /**
     * Loads more job listings by incrementing the current page and fetching more jobs.
     * Only triggers if not already loading and if there are more jobs to load.
     */
    function loadMoreJobs() {
        // Don't load more if already loading or all jobs are loaded
        if (currentState.loading || !currentState.hasMoreJobs) {
            return;
        }
        
        // Load next page
        currentState.currentPage++;
        fetchJobs(true);
    }

    /**
     * Fetches job listings from the server based on the search term and pagination state.
     * Updates the UI with loading indicators, job listings, and search information.
     * @param {boolean} append - Whether to append new jobs to existing ones (true) when loading additional jobs 
     * or replace them (false) when loading results for a new search term or the initial load.
     */
    async function fetchJobs(append = false) {
        if (currentState.loading) return;
        
        currentState.loading = true;
        if (loadingIndicator) {
            loadingIndicator.style.display = 'block';
        }
        if (searchInfoText && !append) {
            searchInfoText.textContent = 'Loading jobs...';
        }
        
        try {
            // Get search term from URL or query parameter
            const urlParams = new URLSearchParams(window.location.search);
            const pathParts = window.location.pathname.split('/');
            
            // Try to get search term from URL paths, query parameters, or input field
            let searchTerm = '';
            if (urlParams.has('search_term')) {
                searchTerm = urlParams.get('search_term');
            } else if (pathParts.length > 2 && pathParts[1] === 'jobs' && pathParts[2]) {
                searchTerm = decodeURIComponent(pathParts[2]);
            } else if (searchInput && searchInput.value) {
                searchTerm = searchInput.value.trim();
            }
            
            currentState.searchTerm = searchTerm;
            
            if (!searchTerm) {
                if (searchInfoText) {
                    searchInfoText.textContent = 'Please enter a search term';
                }
                if (loadingIndicator) {
                    loadingIndicator.style.display = 'none';
                }
                currentState.loading = false;
                return;
            }

            // Collect selected job level filters
            const selectedLevels = Array.from(document.querySelectorAll('.job-level-filter:checked')).map(cb => cb.value);
            const levelParams = selectedLevels.map(l => `&job_level=${encodeURIComponent(l)}`).join('');

            // Construct fetch URL with pagination and filter parameters
            const fetchUrl = `/jobs?search_term=${encodeURIComponent(searchTerm)}&page=${currentState.currentPage}&limit=${currentState.limit}${levelParams}`;
            
            const response = await fetch(fetchUrl, {
                headers: {
                    'Accept': 'application/json'
                }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Update state
            currentState.totalJobs = data.total_count;
            
            // If appending, add to loadedJobs count, otherwise set it to current batch size
            if (append) {
                currentState.loadedJobs += data.jobs.length;
            } else {
                currentState.loadedJobs = data.jobs.length;
            }
            
            currentState.hasMoreJobs = data.has_more;
            
            // Update search info to show loaded/total counts
            if (searchInfoText) {
                searchInfoText.textContent = `Found ${currentState.loadedJobs} of ${data.total_count} jobs for "${searchTerm}"`;
            }
            
            // Update description (only on first load)
            if (!append) {
                const dbDescription = document.getElementById('db-description');
                if (dbDescription && data.db_description) {
                    // Clear previous content
                    dbDescription.innerHTML = '';
                    
                    // Only add title if description exists and is not "No description available"
                    if (data.db_description && data.db_description !== 'No matching job descriptions found.') {
                        // Create title
                        const title = document.createElement('h2');
                        title.textContent = 'Search Description';
                        title.style.color = 'black';
                        title.style.fontWeight = 'bold';
                        title.style.fontSize = '18px';
                        
                        // Add title to description box
                        dbDescription.appendChild(title);
                    }
                    
                    // Add description text
                    const descText = document.createElement('div');
                    descText.textContent = data.db_description;
                    dbDescription.appendChild(descText);
                }
            }
            
            // Render jobs
            renderJobs(data.jobs, append);
            
            // Hide loading indicator if no more jobs
            if (!currentState.hasMoreJobs && loadingIndicator) {
                loadingIndicator.style.display = 'none';
                
                // Add "end of results" message if we've loaded all jobs
                if (jobsList && currentState.loadedJobs > 0) {
                    const endMessage = document.createElement('div');
                    endMessage.className = 'end-of-results';
                    endMessage.textContent = 'End of results';
                    jobsList.appendChild(endMessage);
                }
            }
            
        } catch (error) {
            console.error('Error fetching jobs:', error);
            if (searchInfoText) {
                searchInfoText.textContent = 'Error loading jobs. Please try again.';
            }
            if (jobsList && !append) {
                jobsList.innerHTML = '<div class="error">Failed to load jobs. Please try again.</div>';
            }
        } finally {
            currentState.loading = false;
            if (loadingIndicator && !currentState.hasMoreJobs) {
                loadingIndicator.style.display = 'none';
            }
        }
    }

    /**
     * Renders job listings to the DOM (Document Object Model), either appending to existing jobs or replacing them.
     * Creates job cards with job details and adds click handlers for showing job details.
     * @param {Array} jobs - Array of job objects to render
     * @param {boolean} append - Whether to append new jobs to existing ones (true) when loading additional jobs 
     * or replace them (false) when loading results for a new search term or the initial load.
     */
    function renderJobs(jobs, append = false) {
        // Get jobs list container initially or clear existing jobs
        // Clears all child elements from the jobsList container when not in append mode.
        if (!append) {
            jobsList.innerHTML = '';
        }
        
        if (jobs.length === 0 && !append) {
            jobsList.innerHTML = '<div class="no-jobs">No jobs found. Try a different search.</div>';
            return;
        }
        
        jobs.forEach(job => {
            // Create unique ID for the job
            const jobId = `job-${job.site}-${Math.random().toString(36).substr(2, 9)}`;
            
            // This creates a new DOM element (the job card) to represent each job listing.
            const jobCard = document.createElement('div');
            jobCard.className = 'job-card';
            jobCard.dataset.jobId = jobId;
            jobCard.dataset.index = currentState.loadedJobs + jobs.indexOf(job);
            
            // Store job data as attribute (serialize data)
            jobCard.dataset.jobData = JSON.stringify(job);
            
            jobCard.innerHTML = `
                <div class="job-header">
                    <img src="${job.company_logo || 'https://cdn-icons-png.flaticon.com/512/25/25256.png'}" alt="${job.company}" class="job-logo">
                    <div>
                        <h3 class="job-title">${job.title}</h3>
                        <div class="job-company">${job.company}</div>
                        <div class="job-location">${job.location || 'Location not specified'}</div>
                    </div>
                </div>
                <div class="job-tags">
                    ${job.date_posted ? `<span class="job-tag ${isNewJob(job.date_posted) ? 'new' : ''}">${isNewJob(job.date_posted) ? 'NEW · ' : ''}${job.date_posted}</span>` : ''}
                    ${job.job_type ? `<span class="job-tag">${job.job_type}</span>` : ''}
                    ${job.is_remote ? `<span class="job-tag">Remote</span>` : ''}
                </div>
                <div class="job-source">
                    <span>Source: </span><a href="${job.job_url_direct || job.job_url}" target="_blank">${job.site}</a>
                </div>
            `;
            
            jobCard.addEventListener('click', () => showJobDetail(jobId));
            
            jobsList.appendChild(jobCard);
        });
    }

    /**
     * Displays detailed information for a selected job in the detail panel.
     * Highlights the selected job card and populates the detail panel with job information.
     * @param {string} jobId - The unique identifier for the job to display
     */
    function showJobDetail(jobId) {
        // Deselect previous job
        if (currentState.selectedJobId) {
            const prevSelected = document.querySelector(`.job-card[data-job-id="${currentState.selectedJobId}"]`);
            if (prevSelected) {
                prevSelected.classList.remove('selected');
            }
        }
        
        // Select current job
        const jobCard = document.querySelector(`.job-card[data-job-id="${jobId}"]`);
        if (!jobCard) return;
        
        jobCard.classList.add('selected');
        currentState.selectedJobId = jobId;
        
        // Get job data
        const jobData = JSON.parse(jobCard.dataset.jobData);
        
        // Render job detail
        jobDetail.innerHTML = `
            <div class="detail-header">
                <img src="${jobData.company_logo || 'https://cdn-icons-png.flaticon.com/512/25/25256.png'}" alt="${jobData.company}" class="detail-logo">
                <div>
                    <h2 class="detail-title">${jobData.title}</h2>
                    <div class="detail-company">${jobData.company}</div>
                    <div class="detail-location">${jobData.location || 'Location not specified'}</div>
                </div>
            </div>
            
            <div class="detail-source">
                <a href="${jobData.job_url_direct || jobData.job_url}" target="_blank">
                    <button>Apply Now</button>
                </a>
            </div>
            
            <h5>Job Details</h5>
            <div class="detail-meta">
                ${jobData.job_type ? `<div class="detail-meta-item">Type: ${jobData.job_type}</div>` : ''}
                ${jobData.is_remote !== null ? `<div class="detail-meta-item">Remote: ${jobData.is_remote ? 'Yes' : 'No'}</div>` : ''}
                ${jobData.job_level ? `<div class="detail-meta-item">Level: ${jobData.job_level}</div>` : ''}
                ${jobData.job_function ? `<div class="detail-meta-item">Function: ${jobData.job_function}</div>` : ''}
                ${jobData.company_industry ? `<div class="detail-meta-item">Industry: ${jobData.company_industry}</div>` : ''}
                ${jobData.date_posted ? `<div class="detail-meta-item">Posted: ${jobData.date_posted}</div>` : ''}
            </div>
            
            <h5>Job Description</h5>
            <div class="detail-description">
                ${jobData.description || 'No description available'}
            </div>
        `;
        
        // Show detail panel
        jobDetailPanel.classList.add('active');
    }

    /**
     * Closes the job detail panel and removes the selected state from the current job card.
     */
    function closeJobDetail() {
        // Deselect current job
        if (currentState.selectedJobId) {
            const selected = document.querySelector(`.job-card[data-job-id="${currentState.selectedJobId}"]`);
            if (selected) {
                selected.classList.remove('selected');
            }
            currentState.selectedJobId = null;
        }
        
        // Hide detail panel
        jobDetailPanel.classList.remove('active');
    }

    /**
     * Determines if a job is considered "new" based on its posted date.
     * Handles various date formats in German and English.
     * @param {string} dateString - The date string to check
     * @returns {boolean} - True if the job is considered new (posted within last 7 days), false otherwise
     */
    function isNewJob(dateString) {
        if (!dateString) return false;
        
        try {
            // Handle different date formats
            
            // Case 1: Handle "vor X Tagen/Wochen/etc." format (German)
            const germanMatch = dateString.match(/vor\s+(\d+)\s+(Tag|Tage|Woche|Wochen|Monat|Monate|Stunde|Stunden)/i);
            if (germanMatch) {
                const value = parseInt(germanMatch[1]);
                const unit = germanMatch[2].toLowerCase();
                
                if (unit.includes('stunde')) {
                    return true;
                } else if (unit.includes('tag')) {
                    return value <= 7;
                } else {
                    return false;
                }
            }
            
            // Case 2: Handle "X days/weeks ago" format
            const englishMatch = dateString.match(/(\d+)\s+(day|days|week|weeks|month|months|hour|hours)\s+ago/i);
            if (englishMatch) {
                const value = parseInt(englishMatch[1]);
                const unit = englishMatch[2].toLowerCase();
                
                if (unit.includes('hour')) {
                    return true;
                } else if (unit.includes('day')) {
                    return value <= 7;
                } else {
                    return false;
                }
            }
            
            // Case 3: Handle standard date format (YYYY-MM-DD or similar)
            // Parse the date string to a Date object
            const date = new Date(dateString);
            if (!isNaN(date.getTime())) {
                const now = new Date();
                const diffTime = Math.abs(now - date);
                const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
                return diffDays <= 7;
            }
            
            return false;
            
        } catch (error) {
            return false;
        }
    }
});