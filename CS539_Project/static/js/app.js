// ==================== Configuration ====================
const API_BASE_URL = window.location.origin;

// ==================== State Management ====================
let selectedFile = null;
let currentArtifactId = null;
let currentVisualizations = [];

// ==================== DOM Elements ====================
const elements = {
    uploadArea: document.getElementById('uploadArea'),
    fileInput: document.getElementById('fileInput'),
    fileInfo: document.getElementById('fileInfo'),
    fileName: document.getElementById('fileName'),
    removeFile: document.getElementById('removeFile'),
    queryInput: document.getElementById('queryInput'),
    analyzeBtn: document.getElementById('analyzeBtn'),
    uploadSection: document.getElementById('uploadSection'),
    resultsSection: document.getElementById('resultsSection'),
    errorSection: document.getElementById('errorSection'),
    errorMessage: document.getElementById('errorMessage'),
    statusDot: document.getElementById('statusDot'),
    statusText: document.getElementById('statusText'),
    newAnalysisBtn: document.getElementById('newAnalysisBtn'),
    tryAgainBtn: document.getElementById('tryAgainBtn'),
    summaryText: document.getElementById('summaryText'),
    latencyValue: document.getElementById('latencyValue'),
    stepsValue: document.getElementById('stepsValue'),
    vizValue: document.getElementById('vizValue'),
    qualityBadge: document.getElementById('qualityBadge'),
    visualizationsContainer: document.getElementById('visualizationsContainer'),
    evaluationCard: document.getElementById('evaluationCard'),
    evaluationContent: document.getElementById('evaluationContent'),
    downloadArtifactBtn: document.getElementById('downloadArtifactBtn'),
    downloadVisualizationsBtn: document.getElementById('downloadVisualizationsBtn')
};

// ==================== Initialization ====================
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    setupEventListeners();
    elements.queryInput.value = '';
});

// ==================== Health Check ====================
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        if (data.status === 'healthy') {
            elements.statusDot.classList.add('healthy');
            elements.statusText.textContent = 'API Ready';
        } else {
            const detail = typeof data.detail === 'string' && data.detail.trim()
                ? `: ${data.detail}`
                : '';
            elements.statusText.textContent = `API Unavailable${detail}`;
        }
    } catch (error) {
        elements.statusText.textContent = 'Connection Error';
        console.error('Health check failed:', error);
    }
}

// ==================== Event Listeners ====================
function setupEventListeners() {
    // Upload area click
    elements.uploadArea.addEventListener('click', (e) => {
        if (!e.target.closest('.btn-remove')) {
            elements.fileInput.click();
        }
    });

    // File input change
    elements.fileInput.addEventListener('change', handleFileSelect);

    // Remove file button
    elements.removeFile.addEventListener('click', (e) => {
        e.stopPropagation();
        clearSelectedFile();
    });

    // Drag and drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        elements.uploadArea.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        elements.uploadArea.addEventListener(eventName, () => {
            elements.uploadArea.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        elements.uploadArea.addEventListener(eventName, () => {
            elements.uploadArea.classList.remove('dragover');
        });
    });

    elements.uploadArea.addEventListener('drop', handleDrop);

    // Analyze button
    elements.analyzeBtn.addEventListener('click', handleAnalyze);

    // Query chips
    document.querySelectorAll('.query-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            elements.queryInput.value = chip.dataset.query;
        });
    });

    // Navigation buttons
    elements.newAnalysisBtn.addEventListener('click', resetToUpload);
    elements.tryAgainBtn.addEventListener('click', resetToUpload);

    // Download buttons
    elements.downloadArtifactBtn.addEventListener('click', downloadArtifact);
    elements.downloadVisualizationsBtn.addEventListener('click', downloadAllVisualizations);
}

// ==================== File Handling ====================
function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFile(file) {
    // Validate file type
    if (!file.name.toLowerCase().endsWith('.csv')) {
        showError('Please upload a CSV file only.');
        return;
    }

    // Validate file size (max 50MB)
    const maxSize = 50 * 1024 * 1024;
    if (file.size > maxSize) {
        showError('File size exceeds 50MB limit.');
        return;
    }

    selectedFile = file;
    elements.fileName.textContent = file.name;
    elements.fileInfo.style.display = 'flex';
    elements.analyzeBtn.disabled = false;
    
    // Update upload area UI
    elements.uploadArea.querySelector('h2').textContent = 'File selected!';
    elements.uploadArea.querySelector('p').textContent = 'Click "Analyze Dataset" to continue';
}

function clearSelectedFile() {
    selectedFile = null;
    elements.fileInput.value = '';
    elements.fileInfo.style.display = 'none';
    elements.analyzeBtn.disabled = true;
    elements.uploadArea.querySelector('h2').textContent = 'Drop your CSV file here';
    elements.uploadArea.querySelector('p').textContent = 'or click to browse';
}

// ==================== Analysis ====================
async function handleAnalyze() {
    if (!selectedFile) {
        showError('Please select a file first.');
        return;
    }

    const query = elements.queryInput.value.trim();
    if (!query) {
        showError('Please enter a query.');
        return;
    }

    // Show loading state
    elements.analyzeBtn.disabled = true;
    elements.analyzeBtn.querySelector('.btn-text').style.display = 'none';
    elements.analyzeBtn.querySelector('.btn-loading').style.display = 'flex';

    try {
        // Create FormData
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('query', query);

        // Send request
        const response = await fetch(`${API_BASE_URL}/upload-analyze`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Analysis failed');
        }

        const result = await response.json();
        displayResults(result);
        
    } catch (error) {
        console.error('Analysis error:', error);
        showError(error.message || 'An error occurred during analysis. Please try again.');
        elements.analyzeBtn.disabled = false;
        elements.analyzeBtn.querySelector('.btn-text').style.display = 'inline';
        elements.analyzeBtn.querySelector('.btn-loading').style.display = 'none';
    }
}

// ==================== Results Display ====================
function displayResults(data) {
    // Hide upload section, show results
    elements.uploadSection.style.display = 'none';
    elements.errorSection.style.display = 'none';
    elements.resultsSection.style.display = 'block';

    // Store artifact ID and visualizations
    currentArtifactId = data.artifact_id;
    currentVisualizations = data.visualizations || [];

    // Display summary
    elements.summaryText.textContent = data.summary;

    // Display metrics
    elements.latencyValue.textContent = `${data.latency.toFixed(2)}s`;
    elements.stepsValue.textContent = data.steps_executed || 'N/A';
    elements.vizValue.textContent = currentVisualizations.length;

    // Display quality badge if evaluation exists
    if (data.evaluation && data.evaluation.overall_score !== undefined) {
        const score = data.evaluation.overall_score;
        let badgeClass = 'acceptable';
        let badgeText = 'Acceptable';
        
        if (score >= 0.8) {
            badgeClass = 'excellent';
            badgeText = 'Excellent';
        } else if (score >= 0.6) {
            badgeClass = 'good';
            badgeText = 'Good';
        }
        
        elements.qualityBadge.className = `quality-badge ${badgeClass}`;
        elements.qualityBadge.textContent = `${badgeText} (${(score * 100).toFixed(0)}%)`;
    } else {
        elements.qualityBadge.style.display = 'none';
    }

    // Display visualizations
    displayVisualizations(currentVisualizations);

    // Display evaluation details if available
    if (data.evaluation) {
        displayEvaluation(data.evaluation);
    }

    // Reset analyze button
    elements.analyzeBtn.disabled = false;
    elements.analyzeBtn.querySelector('.btn-text').style.display = 'inline';
    elements.analyzeBtn.querySelector('.btn-loading').style.display = 'none';
}

function displayVisualizations(visualizations) {
    elements.visualizationsContainer.innerHTML = '';
    
    if (visualizations.length === 0) {
        elements.visualizationsContainer.innerHTML = '<p style="color: var(--text-secondary);">No visualizations generated.</p>';
        return;
    }

    visualizations.forEach((viz, index) => {
        const vizName = String(viz).split('/').pop().split('\\').pop();
        const vizItem = document.createElement('div');
        vizItem.className = 'viz-item';
        
        const img = document.createElement('img');
        img.src = `${API_BASE_URL}/visualization/${encodeURIComponent(vizName)}`;
        img.alt = `Visualization ${index + 1}`;
        img.onerror = () => {
            img.src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300"><rect width="400" height="300" fill="%23f3f4f6"/><text x="50%" y="50%" text-anchor="middle" fill="%236b7280">Image not available</text></svg>';
        };
        
        const label = document.createElement('div');
        label.className = 'viz-label';
        label.textContent = formatVisualizationName(vizName);
        
        vizItem.appendChild(img);
        vizItem.appendChild(label);
        elements.visualizationsContainer.appendChild(vizItem);
    });
}

function displayEvaluation(evaluation) {
    if (!evaluation || Object.keys(evaluation).length === 0) {
        elements.evaluationCard.style.display = 'none';
        return;
    }

    elements.evaluationCard.style.display = 'block';
    elements.evaluationContent.innerHTML = '';

    // Overall score
    if (evaluation.overall_score !== undefined) {
        const overallDiv = document.createElement('div');
        overallDiv.className = 'metric';
        overallDiv.innerHTML = `
            <span class="metric-label">Overall Score</span>
            <span class="metric-value">${(evaluation.overall_score * 100).toFixed(0)}%</span>
        `;
        elements.evaluationContent.appendChild(overallDiv);
    }

    // Individual scores
    const scores = {
        'Tool Invocation': evaluation.tool_invocation_score,
        'Summary Accuracy': evaluation.summary_accuracy_score,
        'Visualization Quality': evaluation.visualization_quality_score
    };

    Object.entries(scores).forEach(([label, score]) => {
        if (score !== undefined) {
            const scoreDiv = document.createElement('div');
            scoreDiv.className = 'metric';
            scoreDiv.innerHTML = `
                <span class="metric-label">${label}</span>
                <span class="metric-value">${(score * 100).toFixed(0)}%</span>
            `;
            elements.evaluationContent.appendChild(scoreDiv);
        }
    });

    // Feedback
    if (evaluation.feedback && evaluation.feedback.length > 0) {
        const feedbackDiv = document.createElement('div');
        feedbackDiv.style.marginTop = '20px';
        feedbackDiv.innerHTML = '<h4 style="margin-bottom: 10px;">Feedback:</h4>';
        
        const feedbackList = document.createElement('ul');
        feedbackList.style.paddingLeft = '20px';
        evaluation.feedback.forEach(item => {
            const li = document.createElement('li');
            li.textContent = item;
            li.style.marginBottom = '5px';
            feedbackList.appendChild(li);
        });
        
        feedbackDiv.appendChild(feedbackList);
        elements.evaluationContent.appendChild(feedbackDiv);
    }
}

// ==================== Download Functions ====================
async function downloadArtifact() {
    if (!currentArtifactId) return;
    
    try {
        const response = await fetch(`${API_BASE_URL}/artifact/${currentArtifactId}`);
        const data = await response.json();
        
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `analysis_${currentArtifactId}.json`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    } catch (error) {
        console.error('Download failed:', error);
        alert('Failed to download artifact');
    }
}

async function downloadAllVisualizations() {
    if (currentVisualizations.length === 0) return;
    
    // Download each visualization
    for (const viz of currentVisualizations) {
        try {
            const response = await fetch(`${API_BASE_URL}/visualization/${viz}`);
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = viz;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
            // Small delay between downloads
            await new Promise(resolve => setTimeout(resolve, 200));
        } catch (error) {
            console.error(`Failed to download ${viz}:`, error);
        }
    }
}

// ==================== Error Handling ====================
function showError(message) {
    elements.uploadSection.style.display = 'none';
    elements.resultsSection.style.display = 'none';
    elements.errorSection.style.display = 'block';
    elements.errorMessage.textContent = message;
}

// ==================== Navigation ====================
function resetToUpload() {
    clearSelectedFile();
    elements.uploadSection.style.display = 'block';
    elements.resultsSection.style.display = 'none';
    elements.errorSection.style.display = 'none';
    currentArtifactId = null;
    currentVisualizations = [];
}

// ==================== Utility Functions ====================
function formatVisualizationName(filename) {
    // Remove file extension and timestamp
    let name = filename.replace(/\.(png|jpg|jpeg|svg)$/i, '');
    
    // Remove timestamp pattern (e.g., _20230101_120000)
    name = name.replace(/_\d{8}_\d{6}$/, '');
    
    // Replace underscores with spaces and capitalize
    name = name.replace(/_/g, ' ');
    name = name.charAt(0).toUpperCase() + name.slice(1);
    
    return name;
}
