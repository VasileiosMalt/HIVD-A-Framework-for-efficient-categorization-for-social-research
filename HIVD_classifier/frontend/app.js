/**
 * HIVD Classifier - Frontend Application Logic
 * Maltezos, V. (2026). A framework for efficient image categorisation in social research
 */

// Global State
const state = {
    classes: [],
    datasetImages: [],
    currentJobId: null,
    wsConnection: null,
    selectedReferenceImages: [],
    systemStatus: {
        connected: false
    },
    currentJobResults: [],
    currentFilter: {
        confidence: 'high', // 'high' or 'low'
        class: null        // null or class name string
    }
};

// API Client
const api = {
    baseUrl: '/api',

    async get(endpoint) {
        try {
            const response = await fetch(`${this.baseUrl}${endpoint}`);
            if (!response.ok) throw new Error(`API Error: ${response.statusText}`);
            return await response.json();
        } catch (error) {
            console.error('API Get Error:', error);
            ui.toast(error.message, 'error');
            throw error;
        }
    },

    async post(endpoint, data) {
        try {
            const response = await fetch(`${this.baseUrl}${endpoint}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                let detail = errorData.detail;
                if (Array.isArray(detail)) {
                    detail = detail.map(d => `${d.loc.join('.')}: ${d.msg}`).join(', ');
                } else if (typeof detail === 'object') {
                    detail = JSON.stringify(detail);
                }
                throw new Error(detail || `API Error: ${response.statusText}`);
            }
            return await response.json();
        } catch (error) {
            console.error('API Post Error:', error);
            const msg = error.message || 'Unknown server error';
            ui.toast(msg, 'error');
            throw error;
        }
    },

    async put(endpoint, data) {
        try {
            const response = await fetch(`${this.baseUrl}${endpoint}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            if (!response.ok) throw new Error(`API Error: ${response.statusText}`);
            return await response.json();
        } catch (error) {
            ui.toast(error.message, 'error');
            throw error;
        }
    },

    async delete(endpoint) {
        try {
            const response = await fetch(`${this.baseUrl}${endpoint}`, {
                method: 'DELETE'
            });
            if (!response.ok) throw new Error(`API Error: ${response.statusText}`);
            return await response.json();
        } catch (error) {
            ui.toast(error.message, 'error');
            throw error;
        }
    }
};

// User Interface Module
const ui = {
    elements: {
        statusIndicator: document.getElementById('systemStatus'),
        tabs: document.querySelectorAll('.tab-button'),
        tabPanes: document.querySelectorAll('.tab-pane'),
        classList: document.getElementById('classList'),
        classModal: document.getElementById('classModal'),
        imageBrowserModal: document.getElementById('imageBrowserModal'),
        imageBrowser: document.getElementById('imageBrowser'),
        selectedImagesContainer: document.getElementById('selectedImages'),
        classForm: document.getElementById('classForm'),
        classificationForm: document.getElementById('classificationForm'),
        classSelector: document.getElementById('classSelector'),
        progressSection: document.getElementById('progressSection'),
        progressFill: document.getElementById('progressFill'),
        progressPercentage: document.getElementById('progressPercentage'),
        progressMessage: document.getElementById('progressMessage'),
        progressImages: document.getElementById('progressImages'),
        currentImage: document.getElementById('currentImage'),
        jobsList: document.getElementById('jobsList'),
        toastContainer: document.getElementById('toastContainer'),
        selectedCount: document.getElementById('selectedCount'),
        resultsModal: document.getElementById('resultsModal'),
        resultsGrid: document.getElementById('resultsGrid'),
        resTabs: document.querySelectorAll('.res-tab'),
        previewModal: document.getElementById('imagePreviewModal'),
        previewImage: document.getElementById('previewImage')
    },

    init() {
        this.bindEvents();
        this.updateSliders();
        this.checkSystemStatus();
        this.loadClasses();

        // Poll status every 30 seconds
        setInterval(() => this.checkSystemStatus(), 30000);
    },

    bindEvents() {
        // Tab Navigation
        this.elements.tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const targetId = tab.dataset.tab + 'Tab';

                // Update tabs
                this.elements.tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');

                // Update panes
                this.elements.tabPanes.forEach(pane => {
                    pane.classList.remove('active');
                    if (pane.id === targetId) pane.classList.add('active');
                });

                // Refresh data if needed
                if (tab.dataset.tab === 'classify') this.refreshClassSelector();
                if (tab.dataset.tab === 'results') this.loadJobs();
                if (tab.dataset.tab === 'jobs') this.loadAllJobs();
            });
        });

        // Threshold sliders with validation
        const highSlider = document.getElementById('highThreshold');
        const lowSlider = document.getElementById('lowThreshold');
        const highValue = document.getElementById('highThresholdValue');
        const lowValue = document.getElementById('lowThresholdValue');

        // Initialize values
        if (highSlider && highValue) highValue.textContent = parseFloat(highSlider.value).toFixed(2);
        if (lowSlider && lowValue) lowValue.textContent = parseFloat(lowSlider.value).toFixed(2);

        // High threshold slider
        if (highSlider) {
            highSlider.addEventListener('input', (e) => {
                const highVal = parseFloat(e.target.value);
                highValue.textContent = highVal.toFixed(2);

                // Ensure low threshold stays below high threshold
                if (lowSlider && parseFloat(lowSlider.value) >= highVal) {
                    lowSlider.value = Math.max(0, highVal - 0.05).toFixed(2);
                    lowValue.textContent = lowSlider.value;
                }
            });
        }

        // Low threshold slider
        if (lowSlider) {
            lowSlider.addEventListener('input', (e) => {
                const lowVal = parseFloat(e.target.value);
                lowValue.textContent = lowVal.toFixed(2);

                // Ensure high threshold stays above low threshold
                if (highSlider && parseFloat(highSlider.value) <= lowVal) {
                    highSlider.value = Math.min(1, lowVal + 0.05).toFixed(2);
                    highValue.textContent = highSlider.value;
                }
            });
        }

        // Subcategory threshold slider
        const subcatSlider = document.getElementById('subcategoryThreshold');
        const subcatValue = document.getElementById('subcategoryThresholdValue');
        if (subcatSlider && subcatValue) {
            subcatValue.textContent = parseFloat(subcatSlider.value).toFixed(2);
            subcatSlider.addEventListener('input', (e) => {
                subcatValue.textContent = parseFloat(e.target.value).toFixed(2);
            });
        }

        // Strategy toggle buttons
        const strategyRadios = document.querySelectorAll('input[name="strategyToggle"]');
        const strategySelect = document.getElementById('subStrategy');
        const subcatThresholdGroup = document.getElementById('subcategoryThresholdGroup');
        const postInfo = document.getElementById('postInfo');
        const preInfo = document.getElementById('preInfo');

        strategyRadios.forEach(radio => {
            radio.addEventListener('change', (e) => {
                const isPOST = e.target.value === 'post';


                // Update hidden select for form submission
                strategySelect.value = e.target.value;

                // Show/hide subcategory threshold (only relevant for POST)
                if (subcatThresholdGroup) {
                    subcatThresholdGroup.style.display = isPOST ? 'block' : 'none';
                }

                // Toggle info boxes
                if (postInfo) postInfo.style.display = isPOST ? 'block' : 'none';
                if (preInfo) preInfo.style.display = isPOST ? 'none' : 'block';
            });
        });

        // Modal Controls
        document.getElementById('btnNewClass').addEventListener('click', () => this.openClassModal());
        document.getElementById('btnCloseModal').addEventListener('click', () => this.closeClassModal());
        document.getElementById('btnCancelClass').addEventListener('click', () => this.closeClassModal());
        document.getElementById('btnBrowseImages').addEventListener('click', () => this.openImageBrowser());
        document.getElementById('btnCloseImageBrowser').addEventListener('click', () => this.closeImageBrowser());
        document.getElementById('btnConfirmImages').addEventListener('click', () => this.confirmImageSelection());

        // Class Form
        this.elements.classForm.addEventListener('submit', (e) => this.handleClassSubmit(e));

        // Classification Form
        this.elements.classificationForm.addEventListener('submit', (e) => this.handleClassificationSubmit(e));

        // Results Modal Controls
        document.getElementById('btnCloseResults').addEventListener('click', () => this.closeResultsModal());
        document.getElementById('btnClosePreview').addEventListener('click', () => this.closePreviewModal());
        document.getElementById('btnOpenFolder').addEventListener('click', () => this.openResultsFolder());

        // Results Filter Tabs
        this.elements.resTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                this.elements.resTabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                state.currentFilter.confidence = tab.dataset.filter;
                this.renderResultsGrid();
            });
        });

        // Delete All Jobs
        const btnDeleteAll = document.getElementById('btnDeleteAllJobs');
        if (btnDeleteAll) {
            btnDeleteAll.addEventListener('click', () => this.deleteAllJobs());
        }
    },

    updateSliders() {
        // Initialize slider values on page load
        ['highThreshold', 'lowThreshold', 'subcategoryThreshold'].forEach(id => {
            const slider = document.getElementById(id);
            const valueDisplay = document.getElementById(id + 'Value');
            if (slider && valueDisplay) {
                valueDisplay.textContent = parseFloat(slider.value).toFixed(2);
            }
        });
    },

    async checkSystemStatus() {
        try {
            const status = await api.get('/status');
            const indicator = this.elements.statusIndicator;
            const text = indicator.querySelector('.status-text');

            indicator.classList.remove('disconnected');
            indicator.classList.add('connected');
            text.textContent = 'System Ready';

            state.systemStatus = { ...status, connected: true };
        } catch (error) {
            const indicator = this.elements.statusIndicator;
            const text = indicator.querySelector('.status-text');

            indicator.classList.remove('connected');
            indicator.classList.add('disconnected');
            text.textContent = 'Disconnected';

            state.systemStatus.connected = false;
        }
    },

    // --- Class Management ---

    async loadClasses() {
        try {
            state.classes = await api.get('/classes');
            this.renderClasses();
        } catch (error) {
            console.error('Failed to load classes', error);
        }
    },

    renderClasses() {
        const container = this.elements.classList;
        container.innerHTML = '';

        if (state.classes.length === 0) {
            container.innerHTML = `
                <div style="grid-column: 1/-1; text-align: center; padding: 40px; color: var(--text-tertiary);">
                    <h3>No classes defined yet</h3>
                    <p>Click "+ New Class" to create your first image category</p>
                </div>
            `;
            return;
        }

        // Group classes into macro and subcategories (loosely check for falsy parent_id)
        const macroClasses = state.classes.filter(c => !c.parent_id);
        const subClasses = state.classes.filter(c => !!c.parent_id);

        macroClasses.forEach(cls => {
            const card = document.createElement('div');
            card.className = 'class-card macro-category';

            // Find children
            const children = subClasses.filter(c => c.parent_id == cls.id);
            const childrenHtml = children.map(child => `
                <div class="subclass-badge">
                    <span>${child.name}</span>
                    <button class="sub-edit" onclick="ui.editClass(${child.id})">✎</button>
                </div>
            `).join('');

            // Generate thumbnails HTML
            const thumbsHtml = cls.reference_images.map(img =>
                `<img src="/api/images/preview?path=${encodeURIComponent(img)}" class="ref-thumb" alt="Reference">`
            ).join('');

            card.innerHTML = `
                <div class="class-header">
                    <span class="class-name">${cls.name} <small>(Macro)</small></span>
                    <div class="class-actions">
                        <button class="action-btn edit" onclick="ui.editClass(${cls.id})">✎</button>
                        <button class="action-btn delete" onclick="ui.deleteClass(${cls.id}, '${cls.name}')">×</button>
                    </div>
                </div>
                <p class="class-desc">${cls.description}</p>
                <div class="class-subcategories">
                    ${children.length > 0 ? `<strong>Subcategories:</strong> ${childrenHtml}` : '<small>No subcategories</small>'}
                </div>
                <div class="class-refs">
                    <div class="ref-images-preview">
                        ${thumbsHtml}
                    </div>
                    <div class="ref-count">${cls.reference_images.length} reference images</div>
                </div>
            `;
            container.appendChild(card);
        });

        // Render "Orphan" subcategories (where parent wasn't found in macro list)
        subClasses.filter(s => !macroClasses.some(m => m.id == s.parent_id)).forEach(cls => {
            const card = document.createElement('div');
            card.className = 'class-card subclass-orphan';

            const thumbsHtml = cls.reference_images.map(img =>
                `<img src="/api/images/preview?path=${encodeURIComponent(img)}" class="ref-thumb" alt="Reference">`
            ).join('');

            card.innerHTML = `
                <div class="class-header">
                    <span class="class-name">${cls.name} <small>(Unlinked Subcategory)</small></span>
                    <div class="class-actions">
                        <button class="action-btn edit" onclick="ui.editClass(${cls.id})">✎</button>
                        <button class="action-btn delete" onclick="ui.deleteClass(${cls.id}, '${cls.name}')">×</button>
                    </div>
                </div>
                <p class="class-desc">${cls.description}</p>
                <div class="class-refs">
                    <div class="ref-images-preview">${thumbsHtml}</div>
                </div>
            `;
            container.appendChild(card);
        });
    },

    openClassModal(classId = null) {
        const modal = this.elements.classModal;
        const form = this.elements.classForm;
        const title = document.getElementById('modalTitle');
        const parentSelect = document.getElementById('parentClass');

        // Reset form
        form.reset();
        state.selectedReferenceImages = [];
        this.renderSelectedImages();

        // 1. Populate Parent Options FIRST so they are available for selection
        // Clear existing except first ("-- No Parent --")
        while (parentSelect.options.length > 1) parentSelect.remove(1);

        state.classes.filter(c =>
            (c.parent_id === null || c.parent_id === undefined || c.parent_id === "") &&
            (classId === null || c.id != classId)
        ).forEach(p => {
            const opt = document.createElement('option');
            opt.value = p.id;
            opt.textContent = p.name;
            parentSelect.appendChild(opt);
        });

        // 2. Populate data
        if (classId) {
            // Find class - handle both string/number IDs just in case
            const cls = state.classes.find(c => c.id == classId);
            if (cls) {
                document.getElementById('classId').value = cls.id;
                document.getElementById('className').value = cls.name;
                document.getElementById('classDescription').value = cls.description;

                // Set parent value after options are added
                if (cls.parent_id !== null && cls.parent_id !== undefined) {
                    parentSelect.value = cls.parent_id;
                } else {
                    parentSelect.value = "";
                }

                state.selectedReferenceImages = [...cls.reference_images];
                this.renderSelectedImages();
                title.textContent = 'Edit Class';
            }
        } else {
            document.getElementById('classId').value = '';
            parentSelect.value = "";
            title.textContent = 'New Class';
        }

        modal.classList.add('open');
    },

    closeClassModal() {
        this.elements.classModal.classList.remove('open');
    },

    // Global wrappers for onclick handlers
    editClass(id) {
        this.openClassModal(id);
    },

    async deleteClass(id, name) {
        if (confirm(`Are you sure you want to delete class "${name}"?`)) {
            try {
                await api.delete(`/classes/${id}`);
                this.toast(`Class "${name}" deleted`, 'success');
                this.loadClasses();
            } catch (error) {
                // Error handled by api wrapper
            }
        }
    },

    async handleClassSubmit(e) {
        e.preventDefault();

        const id = document.getElementById('classId').value;
        const name = document.getElementById('className').value;
        const description = document.getElementById('classDescription').value;
        const parentId = document.getElementById('parentClass').value;

        if (state.selectedReferenceImages.length > 5) {
            this.toast('Maximum 5 reference images allowed', 'warning');
            return;
        }

        const data = {
            name,
            description,
            reference_images: state.selectedReferenceImages,
            parent_id: parentId ? parseInt(parentId) : null
        };

        try {
            if (id) {
                await api.put(`/classes/${id}`, data);
                this.toast(`Class "${name}" updated`, 'success');
            } else {
                await api.post('/classes', data);
                this.toast(`Class "${name}" created`, 'success');
            }
            this.closeClassModal();
            this.loadClasses();
        } catch (error) {
            // Error handled by api wrapper
        }
    },

    // --- Image Browser ---

    async openImageBrowser() {
        this.elements.imageBrowserModal.classList.add('open');

        // Load images if not loaded
        if (state.datasetImages.length === 0) {
            try {
                const images = await api.get('/images/dataset');
                state.datasetImages = images;
                this.renderImageBrowser();
            } catch (error) {
                this.toast('Failed to load dataset images', 'error');
            }
        } else {
            this.renderImageBrowser();
        }
    },

    closeImageBrowser() {
        this.elements.imageBrowserModal.classList.remove('open');
    },

    renderImageBrowser() {
        const container = this.elements.imageBrowser;
        container.innerHTML = '';

        // Update selection count
        this.updateSelectionCount();

        state.datasetImages.forEach(img => {
            const div = document.createElement('div');
            const isSelected = state.selectedReferenceImages.includes(img.path);

            div.className = `browser-img-container ${isSelected ? 'selected' : ''}`;
            div.onclick = () => this.toggleImageSelection(img.path, div);

            div.innerHTML = `
                <img src="/api/images/preview?path=${encodeURIComponent(img.path)}" loading="lazy" alt="${img.filename}">
            `;

            container.appendChild(div);
        });
    },

    toggleImageSelection(path, element) {
        const index = state.selectedReferenceImages.indexOf(path);

        if (index > -1) {
            // Deselect
            state.selectedReferenceImages.splice(index, 1);
            element.classList.remove('selected');
        } else {
            // Select (check limit)
            if (state.selectedReferenceImages.length >= 5) {
                this.toast('Maximum 5 images allowed', 'warning');
                return;
            }
            state.selectedReferenceImages.push(path);
            element.classList.add('selected');
        }

        this.updateSelectionCount();
    },

    updateSelectionCount() {
        this.elements.selectedCount.textContent = `${state.selectedReferenceImages.length} / 5 selected`;
    },

    confirmImageSelection() {
        this.renderSelectedImages();
        this.closeImageBrowser();
    },

    renderSelectedImages() {
        const container = this.elements.selectedImagesContainer;
        container.innerHTML = '';

        state.selectedReferenceImages.forEach(path => {
            const wrapper = document.createElement('div');
            wrapper.className = 'selected-img-wrapper';

            wrapper.innerHTML = `
                <img src="/api/images/preview?path=${encodeURIComponent(path)}" alt="Reference">
                <div class="remove-img" onclick="ui.removeSelectedImage('${path.replace(/\\/g, '\\\\')}')">×</div>
            `;

            container.appendChild(wrapper);
        });
    },

    removeSelectedImage(path) {
        state.selectedReferenceImages = state.selectedReferenceImages.filter(p => p !== path);
        this.renderSelectedImages();
    },

    // --- Classification ---

    refreshClassSelector() {
        const container = this.elements.classSelector;
        container.innerHTML = '';

        // Only show macro classes in the selector (Stages will handle subcategorisation)
        const macroClasses = state.classes.filter(c => !c.parent_id);

        macroClasses.forEach(cls => {
            const hasSub = state.classes.some(c => c.parent_id == cls.id);
            const label = document.createElement('label');
            label.className = 'class-checkbox';

            label.innerHTML = `
                <input type="checkbox" name="selected_classes" value="${cls.id}" checked>
                <div class="class-selector-info">
                    <strong>${cls.name}</strong>
                    ${hasSub ? `<span class="sub-indicator">↳ Includes subcategories</span>` : ''}
                </div>
            `;

            const input = label.querySelector('input');
            if (input.checked) label.classList.add('selected');

            input.addEventListener('change', () => {
                if (input.checked) label.classList.add('selected');
                else label.classList.remove('selected');
            });

            container.appendChild(label);
        });
    },

    async handleClassificationSubmit(e) {
        e.preventDefault();

        // Gather form data
        const jobName = document.getElementById('jobName').value;
        const strategy = document.getElementById('subStrategy').value;
        const clipModel = document.getElementById('clipModel').value;
        const numAugmentations = parseInt(document.getElementById('numAugmentations').value);
        const highThreshold = parseFloat(document.getElementById('highThreshold').value);
        const lowThreshold = parseFloat(document.getElementById('lowThreshold').value);
        const applySafetyNet = document.getElementById('applySafetyNet').checked;
        const subcategoryThreshold = parseFloat(document.getElementById('subcategoryThreshold').value);

        // Get selected classes
        const checkboxes = document.querySelectorAll('input[name="selected_classes"]:checked');
        const classIds = Array.from(checkboxes).map(cb => parseInt(cb.value));

        if (classIds.length === 0) {
            this.toast('Please select at least one class', 'warning');
            return;
        }

        // Validate thresholds
        if (lowThreshold >= highThreshold) {
            this.toast('Low threshold must be less than high threshold', 'error');
            return;
        }

        const config = {
            job_name: jobName,
            strategy: strategy,
            class_ids: classIds,
            clip_model: clipModel,
            num_augmentations: numAugmentations,
            high_prob_threshold: highThreshold,
            low_prob_threshold: lowThreshold,
            subcategory_threshold: subcategoryThreshold,
            apply_safety_net: applySafetyNet
        };

        try {
            const job = await api.post('/classify/start', config);
            this.startClassificationJob(job.id);
        } catch (error) {
            // Error handled by api wrapper
        }
    },

    startClassificationJob(jobId) {
        state.currentJobId = jobId;

        // UI Updates
        this.elements.progressSection.style.display = 'block';
        this.elements.progressSection.scrollIntoView({ behavior: 'smooth' });

        // Connect WebSocket
        this.connectWebSocket();
    },

    connectWebSocket() {
        if (state.wsConnection) state.wsConnection.close();

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/api/ws/progress`;

        state.wsConnection = new WebSocket(wsUrl);

        state.wsConnection.onmessage = (event) => {
            const update = JSON.parse(event.data);
            if (update.job_id === state.currentJobId) {
                this.updateProgress(update);
            }
        };

        state.wsConnection.onclose = () => {
            console.log('WebSocket connection closed');
        };

        state.wsConnection.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    },

    updateProgress(update) {
        const { status, progress_percentage, message, processed_images, total_images, current_image } = update;

        // Get UI elements
        const progressFill = document.getElementById('progressFill');
        const progressGlow = document.getElementById('progressGlow');
        const progressPercentage = document.getElementById('progressPercentage');
        const progressMessage = document.getElementById('progressMessage');
        const progressImages = document.getElementById('progressImages');
        const currentImageEl = document.getElementById('currentImage');
        const progressStage = document.getElementById('progressStage');
        const statusDot = document.querySelector('.status-dot');

        // Update progress bar
        if (progressFill) {
            progressFill.style.width = `${progress_percentage}%`;
        }
        if (progressGlow) {
            progressGlow.style.width = `${progress_percentage}%`;
        }
        if (progressPercentage) {
            progressPercentage.textContent = `${Math.round(progress_percentage)}%`;
        }

        // Update message
        if (progressMessage) {
            progressMessage.textContent = message || status;
        }

        // Update image counts
        if (progressImages && total_images > 0) {
            progressImages.textContent = `${processed_images} / ${total_images}`;
        }

        // Update current image
        if (currentImageEl && current_image) {
            currentImageEl.textContent = current_image;
        }

        // Determine stage from message
        if (progressStage && message) {
            if (message.includes('Sub-classif') || message.includes('Stage 2')) {
                progressStage.textContent = '2 of 2';
            } else if (message.includes('Preparing') || message.includes('reference')) {
                progressStage.textContent = 'Prep';
            } else if (message.includes('Organizing') || message.includes('Saving')) {
                progressStage.textContent = 'Saving';
            } else {
                progressStage.textContent = '1 of 2';
            }
        }

        // Update status dot animation
        if (statusDot) {
            if (status === 'completed') {
                statusDot.classList.remove('pulsing');
                statusDot.style.background = 'var(--success)';
            } else if (status === 'failed') {
                statusDot.classList.remove('pulsing');
                statusDot.style.background = 'var(--error)';
            }
        }

        // Handle completion
        if (status === 'completed' || status === 'failed') {
            if (status === 'completed') {
                this.toast('Classification completed successfully!', 'success');
                // Auto switch to results tab after a delay
                setTimeout(() => {
                    document.querySelector('.tab-button[data-tab="results"]').click();
                }, 2000);
            } else {
                this.toast(`Classification failed: ${message}`, 'error');
            }

            // Close WS connection
            if (state.wsConnection) state.wsConnection.close();
        }
    },

    // --- Results ---

    async loadJobs() {
        // Since we don't have a list-jobs endpoint in this simplified version,
        // we'll just mock it or show the recent job if available.
        // In a full implementation, we'd add GET /api/jobs

        const container = this.elements.jobsList;
        container.innerHTML = '';

        if (!state.currentJobId) {
            container.innerHTML = `
                <div style="text-align: center; padding: 40px; color: var(--text-tertiary);">
                    <h3>No recent jobs</h3>
                    <p>Run a classification task to see results here</p>
                </div>
            `;
            return;
        }

        // Fetch current job status/results
        try {
            const job = await api.get(`/classify/status/${state.currentJobId}`);
            const results = await api.get(`/classify/results/${state.currentJobId}`);

            const card = document.createElement('div');
            card.className = 'job-card';

            const highConfCount = results.filter(r => r.is_high_confidence).length;
            const lowConfCount = results.length - highConfCount;

            card.innerHTML = `
                <div class="job-info">
                    <h4>${job.job_name}</h4>
                    <div class="job-meta">
                        <span>ID: #${job.id}</span>
                        <span>${new Date().toLocaleDateString()}</span>
                        <span class="job-status status-${job.status.toLowerCase()}">${job.status}</span>
                    </div>
                </div>
                <div class="job-stats">
                    <div class="stat-item">
                        <span class="stat-value">${results.length}</span>
                        <span class="stat-label">Total</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-value" style="color: var(--success)">${highConfCount}</span>
                        <span class="stat-label">High Conf</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-value" style="color: var(--warning)">${lowConfCount}</span>
                        <span class="stat-label">Low Conf</span>
                    </div>
                </div>
                <div>
                     <button class="btn btn-secondary" onclick="ui.viewJobDetails(${job.id})">View Details</button>
                </div>
            `;

            container.appendChild(card);

        } catch (error) {
            console.error('Failed to load job', error);
        }
    },

    async viewJobDetails(jobId) {
        try {
            const job = await api.get(`/classify/status/${jobId}`);
            state.currentJobResults = await api.get(`/classify/results/${jobId}`);
            state.currentJobId = jobId;  // Track current job ID for open folder button

            // Update Summary
            document.getElementById('resultsTitle').textContent = `Results: ${job.job_name}`;
            document.getElementById('resTotal').textContent = state.currentJobResults.length;

            const highConf = state.currentJobResults.filter(r => r.is_high_confidence);
            const lowConf = state.currentJobResults.filter(r => !r.is_high_confidence);

            document.getElementById('resHigh').textContent = highConf.length;
            document.getElementById('resLow').textContent = lowConf.length;

            const subcategorisedCount = state.currentJobResults.filter(r => r.predicted_subclass_name).length;
            const subStat = document.getElementById('subclassStat');
            if (subcategorisedCount > 0) {
                subStat.style.display = 'block';
                document.getElementById('resSub').textContent = subcategorisedCount;
            } else {
                subStat.style.display = 'none';
            }

            // Reset filters
            state.currentFilter.confidence = highConf.length > 0 ? 'high' : 'low';
            state.currentFilter.class = null;

            // Updated tabs UI
            this.elements.resTabs.forEach(t => {
                t.classList.remove('active');
                if (t.dataset.filter === state.currentFilter.confidence) t.classList.add('active');
            });

            this.renderPerClassBreakdown();
            this.renderResultsGrid();
            this.elements.resultsModal.classList.add('open');

        } catch (error) {
            console.error('Failed to load job details', error);
        }
    },

    renderPerClassBreakdown() {
        const stats = {};

        state.currentJobResults.forEach(res => {
            if (!stats[res.predicted_class_name]) {
                stats[res.predicted_class_name] = { high: 0, low: 0 };
            }
            if (res.is_high_confidence) stats[res.predicted_class_name].high++;
            else stats[res.predicted_class_name].low++;
        });

        // Add breakdown to summary bar if not exists
        let breakdownContainer = document.getElementById('resBreakdown');
        if (!breakdownContainer) {
            breakdownContainer = document.createElement('div');
            breakdownContainer.id = 'resBreakdown';
            breakdownContainer.className = 'results-breakdown-grid';
            document.querySelector('.results-summary-bar').after(breakdownContainer);
        }

        breakdownContainer.innerHTML = '';
        Object.entries(stats).forEach(([name, count]) => {
            const div = document.createElement('div');
            div.className = `breakdown-item ${state.currentFilter.class === name ? 'active' : ''}`;
            div.style.cursor = 'pointer';
            div.onclick = () => {
                // Toggle filter
                if (state.currentFilter.class === name) {
                    state.currentFilter.class = null;
                } else {
                    state.currentFilter.class = name;
                }
                this.renderPerClassBreakdown();
                this.renderResultsGrid();
            };

            div.innerHTML = `
                <div class="breakdown-name">${name}</div>
                <div class="breakdown-stats">
                    <span class="success">H: ${count.high}</span>
                    <span class="warning">L: ${count.low}</span>
                </div>
            `;
            breakdownContainer.appendChild(div);
        });
    },

    renderResultsGrid() {
        const container = this.elements.resultsGrid;
        container.innerHTML = '';

        const filter = state.currentFilter;
        let filtered = state.currentJobResults.filter(r =>
            filter.confidence === 'high' ? r.is_high_confidence : !r.is_high_confidence
        );

        if (filter.class) {
            filtered = filtered.filter(r => r.predicted_class_name === filter.class);
        }

        if (filtered.length === 0) {
            const msg = filter.class ? `No images for "${filter.class}" in this category` : `No images in this category`;
            container.innerHTML = `<div style="grid-column: 1/-1; text-align: center; padding: 40px;">${msg}</div>`;
            return;
        }

        filtered.forEach(res => {
            const item = document.createElement('div');
            item.className = 'result-item';

            const filename = res.image_path.split(/[\\/]/).pop();
            const probPercent = Math.round(res.probability * 100);

            item.innerHTML = `
                <div class="result-img-box">
                    <img src="/api/images/preview?path=${encodeURIComponent(res.image_path)}" loading="lazy">
                    <div class="result-confidence">${probPercent}%</div>
                </div>
                <div class="result-info">
                    <span class="result-class">${res.predicted_class_name}</span>
                    ${res.predicted_subclass_name ? `<span class="result-subclass">↳ ${res.predicted_subclass_name}</span>` : ''}
                    <span class="result-filename" title="${res.image_path}">${filename}</span>
                </div>
            `;

            item.onclick = () => this.openPreviewModal(res);
            container.appendChild(item);
        });
    },

    openPreviewModal(res) {
        const filename = res.image_path.split(/[\\/]/).pop();
        const probPercent = Math.round(res.probability * 100);

        document.getElementById('previewTitle').textContent = filename;
        const classChain = res.predicted_subclass_name ? `${res.predicted_class_name} › ${res.predicted_subclass_name}` : res.predicted_class_name;
        document.getElementById('previewMeta').textContent = `Class: ${classChain} | Confidence: ${probPercent}%`;
        this.elements.previewImage.src = `/api/images/preview?path=${encodeURIComponent(res.image_path)}`;
        this.elements.previewModal.classList.add('open');
    },

    closePreviewModal() {
        this.elements.previewModal.classList.remove('open');
    },

    closeResultsModal() {
        this.elements.resultsModal.classList.remove('open');
    },

    async openResultsFolder() {
        // Get the current job ID from the results modal title
        const title = document.getElementById('resultsTitle').textContent;
        // Find the job ID from recent jobs state
        if (!state.currentJobId && state.currentJobResults && state.currentJobResults.length > 0) {
            state.currentJobId = state.currentJobResults[0].job_id;
        }

        if (!state.currentJobId) {
            this.toast('No job selected', 'warning');
            return;
        }

        try {
            const response = await api.post(`/classify/open-folder/${state.currentJobId}`);
            if (response.success) {
                this.toast('Opening folder in file explorer...', 'success');
            }
        } catch (error) {
            this.toast('Failed to open folder. Check if the results exist.', 'error');
        }
    },

    // --- Job Management ---

    async loadAllJobs() {
        try {
            const jobs = await api.get('/jobs');
            this.renderJobsTable(jobs);
        } catch (error) {
            console.error('Failed to load jobs', error);
            this.toast('Failed to load jobs', 'error');
        }
    },

    renderJobsTable(jobs) {
        const tbody = document.getElementById('jobsTableBody');
        tbody.innerHTML = '';

        if (jobs.length === 0) {
            tbody.innerHTML = `<tr><td colspan="5" style="text-align:center; color: var(--text-tertiary); padding: 30px;">No jobs found</td></tr>`;
            return;
        }

        jobs.forEach(job => {
            const tr = document.createElement('tr');

            const date = new Date(job.created_at).toLocaleDateString() + ' ' + new Date(job.created_at).toLocaleTimeString();

            let statusClass = 'pending';
            if (job.status === 'completed') statusClass = 'completed';
            else if (job.status === 'failed') statusClass = 'failed';
            else if (job.status === 'running') statusClass = 'running';

            tr.innerHTML = `
                <td><span class="status-badge ${statusClass}">${job.status}</span></td>
                <td style="font-weight: 500;">${job.job_name}</td>
                <td>${job.processed_images} / ${job.total_images}</td>
                <td style="color: var(--text-tertiary); font-size: 0.9em;">${date}</td>
                <td>
                    <div style="display: flex; gap: 8px;">
                        <button class="btn btn-secondary btn-small" onclick="ui.rerunJob(${job.id}, '${job.job_name}')" title="Rerun Job">
                           🔄 Rerun
                        </button>
                        <button class="btn btn-secondary btn-small" onclick="ui.deleteJob(${job.id}, '${job.job_name}')" style="color: var(--error);" title="Delete Job">
                           ❌ Delete
                        </button>
                    </div>
                </td>
            `;
            tbody.appendChild(tr);
        });
    },

    async deleteJob(id, name) {
        if (confirm(`Are you sure you want to delete job "${name}"? This will delete all results.`)) {
            try {
                await api.delete(`/jobs/${id}`);
                this.toast(`Job "${name}" deleted`, 'success');
                this.loadAllJobs();
                // If this was the current job, clear display
                if (state.currentJobId === id) {
                    state.currentJobId = null;
                    this.closeResultsModal();
                }
            } catch (error) {
                // Error handled by api wrapper
            }
        }
    },

    async deleteAllJobs() {
        if (confirm('WARNING: Are you sure you want to delete ALL jobs? This cannot be undone and will delete all result files.')) {
            // Double confirmation for destructive action
            if (confirm('Really delete everything?')) {
                try {
                    await api.delete('/jobs/all');
                    this.toast('All jobs deleted', 'success');
                    this.loadAllJobs();
                    state.currentJobId = null;
                } catch (error) {
                    // Error handled by api wrapper
                }
            }
        }
    },

    async rerunJob(id, name) {
        if (confirm(`Are you sure you want to rerun job "${name}"? This will DELETE existing results and restart classification.`)) {
            try {
                await api.post(`/jobs/${id}/rerun`, {});
                this.toast(`Restarting job "${name}"...`, 'info');
                // Switch to classify tab/progress view
                this.startClassificationJob(id);
                // Also switch tab to classify to see progress
                document.querySelector('.tab-button[data-tab="classify"]').click();
            } catch (error) {
                // Error handled by api wrapper
            }
        }
    },

    // --- Utilities ---

    toast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;

        const icon = type === 'success' ? '✓' : type === 'error' ? '!' : 'ℹ';

        toast.innerHTML = `
            <span style="font-weight: bold; font-size: 1.2rem;">${icon}</span>
            <span>${message}</span>
        `;

        this.elements.toastContainer.appendChild(toast);

        // Remove after 3 seconds
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(100%)';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
};

// Expose ui to window for global onclick handlers
window.ui = ui;

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    ui.init();
});
