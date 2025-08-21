// Multi-Document DOCX Viewer - Supports multiple documents from sample boxes
class MultiDocxViewer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.currentPage = 1;
        this.totalPages = 1;
        this.zoom = 100;
        this.docxPreviewLoaded = false;
        this.currentDocument = null;
        this.documentPath = null;
        
        this.init();
        this.loadDocxPreview();
    }
    
    async loadDocxPreview() {
        try {
            // Load JSZip first (required by docx-preview)
            const jszipScript = document.createElement('script');
            jszipScript.src = 'https://unpkg.com/jszip/dist/jszip.min.js';
            jszipScript.onload = () => {
                // Then load docx-preview
                const docxScript = document.createElement('script');
                docxScript.src = 'https://unpkg.com/docx-preview@0.3.6/dist/docx-preview.min.js';
                docxScript.onload = () => {
                    this.docxPreviewLoaded = true;
                    console.log('docx-preview loaded successfully');
                    // Show initial state
                    this.showWelcome();
                };
                docxScript.onerror = () => {
                    console.error('Failed to load docx-preview');
                    this.showError('Failed to load document processing library');
                };
                document.head.appendChild(docxScript);
            };
            jszipScript.onerror = () => {
                console.error('Failed to load JSZip');
                this.showError('Failed to load required dependencies');
            };
            document.head.appendChild(jszipScript);
        } catch (error) {
            console.error('Error loading docx-preview:', error);
        }
    }
    
    init() {
        this.createViewerInterface();
        this.bindEvents();
    }
    
    createViewerInterface() {
        this.container.innerHTML = `
            <div class="docx-viewer-wrapper">
                <div class="docx-viewer-toolbar">
                    <div class="docx-info">
                        <span class="docx-title">Document Viewer</span>
                        <span class="docx-path">Select a document to view</span>
                    </div>
                    <div class="docx-controls">
                        <button class="docx-btn" id="prevPage" disabled>‹</button>
                        <span class="page-info">Page <span id="currentPageNum">-</span> of <span id="totalPageNum">-</span></span>
                        <button class="docx-btn" id="nextPage" disabled>›</button>
                        <button class="docx-btn" id="zoomOut">−</button>
                        <span class="zoom-level">100%</span>
                        <button class="docx-btn" id="zoomIn">+</button>
                        <button class="docx-btn" id="fitWidth">Fit Width</button>
                        <button class="docx-btn" id="fullscreen">⛶</button>
                    </div>
                </div>
                <div class="docx-viewer-content" id="docxViewerContent">
                    <div class="docx-pages-container" id="docxContainer">
                        <div class="docx-placeholder">
                            <div class="placeholder-content">
                                <div class="loading-spinner"></div>
                                <h3>Loading Libraries...</h3>
                                <p>Preparing document viewer</p>
                                <small>Using docx-preview for real-time rendering</small>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        this.addStyles();
    }
    
    addStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .docx-viewer-wrapper {
                width: 100%;
                height: 100%;
                display: flex;
                flex-direction: column;
                background: #f5f5f5;
                border-radius: 8px;
                overflow: hidden;
                font-family: 'Inter', sans-serif;
            }
            
            .docx-viewer-toolbar {
                background: #2a2a2a;
                padding: 0.75rem 1rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid #333;
                flex-wrap: wrap;
                gap: 0.5rem;
            }
            
            .docx-info {
                display: flex;
                flex-direction: column;
                gap: 0.25rem;
                flex: 1;
                min-width: 200px;
            }
            
            .docx-title {
                font-weight: 500;
                color: #f0f0f0;
                font-size: 0.9rem;
            }
            
            .docx-path {
                font-size: 0.8rem;
                color: #a0a0a0;
                font-family: 'JetBrains Mono', monospace;
            }
            
            .docx-controls {
                display: flex;
                gap: 0.5rem;
                align-items: center;
                flex-wrap: wrap;
            }
            
            .docx-btn {
                background: #333;
                border: none;
                color: #f0f0f0;
                padding: 0.5rem 0.75rem;
                border-radius: 4px;
                cursor: pointer;
                font-size: 0.9rem;
                transition: background 0.2s ease;
                white-space: nowrap;
            }
            
            .docx-btn:hover:not(:disabled) {
                background: #444;
            }
            
            .docx-btn:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            
            .zoom-level, .page-info {
                color: #f0f0f0;
                font-size: 0.9rem;
                min-width: 40px;
                text-align: center;
                white-space: nowrap;
            }
            
            .docx-viewer-content {
                flex: 1;
                overflow: auto;
                background: #e0e0e0;
                position: relative;
                scroll-behavior: smooth;
            }
            
            .docx-pages-container {
                padding: 2rem;
                min-height: 100%;
            }
            
            .docx-placeholder {
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100%;
                min-height: 400px;
            }
            
            .placeholder-content {
                text-align: center;
                color: #666;
            }
            
            .placeholder-content h3 {
                margin-bottom: 0.5rem;
                color: #333;
            }
            
            .placeholder-content p {
                color: #666;
                margin-bottom: 0.5rem;
            }
            
            .placeholder-content small {
                color: #999;
                font-size: 0.8rem;
            }
            
            .loading-spinner {
                display: inline-block;
                width: 40px;
                height: 40px;
                border: 4px solid #f3f3f3;
                border-top: 4px solid #6200ea;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-bottom: 1rem;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .error-message {
                color: #ff6b6b;
                background: #ffe6e6;
                padding: 1rem;
                border-radius: 4px;
                border-left: 4px solid #ff6b6b;
                margin: 1rem 0;
            }
            
            .welcome-message {
                color: #6200ea;
                background: #f8f4ff;
                padding: 1rem;
                border-radius: 4px;
                border-left: 4px solid #6200ea;
                margin: 1rem 0;
            }
            
            /* docx-preview generated content styling */
            .docx {
                background: white;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                border-radius: 4px;
                overflow: hidden;
                margin: 0 auto;
                max-width: 100%;
                transform-origin: top center;
                transition: transform 0.3s ease;
            }
            
            .docx-wrapper {
                padding: 2.54cm;
                min-height: 297mm;
                width: 210mm;
                background: white;
                margin: 0 auto;
                box-sizing: border-box;
            }
            
            /* Enhanced page styling for better separation */
            .docx .docx-page,
            .docx > div,
            .docx section {
                background: white;
                margin-bottom: 2rem;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
                border-radius: 4px;
                overflow: hidden;
                page-break-after: always;
                min-height: 297mm; /* A4 height */
                width: 210mm; /* A4 width */
                padding: 2.54cm;
                box-sizing: border-box;
                position: relative;
            }
            
            /* Page indicator */
            .page-indicator {
                position: fixed;
                top: 50%;
                right: 2rem;
                transform: translateY(-50%);
                background: rgba(42, 42, 42, 0.9);
                color: white;
                padding: 0.5rem 1rem;
                border-radius: 20px;
                font-size: 0.8rem;
                z-index: 1000;
                backdrop-filter: blur(10px);
            }
            
            /* Responsive design */
            @media (max-width: 768px) {
                .docx-viewer-toolbar {
                    flex-direction: column;
                    align-items: stretch;
                }
                
                .docx-info {
                    text-align: center;
                    min-width: auto;
                }
                
                .docx-controls {
                    justify-content: center;
                }
                
                .docx-pages-container {
                    padding: 1rem;
                }
                
                .docx-wrapper {
                    width: 100%;
                    min-width: 300px;
                    padding: 1rem;
                }
                
                .docx .docx-page,
                .docx > div,
                .docx section {
                    width: 100%;
                    min-width: 300px;
                    padding: 1rem;
                    min-height: auto;
                }
                
                .page-indicator {
                    right: 1rem;
                }
            }
        `;
        document.head.appendChild(style);
    }
    
    bindEvents() {
        // Zoom controls
        document.getElementById('zoomIn').addEventListener('click', () => this.zoomIn());
        document.getElementById('zoomOut').addEventListener('click', () => this.zoomOut());
        document.getElementById('fitWidth').addEventListener('click', () => this.fitWidth());
        
        // Page navigation
        document.getElementById('prevPage').addEventListener('click', () => this.prevPage());
        document.getElementById('nextPage').addEventListener('click', () => this.nextPage());
        
        // Fullscreen
        document.getElementById('fullscreen').addEventListener('click', () => this.toggleFullscreen());
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.target.closest('.docx-viewer-wrapper')) {
                switch(e.key) {
                    case 'ArrowLeft':
                    case 'ArrowUp':
                        e.preventDefault();
                        this.prevPage();
                        break;
                    case 'ArrowRight':
                    case 'ArrowDown':
                        e.preventDefault();
                        this.nextPage();
                        break;
                    case '+':
                    case '=':
                        e.preventDefault();
                        this.zoomIn();
                        break;
                    case '-':
                        e.preventDefault();
                        this.zoomOut();
                        break;
                }
            }
        });
        
        // Scroll detection for page tracking
        const container = document.getElementById('docxViewerContent');
        if (container) {
            container.addEventListener('scroll', () => this.updateCurrentPage());
        }
    }
    
    async loadDocument(filename) {
        if (!this.docxPreviewLoaded) {
            setTimeout(() => this.loadDocument(filename), 100);
            return;
        }
        
        try {
            this.showLoading();
            this.documentPath = `./${filename}`;
            
            // Update document info
            const docxTitleElement = document.querySelector('.docx-title');
            const docxPathElement = document.querySelector('.docx-path');
            if (docxTitleElement) docxTitleElement.textContent = filename.replace('.docx', '');
            if (docxPathElement) docxPathElement.textContent = `Loading: ${filename}`;
            
            // Fetch the document
            const response = await fetch(this.documentPath);
            if (!response.ok) {
                throw new Error(`Failed to load document: ${response.status} ${response.statusText}`);
            }
            
            const arrayBuffer = await response.arrayBuffer();
            
            // Render the document using docx-preview
            const container = document.getElementById('docxContainer');
            if (!container) {
                throw new Error('docxContainer element not found.');
            }
            container.innerHTML = ''; // Clear previous content
            
            const options = {
                className: 'docx',
                inWrapper: true,
                ignoreWidth: false,
                ignoreHeight: false,
                ignoreFonts: false,
                breakPages: true,
                ignoreLastRenderedPageBreak: false,
                experimental: false,
                trimXmlDeclaration: true,
                useBase64URL: false,
                renderHeaders: true,
                renderFooters: true,
                renderFootnotes: true,
                renderEndnotes: true,
                debug: false
            };
            
            await docx.renderAsync(arrayBuffer, container, null, options);
            
            // Update document info after successful load
            if (docxPathElement) docxPathElement.textContent = `Document: ${filename}`;
            
            // Add page indicator
            this.addPageIndicator();
            
            // Count pages and update controls
            this.countPages();
            this.updateControls();
            this.applyZoom();
            
            console.log(`Document ${filename} loaded successfully`);
            
        } catch (error) {
            console.error('Error loading document:', error);
            this.showError(`Error loading document: ${error.message}\\n\\nMake sure the file exists: ${filename}`);
        }
    }
    
    addPageIndicator() {
        // Remove existing indicator
        const existing = document.querySelector('.page-indicator');
        if (existing) existing.remove();
        
        // Add new indicator
        const indicator = document.createElement('div');
        indicator.className = 'page-indicator';
        indicator.textContent = `${this.currentPage} / ${this.totalPages}`;
        document.body.appendChild(indicator);
    }
    
    countPages() {
        // Try to detect pages from rendered content
        const pages = document.querySelectorAll('.docx .docx-page, .docx > div, .docx section');
        if (pages.length > 0) {
            this.totalPages = pages.length;
        } else {
            // Fallback: estimate pages based on content height
            const docx = document.querySelector('.docx');
            if (docx) {
                const pageHeight = 297 * 3.78; // A4 height in pixels (approx)
                const contentHeight = docx.scrollHeight;
                this.totalPages = Math.max(1, Math.ceil(contentHeight / pageHeight));
            }
        }
        
        // Update page counter in toolbar
        document.getElementById('totalPageNum').textContent = this.totalPages;
        this.currentPage = 1;
        document.getElementById('currentPageNum').textContent = this.currentPage;
    }
    
    updateCurrentPage() {
        const container = document.getElementById('docxViewerContent');
        const pages = document.querySelectorAll('.docx .docx-page, .docx > div, .docx section');
        
        if (pages.length === 0) {
            // Fallback: calculate page based on scroll position
            const scrollTop = container.scrollTop;
            const pageHeight = container.clientHeight;
            this.currentPage = Math.max(1, Math.ceil(scrollTop / pageHeight) + 1);
        } else {
            const containerRect = container.getBoundingClientRect();
            const containerCenter = containerRect.top + containerRect.height / 2;
            
            let currentPage = 1;
            pages.forEach((page, index) => {
                const pageRect = page.getBoundingClientRect();
                if (pageRect.top <= containerCenter && pageRect.bottom >= containerCenter) {
                    currentPage = index + 1;
                }
            });
            this.currentPage = currentPage;
        }
        
        this.updatePageIndicator();
        this.updateControls();
        document.getElementById('currentPageNum').textContent = this.currentPage;
    }
    
    updatePageIndicator() {
        const indicator = document.querySelector('.page-indicator');
        if (indicator) {
            indicator.textContent = `${this.currentPage} / ${this.totalPages}`;
        }
    }
    
    showWelcome() {
        const container = document.getElementById('docxContainer');
        if (!container) return;
        
        container.innerHTML = `
            <div class="docx-placeholder">
                <div class="placeholder-content">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">📄</div>
                    <h3>Document Viewer Ready</h3>
                    <div class="welcome-message">
                        Select a document from the sample boxes below to view it with real-time rendering and page scrolling features.
                    </div>
                    <small>Supports DOCX files with original formatting</small>
                </div>
            </div>
        `;
    }
    
    showLoading() {
        const container = document.getElementById('docxContainer');
        if (!container) return;
        
        container.innerHTML = `
            <div class="docx-placeholder">
                <div class="placeholder-content">
                    <div class="loading-spinner"></div>
                    <h3>Loading Document...</h3>
                    <p>Processing DOCX file</p>
                    <small>Rendering with real-time formatting</small>
                </div>
            </div>
        `;
    }
    
    showError(message) {
        const container = document.getElementById('docxContainer');
        if (!container) return;
        
        container.innerHTML = `
            <div class="docx-placeholder">
                <div class="placeholder-content">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">❌</div>
                    <h3>Error Loading Document</h3>
                    <div class="error-message">${message}</div>
                    <button class="docx-btn" onclick="documentViewer.showWelcome()" style="margin-top: 1rem;">
                        Back to Selection
                    </button>
                </div>
            </div>
        `;
    }
    
    zoomIn() {
        this.zoom = Math.min(this.zoom + 25, 300);
        this.applyZoom();
        this.updateZoomDisplay();
    }
    
    zoomOut() {
        this.zoom = Math.max(this.zoom - 25, 25);
        this.applyZoom();
        this.updateZoomDisplay();
    }
    
    fitWidth() {
        const container = document.getElementById('docxViewerContent');
        const docx = document.querySelector('.docx');
        if (docx && container) {
            const containerWidth = container.clientWidth - 80;
            const docxWidth = docx.scrollWidth;
            this.zoom = Math.floor((containerWidth / docxWidth) * 100);
            this.applyZoom();
            this.updateZoomDisplay();
        }
    }
    
    applyZoom() {
        const docx = document.querySelector('.docx');
        if (docx) {
            docx.style.transform = `scale(${this.zoom / 100})`;
            docx.style.marginBottom = `${(this.zoom - 100) * 0.1}rem`;
        }
    }
    
    updateZoomDisplay() {
        const zoomElement = document.querySelector('.zoom-level');
        if (zoomElement) {
            zoomElement.textContent = `${this.zoom}%`;
        }
    }
    
    prevPage() {
        if (this.currentPage > 1) {
            this.scrollToPage(this.currentPage - 1);
        }
    }
    
    nextPage() {
        if (this.currentPage < this.totalPages) {
            this.scrollToPage(this.currentPage + 1);
        }
    }
    
    scrollToPage(pageNumber) {
        const container = document.getElementById('docxViewerContent');
        const pages = document.querySelectorAll('.docx .docx-page, .docx > div, .docx section');
        
        if (pages[pageNumber - 1]) {
            pages[pageNumber - 1].scrollIntoView({ 
                behavior: 'smooth', 
                block: 'start' 
            });
        } else {
            // Fallback: scroll based on estimated page height
            const pageHeight = container.clientHeight;
            const scrollTop = (pageNumber - 1) * pageHeight;
            container.scrollTo({
                top: scrollTop,
                behavior: 'smooth'
            });
        }
        
        this.currentPage = pageNumber;
        this.updatePageIndicator();
        this.updateControls();
        document.getElementById('currentPageNum').textContent = this.currentPage;
    }
    
    updateControls() {
        const prevBtn = document.getElementById('prevPage');
        const nextBtn = document.getElementById('nextPage');
        
        if (prevBtn) prevBtn.disabled = this.currentPage <= 1;
        if (nextBtn) nextBtn.disabled = this.currentPage >= this.totalPages;
    }
    
    toggleFullscreen() {
        if (!document.fullscreenElement) {
            this.container.requestFullscreen();
        } else {
            document.exitFullscreen();
        }
    }
}

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MultiDocxViewer;
}

