/**
 * IntelCLaw Control UI - Main Application
 */

class IntelCLawApp {
    constructor() {
        // State
        this.messages = [];
        this.currentSessionId = null;
        this.sessions = [];
        this.settings = this._loadSettings();
        this.isTyping = false;
        this.currentStreamingMessage = null;

        // WebSocket
        this.ws = new WebSocketManager();

        // DOM Elements
        this.elements = {};
        
        // Initialize when DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.init());
        } else {
            this.init();
        }
    }

    /**
     * Initialize the application
     */
    init() {
        console.log('[App] Initializing IntelCLaw Control UI...');
        
        this._cacheElements();
        this._setupEventListeners();
        this._setupWebSocket();
        this._loadModels();  // Load models from API
        this._applySettings();
        this._createNewSession();
        
        console.log('[App] Initialization complete');
    }

    /**
     * Load models from the API and populate the selector
     */
    async _loadModels() {
        console.log('[App] Loading models from /api/models...');
        try {
            const response = await fetch('/api/models');
            const data = await response.json();
            
            console.log('[App] Models API response:', data);
            
            if (data.models && data.models.length > 0) {
                const selector = this.elements.modelSelector;
                selector.innerHTML = '';  // Clear existing options
                
                // Group models by category
                const categories = {};
                data.models.forEach(model => {
                    const category = model.category || model.provider || 'Other';
                    if (!categories[category]) {
                        categories[category] = [];
                    }
                    categories[category].push(model);
                });
                
                // Sort categories - Copilot first, then others
                const sortedCategories = Object.keys(categories).sort((a, b) => {
                    // Anthropic and OpenAI Copilot models first
                    if (a.includes('Anthropic') && !b.includes('Anthropic')) return -1;
                    if (!a.includes('Anthropic') && b.includes('Anthropic')) return 1;
                    if (a.includes('Copilot') && !b.includes('Copilot')) return -1;
                    if (!a.includes('Copilot') && b.includes('Copilot')) return 1;
                    return a.localeCompare(b);
                });
                
                // Create optgroups for each category
                for (const category of sortedCategories) {
                    const models = categories[category];
                    const optgroup = document.createElement('optgroup');
                    optgroup.label = category;
                    
                    // Filter to only chat models (exclude embeddings)
                    const chatModels = models.filter(m => {
                        const type = m.capabilities?.type;
                        return !type || type === 'chat' || type === 'completion';
                    });
                    
                    chatModels.forEach(model => {
                        const option = document.createElement('option');
                        option.value = model.id;
                        option.textContent = model.name;
                        option.dataset.provider = model.provider || 'github-copilot';
                        if (model.id === data.current) {
                            option.selected = true;
                        }
                        optgroup.appendChild(option);
                    });
                    
                    if (chatModels.length > 0) {
                        selector.appendChild(optgroup);
                    }
                }
                
                // Update current model display and settings
                if (data.current) {
                    this.settings.model = data.current;
                    this.settings.provider = data.provider || 'github-copilot';
                    if (this.elements.currentModelDisplay) {
                        this.elements.currentModelDisplay.textContent = data.current;
                    }
                }
                
                console.log(`[App] Loaded ${data.models.length} models, provider: ${data.provider}, has_copilot: ${data.has_copilot}, dynamic: ${data.dynamic}`);
            } else {
                console.warn('[App] No models received from API');
                // Add a fallback option
                const option = document.createElement('option');
                option.value = 'gpt-4o';
                option.textContent = 'GPT-4o (fallback)';
                this.elements.modelSelector.appendChild(option);
            }
        } catch (error) {
            console.error('[App] Failed to load models:', error);
            // Add a fallback option on error
            const option = document.createElement('option');
            option.value = 'gpt-4o';
            option.textContent = 'GPT-4o (error fallback)';
            this.elements.modelSelector.innerHTML = '';
            this.elements.modelSelector.appendChild(option);
        }
    }

    /**
     * Cache DOM elements for quick access
     */
    _cacheElements() {
        this.elements = {
            // Layout
            app: document.getElementById('app'),
            sidebar: document.getElementById('sidebar'),
            rightPanel: document.getElementById('rightPanel'),
            
            // Messages
            messagesContainer: document.getElementById('messagesContainer'),
            emptyState: document.getElementById('emptyState'),
            workflowPanel: document.getElementById('workflowPanel'),
            workflowStatus: document.getElementById('workflowStatus'),
            workflowNow: document.getElementById('workflowNow'),
            workflowNext: document.getElementById('workflowNext'),
            workflowSteps: document.getElementById('workflowSteps'),
            workflowPhases: document.getElementById('workflowPhases'),
            
            // Input
            messageInput: document.getElementById('messageInput'),
            sendBtn: document.getElementById('sendBtn'),
            attachBtn: document.getElementById('attachBtn'),
            
            // Header
            menuToggle: document.getElementById('menuToggle'),
            modelSelector: document.getElementById('modelSelector'),
            settingsToggle: document.getElementById('settingsToggle'),
            sessionSelector: document.getElementById('sessionSelector'),
            currentSession: document.getElementById('currentSession'),
            
            // Status
            connectionDot: document.getElementById('connectionDot'),
            connectionStatus: document.getElementById('connectionStatus'),
            currentModelDisplay: document.getElementById('currentModelDisplay'),
            taskCount: document.getElementById('taskCount'),
            
            // Settings Panel
            panelClose: document.getElementById('panelClose'),
            streamingToggle: document.getElementById('streamingToggle'),
            showToolsToggle: document.getElementById('showToolsToggle'),
            soundToggle: document.getElementById('soundToggle'),
            temperatureSlider: document.getElementById('temperatureSlider'),
            temperatureValue: document.getElementById('temperatureValue'),
            maxTokensInput: document.getElementById('maxTokensInput'),
            systemPromptInput: document.getElementById('systemPromptInput'),
            
            // Navigation
            navItems: document.querySelectorAll('.nav-item[data-view]'),
            suggestionChips: document.querySelectorAll('.suggestion-chip'),
            panelTabs: document.querySelectorAll('.panel-tab')
        };
    }

    /**
     * Setup event listeners
     */
    _setupEventListeners() {
        // Send message
        this.elements.sendBtn.addEventListener('click', () => this.sendMessage());
        
        // Input handling
        this.elements.messageInput.addEventListener('input', () => this._handleInputChange());
        this.elements.messageInput.addEventListener('keydown', (e) => this._handleInputKeydown(e));
        
        // Menu toggle (mobile)
        this.elements.menuToggle.addEventListener('click', () => this._toggleSidebar());
        
        // Settings toggle
        this.elements.settingsToggle.addEventListener('click', () => this._toggleRightPanel());
        this.elements.panelClose.addEventListener('click', () => this._toggleRightPanel(false));
        
        // Model selector
        this.elements.modelSelector.addEventListener('change', (e) => this._handleModelChange(e));
        
        // Navigation
        this.elements.navItems.forEach(item => {
            item.addEventListener('click', (e) => this._handleNavClick(e, item));
        });
        
        // Suggestion chips
        this.elements.suggestionChips.forEach(chip => {
            chip.addEventListener('click', () => {
                const suggestion = chip.dataset.suggestion;
                this.elements.messageInput.value = suggestion;
                this._handleInputChange();
                this.elements.messageInput.focus();
            });
        });
        
        // Panel tabs
        this.elements.panelTabs.forEach(tab => {
            tab.addEventListener('click', () => this._handleTabSwitch(tab));
        });
        
        // Toggles
        this.elements.streamingToggle.addEventListener('click', (e) => this._handleToggle(e.target, 'streaming'));
        this.elements.showToolsToggle.addEventListener('click', (e) => this._handleToggle(e.target, 'showTools'));
        this.elements.soundToggle.addEventListener('click', (e) => this._handleToggle(e.target, 'sound'));
        
        // Temperature slider
        this.elements.temperatureSlider.addEventListener('input', (e) => {
            this.elements.temperatureValue.textContent = e.target.value;
            this.settings.temperature = parseFloat(e.target.value);
            this._saveSettings();
        });
        
        // New session handler
        document.querySelector('[data-action="new-session"]')?.addEventListener('click', (e) => {
            e.preventDefault();
            this._createNewSession();
        });
    }

    /**
     * Setup WebSocket connection and handlers
     */
    _setupWebSocket() {
        // Connection status handlers
        this.ws.onConnect(() => {
            this._updateConnectionStatus('connected');
            // Request initial state
            this.ws.send('get_state');
        });

        this.ws.onDisconnect(() => {
            this._updateConnectionStatus('disconnected');
        });

        this.ws.onError(() => {
            this._updateConnectionStatus('error');
        });

        // Message handlers
        this.ws.on('chat_response', (data) => this._handleChatResponse(data));
        this.ws.on('chat_stream', (data) => this._handleChatStream(data));
        this.ws.on('chat_complete', (data) => this._handleChatComplete(data));
        this.ws.on('tool_call', (data) => this._handleToolCall(data));
        this.ws.on('tool_result', (data) => this._handleToolResult(data));
        this.ws.on('error', (data) => this._handleError(data));
        this.ws.on('state', (data) => this._handleState(data));
        this.ws.on('workflow', (data) => this._handleWorkflow(data));

        // Connect
        this.ws.connect();
    }

    /**
     * Update connection status UI
     */
    _updateConnectionStatus(status) {
        const statusMap = {
            connected: { text: 'Online', dotClass: '' },
            connecting: { text: 'Connecting', dotClass: 'warning' },
            disconnected: { text: 'Offline', dotClass: 'error' },
            error: { text: 'Error', dotClass: 'error' }
        };

        const { text, dotClass } = statusMap[status] || statusMap.disconnected;
        
        this.elements.connectionStatus.textContent = text;
        this.elements.connectionDot.className = 'status-dot' + (dotClass ? ` ${dotClass}` : '');
    }

    /**
     * Send a message to the agent
     */
    sendMessage() {
        const content = this.elements.messageInput.value.trim();
        if (!content) return;

        // Hide empty state
        this.elements.emptyState.classList.add('hidden');

        // Add user message to UI
        this._addMessage({
            role: 'user',
            content: content,
            timestamp: new Date().toISOString()
        });

        // Clear input
        this.elements.messageInput.value = '';
        this._handleInputChange();

        // Show typing indicator
        this._showTypingIndicator();

        // Send to backend
        this.ws.send('chat', {
            message: content,
            session_id: this.currentSessionId,
            model: this.elements.modelSelector.value,
            settings: {
                temperature: this.settings.temperature,
                max_tokens: parseInt(this.elements.maxTokensInput.value) || 4096,
                stream: this.settings.streaming
            }
        });
    }

    /**
     * Add a message to the chat UI
     */
    _addMessage(message) {
        this.messages.push(message);

        const messageEl = document.createElement('div');
        messageEl.className = `message ${message.role}`;
        messageEl.dataset.id = message.id || Date.now();

        const avatarIcon = {
            user: '👤',
            assistant: '🦅',
            system: '⚙️',
            tool: '🔧'
        }[message.role] || '💬';

        const senderName = {
            user: 'You',
            assistant: 'IntelCLaw',
            system: 'System',
            tool: 'Tool'
        }[message.role] || 'Unknown';

        const time = message.timestamp 
            ? new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            : '';

        messageEl.innerHTML = `
            <div class="message-avatar">${avatarIcon}</div>
            <div class="message-body">
                <div class="message-header">
                    <span class="message-sender">${senderName}</span>
                    ${time ? `<span class="message-time">${time}</span>` : ''}
                    ${message.model ? `<span class="message-model">${message.model}</span>` : ''}
                </div>
                <div class="message-content">${this._formatContent(message.content)}</div>
            </div>
        `;

        this.elements.messagesContainer.appendChild(messageEl);
        this._scrollToBottom();

        return messageEl;
    }

    /**
     * Format message content (basic markdown)
     */
    _formatContent(content) {
        if (!content) return '';
        
        // Escape HTML
        let html = content
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');

        // Code blocks
        html = html.replace(/```(\w*)\n?([\s\S]*?)```/g, (match, lang, code) => {
            return `<pre><code class="language-${lang}">${code.trim()}</code></pre>`;
        });

        // Inline code
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Bold
        html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

        // Italic
        html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

        // Links
        html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

        // Line breaks
        html = html.replace(/\n/g, '<br>');

        // Paragraphs
        html = '<p>' + html.replace(/<br><br>/g, '</p><p>') + '</p>';

        return html;
    }

    /**
     * Show typing indicator
     */
    _showTypingIndicator() {
        this.isTyping = true;
        
        const indicator = document.createElement('div');
        indicator.className = 'message assistant';
        indicator.id = 'typingIndicator';
        indicator.innerHTML = `
            <div class="message-avatar">🦅</div>
            <div class="message-body">
                <div class="typing-indicator">
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                </div>
            </div>
        `;

        this.elements.messagesContainer.appendChild(indicator);
        this._scrollToBottom();
    }

    /**
     * Hide typing indicator
     */
    _hideTypingIndicator() {
        this.isTyping = false;
        const indicator = document.getElementById('typingIndicator');
        if (indicator) {
            indicator.remove();
        }
    }

    /**
     * Scroll chat to bottom
     */
    _scrollToBottom() {
        this.elements.messagesContainer.scrollTop = this.elements.messagesContainer.scrollHeight;
    }

    /**
     * Handle chat response from server
     */
    _handleChatResponse(data) {
        this._hideTypingIndicator();
        
        this._addMessage({
            role: 'assistant',
            content: data.content,
            model: data.model,
            timestamp: new Date().toISOString()
        });
    }

    /**
     * Handle streaming chat response
     */
    _handleChatStream(data) {
        if (!this.currentStreamingMessage) {
            this._hideTypingIndicator();
            
            // Create new message element for streaming
            const messageEl = document.createElement('div');
            messageEl.className = 'message assistant';
            messageEl.id = 'streamingMessage';
            messageEl.innerHTML = `
                <div class="message-avatar">🦅</div>
                <div class="message-body">
                    <div class="message-header">
                        <span class="message-sender">IntelCLaw</span>
                        <span class="message-time">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                        ${data.model ? `<span class="message-model">${data.model}</span>` : ''}
                    </div>
                    <div class="message-content"></div>
                </div>
            `;
            this.elements.messagesContainer.appendChild(messageEl);
            this.currentStreamingMessage = {
                element: messageEl,
                content: ''
            };
        }

        // Append new content
        this.currentStreamingMessage.content += data.delta || data.content || '';
        this.currentStreamingMessage.element.querySelector('.message-content').innerHTML = 
            this._formatContent(this.currentStreamingMessage.content);
        
        this._scrollToBottom();
    }

    /**
     * Handle chat completion
     */
    _handleChatComplete(data) {
        this._hideTypingIndicator();
        
        if (this.currentStreamingMessage) {
            // Finalize the streaming message
            this.messages.push({
                role: 'assistant',
                content: this.currentStreamingMessage.content,
                model: data.model,
                timestamp: new Date().toISOString()
            });
            this.currentStreamingMessage.element.removeAttribute('id');
            this.currentStreamingMessage = null;
        }
    }

    /**
     * Handle tool call from agent
     */
    _handleToolCall(data) {
        if (!this.settings.showTools) return;

        const toolEl = document.createElement('div');
        toolEl.className = 'message tool';
        toolEl.dataset.toolId = data.id;
        toolEl.innerHTML = `
            <div class="message-avatar">🔧</div>
            <div class="message-body">
                <div class="tool-call">
                    <div class="tool-header">
                        <span class="tool-icon">⚡</span>
                        <span class="tool-name">${data.name}</span>
                        <span class="tool-status running">Running</span>
                    </div>
                    <div class="tool-content">${JSON.stringify(data.args, null, 2)}</div>
                </div>
            </div>
        `;

        this.elements.messagesContainer.appendChild(toolEl);
        this._scrollToBottom();
    }

    /**
     * Handle tool result
     */
    _handleToolResult(data) {
        const toolEl = document.querySelector(`[data-tool-id="${data.id}"]`);
        if (toolEl) {
            const statusEl = toolEl.querySelector('.tool-status');
            const contentEl = toolEl.querySelector('.tool-content');
            
            if (data.success) {
                statusEl.className = 'tool-status success';
                statusEl.textContent = 'Success';
            } else {
                statusEl.className = 'tool-status error';
                statusEl.textContent = 'Error';
            }

            if (data.result) {
                contentEl.textContent = typeof data.result === 'string' 
                    ? data.result 
                    : JSON.stringify(data.result, null, 2);
            }
        }
    }

    /**
     * Handle error from server
     */
    _handleError(data) {
        this._hideTypingIndicator();
        
        this._addMessage({
            role: 'system',
            content: `**Error:** ${data.message || 'An unexpected error occurred'}`,
            timestamp: new Date().toISOString()
        });
    }

    /**
     * Handle state update from server
     */
    _handleState(data) {
        if (data.model) {
            this.elements.modelSelector.value = data.model;
            this.elements.currentModelDisplay.textContent = data.model;
        }
        
        if (data.tasks) {
            this.elements.taskCount.textContent = data.tasks.length;
        }

        if (data.workflow) {
            this._renderWorkflow(data.workflow);
        }
    }

    /**
     * Handle workflow update from server
     */
    _handleWorkflow(data) {
        if (data.workflow) {
            this._renderWorkflow(data.workflow);
        }
    }

    /**
     * Render workflow plan, progress, and queue
     */
    _renderWorkflow(workflow) {
        if (!this.elements.workflowPanel) return;

        const plan = workflow.plan || [];
        const completed = new Set(workflow.completed_steps || []);
        const currentIndex = workflow.current_step || 0;
        const computedProgress = plan.length > 0 ? (completed.size / plan.length) * 100 : 0;
        const progress = typeof workflow.progress === 'number' ? workflow.progress : computedProgress;
        const queue = workflow.queue || [];

        // Status with color coding
        if (this.elements.workflowStatus) {
            const status = workflow.status || 'IDLE';
            const statusText = status.toUpperCase();
            this.elements.workflowStatus.textContent = statusText;
            this.elements.workflowStatus.className = 'workflow-status status-' + status.toLowerCase();
        }

        // Phase chips (Plan → Act → Review)
        if (this.elements.workflowPhases) {
            const status = (workflow.status || 'idle').toLowerCase();
            const phases = this.elements.workflowPhases.querySelectorAll('.phase');
            phases.forEach(phase => {
                phase.classList.remove('active', 'done');
                const name = phase.dataset.phase;
                if (status === 'thinking' && name === 'plan') {
                    phase.classList.add('active');
                } else if (status === 'executing' && name === 'act') {
                    phase.classList.add('active');
                } else if ((status === 'idle' || status === 'completed') && name === 'review') {
                    phase.classList.add('active');
                }
            });
            
            // Mark done when progress reaches 100
            if (progress >= 100) {
                phases.forEach(phase => phase.classList.add('done'));
            }
        }

        // Progress bar
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        if (progressFill) {
            progressFill.style.width = `${progress}%`;
            // Color based on progress
            if (progress >= 100) {
                progressFill.classList.add('complete');
            } else {
                progressFill.classList.remove('complete');
            }
        }
        if (progressText) {
            progressText.textContent = `${Math.round(progress)}%`;
        }

        // Current step (use current_step_title if available)
        const nowStep = workflow.current_step_title || plan[currentIndex] || '-';
        const nextStep = workflow.next_step || plan[currentIndex + 1] || '-';
        const truncate = (text, max = 140) => {
            if (!text) return '-';
            if (text.length <= max) return text;
            return text.slice(0, max).trim() + '…';
        };
        
        if (this.elements.workflowNow) {
            this.elements.workflowNow.textContent = truncate(nowStep, 180);
            this.elements.workflowNow.title = nowStep;
            // Add animation when step changes
            this.elements.workflowNow.classList.remove('step-change');
            void this.elements.workflowNow.offsetWidth; // Force reflow
            this.elements.workflowNow.classList.add('step-change');
        }
        if (this.elements.workflowNext) {
            this.elements.workflowNext.textContent = truncate(nextStep, 180);
            this.elements.workflowNext.title = nextStep;
        }

        // Step count
        const stepCount = document.getElementById('stepCount');
        if (stepCount) {
            stepCount.textContent = `${completed.size}/${plan.length}`;
        }

        // Steps list with icons
        if (this.elements.workflowSteps) {
            this.elements.workflowSteps.innerHTML = '';
            plan.forEach((step, idx) => {
                const li = document.createElement('li');
                li.className = 'workflow-step';
                
                // Status icon
                const icon = document.createElement('span');
                icon.className = 'step-icon';
                
                if (completed.has(step)) {
                    li.classList.add('done');
                    icon.innerHTML = '✓';
                    icon.classList.add('icon-done');
                } else if (idx === currentIndex) {
                    li.classList.add('active');
                    icon.innerHTML = '●';
                    icon.classList.add('icon-active', 'pulse');
                } else {
                    icon.innerHTML = String(idx + 1);
                    icon.classList.add('icon-pending');
                }
                
                const text = document.createElement('span');
                text.className = 'step-text';
                text.textContent = truncate(step, 160);
                text.title = step;
                
                li.appendChild(icon);
                li.appendChild(text);
                this.elements.workflowSteps.appendChild(li);
            });
        }

        // Task queue
        const queuePanel = document.getElementById('workflowQueue');
        const queueList = document.getElementById('queueList');
        const queueCount = document.getElementById('queueCount');
        
        if (queuePanel && queue.length > 0) {
            queuePanel.classList.remove('hidden');
            if (queueCount) queueCount.textContent = queue.length;
            
            if (queueList) {
                queueList.innerHTML = '';
                queue.slice(0, 5).forEach((task, idx) => {
                    const li = document.createElement('li');
                    li.className = 'queue-item';
                    if (idx === 0) li.classList.add('current');
                    const label = typeof task === 'string' ? task : task.goal || task.title || 'Task';
                    li.textContent = truncate(label, 140);
                    li.title = label;
                    queueList.appendChild(li);
                });
                if (queue.length > 5) {
                    const li = document.createElement('li');
                    li.className = 'queue-item more';
                    li.textContent = `+${queue.length - 5} more...`;
                    queueList.appendChild(li);
                }
            }
        } else if (queuePanel) {
            queuePanel.classList.add('hidden');
        }

        // Show/hide panel based on activity
        if (this.elements.workflowPanel) {
            const hasActivity = plan.length > 0 || workflow.status !== 'idle';
            if (hasActivity) {
                this.elements.workflowPanel.classList.remove('hidden');
            } else {
                // Keep visible briefly after completion
                setTimeout(() => {
                    if (!this._hasActiveWorkflow()) {
                        this.elements.workflowPanel.classList.add('hidden');
                    }
                }, 3000);
            }
        }
    }

    /**
     * Check if there's an active workflow
     */
    _hasActiveWorkflow() {
        const status = this.elements.workflowStatus?.textContent?.toLowerCase();
        return status && status !== 'idle' && status !== 'completed';
    }

    /**
     * Handle input change
     */
    _handleInputChange() {
        const hasContent = this.elements.messageInput.value.trim().length > 0;
        this.elements.sendBtn.disabled = !hasContent;
        
        // Auto-resize textarea
        this.elements.messageInput.style.height = 'auto';
        this.elements.messageInput.style.height = Math.min(this.elements.messageInput.scrollHeight, 200) + 'px';
    }

    /**
     * Handle input keydown
     */
    _handleInputKeydown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            this.sendMessage();
        }
    }

    /**
     * Handle model change
     */
    async _handleModelChange(e) {
        const model = e.target.value;
        const selectedOption = e.target.options[e.target.selectedIndex];
        
        // Get provider from the option's data attribute or optgroup
        let provider = selectedOption.dataset.provider;
        if (!provider && selectedOption.parentElement.tagName === 'OPTGROUP') {
            const category = selectedOption.parentElement.label || '';
            if (category.includes('Copilot')) {
                provider = 'github-copilot';
            } else {
                provider = 'github-models';
            }
        }
        
        this.settings.model = model;
        this.settings.provider = provider || 'github-copilot';
        this.elements.currentModelDisplay.textContent = model;
        this._saveSettings();
        
        // Notify backend via REST API (more reliable than WebSocket)
        try {
            const response = await fetch('/api/set_model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model, provider: this.settings.provider })
            });
            const data = await response.json();
            console.log(`[App] Model changed to: ${model} (provider: ${this.settings.provider})`, data);
        } catch (err) {
            console.error('[App] Failed to set model via API:', err);
            // Fallback to WebSocket
            this.ws.send('set_model', { model, provider: this.settings.provider });
        }
    }

    /**
     * Handle navigation click
     */
    _handleNavClick(e, item) {
        e.preventDefault();
        
        // Update active state
        this.elements.navItems.forEach(nav => nav.classList.remove('active'));
        item.classList.add('active');
        
        const view = item.dataset.view;
        // TODO: Implement view switching
        console.log(`[App] Switch to view: ${view}`);
    }

    /**
     * Handle tab switch
     */
    _handleTabSwitch(tab) {
        const tabId = tab.dataset.tab;
        
        // Update tab buttons
        this.elements.panelTabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        
        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.add('hidden');
        });
        document.getElementById(`tab-${tabId}`).classList.remove('hidden');
    }

    /**
     * Handle toggle buttons
     */
    _handleToggle(element, setting) {
        element.classList.toggle('active');
        this.settings[setting] = element.classList.contains('active');
        this._saveSettings();
    }

    /**
     * Toggle sidebar
     */
    _toggleSidebar(show = null) {
        if (show === null) {
            this.elements.sidebar.classList.toggle('open');
        } else {
            this.elements.sidebar.classList.toggle('open', show);
        }
    }

    /**
     * Toggle right panel
     */
    _toggleRightPanel(show = null) {
        if (show === null) {
            this.elements.rightPanel.classList.toggle('collapsed');
        } else {
            this.elements.rightPanel.classList.toggle('collapsed', !show);
        }
    }

    /**
     * Create new session
     */
    _createNewSession() {
        this.currentSessionId = 'session_' + Date.now();
        this.messages = [];
        
        // Clear messages UI
        this.elements.messagesContainer.innerHTML = '';
        this.elements.messagesContainer.appendChild(this.elements.emptyState);
        this.elements.emptyState.classList.remove('hidden');
        
        // Update session display
        this.elements.currentSession.textContent = 'New Session';
        
        // Notify backend
        this.ws.send('new_session', { session_id: this.currentSessionId });
    }

    /**
     * Load settings from localStorage
     */
    _loadSettings() {
        const defaults = {
            model: 'gpt-4o-mini',  // Default to fast, free-tier friendly model
            streaming: true,
            showTools: true,
            sound: false,
            temperature: 0.7,
            maxTokens: 4096,
            systemPrompt: ''
        };

        try {
            const saved = localStorage.getItem('intelclaw_settings');
            return saved ? { ...defaults, ...JSON.parse(saved) } : defaults;
        } catch {
            return defaults;
        }
    }

    /**
     * Save settings to localStorage
     */
    _saveSettings() {
        try {
            localStorage.setItem('intelclaw_settings', JSON.stringify(this.settings));
        } catch (error) {
            console.error('[App] Failed to save settings:', error);
        }
    }

    /**
     * Apply settings to UI
     */
    _applySettings() {
        this.elements.modelSelector.value = this.settings.model;
        this.elements.currentModelDisplay.textContent = this.settings.model;
        this.elements.streamingToggle.classList.toggle('active', this.settings.streaming);
        this.elements.showToolsToggle.classList.toggle('active', this.settings.showTools);
        this.elements.soundToggle.classList.toggle('active', this.settings.sound);
        this.elements.temperatureSlider.value = this.settings.temperature;
        this.elements.temperatureValue.textContent = this.settings.temperature;
        this.elements.maxTokensInput.value = this.settings.maxTokens;
        this.elements.systemPromptInput.value = this.settings.systemPrompt || '';
    }
}

// Initialize application
const app = new IntelCLawApp();
