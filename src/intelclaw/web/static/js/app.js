/**
 * IntelCLaw Control UI - Main Application
 */

class IntelCLawApp {
    constructor() {
        // State
        this.messages = [];
        this.currentSessionId = null;
        this.sessions = [];
        this.sessionBusy = false;
        this.skills = [];
        this.mailboxMessages = [];
        this.inboxUnread = 0;
        this.activePanelTab = 'general';

        // Session picker (header dropdown)
        this.sessionPickerQuery = '';
        this.sessionPickerOffset = 0;
        this.sessionPickerLimit = 50;
        this.sessionPickerSessions = [];
        this.sessionPickerHasMore = false;
        this.sessionPickerLoading = false;
        this._sessionSearchDebounce = null;
        this._sessionPickerLoadToken = 0;

        // Avoid race conditions when switching sessions quickly
        this._activeSessionLoadToken = 0;
        this._sessionsRefreshTimer = null;
        this.settings = this._loadSettings();
        this.isTyping = false;
        this.currentStreamingMessage = null;
        this.workflowExpanded = false;

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
    async init() {
        console.log('[App] Initializing IntelCLaw Control UI...');
        
        this._cacheElements();
        this._setupEventListeners();
        this._setupWebSocket();
        this._loadModels();  // Load models from API
        this._applySettings();
        this._fetchSkills(); // Best-effort: pre-load skills list for panel

        // Load persisted sessions from backend; fall back to a new session.
        try {
            await this._loadSessions();
            if (this.sessions.length > 0) {
                const desired = this._getDesiredSessionId();
                if (desired) {
                    const inList = this.sessions.find((s) => s.session_id === desired);
                    if (inList) {
                        await this._switchToSession(desired);
                    } else {
                        // Best-effort: allow switching to older sessions not in the first page.
                        const found = await this._findSessionByQuery(desired);
                        await this._switchToSession((found && found.session_id) ? found.session_id : this.sessions[0].session_id);
                    }
                } else {
                    await this._switchToSession(this.sessions[0].session_id);
                }
            } else {
                await this._createNewSession();
            }
        } catch (e) {
            console.warn('[App] Failed to load sessions, creating new session:', e);
            await this._createNewSession();
        }
        
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
            workflowBar: document.getElementById('workflowBar'),
            workflowBarNow: document.getElementById('workflowBarNow'),
            workflowBarStatus: document.getElementById('workflowBarStatus'),
            workflowBarFill: document.getElementById('workflowBarFill'),
            workflowToggle: document.getElementById('workflowToggle'),
            workflowClose: document.getElementById('workflowClose'),
            
            // Input
            messageInput: document.getElementById('messageInput'),
            sendBtn: document.getElementById('sendBtn'),
            attachBtn: document.getElementById('attachBtn'),
            
            // Header
            menuToggle: document.getElementById('menuToggle'),
            modelSelector: document.getElementById('modelSelector'),
            settingsToggle: document.getElementById('settingsToggle'),
            skillsToggle: document.getElementById('skillsToggle'),
            inboxToggle: document.getElementById('inboxToggle'),
            inboxBadge: document.getElementById('inboxBadge'),
            sessionPicker: document.getElementById('sessionPicker'),
            sessionSelector: document.getElementById('sessionSelector'),
            sessionDropdown: document.getElementById('sessionDropdown'),
            sessionSearchInput: document.getElementById('sessionSearchInput'),
            sessionDropdownList: document.getElementById('sessionDropdownList'),
            loadMoreSessionsBtn: document.getElementById('loadMoreSessionsBtn'),
            newSessionBtn: document.getElementById('newSessionBtn'),
            currentSession: document.getElementById('currentSession'),
            sessionList: document.getElementById('sessionList'),
            
            // Status
            connectionDot: document.getElementById('connectionDot'),
            connectionStatus: document.getElementById('connectionStatus'),
            currentModelDisplay: document.getElementById('currentModelDisplay'),
            taskCount: document.getElementById('taskCount'),
            
            // Settings Panel
            panelClose: document.getElementById('panelClose'),
            panelTitle: document.getElementById('panelTitle'),
            streamingToggle: document.getElementById('streamingToggle'),
            showToolsToggle: document.getElementById('showToolsToggle'),
            soundToggle: document.getElementById('soundToggle'),
            temperatureSlider: document.getElementById('temperatureSlider'),
            temperatureValue: document.getElementById('temperatureValue'),
            maxTokensInput: document.getElementById('maxTokensInput'),
            systemPromptInput: document.getElementById('systemPromptInput'),

            // Skills + Inbox tabs
            skillsList: document.getElementById('skillsList'),
            refreshSkillsBtn: document.getElementById('refreshSkillsBtn'),
            addSkillBtn: document.getElementById('addSkillBtn'),
            inboxList: document.getElementById('inboxList'),
            refreshInboxBtn: document.getElementById('refreshInboxBtn'),
            clearInboxBtn: document.getElementById('clearInboxBtn'),

            // Add skill modal
            addSkillModal: document.getElementById('addSkillModal'),
            addSkillBackdrop: document.getElementById('addSkillBackdrop'),
            addSkillClose: document.getElementById('addSkillClose'),
            skillManifestInput: document.getElementById('skillManifestInput'),
            skillAgentInput: document.getElementById('skillAgentInput'),
            skillEnableOnInstall: document.getElementById('skillEnableOnInstall'),
            skillInstallBtn: document.getElementById('skillInstallBtn'),
            addSkillError: document.getElementById('addSkillError'),

            // Toasts
            toastContainer: document.getElementById('toastContainer'),
            
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
        this.elements.settingsToggle.addEventListener('click', () => this._openPanelTab('general', { toggle: true }));
        this.elements.panelClose.addEventListener('click', () => this._toggleRightPanel(false));

        // Skills / Inbox shortcuts
        if (this.elements.skillsToggle) {
            this.elements.skillsToggle.addEventListener('click', () => this._openPanelTab('skills', { toggle: true }));
        }
        if (this.elements.inboxToggle) {
            this.elements.inboxToggle.addEventListener('click', () => this._openPanelTab('inbox', { toggle: true }));
        }
        
        // Model selector
        this.elements.modelSelector.addEventListener('change', (e) => this._handleModelChange(e));

        // Session picker (header dropdown)
        if (this.elements.sessionSelector) {
            this.elements.sessionSelector.addEventListener('click', async (e) => {
                e.preventDefault();
                await this._toggleSessionDropdown();
            });
        }
        if (this.elements.newSessionBtn) {
            this.elements.newSessionBtn.addEventListener('click', async (e) => {
                e.preventDefault();
                await this._createNewSession();
                await this._toggleSessionDropdown(false);
            });
        }
        if (this.elements.loadMoreSessionsBtn) {
            this.elements.loadMoreSessionsBtn.addEventListener('click', async (e) => {
                e.preventDefault();
                await this._loadSessionPickerPage({ append: true });
            });
        }
        if (this.elements.sessionSearchInput) {
            this.elements.sessionSearchInput.addEventListener('input', () => {
                this._debounceSessionPickerSearch();
            });
        }

        // Close session dropdown when clicking outside (or on ESC)
        document.addEventListener('click', (e) => {
            const picker = this.elements.sessionPicker;
            const dropdown = this.elements.sessionDropdown;
            if (!picker || !dropdown) return;
            if (dropdown.classList.contains('hidden')) return;
            if (!picker.contains(e.target)) {
                this._toggleSessionDropdown(false);
            }
        });
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this._toggleSessionDropdown(false);
            }
        });
        
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

        // Skills tab actions
        if (this.elements.refreshSkillsBtn) {
            this.elements.refreshSkillsBtn.addEventListener('click', () => this._fetchSkills());
        }
        if (this.elements.addSkillBtn) {
            this.elements.addSkillBtn.addEventListener('click', () => this._showAddSkillModal(true));
        }

        // Inbox tab actions
        if (this.elements.refreshInboxBtn) {
            this.elements.refreshInboxBtn.addEventListener('click', () => this._fetchMailbox());
        }
        if (this.elements.clearInboxBtn) {
            this.elements.clearInboxBtn.addEventListener('click', () => {
                this.mailboxMessages = [];
                this._renderInbox();
                this._setInboxUnread(0);
            });
        }

        // Add skill modal actions
        if (this.elements.addSkillBackdrop) {
            this.elements.addSkillBackdrop.addEventListener('click', () => this._showAddSkillModal(false));
        }
        if (this.elements.addSkillClose) {
            this.elements.addSkillClose.addEventListener('click', () => this._showAddSkillModal(false));
        }
        if (this.elements.skillInstallBtn) {
            this.elements.skillInstallBtn.addEventListener('click', () => this._installSkillFromModal());
        }

        // Workflow panel toggle
        if (this.elements.workflowToggle) {
            this.elements.workflowToggle.addEventListener('click', () => this._toggleWorkflowPanel());
        }
        if (this.elements.workflowClose) {
            this.elements.workflowClose.addEventListener('click', () => this._toggleWorkflowPanel(false));
        }
        
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
        document.querySelector('[data-action="new-session"]')?.addEventListener('click', async (e) => {
            e.preventDefault();
            await this._createNewSession();
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
        this.ws.on('skills', () => this._fetchSkills());
        this.ws.on('mailbox', (data) => this._handleMailboxMessage(data));
        this.ws.on('notification', (data) => this._handleNotification(data));

        // Connect
        this.ws.connect();
    }

    _openPanelTab(tabId, { toggle = false } = {}) {
        const t = String(tabId || '').trim();
        if (!t) return;

        const isCollapsed = this.elements.rightPanel.classList.contains('collapsed');
        if (toggle && !isCollapsed && this.activePanelTab === t) {
            this._toggleRightPanel(false);
            return;
        }

        this._toggleRightPanel(true);
        const tabBtn = document.querySelector(`.panel-tab[data-tab="${t}"]`);
        if (tabBtn) {
            this._handleTabSwitch(tabBtn);
        }
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
        if (this.sessionBusy) return;
        if (!this.currentSessionId) {
            console.warn('[App] Cannot send message: no active session');
            return;
        }
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
    _addMessage(message, options = {}) {
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
        if (!options || options.scroll !== false) {
            this._scrollToBottom();
        }

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
        if (data && data.session_id && this.currentSessionId && data.session_id !== this.currentSessionId) {
            return;
        }
        this._hideTypingIndicator();
        
        this._addMessage({
            role: 'assistant',
            content: data.content,
            model: data.model,
            timestamp: new Date().toISOString()
        });

        this._scheduleSessionsRefresh();
    }

    /**
     * Handle streaming chat response
     */
    _handleChatStream(data) {
        if (data && data.session_id && this.currentSessionId && data.session_id !== this.currentSessionId) {
            return;
        }
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
        if (data && data.session_id && this.currentSessionId && data.session_id !== this.currentSessionId) {
            return;
        }
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

        this._scheduleSessionsRefresh();
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
        if (data && data.session_id && this.currentSessionId && data.session_id !== this.currentSessionId) {
            return;
        }
        this._hideTypingIndicator();
        
        this._addMessage({
            role: 'system',
            content: `**Error:** ${data.message || 'An unexpected error occurred'}`,
            timestamp: new Date().toISOString()
        });

        this._scheduleSessionsRefresh();
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
        const truncate = (text, max = 140) => {
            if (!text) return '-';
            if (text.length <= max) return text;
            return text.slice(0, max).trim() + '…';
        };
        const statusRaw = workflow.status || 'idle';
        const status = statusRaw.toLowerCase();
        const statusUpper = statusRaw.toUpperCase();
        const barPhaseLabel = status === 'thinking'
            ? 'PLAN'
            : status === 'executing'
                ? 'ACT'
                : status === 'completed' || status === 'idle'
                    ? 'REVIEW'
                    : statusUpper;
        const statusFallback = status === 'thinking'
            ? 'Planning'
            : status === 'executing'
                ? 'Executing'
                : status === 'waiting'
                    ? 'Waiting'
                    : '-';

        // Compact workflow bar
        if (this.elements.workflowBar) {
            if (plan.length > 0 || status !== 'idle') {
                this.elements.workflowBar.classList.remove('hidden');
            } else {
                this.elements.workflowBar.classList.add('hidden');
            }
        }
        if (this.elements.workflowBarNow) {
            const barNow = workflow.current_step_title || plan[currentIndex] || statusFallback;
            this.elements.workflowBarNow.textContent = truncate(barNow, 120);
            this.elements.workflowBarNow.title = barNow;
        }
        if (this.elements.workflowBarStatus) {
            this.elements.workflowBarStatus.textContent = barPhaseLabel;
            this.elements.workflowBarStatus.className = 'workflow-bar-status status-' + status;
            this.elements.workflowBarStatus.title = statusUpper;
        }
        if (this.elements.workflowBarFill) {
            this.elements.workflowBarFill.style.width = `${progress}%`;
        }

        // Status with color coding
        if (this.elements.workflowStatus) {
            this.elements.workflowStatus.textContent = statusUpper;
            this.elements.workflowStatus.className = 'workflow-status status-' + status;
        }

        // Phase chips (Plan → Act → Review)
        if (this.elements.workflowPhases) {
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
        const nowStep = workflow.current_step_title || plan[currentIndex] || statusFallback;
        const nextStep = workflow.next_step || plan[currentIndex + 1] || '-';
        
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
            const hasActivity = plan.length > 0 || status !== 'idle';
            if (hasActivity) {
                if (this.workflowExpanded) {
                    this.elements.workflowPanel.classList.remove('hidden');
                }
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
     * Toggle workflow panel visibility
     */
    _toggleWorkflowPanel(show = null) {
        if (!this.elements.workflowPanel) return;
        if (show === null) {
            this.workflowExpanded = !this.workflowExpanded;
        } else {
            this.workflowExpanded = show;
        }
        if (this.workflowExpanded) {
            this.elements.workflowPanel.classList.remove('hidden');
            if (this.elements.workflowToggle) {
                this.elements.workflowToggle.textContent = 'Hide steps';
                this.elements.workflowToggle.setAttribute('aria-expanded', 'true');
            }
        } else {
            this.elements.workflowPanel.classList.add('hidden');
            if (this.elements.workflowToggle) {
                this.elements.workflowToggle.textContent = 'View steps';
                this.elements.workflowToggle.setAttribute('aria-expanded', 'false');
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
        this.activePanelTab = tabId;
        
        // Update tab buttons
        this.elements.panelTabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        
        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.add('hidden');
        });
        document.getElementById(`tab-${tabId}`).classList.remove('hidden');

        // Title + on-open refresh
        const titleMap = {
            general: 'Settings',
            model: 'Settings',
            tools: 'Settings',
            skills: 'Skills',
            inbox: 'Inbox'
        };
        if (this.elements.panelTitle) {
            this.elements.panelTitle.textContent = titleMap[tabId] || 'Panel';
        }
        if (tabId === 'skills') {
            this._fetchSkills();
        }
        if (tabId === 'inbox') {
            this._fetchMailbox();
            this._setInboxUnread(0);
        }
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

    async _fetchSkills() {
        try {
            const resp = await fetch('/api/skills');
            const data = await resp.json();
            this.skills = Array.isArray(data.skills) ? data.skills : [];
            this._renderSkills();
        } catch (e) {
            console.warn('[App] Failed to load skills:', e);
        }
    }

    _renderSkills() {
        const container = this.elements.skillsList;
        if (!container) return;

        const skills = Array.isArray(this.skills) ? this.skills : [];
        container.innerHTML = '';

        if (skills.length === 0) {
            const empty = document.createElement('div');
            empty.className = 'panel-hint';
            empty.textContent = 'No skills found.';
            container.appendChild(empty);
            return;
        }

        skills.forEach((s) => {
            const sid = String(s.id || '').trim();
            if (!sid) return;

            const enabled = !!s.enabled;
            const icon = s.icon || '🧩';
            const name = s.name || sid;
            const version = s.version || '';
            const description = s.description || '';
            const toolCount = (typeof s.tool_count === 'number') ? s.tool_count : 0;
            const sourceKind = s.source_kind || '';

            const health = s.health || { healthy: true, last_error: null };
            const healthy = !!health.healthy;
            const healthClass = healthy ? 'healthy' : 'unhealthy';
            const healthText = enabled ? (healthy ? 'Healthy' : 'Unhealthy') : 'Disabled';

            const row = document.createElement('div');
            row.className = 'skill-row';

            const meta = document.createElement('div');
            meta.className = 'skill-meta';
            meta.innerHTML = `
                <div class="skill-title">
                    <span class="skill-icon">${icon}</span>
                    <span class="skill-name">${this._escapeHtml(name)}</span>
                    ${version ? `<span class="skill-version">v${this._escapeHtml(version)}</span>` : ''}
                </div>
                ${description ? `<div class="skill-desc">${this._escapeHtml(description)}</div>` : ''}
                <div class="skill-sub">
                    ${sourceKind ? `<span class="skill-source">${this._escapeHtml(sourceKind)}</span>` : ''}
                    <span class="skill-health ${healthClass}">${healthText}</span>
                    <span class="skill-tools">${toolCount} tool${toolCount === 1 ? '' : 's'}</span>
                </div>
                ${enabled && !healthy && health.last_error ? `<div class="skill-desc">${this._escapeHtml(String(health.last_error).slice(0, 240))}</div>` : ''}
            `;

            const toggle = document.createElement('button');
            toggle.className = 'toggle' + (enabled ? ' active' : '');
            toggle.type = 'button';
            toggle.title = enabled ? 'Disable' : 'Enable';
            toggle.addEventListener('click', async () => {
                toggle.disabled = true;
                try {
                    await this._setSkillEnabled(sid, !enabled);
                } finally {
                    toggle.disabled = false;
                }
            });

            row.appendChild(meta);
            row.appendChild(toggle);
            container.appendChild(row);
        });
    }

    async _setSkillEnabled(skillId, enabled) {
        const sid = String(skillId || '').trim();
        if (!sid) return;
        try {
            const url = enabled
                ? `/api/skills/${encodeURIComponent(sid)}/enable`
                : `/api/skills/${encodeURIComponent(sid)}/disable`;
            const resp = await fetch(url, { method: 'POST' });
            const data = await resp.json().catch(() => ({}));
            if (!resp.ok || !data.success) {
                const title = enabled ? 'Enable failed' : 'Disable failed';
                const body = data.error || `HTTP ${resp.status}`;
                this._toast('error', title, body);
            }
        } catch (e) {
            this._toast('error', 'Skill update failed', String(e));
        } finally {
            await this._fetchSkills();
        }
    }

    _showAddSkillModal(show) {
        const modal = this.elements.addSkillModal;
        if (!modal) return;
        const shouldShow = !!show;
        modal.classList.toggle('hidden', !shouldShow);
        modal.setAttribute('aria-hidden', shouldShow ? 'false' : 'true');
        if (this.elements.addSkillError) {
            this.elements.addSkillError.classList.add('hidden');
            this.elements.addSkillError.textContent = '';
        }
        if (shouldShow && this.elements.skillManifestInput) {
            this.elements.skillManifestInput.focus();
        }
    }

    async _installSkillFromModal() {
        const yamlText = this.elements.skillManifestInput ? this.elements.skillManifestInput.value : '';
        const agentText = this.elements.skillAgentInput ? this.elements.skillAgentInput.value : '';
        const enable = this.elements.skillEnableOnInstall ? !!this.elements.skillEnableOnInstall.checked : false;

        try {
            const resp = await fetch('/api/skills/install', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    manifest_yaml: yamlText || '',
                    agent_md: agentText || '',
                    enable
                })
            });
            const data = await resp.json().catch(() => ({}));
            if (!resp.ok || !data.success) {
                const err = data.error || `HTTP ${resp.status}`;
                if (this.elements.addSkillError) {
                    this.elements.addSkillError.classList.remove('hidden');
                    this.elements.addSkillError.textContent = err;
                }
                return;
            }

            this._toast('success', 'Skill installed', `Installed: ${data.skill_id || 'unknown'}`);
            this._showAddSkillModal(false);
            await this._fetchSkills();
        } catch (e) {
            if (this.elements.addSkillError) {
                this.elements.addSkillError.classList.remove('hidden');
                this.elements.addSkillError.textContent = String(e);
            }
        }
    }

    async _fetchMailbox() {
        const sid = String(this.currentSessionId || '').trim();
        if (!sid) return;
        try {
            const resp = await fetch(`/api/mailbox?session_id=${encodeURIComponent(sid)}&limit=200`);
            const data = await resp.json();
            this.mailboxMessages = Array.isArray(data.messages) ? data.messages : [];
            this._renderInbox();
        } catch (e) {
            console.warn('[App] Failed to load mailbox:', e);
        }
    }

    _renderInbox() {
        const container = this.elements.inboxList;
        if (!container) return;
        const msgs = Array.isArray(this.mailboxMessages) ? this.mailboxMessages : [];
        container.innerHTML = '';

        if (msgs.length === 0) {
            const empty = document.createElement('div');
            empty.className = 'panel-hint';
            empty.textContent = 'Inbox is empty.';
            container.appendChild(empty);
            return;
        }

        msgs.forEach((m) => {
            const item = document.createElement('div');
            item.className = 'inbox-item';
            const kind = String(m.kind || 'info');
            const fromAgent = String(m.from_agent || 'agent');
            const ts = m.ts ? new Date(m.ts) : null;
            const time = ts ? ts.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';
            const title = String(m.title || '');
            const body = String(m.body || '');

            item.innerHTML = `
                <div class="inbox-top">
                    <span class="inbox-agent">${this._escapeHtml(fromAgent)}</span>
                    <span class="inbox-kind ${this._escapeHtml(kind)}">${this._escapeHtml(kind)}</span>
                    <span class="inbox-time">${this._escapeHtml(time)}</span>
                </div>
                ${title ? `<div class="inbox-title">${this._escapeHtml(title)}</div>` : ''}
                ${body ? `<div class="inbox-body">${this._escapeHtml(body).slice(0, 4000)}</div>` : ''}
            `;
            container.appendChild(item);
        });
    }

    _handleMailboxMessage(data) {
        const msg = data && data.message ? data.message : null;
        if (!msg) return;
        const currentSid = String(this.currentSessionId || '').trim();
        const msgSid = String(msg.session_id || '').trim();
        if (currentSid && msgSid && currentSid !== msgSid) return;

        this.mailboxMessages = Array.isArray(this.mailboxMessages) ? this.mailboxMessages : [];
        this.mailboxMessages.push(msg);
        // Keep last 500 (matches server default)
        if (this.mailboxMessages.length > 500) {
            this.mailboxMessages = this.mailboxMessages.slice(-500);
        }

        if (this.activePanelTab === 'inbox' && !this.elements.rightPanel.classList.contains('collapsed')) {
            this._renderInbox();
            return;
        }

        this._setInboxUnread(this.inboxUnread + 1);
    }

    _handleNotification(data) {
        const level = (data && data.level) ? String(data.level) : 'success';
        const title = (data && data.title) ? String(data.title) : 'Notification';
        const message = (data && (data.message || data.body)) ? String(data.message || data.body) : '';
        this._toast(level, title, message);
        if (this.settings.sound) {
            this._playNotificationSound();
        }
    }

    _setInboxUnread(count) {
        const n = Math.max(0, parseInt(count || 0, 10) || 0);
        this.inboxUnread = n;
        if (!this.elements.inboxBadge) return;
        this.elements.inboxBadge.textContent = String(n);
        this.elements.inboxBadge.classList.toggle('hidden', n <= 0);
    }

    _toast(level, title, body) {
        const container = this.elements.toastContainer;
        if (!container) return;

        const toast = document.createElement('div');
        toast.className = `toast ${this._escapeHtml(String(level || 'success'))}`;
        toast.innerHTML = `
            <div class="toast-title">${this._escapeHtml(String(title || ''))}</div>
            ${body ? `<div class="toast-body">${this._escapeHtml(String(body)).slice(0, 2000)}</div>` : ''}
        `;
        container.appendChild(toast);

        setTimeout(() => {
            try { toast.remove(); } catch { /* ignore */ }
        }, 6500);
    }

    _playNotificationSound() {
        try {
            const Ctx = window.AudioContext || window.webkitAudioContext;
            if (!Ctx) return;
            const ctx = new Ctx();
            const o = ctx.createOscillator();
            const g = ctx.createGain();
            o.type = 'sine';
            o.frequency.value = 880;
            g.gain.value = 0.02;
            o.connect(g);
            g.connect(ctx.destination);
            o.start();
            setTimeout(() => {
                try { o.stop(); } catch { /* ignore */ }
                try { ctx.close(); } catch { /* ignore */ }
            }, 120);
        } catch {
            // ignore
        }
    }

    _escapeHtml(text) {
        const s = String(text || '');
        return s
            .replaceAll('&', '&amp;')
            .replaceAll('<', '&lt;')
            .replaceAll('>', '&gt;')
            .replaceAll('"', '&quot;')
            .replaceAll("'", '&#039;');
    }

    _setSessionBusy(busy) {
        this.sessionBusy = !!busy;
        if (this.elements.sendBtn) this.elements.sendBtn.disabled = this.sessionBusy;
        if (this.elements.messageInput) this.elements.messageInput.disabled = this.sessionBusy;
    }

    _getDesiredSessionId() {
        try {
            const url = new URL(window.location.href);
            const sid = url.searchParams.get('session_id');
            if (sid && sid.trim()) return sid.trim();
        } catch {
            // ignore
        }

        try {
            const sid = localStorage.getItem('intelclaw_last_session_id');
            return sid && sid.trim() ? sid.trim() : null;
        } catch {
            return null;
        }
    }

    _persistActiveSessionId(sessionId) {
        try {
            if (sessionId) {
                localStorage.setItem('intelclaw_last_session_id', String(sessionId));
            }
        } catch {
            // ignore
        }
    }

    _setUrlSessionId(sessionId) {
        try {
            const url = new URL(window.location.href);
            if (sessionId && String(sessionId).trim()) {
                url.searchParams.set('session_id', String(sessionId).trim());
            } else {
                url.searchParams.delete('session_id');
            }
            window.history.replaceState({}, '', url.toString());
        } catch {
            // ignore
        }
    }

    _isSessionDropdownOpen() {
        return !!this.elements.sessionDropdown && !this.elements.sessionDropdown.classList.contains('hidden');
    }

    async _toggleSessionDropdown(show = null) {
        const dropdown = this.elements.sessionDropdown;
        const button = this.elements.sessionSelector;
        if (!dropdown || !button) return;

        const isOpen = !dropdown.classList.contains('hidden');
        const open = show === null ? !isOpen : !!show;

        if (!open) {
            dropdown.classList.add('hidden');
            dropdown.setAttribute('aria-hidden', 'true');
            button.setAttribute('aria-expanded', 'false');
            return;
        }

        dropdown.classList.remove('hidden');
        dropdown.setAttribute('aria-hidden', 'false');
        button.setAttribute('aria-expanded', 'true');

        // Populate list on open.
        await this._loadSessionPickerPage({ reset: true });

        // Focus search input.
        if (this.elements.sessionSearchInput) {
            setTimeout(() => this.elements.sessionSearchInput.focus(), 0);
        }
    }

    _debounceSessionPickerSearch() {
        if (this._sessionSearchDebounce) {
            clearTimeout(this._sessionSearchDebounce);
        }
        this._sessionSearchDebounce = setTimeout(() => {
            this._loadSessionPickerPage({ reset: true });
        }, 200);
    }

    async _loadSessionPickerPage({ reset = false, append = false } = {}) {
        const listEl = this.elements.sessionDropdownList;
        if (!listEl) return;
        if (this.sessionPickerLoading) return;

        const token = ++this._sessionPickerLoadToken;

        this.sessionPickerLoading = true;
        try {
            const q = (this.elements.sessionSearchInput?.value || '').trim();
            const queryChanged = q !== this.sessionPickerQuery;
            if (reset || queryChanged) {
                this.sessionPickerQuery = q;
                this.sessionPickerOffset = 0;
                this.sessionPickerSessions = [];
                this.sessionPickerHasMore = false;
            }

            const limit = this.sessionPickerLimit;
            const offset = append ? this.sessionPickerOffset : 0;
            const url = q
                ? `/api/sessions?limit=${encodeURIComponent(limit)}&offset=${encodeURIComponent(offset)}&q=${encodeURIComponent(q)}`
                : `/api/sessions?limit=${encodeURIComponent(limit)}&offset=${encodeURIComponent(offset)}`;

            const resp = await fetch(url);
            const data = await resp.json();
            const sessions = Array.isArray(data.sessions) ? data.sessions : [];

            if (token !== this._sessionPickerLoadToken) return;

            if (append && this.sessionPickerSessions.length > 0) {
                this.sessionPickerSessions = this.sessionPickerSessions.concat(sessions);
            } else {
                this.sessionPickerSessions = sessions;
            }
            this.sessionPickerOffset = this.sessionPickerSessions.length;
            this.sessionPickerHasMore = sessions.length === limit;

            this._renderSessionPickerList();
        } catch (e) {
            console.warn('[App] Failed to load session picker sessions:', e);
        } finally {
            if (token === this._sessionPickerLoadToken) {
                this.sessionPickerLoading = false;
                if (this.elements.loadMoreSessionsBtn) {
                    this.elements.loadMoreSessionsBtn.disabled = !this.sessionPickerHasMore;
                }
            }
        }
    }

    _renderSessionPickerList() {
        const listEl = this.elements.sessionDropdownList;
        if (!listEl) return;

        listEl.innerHTML = '';

        if (!this.sessionPickerSessions || this.sessionPickerSessions.length === 0) {
            const empty = document.createElement('div');
            empty.className = 'session-dropdown-empty';
            empty.textContent = 'No sessions found';
            listEl.appendChild(empty);
        } else {
            this.sessionPickerSessions.forEach((s) => {
                const sid = s.session_id;
                const title = (s.title && s.title.trim()) ? s.title.trim() : sid;
                const count = s.message_count ?? 0;

                const item = document.createElement('button');
                item.type = 'button';
                item.className = 'session-dropdown-item' + (sid === this.currentSessionId ? ' active' : '');
                item.dataset.sessionId = sid;

                const titleEl = document.createElement('div');
                titleEl.className = 'session-dropdown-title';
                titleEl.textContent = title;

                const metaEl = document.createElement('div');
                metaEl.className = 'session-dropdown-meta';
                metaEl.textContent = `${count} msg${count === 1 ? '' : 's'}`;

                item.appendChild(titleEl);
                item.appendChild(metaEl);

                item.addEventListener('click', async (e) => {
                    e.preventDefault();
                    await this._switchToSession(sid);
                    await this._toggleSessionDropdown(false);
                });

                listEl.appendChild(item);
            });
        }

        if (this.elements.loadMoreSessionsBtn) {
            const show = !!this.sessionPickerHasMore;
            this.elements.loadMoreSessionsBtn.style.display = show ? 'block' : 'none';
            this.elements.loadMoreSessionsBtn.disabled = !show || this.sessionPickerLoading;
        }
    }

    async _findSessionByQuery(query) {
        const q = String(query || '').trim();
        if (!q) return null;
        try {
            const resp = await fetch(`/api/sessions?limit=1&offset=0&q=${encodeURIComponent(q)}`);
            const data = await resp.json();
            const sessions = Array.isArray(data.sessions) ? data.sessions : [];
            return sessions.length > 0 ? sessions[0] : null;
        } catch {
            return null;
        }
    }

    _scheduleSessionsRefresh() {
        if (this._sessionsRefreshTimer) return;
        this._sessionsRefreshTimer = setTimeout(async () => {
            this._sessionsRefreshTimer = null;
            try {
                await this._loadSessions();
            } catch {
                // ignore
            }
        }, 400);
    }

    /**
     * Create new session
     */
    async _createNewSession() {
        if (this.sessionBusy) return;
        const token = ++this._activeSessionLoadToken;
        this._setSessionBusy(true);

        try {
            // Create on backend so it is persisted and shows up in the session list.
            try {
                const resp = await fetch('/api/sessions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({})
                });
                const data = await resp.json();
                if (token !== this._activeSessionLoadToken) return;
                this.currentSessionId = data.session_id || ('session_' + Date.now());
            } catch (e) {
                console.warn('[App] Failed to create session via API, using fallback id:', e);
                if (token !== this._activeSessionLoadToken) return;
                this.currentSessionId = 'session_' + Date.now();
            }

            if (token !== this._activeSessionLoadToken) return;

            this._persistActiveSessionId(this.currentSessionId);
            this._setUrlSessionId(this.currentSessionId);

            this._hideTypingIndicator();
            this.currentStreamingMessage = null;

            this.messages = [];

            // Clear messages UI
            this.elements.messagesContainer.innerHTML = '';
            this.elements.messagesContainer.appendChild(this.elements.emptyState);
            this.elements.emptyState.classList.remove('hidden');

            // Update session display
            this.elements.currentSession.textContent = 'New Session';

            // Notify backend (WebSocket current session)
            this.ws.send('new_session', { session_id: this.currentSessionId });

            // Reset inbox state for the new session.
            this.mailboxMessages = [];
            this._renderInbox();
            this._setInboxUnread(0);

            await this._loadSessions();
            this._renderSessionList();

            // Keep session picker list fresh if it is open.
            if (this._isSessionDropdownOpen()) {
                await this._loadSessionPickerPage({ reset: true });
            }
        } finally {
            if (token === this._activeSessionLoadToken) {
                this._setSessionBusy(false);
            }
        }
    }

    /**
     * Load sessions from backend and render in sidebar.
     */
    async _loadSessions() {
        const resp = await fetch('/api/sessions?limit=50&offset=0');
        const data = await resp.json();
        this.sessions = Array.isArray(data.sessions) ? data.sessions : [];
        this._renderSessionList();
    }

    _renderSessionList() {
        const container = this.elements.sessionList;
        if (!container) return;
        container.innerHTML = '';

        this.sessions.forEach((s) => {
            const sid = s.session_id;
            const title = (s.title && s.title.trim()) ? s.title.trim() : sid;
            const count = s.message_count ?? 0;

            const item = document.createElement('a');
            item.href = '#';
            item.className = 'nav-item' + (sid === this.currentSessionId ? ' active' : '');
            item.dataset.sessionId = sid;
            item.innerHTML = `
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                </svg>
                <span class="nav-item-text">${title}</span>
                <span class="nav-badge">${count}</span>
            `;
            item.addEventListener('click', async (e) => {
                e.preventDefault();
                await this._switchToSession(sid);
            });
            container.appendChild(item);
        });
    }

    /**
     * Switch to an existing session and load its messages.
     */
    async _switchToSession(sessionId) {
        const sid = String(sessionId || '').trim();
        if (!sid) return;

        const token = ++this._activeSessionLoadToken;
        this._setSessionBusy(true);

        try {
            this.currentSessionId = sid;
            this._persistActiveSessionId(this.currentSessionId);
            this._setUrlSessionId(this.currentSessionId);

            this._hideTypingIndicator();
            this.currentStreamingMessage = null;

            this.ws.send('new_session', { session_id: this.currentSessionId });

            // Load messages from backend
            const resp = await fetch(`/api/sessions/${encodeURIComponent(sid)}`);
            const data = await resp.json();
            if (token !== this._activeSessionLoadToken) return;

            const msgs = Array.isArray(data.messages) ? data.messages : [];

            // Render
            this.messages = [];
            this.elements.messagesContainer.innerHTML = '';
            this.elements.messagesContainer.appendChild(this.elements.emptyState);

            if (msgs.length === 0) {
                this.elements.emptyState.classList.remove('hidden');
            } else {
                this.elements.emptyState.classList.add('hidden');
                msgs.forEach((m) => {
                    this._addMessage(
                        {
                            role: m.role,
                            content: m.content,
                            timestamp: m.created_at,
                            id: m.id
                        },
                        { scroll: false }
                    );
                });
                this._scrollToBottom();
            }

            // Update session title in header (API preferred; fallback to list).
            const titleFromApi = (data && typeof data.title === 'string') ? data.title.trim() : '';
            const session = this.sessions.find((s) => s.session_id === sid);
            const title = titleFromApi || ((session && session.title) ? session.title : sid);
            this.elements.currentSession.textContent = title || 'Session';

            this._renderSessionList();
            if (this._isSessionDropdownOpen()) {
                this._renderSessionPickerList();
            }

            // Refresh mailbox for this session.
            await this._fetchMailbox();
            this._setInboxUnread(0);
        } catch (e) {
            console.warn('[App] Failed to switch session:', e);
        } finally {
            if (token === this._activeSessionLoadToken) {
                this._setSessionBusy(false);
            }
        }
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
