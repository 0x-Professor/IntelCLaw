/**
 * IntelCLaw WebSocket Manager
 * Handles real-time communication with the backend agent
 */

class WebSocketManager {
    constructor(url = null) {
        this.url = url || `ws://${window.location.host}/ws`;
        this.socket = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.messageHandlers = new Map();
        this.connectionHandlers = {
            onOpen: [],
            onClose: [],
            onError: []
        };
        this.messageQueue = [];
        this.isConnected = false;
    }

    /**
     * Connect to the WebSocket server
     */
    connect() {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            console.log('[WS] Already connected');
            return;
        }

        console.log(`[WS] Connecting to ${this.url}...`);
        
        try {
            this.socket = new WebSocket(this.url);
            this._setupEventHandlers();
        } catch (error) {
            console.error('[WS] Connection error:', error);
            this._handleError(error);
        }
    }

    /**
     * Disconnect from the WebSocket server
     */
    disconnect() {
        if (this.socket) {
            this.socket.close();
            this.socket = null;
            this.isConnected = false;
        }
    }

    /**
     * Setup WebSocket event handlers
     */
    _setupEventHandlers() {
        this.socket.onopen = (event) => {
            console.log('[WS] Connected');
            this.isConnected = true;
            this.reconnectAttempts = 0;
            
            // Send queued messages
            while (this.messageQueue.length > 0) {
                const msg = this.messageQueue.shift();
                this._sendRaw(msg);
            }

            // Notify handlers
            this.connectionHandlers.onOpen.forEach(handler => handler(event));
        };

        this.socket.onclose = (event) => {
            console.log(`[WS] Disconnected (code: ${event.code})`);
            this.isConnected = false;

            // Notify handlers
            this.connectionHandlers.onClose.forEach(handler => handler(event));

            // Attempt reconnection
            if (this.reconnectAttempts < this.maxReconnectAttempts) {
                this.reconnectAttempts++;
                const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
                console.log(`[WS] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
                setTimeout(() => this.connect(), delay);
            } else {
                console.error('[WS] Max reconnection attempts reached');
            }
        };

        this.socket.onerror = (error) => {
            console.error('[WS] Error:', error);
            this.connectionHandlers.onError.forEach(handler => handler(error));
        };

        this.socket.onmessage = (event) => {
            this._handleMessage(event.data);
        };
    }

    /**
     * Handle incoming message
     */
    _handleMessage(data) {
        try {
            const message = JSON.parse(data);
            const { type, ...payload } = message;

            console.log(`[WS] Received: ${type}`, payload);

            // Call registered handlers for this message type
            if (this.messageHandlers.has(type)) {
                this.messageHandlers.get(type).forEach(handler => handler(payload));
            }

            // Call wildcard handlers
            if (this.messageHandlers.has('*')) {
                this.messageHandlers.get('*').forEach(handler => handler(message));
            }
        } catch (error) {
            console.error('[WS] Failed to parse message:', error, data);
        }
    }

    /**
     * Handle connection errors
     */
    _handleError(error) {
        this.connectionHandlers.onError.forEach(handler => handler(error));
    }

    /**
     * Send a message to the server
     */
    send(type, payload = {}) {
        const message = { type, ...payload };
        
        if (this.isConnected) {
            this._sendRaw(message);
        } else {
            console.log('[WS] Queuing message (not connected):', type);
            this.messageQueue.push(message);
        }
    }

    /**
     * Send raw message
     */
    _sendRaw(message) {
        try {
            this.socket.send(JSON.stringify(message));
            console.log('[WS] Sent:', message.type);
        } catch (error) {
            console.error('[WS] Send error:', error);
        }
    }

    /**
     * Register a handler for a specific message type
     */
    on(type, handler) {
        if (!this.messageHandlers.has(type)) {
            this.messageHandlers.set(type, []);
        }
        this.messageHandlers.get(type).push(handler);
        return () => this.off(type, handler);
    }

    /**
     * Remove a handler
     */
    off(type, handler) {
        if (this.messageHandlers.has(type)) {
            const handlers = this.messageHandlers.get(type);
            const index = handlers.indexOf(handler);
            if (index > -1) {
                handlers.splice(index, 1);
            }
        }
    }

    /**
     * Register connection event handlers
     */
    onConnect(handler) {
        this.connectionHandlers.onOpen.push(handler);
        return () => {
            const index = this.connectionHandlers.onOpen.indexOf(handler);
            if (index > -1) this.connectionHandlers.onOpen.splice(index, 1);
        };
    }

    onDisconnect(handler) {
        this.connectionHandlers.onClose.push(handler);
        return () => {
            const index = this.connectionHandlers.onClose.indexOf(handler);
            if (index > -1) this.connectionHandlers.onClose.splice(index, 1);
        };
    }

    onError(handler) {
        this.connectionHandlers.onError.push(handler);
        return () => {
            const index = this.connectionHandlers.onError.indexOf(handler);
            if (index > -1) this.connectionHandlers.onError.splice(index, 1);
        };
    }

    /**
     * Get connection status
     */
    getStatus() {
        if (!this.socket) return 'disconnected';
        switch (this.socket.readyState) {
            case WebSocket.CONNECTING: return 'connecting';
            case WebSocket.OPEN: return 'connected';
            case WebSocket.CLOSING: return 'closing';
            case WebSocket.CLOSED: return 'disconnected';
            default: return 'unknown';
        }
    }
}

// Export for use in other modules
window.WebSocketManager = WebSocketManager;
