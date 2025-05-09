/**
 * WebSocket client for CollabGPT real-time updates
 * This script manages WebSocket connections for real-time document updates
 */

// Initialize socket connection when the document is ready
let socket = null;
let connected = false;
let currentRoom = null;
let heartbeatCount = 0;
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 10;

// Function to initialize the socket connection
function initSocketConnection() {
    if (socket) return; // Already initialized
    
    console.log('Initializing WebSocket connection...');
    
    // Connect to the WebSocket server using the current window location
    // This ensures it works regardless of which port the server is running on
    const socketUrl = window.location.protocol + '//' + window.location.host;
    console.log('Connecting to WebSocket at:', socketUrl);
    
    socket = io.connect(socketUrl, {
        reconnection: true,
        reconnectionDelay: 1000,
        reconnectionDelayMax: 5000,
        reconnectionAttempts: Infinity,
        timeout: 60000,  // Longer timeout (60s)
        pingTimeout: 60000,  // Longer ping timeout
        pingInterval: 25000  // More frequent pings
    });
    
    // Connection established
    socket.on('connect', function() {
        console.log('WebSocket connected');
        connected = true;
        reconnectAttempts = 0;
        
        // Join the appropriate room based on the current page
        joinAppropriateRoom();
    });
    
    // Connection lost
    socket.on('disconnect', function(reason) {
        console.log('WebSocket disconnected:', reason);
        connected = false;
        currentRoom = null;
        
        // If the disconnect was due to a timeout, try to reconnect manually
        if (reason === 'transport close' || reason === 'ping timeout') {
            handleReconnect();
        }
    });
    
    // Handle server heartbeats
    socket.on('server_heartbeat', function(data) {
        console.log('Server heartbeat received:', data.count);
        heartbeatCount = data.count;
        
        // Send a response heartbeat to keep the connection alive
        socket.emit('client_heartbeat', { count: heartbeatCount });
        
        // Update connection status indicator if it exists
        updateConnectionStatus(true);
    });
    
    // Handle connection error
    socket.on('connect_error', function(error) {
        console.error('Connection error:', error);
        updateConnectionStatus(false);
        
        // Try to reconnect manually if standard reconnection fails
        handleReconnect();
    });
    
    // Handle document updates
    socket.on('document_update', function(data) {
        console.log('Document update received:', data);
        handleDocumentUpdate(data);
    });
    
    // Handle activity updates (for dashboard)
    socket.on('activity_update', function(data) {
        console.log('Activity update received:', data);
        handleActivityUpdate(data);
    });
    
    // Handle document activity updates (specifically for activity log)
    socket.on('document_activity', function(data) {
        console.log('Document activity received:', data);
        handleDocumentActivityUpdate(data);
    });
    
    // Handle room join confirmation
    socket.on('join_confirmation', function(data) {
        console.log('Joined room:', data.room);
        currentRoom = data.room;
    });
}

// Manual reconnection handler
function handleReconnect() {
    if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
        reconnectAttempts++;
        console.log(`Attempting manual reconnect (${reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS})...`);
        
        // Update UI to show reconnecting status
        updateConnectionStatus(false, true);
        
        // Try to reconnect
        setTimeout(() => {
            if (!connected && socket) {
                socket.connect();
            }
        }, reconnectAttempts * 1000); // Increasing backoff
    } else {
        console.error('Maximum reconnection attempts reached');
        updateConnectionStatus(false, false);
        
        // Show a message to the user that they need to refresh
        addUpdateToRealtimeContainer({
            id: 'notification-reconnect-' + Date.now(),
            type: 'error',
            title: 'Connection Lost',
            message: 'Could not reconnect to the server. Please refresh the page.',
            timestamp: new Date().toISOString()
        });
    }
}

// Update connection status indicator
function updateConnectionStatus(isConnected, isReconnecting = false) {
    const indicator = document.getElementById('real-time-indicator');
    if (!indicator) return;
    
    if (isConnected) {
        indicator.className = 'inline-flex items-center mr-3 text-sm bg-green-100 text-green-800 px-2 py-1 rounded-full';
        indicator.innerHTML = '<span class="h-2 w-2 bg-green-500 rounded-full mr-1 animate-pulse"></span>Real-time updates';
    } else if (isReconnecting) {
        indicator.className = 'inline-flex items-center mr-3 text-sm bg-yellow-100 text-yellow-800 px-2 py-1 rounded-full';
        indicator.innerHTML = '<span class="h-2 w-2 bg-yellow-500 rounded-full mr-1 animate-pulse"></span>Reconnecting...';
    } else {
        indicator.className = 'inline-flex items-center mr-3 text-sm bg-red-100 text-red-800 px-2 py-1 rounded-full';
        indicator.innerHTML = '<span class="h-2 w-2 bg-red-500 rounded-full mr-1"></span>Disconnected';
    }
}

// Join the appropriate room based on the current page
function joinAppropriateRoom() {
    if (!connected || !socket) return;
    
    // Check if we're on a document page
    const documentId = getDocumentIdFromUrl();
    if (documentId) {
        console.log('Joining document room for:', documentId);
        socket.emit('join', { document_id: documentId });
        return;
    }
    
    // If not on a document page, join the dashboard room for general updates
    console.log('Joining dashboard room');
    socket.emit('join', { room: 'dashboard' });
}

// Extract document ID from the current URL
function getDocumentIdFromUrl() {
    const path = window.location.pathname;
    const matches = path.match(/\/document\/([^\/]+)/);
    return matches ? matches[1] : null;
}

// Handle a document update event
function handleDocumentUpdate(data) {
    if (!data || !data.document_id) return;
    
    // Create notification
    const notification = createNotification(data);
    
    // Add notification to the real-time updates container
    addUpdateToRealtimeContainer(notification);
    
    // Update relevant UI components based on the update type
    updateDocumentUI(data);
}

// Create a notification object from update data
function createNotification(data) {
    return {
        id: 'notification-' + Date.now(),
        type: data.type || 'info',
        title: 'Document Updated',
        message: data.summary || 'Document was updated',
        timestamp: data.timestamp || new Date().toISOString(),
        user: data.user || 'Unknown user',
        documentId: data.document_id
    };
}

// Add update to the real-time updates container
function addUpdateToRealtimeContainer(notification) {
    // Get the updates container
    const container = document.getElementById('realtime-updates-container');
    if (!container) return;
    
    // Remove the placeholder message if it exists
    const placeholder = container.querySelector('.bg-blue-50.text-blue-800');
    if (placeholder && placeholder.textContent.includes('Waiting for document updates')) {
        placeholder.remove();
    }
    
    // Create update element
    const updateElement = document.createElement('div');
    updateElement.id = notification.id;
    updateElement.className = 'notification p-3 border-b border-gray-100 recent-update';
    
    // Style based on notification type
    let bgColor = 'bg-white';
    let textColor = 'text-gray-800';
    
    if (notification.type === 'error') {
        bgColor = 'bg-red-50';
        textColor = 'text-red-800';
    } else if (notification.type === 'document_change') {
        bgColor = 'bg-blue-50';
        textColor = 'text-blue-800';
    }
    
    updateElement.classList.add(bgColor, textColor);
    
    // Format timestamp
    const formattedTime = new Date(notification.timestamp).toLocaleTimeString();
    
    updateElement.innerHTML = `
        <div class="flex items-center justify-between">
            <span class="font-medium">${notification.title}</span>
            <button class="text-gray-400 hover:text-gray-600" onclick="this.parentElement.parentElement.remove()">
                &times;
            </button>
        </div>
        <p class="mt-1">${notification.message}</p>
        <div class="text-xs text-gray-500 mt-1">
            ${notification.user ? notification.user + ' | ' : ''}${formattedTime}
        </div>
    `;
    
    // Add to the container at the top
    if (container.firstChild) {
        container.insertBefore(updateElement, container.firstChild);
    } else {
        container.appendChild(updateElement);
    }
    
    // Limit the number of items to prevent the list from growing too large
    const maxItems = 20;
    const items = container.querySelectorAll('.notification');
    if (items.length > maxItems) {
        for (let i = maxItems; i < items.length; i++) {
            items[i].remove();
        }
    }
}

// Update UI components based on document update
function updateDocumentUI(data) {
    // Update suggestions if available and if we're on the suggestions tab
    const resultsPanel = document.getElementById('results-panel');
    if (resultsPanel && resultsPanel.querySelector('.suggestions-container')) {
        // Refresh suggestions since document content has changed
        const suggestionsBtn = document.querySelector('button[hx-get*="/suggestions/"]');
        if (suggestionsBtn) {
            htmx.trigger(suggestionsBtn, 'click');
        }
    }
}

// Handle document activity specific update
function handleDocumentActivityUpdate(data) {
    if (!data || !data.event) return;
    
    // Create a notification from the activity data
    const notification = {
        id: 'activity-' + Date.now(),
        type: data.event.type || 'activity',
        title: data.event.type || 'Activity',
        message: data.event.description || 'New activity detected',
        timestamp: data.event.timestamp || new Date().toISOString(),
        user: data.event.user || 'Unknown user',
        documentId: data.event.document_id
    };
    
    // Add to real-time updates container
    addUpdateToRealtimeContainer(notification);
}

// Handle activity update (for dashboard)
function handleActivityUpdate(data) {
    // Reload recent activity feed on dashboard
    const activityFeed = document.getElementById('activity-feed');
    if (activityFeed && activityFeed.hasAttribute('hx-get')) {
        htmx.trigger(activityFeed, 'load');
    }
    
    // On document pages, add the update to our real-time updates section
    const documentId = getDocumentIdFromUrl();
    if (documentId && documentId === data.document_id) {
        const notification = {
            id: 'activity-' + Date.now(),
            type: 'activity',
            title: data.type || 'Activity',
            message: data.description || 'New activity detected',
            timestamp: data.timestamp || new Date().toISOString(),
            user: data.user || 'Unknown user',
            documentId: data.document_id
        };
        
        addUpdateToRealtimeContainer(notification);
    }
}

// Initialize when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    initSocketConnection();
});

// Reinitialize connection when coming back from a different tab
document.addEventListener('visibilitychange', function() {
    if (document.visibilityState === 'visible') {
        if (!connected && socket) {
            console.log('Page visible again, reconnecting...');
            socket.connect();
        }
        updateConnectionStatus(connected);
    }
});

// Handle navigation via HTMX to ensure proper room joining
document.body.addEventListener('htmx:afterOnLoad', function() {
    if (connected) {
        // Small timeout to ensure any URL changes have completed
        setTimeout(joinAppropriateRoom, 100);
    }
});

// Re-export functions for global use
window.collabSocket = {
    init: initSocketConnection,
    joinRoom: joinAppropriateRoom,
    checkConnection: function() {
        return connected;
    },
    reconnect: function() {
        if (!connected && socket) {
            reconnectAttempts = 0;
            socket.connect();
            return true;
        }
        return false;
    }
};