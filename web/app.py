#!/usr/bin/env python3
"""
Flask web application for CollabGPT.
This provides a web interface to CollabGPT's functionality.
"""

import sys
import os
import json
import threading
import time
from datetime import datetime
from flask import Flask, render_template, jsonify, request, redirect, url_for, flash
from flask_socketio import SocketIO, emit, join_room
import eventlet

# Set up eventlet to avoid timeout issues
eventlet.monkey_patch()

# Add the root directory to Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.app import CollabGPT
from src.utils import logger
from src.services.auto_webhook import WebhookManager

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev_key_change_in_production')

# Configure SocketIO with more resilient settings
socketio = SocketIO(
    app, 
    cors_allowed_origins="*",
    async_mode='eventlet',
    ping_timeout=60,  # Increase ping timeout
    ping_interval=25,  # More frequent pings to keep connection alive
    reconnection=True,
    reconnection_attempts=float('inf'),
    reconnection_delay=1000,
    reconnection_delay_max=5000
)

# Configure logging
log = logger.get_logger("web")

# Initialize CollabGPT instance
collabgpt = CollabGPT()

# Override webhook port to avoid conflicts with other services
os.environ['WEBHOOK_PORT'] = '5002'  # Use a different port than the web app's 5001

# Heartbeat mechanism to keep the server alive
def heartbeat():
    """Send periodic heartbeat to prevent timeouts"""
    count = 0
    while True:
        count += 1
        time.sleep(20)  # Send heartbeat every 20 seconds
        log.debug(f"Server heartbeat: {count}")
        # Emit a keepalive event to all clients
        socketio.emit('server_heartbeat', {'count': count}, namespace='/')

# In newer Flask versions, we need to use a different approach
# instead of @app.before_first_request
def initialize_app():
    """Initialize CollabGPT before handling requests."""
    if not collabgpt.initialize():
        log.error("Failed to initialize CollabGPT")
        sys.exit(1)

    if not collabgpt.google_docs_api.authenticate():
        log.error("Failed to authenticate with Google Docs API")
        sys.exit(1)

    # Start background services
    if not collabgpt.start():
        log.error("Failed to start CollabGPT services")
        sys.exit(1)

    # Set up document change callback to broadcast to WebSocket clients
    collabgpt.webhook_handler.register_callback('change', handle_document_change_for_websocket)
    
    # Initialize automatic webhook registration for all monitored documents
    webhook_url = os.environ.get('WEBHOOK_EXTERNAL_URL')
    if webhook_url:
        try:
            log.info("Setting up automatic webhook registration")
            webhook_manager = WebhookManager(collabgpt.google_docs_api)
            
            # Ensure API is authenticated before calling webhook manager
            # The GoogleDocsAPI class doesn't have is_authenticated() method, it has authenticate()
            if collabgpt.google_docs_api.service is None:
                log.info("Re-authenticating Google Docs API for webhook registration")
                collabgpt.google_docs_api.authenticate()
                
            success = webhook_manager.register_all_webhooks()
            if success:
                log.info("Automatic webhook registration successful")
            else:
                log.warning("Some webhooks could not be registered. Check logs for details.")
        except Exception as e:
            log.error(f"Error during automatic webhook setup: {e}")
    else:
        log.warning("WEBHOOK_EXTERNAL_URL not set in environment. Skipping webhook registration.")
    
    # Start the heartbeat thread to keep the server alive
    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()
    
    log.info("CollabGPT web interface started with continuous running mode")

# WebSocket document change handler
def handle_document_change_for_websocket(change_data):
    """Handle document changes and broadcast to connected WebSocket clients."""
    try:
        document_id = change_data.get('document_id')
        user_id = change_data.get('metadata', {}).get('user_id', 'unknown')
        
        log.info(f"Broadcasting document change for {document_id} by {user_id}")
        # Debug information to track payload
        log.debug(f"Change data payload: {json.dumps(change_data, default=str)[:500]}")
        
        # Prepare data for WebSocket clients
        event_data = {
            'document_id': document_id,
            'user': user_id,
            'timestamp': datetime.now().isoformat(),
            'type': 'document_change'
        }
        
        # Add summary if available
        if collabgpt.llm_interface:
            try:
                current_content = change_data.get('current_content', '')
                previous_content = change_data.get('previous_content', '')
                if current_content and previous_content and current_content != previous_content:
                    summary_response = collabgpt.llm_interface.generate_with_template(
                        "summarize_changes",
                        previous_content=previous_content,
                        current_content=current_content
                    )
                    if summary_response.success:
                        event_data['summary'] = summary_response.text
                    else:
                        event_data['summary'] = "Document was updated"
                else:
                    event_data['summary'] = "Minor document update"
            except Exception as e:
                log.error(f"Error generating change summary: {e}")
                event_data['summary'] = "Document was updated"
        else:
            event_data['summary'] = "Document was updated"
            
        # Create notification for activity feed
        activity_event = {
            'type': 'EDIT',
            'document_id': document_id,
            'document_name': collabgpt.monitored_documents.get(document_id, {}).get('name', 'Unknown Document'),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'user': user_id,
            'description': event_data['summary'],
            'id': f"event_{int(datetime.now().timestamp())}"  # Add unique ID for deduplication
        }
        
        # Debug info to confirm broadcasting attempts
        log.debug(f"Broadcasting to room: document_{document_id}")
        
        # Broadcast to the document-specific room
        socketio.emit('document_update', event_data, room=f'document_{document_id}')
        
        # Broadcast to the activity feed on the dashboard
        socketio.emit('activity_update', activity_event, room='dashboard')
        log.debug(f"Broadcast activity_update to dashboard")
        
        # Render the activity item HTML and broadcast specifically for the activity log
        try:
            with app.app_context():
                # Create a temporary event context for the template
                event = activity_event
                is_new = True
                rendered_html = render_template('partials/activity_item.html', event=event, is_new=is_new)
                
                # Send the document activity update with the rendered HTML
                activity_data = {
                    'event': activity_event,
                    'html': rendered_html
                }
                socketio.emit('document_activity', activity_data, room=f'document_{document_id}')
                log.debug(f"Broadcast document_activity to room: document_{document_id}")
        except Exception as e:
            log.error(f"Error rendering activity item: {e}")
        
    except Exception as e:
        log.error(f"Error in WebSocket broadcast: {e}", exc_info=True)

# Call initialize on startup
with app.app_context():
    initialize_app()

# Register cleanup on exit
import atexit
def cleanup():
    if collabgpt.running:
        collabgpt.stop()
        log.info("CollabGPT services stopped")
atexit.register(cleanup)

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    """Handle client connection to WebSocket."""
    log.info(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection from WebSocket."""
    log.info(f"Client disconnected: {request.sid}")

@socketio.on('join')
def handle_join(data):
    """Handle client joining a specific document room."""
    if 'document_id' in data:
        document_id = data['document_id']
        room = f"document_{document_id}"
        join_room(room)
        log.info(f"Client {request.sid} joined room: {room}")
        emit('join_confirmation', {'status': 'success', 'room': room})
    elif data.get('room') == 'dashboard':
        join_room('dashboard')
        log.info(f"Client {request.sid} joined dashboard room")
        emit('join_confirmation', {'status': 'success', 'room': 'dashboard'})

@socketio.on('leave')
def handle_leave(data):
    """Handle client leaving a specific document room."""
    if 'document_id' in data:
        document_id = data['document_id']
        room = f"document_{document_id}"
        socketio.leave_room(room)
        log.info(f"Client {request.sid} left room: {room}")

# Routes
@app.route('/')
def index():
    """Dashboard showing monitored documents and system status."""
    return render_template('index.html', documents=collabgpt.monitored_documents)

@app.route('/document/<doc_id>')
def document_view(doc_id):
    """View a specific document with Google Docs iframe and suggestions."""
    # Get document info
    doc_info = collabgpt.monitored_documents.get(doc_id, {})
    if not doc_info and doc_id:
        # Try to get info from Google Docs API
        try:
            # Get document info using the get_document method instead of get_document_metadata
            doc_data = collabgpt.google_docs_api.get_document(doc_id)
            if doc_data:
                doc_info = {
                    'title': doc_data.get('title', 'Untitled Document'),
                    'name': doc_data.get('title', 'Untitled Document'),
                    'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
        except Exception as e:
            log.error(f"Error getting document info: {e}")
            flash(f"Error: Unable to retrieve document information.", "error")
            return redirect(url_for('index'))
    
    # Determine monitoring status
    monitoring_status = "active" if doc_info.get('webhook_enabled', False) else "paused"
    
    return render_template('document.html', 
                           doc_id=doc_id, 
                           doc_info=doc_info,
                           monitoring_status=monitoring_status)

@app.route('/analyze/<doc_id>')
def analyze_document(doc_id):
    """Run document analysis and return results."""
    try:
        results = collabgpt.process_document(doc_id)
        # Return formatted HTML using the template instead of raw JSON
        return render_template('partials/analysis_results.html', 
                              analysis=results, 
                              doc_id=doc_id,
                              now=datetime.now())
    except Exception as e:
        log.error(f"Error analyzing document: {e}")
        return f"<div class='text-red-500'>Error analyzing document: {str(e)}</div>", 500

@app.route('/suggestions/<doc_id>')
def document_suggestions(doc_id):
    """Get and display smart suggestions for a document."""
    try:
        max_suggestions = int(request.args.get('max', 3))
        suggestions = collabgpt.generate_smart_edit_suggestions(
            doc_id, 
            max_suggestions=max_suggestions,
            feedback_enabled=True
        )
        return render_template('partials/suggestions.html', 
                               suggestions=suggestions, 
                               doc_id=doc_id)
    except Exception as e:
        log.error(f"Error generating suggestions: {e}")
        return f"<div class='text-red-500'>Error: {str(e)}</div>", 500

@app.route('/document-map/<doc_id>')
def document_map(doc_id):
    """Get document structure map."""
    try:
        doc_map = collabgpt.get_document_map(doc_id)
        return render_template('partials/document_map.html', 
                               doc_map=doc_map, 
                               doc_id=doc_id)
    except Exception as e:
        log.error(f"Error generating document map: {e}")
        return f"<div class='text-red-500'>Error: {str(e)}</div>", 500

@app.route('/system-status')
def system_status():
    """Get system status information."""
    status = {
        "running": collabgpt.running,
        "monitored_documents": len(collabgpt.monitored_documents),
        "system_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "memory_usage": "N/A"  # Could add actual memory monitoring here
    }
    return render_template('partials/system_status.html', status=status)

@app.route('/recent-activity')
def recent_activity():
    """Get recent activity across all documents."""
    # Combine recent activity from all monitored documents
    all_activity = []
    for doc_id, doc_info in collabgpt.monitored_documents.items():
        try:
            # Get last 24 hours of activity
            activity = collabgpt.get_document_activity(doc_id, hours=24)
            if activity and "events" in activity:
                for event in activity.get("events", []):
                    event["document_id"] = doc_id
                    event["document_name"] = doc_info.get("name", "Unknown")
                    all_activity.append(event)
        except Exception as e:
            log.error(f"Error getting activity for document {doc_id}: {e}")
    
    # Sort by timestamp descending
    all_activity.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    # Limit to 20 most recent events
    recent_events = all_activity[:20]
    
    return render_template('partials/activity_feed.html', events=recent_events)

@app.route('/add-document', methods=['POST'])
def add_document():
    """Add a new document to monitor."""
    doc_id = request.form.get('doc_id')
    if not doc_id:
        flash("Document ID is required", "error")
        return redirect(url_for('index'))
    
    try:
        # Validate document ID and setup monitoring
        # Use get_document instead of get_document_metadata
        doc_data = collabgpt.google_docs_api.get_document(doc_id)
        if not doc_data:
            flash("Invalid document ID or document not accessible", "error")
            return redirect(url_for('index'))
            
        collabgpt._setup_document_webhook(doc_id)
        flash(f"Document '{doc_data.get('title', doc_id)}' added successfully", "success")
    except Exception as e:
        log.error(f"Error adding document: {e}")
        flash(f"Error: {str(e)}", "error")
    
    return redirect(url_for('index'))

@app.route('/add-document-form')
def add_document_form():
    """Return the form for adding a new document."""
    return render_template('partials/add_document_form.html')

@app.route('/activity/<doc_id>')
def document_activity(doc_id):
    """Get activity for a specific document."""
    try:
        hours = int(request.args.get('hours', 24))
        activity = collabgpt.get_document_activity(doc_id, hours=hours)
        
        # Ensure we have events array to avoid errors
        if not activity:
            activity = {"events": []}
        elif "events" not in activity:
            activity["events"] = []
            
        # Pass enumeration data to help template generate indexes
        # This provides a substitute for loop.index
        return render_template('partials/document_activity.html', 
                               activity=activity, 
                               doc_id=doc_id)
    except Exception as e:
        log.error(f"Error getting document activity: {e}")
        return f"<div class='text-red-500'>Error: {str(e)}</div>", 500

@app.route('/monitoring-toggle/<doc_id>', methods=['GET'])
def toggle_monitoring(doc_id):
    """Toggle monitoring for a document."""
    # Get current status from document info, not just presence in dictionary
    doc_info = collabgpt.monitored_documents.get(doc_id, {})
    currently_monitored = doc_info.get('webhook_enabled', True) if doc_id in collabgpt.monitored_documents else False
    
    try:
        if currently_monitored:
            # Stop monitoring
            collabgpt._remove_document_webhook(doc_id)
            message = "Document monitoring paused"
            status = "paused"
        else:
            # Start monitoring
            collabgpt._setup_document_webhook(doc_id)
            message = "Document monitoring activated"
            status = "active"
            
        # Use the explicitly determined status rather than inferring it
        return render_template('partials/monitoring_status.html', 
                               status=status, 
                               message=message,
                               doc_id=doc_id)
    except Exception as e:
        log.error(f"Error toggling monitoring: {e}")
        return f"<div class='text-red-500'>Error: {str(e)}</div>", 500

@app.route('/apply-suggestion/<doc_id>', methods=['POST'])
def apply_suggestion(doc_id):
    """Apply a suggestion to the document."""
    try:
        suggestion_id = request.form.get('suggestion_id')
        if not suggestion_id:
            return jsonify({"error": "No suggestion ID provided"}), 400
        
        log.info(f"Applying suggestion {suggestion_id} to document {doc_id}")
        
        # Get the suggestion from the edit suggestion system
        suggestion = collabgpt.edit_suggestion_system.get_suggestion_by_id(suggestion_id)
        if not suggestion:
            return jsonify({"error": "Suggestion not found"}), 404
            
        # Apply the suggestion to the document
        # This would typically involve calling Google Docs API to make the edit
        try:
            # Get document content
            doc_content = collabgpt.google_docs_api.get_document_content(doc_id)
            if not doc_content:
                return jsonify({"error": "Could not retrieve document content"}), 500
                
            # In a real implementation, you would locate the text to replace and 
            # call the appropriate Google Docs API method to replace it
            # For now, we'll just record that the suggestion was "applied"
            
            # Record feedback that the suggestion was accepted
            collabgpt.record_suggestion_feedback(
                suggestion_id=suggestion_id,
                accepted=True,
                user_feedback="Applied via web interface",
                user_id=request.remote_addr  # Use IP as a simple user identifier
            )
            
            # Return success message and prompt user to refresh
            return render_template('partials/suggestion_applied.html', 
                                  suggestion=suggestion,
                                  doc_id=doc_id)
                                  
        except Exception as e:
            log.error(f"Error applying suggestion: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500
    
    except Exception as e:
        log.error(f"Error in apply_suggestion: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/feedback/<doc_id>', methods=['POST'])
def record_feedback(doc_id):
    """Record user feedback on a suggestion."""
    try:
        suggestion_id = request.form.get('suggestion_id')
        feedback = request.form.get('feedback', '')
        
        if not suggestion_id:
            return jsonify({"error": "No suggestion ID provided"}), 400
            
        log.info(f"Recording feedback '{feedback}' for suggestion {suggestion_id}")
        
        # Record the feedback
        accepted = feedback.lower() != "reject"
        success = collabgpt.record_suggestion_feedback(
            suggestion_id=suggestion_id,
            accepted=accepted,
            user_feedback=feedback,
            user_id=request.remote_addr  # Use IP as a simple user identifier
        )
        
        if success:
            return render_template('partials/feedback_recorded.html',
                                  feedback_type=feedback,
                                  doc_id=doc_id)
        else:
            return jsonify({"error": "Failed to record feedback"}), 500
            
    except Exception as e:
        log.error(f"Error recording feedback: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/simulate-change/<doc_id>', methods=['GET'])
def simulate_document_change(doc_id):
    """Simulate a document change event for testing."""
    try:
        log.info(f"Simulating change event for document {doc_id}")
        collabgpt.webhook_handler.simulate_change_event(doc_id)
        log.info("Simulation completed")
        return jsonify({"status": "success", "message": "Document change event simulated"})
    except Exception as e:
        log.error(f"Error simulating change: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/refresh-webhook/<doc_id>', methods=['GET'])
def refresh_webhook(doc_id):
    """Force refresh of the webhook for a document."""
    try:
        log.info(f"Forcing webhook refresh for document {doc_id}")
        
        # Stop existing webhook if present
        if hasattr(collabgpt, '_remove_document_webhook'):
            collabgpt._remove_document_webhook(doc_id)
        
        # Set up new webhook
        success = collabgpt._setup_document_webhook(doc_id)
        
        if success:
            # Also register in webhook manager
            webhook_manager = WebhookManager(collabgpt.google_docs_api)
            webhook_manager.port = int(os.environ.get('WEBHOOK_PORT', 5002))  # Ensure correct port
            channel_id = webhook_manager._register_webhook_for_document(doc_id)
            
            if channel_id:
                return jsonify({
                    "status": "success", 
                    "message": "Webhook successfully refreshed"
                })
            else:
                return jsonify({
                    "status": "partial", 
                    "message": "CollabGPT webhook set up but webhook manager registration failed"
                })
        else:
            return jsonify({
                "status": "error", 
                "message": "Failed to refresh webhook"
            }), 500
    except Exception as e:
        log.error(f"Error refreshing webhook: {e}")
        return jsonify({
            "status": "error", 
            "message": str(e)
        }), 500

@app.route('/webhook', methods=['POST'])
def handle_webhook():
    """Handle incoming webhook requests from Google Docs."""
    try:
        log.info(f"Received webhook request from: {request.remote_addr}")
        
        # Get headers and payload
        headers = dict(request.headers)
        payload = request.get_data()
        
        # Process the webhook through CollabGPT's webhook handler
        success = collabgpt.webhook_handler.process_webhook(headers, payload)
        
        if success:
            return "", 200  # Empty response with 200 OK status
        else:
            log.warning("Webhook processing returned False")
            return "", 202  # Accepted but not processed
            
    except Exception as e:
        log.error(f"Error processing webhook: {e}", exc_info=True)
        return "", 500  # Server error

if __name__ == "__main__":
    # This is used when running flask directly
    socketio.run(app, debug=True, host="0.0.0.0", port=5001)