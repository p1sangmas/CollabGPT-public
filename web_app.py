#!/usr/bin/env python3
"""
Entry point script for CollabGPT Web Interface.

This script provides a web-based interface to CollabGPT's functionality.
Usage:
  python web_app.py  # Start the CollabGPT web server
"""

# Apply eventlet monkey patch before importing any other modules
import eventlet
eventlet.monkey_patch()

import sys
import socket
import signal
import time
import os
from web.app import app, socketio, log

# Import WebhookManager for automatic webhook registration
from src.services.auto_webhook import WebhookManager
from src.api.google_docs import GoogleDocsAPI

# Add signal handlers for graceful shutdown
def signal_handler(sig, frame):
    """Handle termination signals for graceful shutdown."""
    log.info(f"Received signal {sig}, shutting down gracefully...")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def find_available_port(start_port=5000, max_attempts=10):
    """Find an available port starting from the given port."""
    port = start_port
    for attempt in range(max_attempts):
        try:
            # Try to create a socket with the current port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                # Setting SO_REUSEADDR option to avoid "Address already in use" error
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('0.0.0.0', port))
                return port
        except OSError:
            print(f"Port {port} is already in use, trying {port+1}...")
            port += 1
    
    # If we couldn't find an available port after max_attempts, raise an error
    raise RuntimeError(f"Could not find an available port after {max_attempts} attempts")

def setup_auto_webhook(port):
    """Set up automatic webhook registration with the current ngrok URL."""
    try:
        log.info("Setting up automatic webhook registration")
        # Initialize the Google Docs API with credentials
        google_docs_api = GoogleDocsAPI()
        
        # Make sure we're authenticated before proceeding
        if not google_docs_api.authenticate():
            log.warning("Failed to authenticate with Google Docs API. Skipping webhook registration.")
            return None
            
        # Initialize the WebhookManager with the port that the web app is running on
        webhook_manager = WebhookManager(
            google_docs_api=google_docs_api,
            port=port,
            webhook_endpoint="/webhook"
        )
        
        # Register webhooks for all monitored documents
        success = webhook_manager.register_all_webhooks()
        if success:
            log.info("Successfully registered all webhooks with new ngrok URL")
        else:
            log.warning("Some webhooks could not be registered. Check logs for details.")
            
        # Store the webhook manager in app.config for potential future use
        app.config['webhook_manager'] = webhook_manager
        
        return webhook_manager
    except Exception as e:
        log.error(f"Error setting up automatic webhook registration: {e}")
        return None

def run_server_with_retry(max_retries=5):
    """Run the server with automatic retry on failure."""
    retry_count = 0
    while retry_count < max_retries:
        try:
            # Find an available port
            port = find_available_port()
            log.info(f"Starting server on port {port}")
            
            # Set up automatic webhook registration with the new port
            webhook_manager = setup_auto_webhook(port)
            
            # Use the available port with Flask-SocketIO
            # Set larger timeouts to prevent disconnects
            socketio.run(
                app, 
                debug=True, 
                host="0.0.0.0", 
                port=port, 
                allow_unsafe_werkzeug=True
            )
            
            # If socketio.run() returns without error, exit the loop
            break
            
        except Exception as e:
            retry_count += 1
            log.error(f"Error in server (attempt {retry_count}/{max_retries}): {e}")
            
            if retry_count < max_retries:
                # Wait before retrying to avoid rapid cycling
                wait_time = min(30, retry_count * 5)  # Progressive backoff
                log.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                log.error("Maximum retry attempts reached. Exiting.")
                sys.exit(1)

if __name__ == "__main__":
    print("Starting CollabGPT Web Interface with continuous operation mode")
    run_server_with_retry()