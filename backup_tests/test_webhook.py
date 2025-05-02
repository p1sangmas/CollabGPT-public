#!/usr/bin/env python3
"""
Test script for webhook integration with Google Docs.
This script will start the CollabGPT application and monitor logs
for webhook notifications when you make changes to your Google Docs.
"""

import os
import sys
import time
import logging
import socket
import dotenv

# Load environment variables from .env file (important for webhook settings)
dotenv.load_dotenv()

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Use DEBUG level for more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/webhook_test.log")
    ]
)

logger = logging.getLogger("webhook_test")

# Import the application and settings
from src.app import CollabGPT
from src.config import settings

def get_local_ip():
    """Get the local IP address for use in webhook testing."""
    try:
        # Create a socket to determine the local IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Doesn't need to be reachable
        s.connect(('10.255.255.255', 1))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return '127.0.0.1'

def test_webhook_integration():
    """
    Run a test of the webhook integration with Google Docs.
    """
    logger.info("Starting webhook integration test")
    
    # Get webhook settings from environment (.env)
    webhook_port = int(os.environ.get('WEBHOOK_PORT', 8000))
    external_url = os.environ.get('WEBHOOK_EXTERNAL_URL', '')
    
    logger.info(f"Using webhook port: {webhook_port}")
    logger.info(f"Using external URL: {external_url}")
    
    # Ensure settings are properly loaded from environment
    settings.WEBHOOK['port'] = webhook_port
    settings.WEBHOOK['external_url'] = external_url
    
    if not external_url:
        logger.error("WEBHOOK_EXTERNAL_URL not set in environment. Webhooks won't work!")
        return 1
        
    # Create and initialize the application
    app = CollabGPT()
    if not app.initialize():
        logger.error("Failed to initialize application")
        return 1
        
    # Increase logging level for webhook handler
    webhook_logger = logging.getLogger("collabgpt.webhook_handler")
    webhook_logger.setLevel(logging.DEBUG)
        
    # Start the application
    if not app.start():
        logger.error("Failed to start application")
        return 1
    
    logger.info("CollabGPT application started successfully")
    logger.info(f"Webhook server listening on port {webhook_port}")
    logger.info(f"External webhook URL: {external_url}")
    
    # Get the list of monitored documents
    monitored_docs = app.monitored_documents
    doc_count = len(monitored_docs)
    
    if doc_count == 0:
        logger.warning("No documents are being monitored. Add a document first.")
        app.stop()
        return 1
        
    logger.info(f"Monitoring {doc_count} documents:")
    for doc_id, doc_info in monitored_docs.items():
        logger.info(f"  - {doc_info.get('name', doc_id)} ({doc_id})")
        
        # Manually set up webhooks for each document to ensure they're active
        logger.info(f"Setting up webhook for document: {doc_id}")
        success = app._setup_document_webhook(doc_id)
        if success:
            logger.info(f"Successfully set up webhook for {doc_id}")
        else:
            logger.warning(f"Failed to set up webhook for {doc_id}")
    
    # Add a manual test option to simulate a webhook event
    logger.info("\nTest Instructions:")
    logger.info("1. The application is now running and listening for webhook notifications")
    logger.info("2. Open one of the monitored Google Docs in your browser")
    logger.info("3. Make some changes to the document and save")
    logger.info("4. Watch the logs for webhook notifications")
    logger.info("5. If no notifications appear, try simulating a change with the 'c' key")
    logger.info("6. Press Ctrl+C to stop the test when finished")
    
    print("\nPress 'c' to simulate a document change for the first document (for testing)")
    print("Press 'd' to dump the current document content (for debugging)")
    print("Press Ctrl+C to exit")
    
    try:
        # Keep the application running and handle user input
        while True:
            # Check if a key is pressed (non-blocking)
            if os.name == 'nt':  # Windows
                import msvcrt
                if msvcrt.kbhit():
                    key = msvcrt.getch().decode('utf-8').lower()
                    handle_key_press(key, app, monitored_docs, doc_count)
            else:  # Unix/Mac
                import select
                # Check if input is available
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1).lower()
                    handle_key_press(key, app, monitored_docs, doc_count)
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    finally:
        # Stop the application
        app.stop()
    
    logger.info("Webhook integration test completed")
    return 0

def handle_key_press(key, app, monitored_docs, doc_count):
    """Handle key presses for interactive testing."""
    if key == 'c' and doc_count > 0:
        # Simulate a change event for the first document
        doc_id = list(monitored_docs.keys())[0]
        logger.info(f"Simulating change for document: {doc_id}")
        app.webhook_handler.simulate_change_event(doc_id)
        logger.info("Change event simulation completed")
    elif key == 'd' and doc_count > 0:
        # Dump current document content for debugging
        doc_id = list(monitored_docs.keys())[0]
        logger.info(f"Dumping content for document: {doc_id}")
        content = app.google_docs_api.get_document_content(doc_id)
        if content:
            logger.info(f"Document content length: {len(content)} characters")
            logger.info(f"First 200 characters: {content[:200]}")
        else:
            logger.warning("Failed to retrieve document content")

if __name__ == "__main__":
    sys.exit(test_webhook_integration())