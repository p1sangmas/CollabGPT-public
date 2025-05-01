"""
Main application file for CollabGPT.

This module coordinates all components of the application,
initializing services and handling the main application flow.
"""

import os
import sys
import signal
import time
import json
from typing import Dict, Any, List, Optional, Callable
from threading import Thread
import asyncio
import http.server
import socketserver
from urllib.parse import urlparse, parse_qs

from .api.google_docs import GoogleDocsAPI
from .services.webhook_handler import WebhookHandler
from .services.document_analyzer import DocumentAnalyzer
from .models.rag_system import RAGSystem
from .models.llm_interface import LLMInterface, LLMResponse
from .utils import logger
from .config import settings


class CollabGPT:
    """
    Main application class that coordinates all components.
    """
    
    def __init__(self):
        """Initialize the CollabGPT application."""
        self.logger = logger.get_logger("app")
        self.logger.info("Initializing CollabGPT application")
        
        # Initialize components
        self.google_docs_api = None
        self.webhook_handler = None
        self.document_analyzer = None
        self.rag_system = None
        self.llm_interface = None
        
        # Webhook server
        self.webhook_server = None
        self.webhook_thread = None
        
        # State
        self.running = False
        self.monitored_documents = {}
        
    def initialize(self) -> bool:
        """
        Initialize all components of the application.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            # Initialize Google Docs API
            self.logger.info("Initializing Google Docs API")
            self.google_docs_api = GoogleDocsAPI(
                credentials_path=settings.GOOGLE_API.get('credentials_path')
            )
            
            # Initialize document analyzer
            self.logger.info("Initializing Document Analyzer")
            self.document_analyzer = DocumentAnalyzer(
                language=settings.DOCUMENT.get('default_language', 'english')
            )
            
            # Initialize RAG system if enabled
            if settings.AI.get('rag_enabled', True):
                self.logger.info("Initializing RAG System")
                self.rag_system = RAGSystem()
            
            # Initialize LLM interface
            self.logger.info("Initializing LLM Interface")
            self.llm_interface = LLMInterface()
            
            # Initialize webhook handler
            self.logger.info("Initializing Webhook Handler")
            self.webhook_handler = WebhookHandler(
                self.google_docs_api,
                secret_key=settings.WEBHOOK.get('secret_key', '')
            )
            
            # Register event handlers
            self._register_event_handlers()
            
            # Load monitored documents
            self._load_monitored_documents()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization error: {e}")
            return False
    
    def start(self) -> bool:
        """
        Start the application services.
        
        Returns:
            True if startup was successful, False otherwise
        """
        try:
            self.running = True
            
            # Authenticate with Google Docs API
            if not self.google_docs_api.authenticate(
                use_service_account=settings.GOOGLE_API.get('use_service_account', False)
            ):
                self.logger.error("Failed to authenticate with Google Docs API")
                return False
                
            # Start webhook server if real-time monitoring is enabled
            if settings.FEATURES.get('real_time_monitoring', True):
                self._start_webhook_server()
                
                # Set up webhooks for all monitored documents
                for doc_id, doc_info in self.monitored_documents.items():
                    if doc_info.get('webhook_enabled', True):
                        self._setup_document_webhook(doc_id)
                        
            self.logger.info("CollabGPT application started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Startup error: {e}")
            self.running = False
            return False
    
    def stop(self) -> None:
        """Stop all application services."""
        self.logger.info("Stopping CollabGPT application")
        self.running = False
        
        # Stop the webhook server if running
        if self.webhook_server:
            self.logger.info("Stopping webhook server")
            self.webhook_server.shutdown()
            if self.webhook_thread and self.webhook_thread.is_alive():
                self.webhook_thread.join(timeout=5.0)
                
        self.logger.info("CollabGPT application stopped")
    
    def process_document(self, document_id: str) -> Dict[str, Any]:
        """
        Process a document to generate insights.
        
        Args:
            document_id: The Google Doc ID to process
            
        Returns:
            Dictionary containing analysis results
        """
        self.logger.info(f"Processing document: {document_id}")
        
        # Get the document content
        document = self.google_docs_api.get_document(document_id)
        if not document:
            self.logger.error(f"Failed to retrieve document: {document_id}")
            return {"error": "Document retrieval failed"}
            
        content = self.google_docs_api.get_document_content(document_id)
        
        # Analyze the document
        analysis = self.document_analyzer.analyze_document(document_id, content)
        
        # Process for RAG if enabled
        if self.rag_system:
            metadata = {
                "title": document.get('title', document_id),
                "last_updated": document.get('modifiedTime', ''),
            }
            self.rag_system.process_document(document_id, content, metadata)
            
        # Generate summary using LLM
        if self.llm_interface:
            summary_response = self.llm_interface.generate_with_template(
                "summarize_document", 
                document_content=content
            )
            
            if summary_response.success:
                analysis['ai_summary'] = summary_response.text
            else:
                analysis['ai_summary'] = "Failed to generate AI summary"
                
        # Update monitored documents
        if document_id not in self.monitored_documents:
            self.monitored_documents[document_id] = {
                'id': document_id,
                'name': document.get('title', document_id),
                'last_processed': time.time()
            }
            # Save to settings
            settings.save_monitored_document(
                document_id, 
                document.get('title', document_id)
            )
        
        return analysis
    
    def process_document_changes(self, document_id: str, 
                                previous_content: str, 
                                current_content: str) -> Dict[str, Any]:
        """
        Process changes between document versions.
        
        Args:
            document_id: The document identifier
            previous_content: The previous version content
            current_content: The current version content
            
        Returns:
            Dictionary containing change analysis results
        """
        self.logger.info(f"Analyzing changes for document: {document_id}")
        
        # Analyze changes
        change_analysis = self.document_analyzer.analyze_changes(
            document_id, previous_content, current_content
        )
        
        # Generate change summary using LLM
        if self.llm_interface:
            summary_response = self.llm_interface.generate_with_template(
                "summarize_changes",
                previous_content=previous_content,
                current_content=current_content
            )
            
            if summary_response.success:
                change_analysis['ai_change_summary'] = summary_response.text
                
        # Update RAG system with new content
        if self.rag_system:
            if document_id in self.monitored_documents:
                metadata = {
                    "title": self.monitored_documents[document_id].get('name', document_id),
                    "last_updated": time.time(),
                }
                self.rag_system.process_document(document_id, current_content, metadata)
        
        return change_analysis
    
    def suggest_edits(self, document_id: str, section_title: str = None) -> LLMResponse:
        """
        Generate edit suggestions for a document or section.
        
        Args:
            document_id: The document identifier
            section_title: Optional specific section to generate suggestions for
            
        Returns:
            LLM response containing suggestions
        """
        if not self.llm_interface:
            return LLMResponse("", error="LLM interface not initialized")
            
        # Get document content
        content = self.google_docs_api.get_document_content(document_id)
        if not content:
            return LLMResponse("", error="Failed to retrieve document content")
            
        # Get section content if specified
        if section_title:
            sections = self.document_analyzer._identify_sections(content)
            section_content = None
            
            for section in sections:
                if section['title'] == section_title:
                    section_content = section['content']
                    break
                    
            if not section_content:
                return LLMResponse("", error=f"Section not found: {section_title}")
                
            # Generate suggestions for the specific section
            return self.llm_interface.generate_with_template(
                "suggest_edits",
                section_title=section_title,
                section_content=section_content
            )
        else:
            # Get relevant context from RAG if available
            context = ""
            if self.rag_system:
                context = self.rag_system.get_relevant_context(
                    "document suggestions improvements", 
                    doc_id=document_id
                )
            
            # Generate suggestions for the whole document
            prompt = (
                f"Please suggest improvements or additions to the following document.\n\n"
                f"Document content:\n{content[:2000]}...\n\n"  # Limit content length
            )
            
            if context:
                prompt += f"Additional context:\n{context}\n\n"
                
            prompt += "Suggestions:"
            
            return self.llm_interface.generate(prompt)
    
    def _register_event_handlers(self) -> None:
        """Register event handlers for webhooks."""
        if not self.webhook_handler:
            return
            
        # Register handler for document changes
        self.webhook_handler.register_callback(
            'change', self._handle_document_change
        )
        
        # Register handler for comments
        self.webhook_handler.register_callback(
            'comment', self._handle_document_comment
        )
    
    def _handle_document_change(self, change_data: Dict[str, Any]) -> None:
        """
        Handle a document change event.
        
        Args:
            change_data: Information about the document change
        """
        document_id = change_data.get('document_id')
        current_content = change_data.get('current_content', '')
        previous_content = change_data.get('previous_content', '')
        
        self.logger.info(f"Document change detected: {document_id}")
        
        # Skip processing if no significant changes
        if previous_content and current_content == previous_content:
            self.logger.info("No significant changes detected")
            return
            
        # Process the changes
        change_analysis = self.process_document_changes(
            document_id, previous_content, current_content
        )
        
        self.logger.info(f"Change analysis completed: {document_id}")
        if 'ai_change_summary' in change_analysis:
            self.logger.info(f"AI summary: {change_analysis['ai_change_summary']}")
            
        # Here you would implement notification logic, UI updates, etc.
        # This is a simple placeholder that logs the summary
    
    def _handle_document_comment(self, comment_data: Dict[str, Any]) -> None:
        """
        Handle a document comment event.
        
        Args:
            comment_data: Information about the document comment
        """
        document_id = comment_data.get('document_id')
        comments = comment_data.get('comments', [])
        
        self.logger.info(f"Document comment event: {document_id}, {len(comments)} comments")
        
        # Check if any comments are addressed to CollabGPT
        for comment in comments:
            content = comment.get('content', '').lower()
            # Look for mentions or commands
            if '@collabgpt' in content or '#collabgpt' in content:
                self.logger.info(f"CollabGPT mentioned in comment: {content}")
                # Here you would implement the logic to respond to comments
                # This is a placeholder
    
    def _start_webhook_server(self) -> None:
        """Start the webhook server to receive document change notifications."""
        port = settings.WEBHOOK.get('port', 8000)
        path = settings.WEBHOOK.get('path', '/webhook')
        
        # Remove leading slash if present for comparison
        if path.startswith('/'):
            path_no_slash = path[1:]
        else:
            path_no_slash = path
            path = '/' + path
            
        self.logger.info(f"Starting webhook server on port {port}, path: {path}")
        
        class WebhookHandler(http.server.BaseHTTPRequestHandler):
            collabgpt_app = self  # Store reference to the app
            webhook_path = path
            webhook_path_no_slash = path_no_slash
            
            def do_POST(self):
                # More flexible path matching to handle variations in how Google sends webhooks
                request_path = self.path.strip()
                if request_path.startswith('/'):
                    request_path_no_slash = request_path[1:]
                else:
                    request_path_no_slash = request_path
                
                # Check if the path matches any of our expected formats
                path_matches = (
                    request_path == self.webhook_path or 
                    request_path_no_slash == self.webhook_path_no_slash or
                    request_path == '/' or  # Some webhook providers send to root
                    (request_path.startswith(self.webhook_path) and '?' in request_path)  # Handle query params
                )
                
                # Debug logging for webhook paths
                self.collabgpt_app.logger.info(f"Webhook request to path: '{request_path}', expecting: '{self.webhook_path}'")
                
                if not path_matches:
                    self.collabgpt_app.logger.warning(f"Webhook path mismatch: got '{request_path}', expected '{self.webhook_path}'")
                    self.send_response(404)
                    self.end_headers()
                    return
                
                # Get content length
                content_length = int(self.headers.get('Content-Length', 0))
                
                # Read the payload (even if empty)
                payload = self.rfile.read(content_length) if content_length > 0 else b''
                
                # Debug information
                self.collabgpt_app.logger.info(f"Webhook received with method: POST")
                self.collabgpt_app.logger.info(f"Webhook headers: {dict(self.headers)}")
                self.collabgpt_app.logger.info(f"Webhook payload size: {content_length} bytes")
                
                # Process Google Drive notifications first
                if 'X-Goog-Channel-ID' in self.headers or 'X-Goog-Resource-ID' in self.headers:
                    # For Google push notifications, immediately respond with 200 OK
                    # This is required by Google's Push Notifications API
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(b'{"status":"ok"}')
                    
                    # Process with webhook handler
                    success = self.collabgpt_app.webhook_handler.process_google_push_notification(dict(self.headers))
                    if success:
                        self.collabgpt_app.logger.info("Successfully processed Google push notification")
                    else:
                        self.collabgpt_app.logger.warning("Failed to process Google push notification")
                    
                    return
                
                # Handle non-Google webhook calls (your custom webhook format)
                try:
                    # For other types of webhooks that expect JSON payloads
                    if payload:
                        payload_str = payload.decode('utf-8')
                        self.collabgpt_app.logger.info(f"Webhook payload: {payload_str[:200]}..." if len(payload_str) > 200 else payload_str)
                        
                        # Process with regular webhook handler
                        success = self.collabgpt_app.webhook_handler.process_webhook(
                            dict(self.headers), payload
                        )
                        
                        # Send response
                        if success:
                            self.send_response(200)
                            self.end_headers()
                            self.wfile.write(b'{"status":"ok"}')
                        else:
                            self.send_response(400)
                            self.end_headers()
                            self.wfile.write(b'{"status":"error", "message":"Failed to process webhook"}')
                    else:
                        # Empty payload may still be a valid notification from some services
                        self.collabgpt_app.logger.info("Received webhook with empty payload")
                        self.send_response(200)
                        self.end_headers()
                        self.wfile.write(b'{"status":"ok"}')
                        
                except Exception as e:
                    self.collabgpt_app.logger.error(f"Error parsing webhook payload: {e}", exc_info=True)
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(f'{{"status":"error", "message":"{str(e)}"}}').encode('utf-8')
            
            def log_message(self, format, *args):
                # Redirect logs to our logger
                self.collabgpt_app.logger.info(f"Webhook server: {format % args}")
        
        try:
            # Create a custom TCP server with address reuse enabled
            class ReuseAddressTCPServer(socketserver.TCPServer):
                allow_reuse_address = True
            
            # Create and start the webhook server in a separate thread
            self.webhook_server = ReuseAddressTCPServer(("", port), WebhookHandler)
            self.webhook_thread = Thread(target=self.webhook_server.serve_forever)
            self.webhook_thread.daemon = True
            self.webhook_thread.start()
            
            self.logger.info(f"Webhook server started on port {port}")
            external_url = settings.WEBHOOK.get('external_url', 'Not configured')
            self.logger.info(f"External webhook URL: {external_url}{path}")
            
        except Exception as e:
            self.logger.error(f"Failed to start webhook server: {e}", exc_info=True)
    
    def _setup_document_webhook(self, document_id: str) -> bool:
        """
        Set up a webhook for a specific document.
        
        Args:
            document_id: The document identifier
            
        Returns:
            True if successful, False otherwise
        """
        external_url = settings.WEBHOOK.get('external_url', '')
        if not external_url:
            self.logger.error("Cannot set up webhook: No external URL configured")
            return False
            
        webhook_path = settings.WEBHOOK.get('path', '/webhook')
        webhook_url = f"{external_url}{webhook_path}"
        
        try:
            response = self.google_docs_api.watch_document(document_id, webhook_url)
            
            if response:
                self.logger.info(f"Webhook set up for document: {document_id}")
                
                # Store webhook info in monitored documents
                if document_id in self.monitored_documents:
                    self.monitored_documents[document_id].update({
                        'webhook': {
                            'channel_id': response.get('id', ''),
                            'resource_id': response.get('resourceId', ''),
                            'expiration': response.get('expiration', '')
                        }
                    })
                return True
            else:
                self.logger.error(f"Failed to set up webhook for document: {document_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Webhook setup error for document {document_id}: {e}")
            return False
    
    def _load_monitored_documents(self) -> None:
        """Load the list of monitored documents from settings."""
        docs = settings.get_monitored_documents()
        
        for doc in docs:
            doc_id = doc.get('id')
            if doc_id:
                self.monitored_documents[doc_id] = doc
                self.logger.info(f"Loaded monitored document: {doc.get('name', doc_id)}")


def signal_handler(sig, frame):
    """Handle termination signals to gracefully shut down."""
    print("\nShutting down CollabGPT...")
    if app and app.running:
        app.stop()
    sys.exit(0)


# Global application instance
app = None

def main():
    """Main entry point for the application."""
    global app
    
    logger.info("Starting CollabGPT")
    
    # Create and initialize the application
    app = CollabGPT()
    if not app.initialize():
        logger.error("Failed to initialize application")
        return 1
        
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start the application
    if not app.start():
        logger.error("Failed to start application")
        return 1
    
    try:
        # Keep the main thread alive
        while app.running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        pass
    finally:
        # Stop the application
        if app.running:
            app.stop()
    
    logger.info("CollabGPT terminated")
    return 0


if __name__ == "__main__":
    sys.exit(main())