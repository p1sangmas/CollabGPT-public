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
from .services.conflict_detector import ConflictDetector, ConflictType
from .services.activity_tracker import UserActivityTracker, ActivityType
from .services.comment_analyzer import CommentAnalyzer, CommentCategory
from .models.rag_system import RAGSystem
from .models.llm_interface import LLMInterface, LLMResponse
from .utils import logger
from .utils.performance import get_performance_monitor
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
        self.conflict_detector = None
        self.activity_tracker = None
        self.comment_analyzer = None
        self.rag_system = None
        self.llm_interface = None
        self.performance_monitor = get_performance_monitor()
        
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
            
            # Initialize Phase 2 components
            
            # Initialize conflict detector
            self.logger.info("Initializing Conflict Detector")
            self.conflict_detector = ConflictDetector(
                conflict_window_seconds=settings.COLLABORATION.get('conflict_window_seconds', 60)
            )
            
            # Initialize activity tracker
            self.logger.info("Initializing User Activity Tracker")
            self.activity_tracker = UserActivityTracker(
                activity_retention_days=settings.COLLABORATION.get('activity_retention_days', 14)
            )
            
            # Initialize comment analyzer
            self.logger.info("Initializing Comment Analyzer")
            self.comment_analyzer = CommentAnalyzer()
            
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
            self.logger.error(f"Initialization error: {e}", exc_info=True)
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
            self.logger.error(f"Startup error: {e}", exc_info=True)
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
                
        # Log performance metrics before shutdown
        if self.performance_monitor:
            metrics = self.performance_monitor.get_metrics()
            self.logger.info(f"Performance metrics at shutdown: {metrics}")
                
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
                
        # Process document comments if any
        if 'comments' in document:
            comments = document.get('comments', [])
            if comments:
                self.logger.info(f"Processing {len(comments)} comments for document {document_id}")
                comment_threads = self.comment_analyzer.process_comments(document_id, comments)
                analysis['comment_analysis'] = {
                    'thread_count': len(comment_threads),
                    'unresolved_threads': len(self.comment_analyzer.get_unresolved_threads(document_id)),
                    'categories': {cat.name: len(self.comment_analyzer.get_threads_by_category(document_id, cat)) 
                                  for cat in CommentCategory}
                }
                
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
            
        # Track system user viewing activity (for initial processing)
        self.activity_tracker.track_activity(
            user_id="system",
            document_id=document_id,
            activity_type=ActivityType.VIEW,
            sections=[section['title'] for section in analysis.get('sections', [])]
        )
        
        return analysis
    
    def process_document_changes(self, document_id: str, 
                                previous_content: str, 
                                current_content: str,
                                user_id: str = "unknown") -> Dict[str, Any]:
        """
        Process changes between document versions.
        
        Args:
            document_id: The document identifier
            previous_content: The previous version content
            current_content: The current version content
            user_id: The identifier of the user who made the changes
            
        Returns:
            Dictionary containing change analysis results
        """
        with self.performance_monitor.measure_latency("process_document_changes"):
            self.logger.info(f"Analyzing changes for document: {document_id}")
            
            # Analyze changes
            change_analysis = self.document_analyzer.analyze_changes(
                document_id, previous_content, current_content
            )
            
            # Record edit in conflict detector
            if self.conflict_detector:
                affected_sections = [section['title'] for section in 
                                    change_analysis.get('changes', {}).get('changed_sections', [])]
                
                edit_id = self.conflict_detector.record_edit(
                    document_id, 
                    user_id, 
                    previous_content, 
                    current_content,
                    affected_sections
                )
                
                # Check for conflicts
                conflicts = self.conflict_detector.get_conflicts(document_id)
                if conflicts:
                    self.logger.warning(f"Detected {len(conflicts)} conflicts in document {document_id}")
                    change_analysis['conflicts'] = [
                        {
                            'id': conflict.conflict_id,
                            'type': conflict.conflict_type.name,
                            'severity': conflict.severity,
                            'description': conflict.description,
                            'suggested_resolution': conflict.suggested_resolution
                        }
                        for conflict in conflicts
                    ]
            
            # Track user activity
            if self.activity_tracker:
                content_length_diff = len(current_content) - len(previous_content)
                affected_sections = [section['title'] for section in 
                                    change_analysis.get('changes', {}).get('changed_sections', [])]
                
                # Create metadata with content length instead of passing it as parameter
                activity_metadata = {
                    'content_length_diff': content_length_diff,
                    'importance': change_analysis.get('changes', {}).get('importance', {})
                }
                
                self.activity_tracker.track_activity(
                    user_id=user_id,
                    document_id=document_id,
                    activity_type=ActivityType.EDIT,
                    sections=affected_sections,
                    metadata=activity_metadata
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
                
            # Get section history from RAG system if available
            section_history = ""
            if self.rag_system:
                # Get document history for the specific section
                history_data = self.rag_system.get_document_history(document_id, section_title)
                
                if history_data and "sections" in history_data and history_data["sections"]:
                    section_history = "Document History:\n"
                    for section_info in history_data["sections"]:
                        section_history += f"- Section: {section_info['section']}\n"
                        section_history += f"- Current Version: {section_info['current_version']}\n"
                        
                        if section_info.get("changes"):
                            section_history += "- Previous changes:\n"
                            for change in section_info["changes"]:
                                section_history += f"  - Version {change['version']} ({change['timestamp']}): "
                                section_history += f"{change['content_snippet']}\n"
            
            # Generate suggestions for the specific section with history context
            return self.llm_interface.generate_with_template(
                "suggest_edits",
                section_title=section_title,
                section_content=section_content,
                section_history=section_history if section_history else "No previous versions available."
            )
        else:
            # Get relevant context from RAG with history included
            context = ""
            if self.rag_system:
                context = self.rag_system.get_relevant_context(
                    "document suggestions improvements", 
                    doc_id=document_id,
                    include_history=True
                )
            
            # Get collaboration context if available
            collab_context = self._get_collaboration_context(document_id)
            
            # Generate suggestions for the whole document
            prompt = (
                f"Please suggest improvements or additions to the following document.\n\n"
                f"Document content:\n{content[:2000]}...\n\n"  # Limit content length
            )
            
            if context:
                prompt += f"Additional context with document history:\n{context}\n\n"
                
            if collab_context:
                prompt += f"Collaboration context:\n{collab_context}\n\n"
                
            prompt += "Suggestions:"
            
            return self.llm_interface.generate(prompt)
    
    def get_document_activity(self, document_id: str, hours: int = 24) -> Dict[str, Any]:
        """
        Get activity information for a document.
        
        Args:
            document_id: The document identifier
            hours: Time window in hours
            
        Returns:
            Dictionary with activity information
        """
        if not self.activity_tracker:
            return {"error": "Activity tracker not initialized"}
            
        # Get active users
        active_users = self.activity_tracker.get_active_users(document_id, hours)
        
        # Get comment statistics if comment analyzer is available
        comment_stats = {}
        if self.comment_analyzer:
            comment_stats = self.comment_analyzer.get_comment_statistics(document_id)
            
        # Get conflicts if conflict detector is available
        conflicts = []
        if self.conflict_detector:
            conflicts = [
                {
                    'id': conflict.conflict_id,
                    'type': conflict.conflict_type.name,
                    'severity': conflict.severity,
                    'description': conflict.description,
                    'resolved': conflict.resolved
                }
                for conflict in self.conflict_detector.get_conflicts(document_id, resolved=True)
            ]
        
        return {
            'document_id': document_id,
            'window_hours': hours,
            'active_users': active_users,
            'comment_statistics': comment_stats,
            'conflicts': conflicts
        }
    
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
        with self.performance_monitor.measure_latency("handle_document_change"):
            document_id = change_data.get('document_id')
            current_content = change_data.get('current_content', '')
            previous_content = change_data.get('previous_content', '')
            
            # Extract user information if available
            metadata = change_data.get('metadata', {})
            user_id = metadata.get('user_id', 'unknown')
            
            self.logger.info(f"Document change detected: {document_id} by user {user_id}")
            
            # Skip processing if no significant changes
            if previous_content and current_content == previous_content:
                self.logger.info("No significant changes detected")
                return
                
            # Process the changes
            change_analysis = self.process_document_changes(
                document_id, previous_content, current_content, user_id
            )
            
            self.logger.info(f"Change analysis completed: {document_id}")
            
            # Extract importance level for logging
            importance = change_analysis.get('changes', {}).get('importance', {})
            importance_level = importance.get('level', 'UNKNOWN')
            
            self.logger.info(f"Change importance: {importance_level}")
            
            if 'ai_change_summary' in change_analysis:
                self.logger.info(f"AI summary: {change_analysis['ai_change_summary']}")
                
            # Check for conflicts
            if 'conflicts' in change_analysis and change_analysis['conflicts']:
                conflict_count = len(change_analysis['conflicts'])
                self.logger.warning(f"Detected {conflict_count} conflicts in document {document_id}")
                
                # Here you would implement notification logic for conflicts
                
            # Here you would implement notification logic, UI updates, etc.
            # This is a simple placeholder that logs the summary
    
    def _handle_document_comment(self, comment_data: Dict[str, Any]) -> None:
        """
        Handle a document comment event.
        
        Args:
            comment_data: Information about the document comment
        """
        with self.performance_monitor.measure_latency("handle_document_comment"):
            document_id = comment_data.get('document_id')
            comments = comment_data.get('comments', [])
            
            self.logger.info(f"Document comment event: {document_id}, {len(comments)} comments")
            
            # Process comments with the comment analyzer
            if self.comment_analyzer and comments:
                threads = self.comment_analyzer.process_comments(document_id, comments)
                self.logger.info(f"Processed {len(threads)} comment threads for document {document_id}")
                
                # Track user activity for each comment author
                if self.activity_tracker:
                    for comment in comments:
                        author = comment.get('author', {}).get('id')
                        if author:
                            self.activity_tracker.track_activity(
                                user_id=author,
                                document_id=document_id,
                                activity_type=ActivityType.COMMENT,
                                metadata=comment
                            )
            
            # Check if any comments are addressed to CollabGPT
            for comment in comments:
                content = comment.get('content', '').lower()
                author = comment.get('author', {}).get('id', 'unknown')
                
                # Look for mentions or commands
                if '@collabgpt' in content or '#collabgpt' in content:
                    self.logger.info(f"CollabGPT mentioned in comment by {author}: {content}")
                    # Here you would implement the logic to respond to comments
                    # This is a placeholder
                    
                    # Track as a user activity for the AI agent to help with contextual awareness
                    if self.activity_tracker:
                        self.activity_tracker.track_activity(
                            user_id="collabgpt",
                            document_id=document_id,
                            activity_type=ActivityType.COMMENT,
                            metadata={"reply_to": author}
                        )
    
    def _get_collaboration_context(self, document_id: str) -> str:
        """
        Get collaboration context for a document to inform the LLM.
        
        Args:
            document_id: The document identifier
            
        Returns:
            Collaborative context as a string
        """
        context_parts = []
        
        # Get active users if activity tracker is available
        if self.activity_tracker:
            active_users = self.activity_tracker.get_active_users(document_id, 24)
            if active_users:
                context_parts.append(f"Document has {len(active_users)} active users in the last 24 hours.")
                
                # Get most active user
                most_active = active_users[0] if active_users else None
                if most_active:
                    user_id = most_active.get('user_id')
                    context_parts.append(f"User {user_id} is most active with {most_active.get('total_activities', 0)} activities.")
                    
                    # Add focused sections
                    focused_sections = most_active.get('focused_sections', [])
                    if focused_sections:
                        context_parts.append(f"Main focus has been on sections: {', '.join(focused_sections[:3])}.")
        
        # Get unresolved comments if comment analyzer is available
        if self.comment_analyzer:
            unresolved = self.comment_analyzer.get_unresolved_threads(document_id)
            if unresolved:
                context_parts.append(f"Document has {len(unresolved)} unresolved comment threads.")
                
                # Categorize by type
                questions = [t for t in unresolved if CommentCategory.QUESTION in t.categories]
                suggestions = [t for t in unresolved if CommentCategory.SUGGESTION in t.categories]
                
                if questions:
                    context_parts.append(f"There are {len(questions)} unanswered questions.")
                if suggestions:
                    context_parts.append(f"There are {len(suggestions)} pending suggestions.")
        
        # Get conflicts if conflict detector is available
        if self.conflict_detector:
            conflicts = self.conflict_detector.get_conflicts(document_id)
            if conflicts:
                context_parts.append(f"Document has {len(conflicts)} unresolved editing conflicts.")
                
                # Add most severe conflict
                severe_conflicts = [c for c in conflicts if c.severity >= 4]
                if severe_conflicts:
                    context_parts.append(f"Most severe conflict: {severe_conflicts[0].description}")
        
        # Combine all context parts
        if context_parts:
            return "\n".join(context_parts)
        else:
            return ""
    
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