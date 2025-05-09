"""
Webhook handler for CollabGPT.

This module handles incoming webhooks from Google Docs API,
processes document change notifications, and triggers the appropriate actions.
"""

import json
import hashlib
import hmac
import time
import asyncio
import os
import ssl
import http.client
import threading
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from ..api.google_docs import GoogleDocsAPI, DocumentContentFetcher
from ..utils import logger
from ..utils.performance import get_performance_monitor, measure_latency


class WebhookHandler:
    """
    Handler for processing webhooks from Google Docs API.
    """
    
    def __init__(self, google_docs_api: GoogleDocsAPI, secret_key: str = None):
        """
        Initialize the webhook handler.
        
        Args:
            google_docs_api: An authenticated GoogleDocsAPI instance
            secret_key: Optional secret key for webhook verification
        """
        self.google_docs_api = google_docs_api
        self.secret_key = secret_key
        self.callbacks = {}
        self.document_history = {}
        self.logger = logger.get_logger("webhook_handler")
        self.performance_monitor = get_performance_monitor()
        self.thread_pool = ThreadPoolExecutor(max_workers=5)  # Pool for parallel processing
        self.latency_callback_queue = asyncio.Queue()
        self.content_fetcher = DocumentContentFetcher(google_docs_api, self.logger)
        self.fetch_lock = threading.Lock()  # To prevent multiple simultaneous fetches for the same document
        self.ongoing_fetches = {}  # document_id -> request_id
        
    def register_callback(self, event_type: str, callback: Callable) -> None:
        """
        Register a callback function for a specific event type.
        
        Args:
            event_type: Type of event (e.g., 'change', 'comment', 'suggest')
            callback: Function to call when event occurs
        """
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        self.callbacks[event_type].append(callback)
        self.logger.info(f"Registered callback for event type: {event_type}")
    
    def verify_signature(self, payload: bytes, signature: str) -> bool:
        """
        Verify the webhook signature if a secret key is set.
        
        Args:
            payload: The raw webhook payload
            signature: The signature from the webhook header
            
        Returns:
            True if signature is valid, False otherwise
        """
        if not self.secret_key:
            return True  # No verification if no secret key is set
            
        computed_signature = hmac.new(
            self.secret_key.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(computed_signature, signature)
    
    def process_webhook(self, headers: Dict[str, str], payload: bytes) -> bool:
        """
        Process an incoming webhook from Google Docs API.
        
        Args:
            headers: The webhook request headers
            payload: The raw webhook payload
            
        Returns:
            True if the webhook was processed successfully, False otherwise
        """
        with measure_latency("webhook_processing", self.performance_monitor):
            # Debug information
            self.logger.info(f"Received webhook with headers: {headers}")
            
            # Check if this is a Google Drive push notification (case-insensitive header check)
            google_headers = ['X-Goog-Channel', 'X-Goog-Resource']
            is_google_push = any(any(key.lower().startswith(h.lower()) for h in google_headers) for key in headers.keys())
            
            if is_google_push:
                self.logger.info("Identified as Google Drive push notification")
                return self.process_google_push_notification(headers)
                
            # For standard webhooks with JSON payload
            try:
                payload_str = payload.decode('utf-8') if payload else ""
                self.logger.info(f"Webhook payload: {payload_str[:200]}..." if len(payload_str) > 200 else payload_str)
                
                # Verify signature if needed
                if 'X-Goog-Signature' in headers and not self.verify_signature(
                        payload, headers['X-Goog-Signature']):
                    self.logger.error("Invalid webhook signature")
                    return False
                    
                # Parse the payload
                data = json.loads(payload_str) if payload_str else {}
                
                # Extract document ID from resource URI
                # Format: https://www.googleapis.com/drive/v3/files/DOCUMENT_ID
                resource_uri = data.get('resourceUri', '')
                document_id = resource_uri.split('/')[-1] if resource_uri else None
                
                if not document_id:
                    self.logger.error("No document ID found in webhook payload")
                    return False
                    
                # Process based on event type
                event_type = data.get('eventType', '').lower()
                
                # Process the webhook asynchronously to minimize response latency
                # This allows us to return a response to Google quickly while processing continues
                if event_type == 'change':
                    self.logger.info(f"Processing change event for document {document_id}")
                    self.thread_pool.submit(self._process_document_change, document_id, data)
                elif event_type == 'comment':
                    self.logger.info(f"Processing comment event for document {document_id}")
                    self.thread_pool.submit(self._process_document_comment, document_id, data)
                else:
                    self.logger.warning(f"Unhandled event type: {event_type}")
                    
                return True
                
            except json.JSONDecodeError:
                self.logger.error(f"Invalid JSON payload in webhook: {payload}")
                return False
            except Exception as e:
                self.logger.error(f"Error processing webhook: {e}", exc_info=True)
                return False
    
    def process_google_push_notification(self, headers: Dict[str, str]) -> bool:
        """
        Process a Google Drive push notification.
        
        Args:
            headers: The webhook request headers containing Google-specific headers
            
        Returns:
            True if the notification was processed successfully, False otherwise
        """
        with measure_latency("google_push_notification", self.performance_monitor):
            try:
                # Extract key headers
                channel_id = headers.get('X-Goog-Channel-ID', headers.get('X-Goog-Channel-Id', ''))
                resource_id = headers.get('X-Goog-Resource-ID', headers.get('X-Goog-Resource-Id', ''))
                resource_state = headers.get('X-Goog-Resource-State', '')
                resource_uri = headers.get('X-Goog-Resource-URI', headers.get('X-Goog-Resource-Uri', ''))
                changed = headers.get('X-Goog-Changed', '')
                token = headers.get('X-Goog-Channel-Token', '')  # Document ID sent in token
                
                # Log the complete headers for debugging
                self.logger.info(f"Complete Google notification headers: {headers}")
                
                # Log the individual extracted headers
                self.logger.info(f"Google notification details - State: {resource_state}, Channel: {channel_id}, Resource: {resource_id}, URI: {resource_uri}, Token: {token}")
                if changed:
                    self.logger.info(f"Changed components: {changed}")
                
                # Handle sync notification (webhook setup verification)
                if resource_state == 'sync':
                    self.logger.info("Received webhook sync notification (verification)")
                    return True
                
                # Extract document ID from token if available, otherwise from URI
                doc_id = None
                
                # First try to get document ID from token
                if token:
                    doc_id = token
                    self.logger.info(f"Extracted document ID from token: {doc_id}")
                # Then try from resource URI
                elif resource_uri:
                    # Extract document ID from the resource URI
                    # URI format: https://www.googleapis.com/drive/v3/files/document_id
                    doc_id = resource_uri.split('/')[-1].split('?')[0]  # Handle any query parameters
                    self.logger.info(f"Extracted document ID from URI: {doc_id}")
                
                # If we couldn't extract the document ID, check if the channel ID contains it
                if not doc_id and channel_id and 'collabgpt-channel-' in channel_id:
                    # Extract doc ID from channel ID format: collabgpt-channel-{doc_id}-{timestamp}
                    parts = channel_id.split('-')
                    if len(parts) > 2:
                        # The document ID should be after "collabgpt-channel-"
                        potential_id = parts[2]
                        if potential_id:
                            doc_id = potential_id
                            self.logger.info(f"Extracted document ID from channel ID: {doc_id}")
                
                if not doc_id:
                    self.logger.error("No document ID found in notification")
                    return False
                
                self.logger.info(f"Document notification for doc_id={doc_id}")
                
                # For 'change' or 'update' state, process document changes
                if resource_state in ['change', 'update']:
                    self.logger.info(f"Processing document change from Google notification for doc: {doc_id}")
                    # Process asynchronously to minimize latency
                    self.thread_pool.submit(
                        self._process_document_change, 
                        doc_id, 
                        {'eventType': 'change', 'source': 'google_push', 'changed': changed}
                    )
                    return True
                else:
                    self.logger.warning(f"Unhandled resource state: {resource_state}")
                    return True  # Still return true to acknowledge receipt
                    
            except Exception as e:
                self.logger.error(f"Error processing Google notification: {e}", exc_info=True)
                return False
    
    def _process_document_change(self, document_id: str, data: Dict[str, Any] = None) -> None:
        """
        Process a document change event.
        
        Args:
            document_id: The ID of the changed document
            data: Additional data about the change
        """
        self.logger.info(f"Processing document change for {document_id}")
        
        try:
            # Get the previous content from cache if available
            previous_content = ""
            previous_content_path = os.path.join(
                "data", f"test_document_{document_id}_content.txt"
            )
            if os.path.exists(previous_content_path):
                with open(previous_content_path, 'r', encoding='utf-8') as f:
                    previous_content = f.read()
            
            # Get the current content using our continuous read operation pattern
            # This will handle the timeout errors gracefully
            current_content = self._get_document_content_continuous(document_id)
            
            if not current_content:
                self.logger.error(f"Failed to retrieve content for document {document_id} after multiple attempts")
                return
                
            self.logger.debug(f"Retrieved document content: {len(current_content)} characters")
            
            # Save current content for future reference
            with open(previous_content_path, 'w', encoding='utf-8') as f:
                f.write(current_content)
                
            # Store current content in history
            self.document_history[document_id] = {
                'content': current_content,
                'last_updated': datetime.now().isoformat()
            }
            
            # Skip if no previous content (first notification)
            if not previous_content:
                self.logger.info(f"First change notification, no previous content to compare")
                # Still call callbacks with empty previous content
            
            # Call registered callbacks
            if 'change' in self.callbacks:
                change_data = {
                    'document_id': document_id,
                    'current_content': current_content,
                    'previous_content': previous_content,
                    'timestamp': time.time(),
                    'metadata': data or {}
                }
                
                self.logger.info(f"Calling {len(self.callbacks['change'])} registered change callbacks")
                
                for callback in self.callbacks['change']:
                    try:
                        with measure_latency("handle_document_change", self.performance_monitor):
                            self.logger.debug(f"Calling change callback: {callback.__name__ if hasattr(callback, '__name__') else 'anonymous'}")
                            callback(change_data)
                            self.logger.debug("Callback executed successfully")
                    except Exception as e:
                        self.logger.error(f"Error in change callback: {e}", exc_info=True)
                        
            else:
                self.logger.warning("No callbacks registered for 'change' events")
                
        except Exception as e:
            self.logger.error(f"Error processing document change: {e}", exc_info=True)
    
    def _process_document_comment(self, document_id: str, data: Dict[str, Any]) -> None:
        """
        Process a document comment event.
        
        Args:
            document_id: The ID of the document with new comments
            data: Additional data about the comments
        """
        self.logger.info(f"Processing document comment for {document_id}")
        
        try:
            # Get comments from the document
            self.logger.debug(f"Fetching comments for document {document_id}")
            comments = self.google_docs_api.get_document_comments(document_id)
            
            if not comments:
                self.logger.warning(f"No comments found for document {document_id}")
                return
                
            self.logger.debug(f"Retrieved {len(comments)} comments")
                
            # Call registered callbacks
            if 'comment' in self.callbacks:
                comment_data = {
                    'document_id': document_id,
                    'comments': comments,
                    'timestamp': time.time(),
                    'metadata': data
                }
                
                self.logger.info(f"Calling {len(self.callbacks['comment'])} registered comment callbacks")
                
                for callback in self.callbacks['comment']:
                    try:
                        self.logger.debug(f"Calling comment callback: {callback.__name__ if hasattr(callback, '__name__') else 'anonymous'}")
                        callback(comment_data)
                        self.logger.debug("Callback executed successfully")
                    except Exception as e:
                        self.logger.error(f"Error in comment callback: {e}", exc_info=True)
                        
            else:
                self.logger.warning("No callbacks registered for 'comment' events")
                
        except Exception as e:
            self.logger.error(f"Error processing document comment: {e}", exc_info=True)
    
    def simulate_change_event(self, document_id: str) -> None:
        """
        Simulate a document change event for testing or manual triggering.
        
        Args:
            document_id: The ID of the document to simulate a change for
        """
        self.logger.info(f"Simulating change event for document {document_id}")
        # Use a special metadata attribute to identify this as a simulated change
        self._process_document_change(
            document_id, 
            {
                'eventType': 'change', 
                'simulated': True,
                'user_id': 'simulator', 
                'timestamp': datetime.now().isoformat()
            }
        )
        self.logger.info(f"Simulation completed for document {document_id}")

    def _get_document_content_continuous(self, document_id: str) -> str:
        """
        Get document content using the continuous read operation pattern.
        This method provides fault tolerance against timeouts and connection issues.
        
        Args:
            document_id: The ID of the document to retrieve
            
        Returns:
            The document content as text, or empty string on complete failure
        """
        try:
            with self.fetch_lock:  # Ensure we don't start multiple fetches for the same document
                # Check if we already have an ongoing fetch for this document
                if document_id in self.ongoing_fetches:
                    request_id = self.ongoing_fetches[document_id]
                    self.logger.info(f"Using existing fetch request {request_id} for document {document_id}")
                else:
                    # Start a new asynchronous fetch
                    self.logger.info(f"Starting continuous read operation for document {document_id}")
                    request_id, _ = self.content_fetcher.fetch_content_async(document_id, max_retries=5)
                    self.ongoing_fetches[document_id] = request_id
            
            # Wait for the content with a reasonable timeout
            content = self.content_fetcher.get_content(request_id, timeout=300)  # 5 minutes max wait
            
            # Clean up completed request
            with self.fetch_lock:
                if document_id in self.ongoing_fetches and self.ongoing_fetches[document_id] == request_id:
                    del self.ongoing_fetches[document_id]
                self.content_fetcher.cleanup(request_id)
            
            if content:
                self.logger.info(f"Successfully retrieved content for document {document_id} ({len(content)} characters)")
                return content
            else:
                self.logger.error(f"Failed to retrieve content for document {document_id} after multiple attempts")
                return ""
                
        except Exception as e:
            self.logger.error(f"Error in continuous read operation for document {document_id}: {e}", exc_info=True)
            return ""