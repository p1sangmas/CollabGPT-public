#!/usr/bin/env python3
"""
Automatic webhook registration and management for Google Docs integration.
This module provides functionality to automatically set up and manage webhooks
for Google Docs, eliminating the need for manual ngrok URL updates.
"""

import os
import json
import uuid
import socket
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta
from pyngrok import ngrok

from src.utils.logger import get_logger
from src.api.google_docs import GoogleDocsAPI

class WebhookManager:
    """
    Manager for automatically registering and maintaining Google Docs webhooks.
    Eliminates the need for manual ngrok URL updates by automatically detecting
    the server's URL and registering webhooks for all monitored documents.
    """
    
    def __init__(self, google_docs_api: GoogleDocsAPI, webhook_endpoint: str = "/webhook", 
                 port: int = 5001, monitored_docs_path: str = None):
        """
        Initialize the webhook manager.
        
        Args:
            google_docs_api: An authenticated GoogleDocsAPI instance
            webhook_endpoint: The endpoint that will receive webhook notifications
            port: The port the server is running on
            monitored_docs_path: Path to the monitored documents JSON file
        """
        self.google_docs_api = google_docs_api
        self.webhook_endpoint = webhook_endpoint
        self.port = port
        self.monitored_docs_path = monitored_docs_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'data', 'monitored_documents.json'
        )
        self.log = get_logger("webhook_manager")
        self.base_url = self._detect_webhook_url()
        self.full_webhook_url = f"{self.base_url}{webhook_endpoint}"
        
        self.log.info(f"Webhook manager initialized with URL: {self.full_webhook_url}")
    
    def _detect_webhook_url(self) -> str:
        """
        Detect the server's public URL.
        
        Returns:
            The base URL for webhooks
        """

            
        # Check if NGROK_AUTHTOKEN is set
        ngrok_token = os.environ.get('NGROK_AUTHTOKEN')
        if ngrok_token:
            try:
                # Configure ngrok with the auth token
                ngrok.set_auth_token(ngrok_token)
                
                # Start ngrok tunnel to the webhook port
                self.log.info(f"Starting ngrok tunnel to port {self.port}")
                tunnel = ngrok.connect(self.port, bind_tls=True)
                
                # Extract the public URL
                public_url = tunnel.public_url
                self.log.info(f"Ngrok tunnel established: {public_url}")
                
                # Save the tunnel for later cleanup
                self.ngrok_tunnel = tunnel
                
                return public_url
            except Exception as e:
                self.log.error(f"Failed to establish ngrok tunnel: {e}")
        else:
            self.log.warning("NGROK_AUTHTOKEN not set in environment variables. For webhook registration to work, set this variable.")
            
        # Try to get the public IP address for this server
        try:
            response = requests.get("https://api.ipify.org", timeout=5)
            if response.status_code == 200:
                public_ip = response.text
                self.log.info(f"Detected public IP: {public_ip}")
                self.log.warning("Using HTTP URL which Google Drive API will reject. Set up NGROK_AUTHTOKEN for HTTPS URLs.")
                return f"http://{public_ip}:{self.port}"
        except Exception as e:
            self.log.warning(f"Could not detect public IP: {e}")
            
        # Fall back to localhost with a warning
        self.log.warning("Could not detect public URL. Using localhost, which will not work for Google Drive webhooks.")
        self.log.warning("For webhooks to work, set NGROK_AUTHTOKEN environment variable.")
        return f"http://localhost:{self.port}"
    
    def _load_monitored_documents(self) -> Dict[str, Any]:
        """
        Load the list of monitored documents from storage.
        
        Returns:
            Dictionary of monitored documents
        """
        try:
            if os.path.exists(self.monitored_docs_path):
                with open(self.monitored_docs_path, 'r') as f:
                    data = json.load(f)
                    # Handle both formats: either a list of documents or a dict with "documents" key
                    if isinstance(data, list):
                        return {"documents": data}
                    return data
            else:
                return {"documents": []}
        except Exception as e:
            self.log.error(f"Error loading monitored documents: {e}")
            return {"documents": []}
    
    def _save_monitored_documents(self, data: Dict[str, Any]) -> bool:
        """
        Save the list of monitored documents to storage.
        
        Args:
            data: Dictionary of monitored documents
            
        Returns:
            True if save was successful, False otherwise
        """
        try:
            # Make sure the directory exists
            os.makedirs(os.path.dirname(self.monitored_docs_path), exist_ok=True)
            
            with open(self.monitored_docs_path, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            self.log.error(f"Error saving monitored documents: {e}")
            return False
    
    def register_all_webhooks(self) -> bool:
        """
        Register webhooks for all monitored documents.
        
        Returns:
            True if all webhooks were registered successfully, False otherwise
        """
        monitored_documents = self._load_monitored_documents()
        documents = monitored_documents.get("documents", [])
        
        if not documents:
            self.log.warning("No monitored documents found to register webhooks for")
            return True  # No documents to register is technically a success
        
        self.log.info(f"Registering webhooks for {len(documents)} documents")
        success = True
        
        # Process each document
        for doc in documents:
            doc_id = doc.get("id")
            if not doc_id:
                continue
                
            try:
                # Check if webhook needs refreshing
                webhook_info = doc.get("webhook", {})
                expires_at = webhook_info.get("expiration")
                webhook_url = webhook_info.get("webhook_url", "")
                
                needs_refresh = (
                    not webhook_info or
                    not expires_at or
                    datetime.fromisoformat(expires_at) <= datetime.now() + timedelta(days=1) or
                    webhook_url != self.full_webhook_url
                )
                
                if needs_refresh:
                    self.log.info(f"Refreshing webhook for document {doc_id}")
                    channel_id = self._register_webhook_for_document(doc_id)
                    if channel_id:
                        # Update the document's webhook info
                        doc["webhook"] = {
                            "channel_id": channel_id,
                            "webhook_url": self.full_webhook_url,
                            "expiration": (datetime.now() + timedelta(days=7)).isoformat(),
                            "last_updated": datetime.now().isoformat()
                        }
                    else:
                        self.log.error(f"Failed to register webhook for document {doc_id}")
                        success = False
                else:
                    self.log.info(f"Webhook for document {doc_id} is still valid, no refresh needed")
            except Exception as e:
                self.log.error(f"Error registering webhook for document {doc_id}: {e}")
                success = False
                
        # Save the updated monitored documents info
        self._save_monitored_documents(monitored_documents)
        return success
    
    def _register_webhook_for_document(self, doc_id: str) -> Optional[str]:
        """
        Register a webhook for a specific document.
        
        Args:
            doc_id: Google Doc ID
            
        Returns:
            Channel ID if successful, None otherwise
        """
        try:
            # Generate a unique channel ID
            channel_id = str(uuid.uuid4())
            
            # Register the webhook
            self.log.info(f"Registering webhook for document {doc_id} with channel {channel_id} at {self.full_webhook_url}")
            
            # Use the Google Docs API to register the webhook with token for verification
            # Include WEBHOOK_SECRET_KEY from environment if available
            secret_token = os.environ.get('WEBHOOK_SECRET_KEY', doc_id)
            
            result = self.google_docs_api.watch_document(
                document_id=doc_id,
                webhook_url=self.full_webhook_url
            )
            
            if result and 'id' in result:
                self.log.info(f"Successfully registered webhook for document {doc_id}")
                
                # Also add the resource_id to the webhook info
                resource_id = result.get('resourceId', '')
                
                # Store the webhook details in a file for reference and troubleshooting
                webhook_details_path = os.path.join(
                    os.path.dirname(self.monitored_docs_path),
                    f"webhook_subscription_{doc_id}.txt"
                )
                
                os.makedirs(os.path.dirname(webhook_details_path), exist_ok=True)
                
                with open(webhook_details_path, "w") as f:
                    f.write(f"Document ID: {doc_id}\n")
                    f.write(f"Channel ID: {result.get('id')}\n")
                    f.write(f"Resource ID: {resource_id}\n")
                    f.write(f"Webhook URL: {self.full_webhook_url}\n")
                    f.write(f"Created: {datetime.now().isoformat()}\n")
                    f.write(f"Expiration: {result.get('expiration', 'Unknown')}\n")
                
                return channel_id
            else:
                self.log.error(f"Failed to register webhook for document {doc_id}: {result}")
                return None
        except Exception as e:
            self.log.error(f"Error in _register_webhook_for_document: {e}")
            return None
    
    def add_document(self, doc_id: str, doc_name: str = "") -> bool:
        """
        Add a document to be monitored and register a webhook for it.
        
        Args:
            doc_id: Google Doc ID
            doc_name: Name of the document (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load current monitored documents
            monitored_documents = self._load_monitored_documents()
            
            # Check if document is already being monitored
            doc_exists = False
            for doc in monitored_documents.get("documents", []):
                if doc.get("id") == doc_id:
                    doc_exists = True
                    break
            
            if not doc_exists:
                # Register webhook for the document
                channel_id = self._register_webhook_for_document(doc_id)
                if not channel_id:
                    return False
                
                # Add document to the monitored list
                doc_info = {
                    "id": doc_id,
                    "name": doc_name,
                    "added_at": datetime.now().isoformat(),
                    "webhook": {
                        "channel_id": channel_id,
                        "webhook_url": self.full_webhook_url,
                        "expiration": (datetime.now() + timedelta(days=7)).isoformat(),
                        "last_updated": datetime.now().isoformat()
                    }
                }
                
                monitored_documents.setdefault("documents", []).append(doc_info)
                
                # Save updated list
                if self._save_monitored_documents(monitored_documents):
                    self.log.info(f"Document {doc_id} added to monitored list")
                    return True
                else:
                    self.log.error(f"Failed to save updated monitored documents list")
                    return False
            else:
                self.log.info(f"Document {doc_id} is already being monitored")
                # Document already exists, refresh its webhook
                return self.register_all_webhooks()
        except Exception as e:
            self.log.error(f"Error adding document {doc_id}: {e}")
            return False