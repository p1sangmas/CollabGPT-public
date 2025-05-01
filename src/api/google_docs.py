"""
Google Docs API Integration for CollabGPT.

This module handles all interactions with the Google Docs API, including:
- Authentication and access management
- Document retrieval and modification
- Real-time change monitoring
- Comment and suggestion management
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

class GoogleDocsAPI:
    """
    A class to handle all interactions with the Google Docs API.
    """
    
    # Define the scopes needed for Google Docs API
    SCOPES = [
        'https://www.googleapis.com/auth/documents',
        'https://www.googleapis.com/auth/drive',
        'https://www.googleapis.com/auth/drive.file',
    ]
    
    def __init__(self, credentials_path: str = None):
        """
        Initialize the GoogleDocsAPI with optional credentials path.
        
        Args:
            credentials_path: Path to the credentials JSON file (either service account or OAuth client)
        """
        self.credentials_path = credentials_path
        self.credentials = None
        self.service = None
        self.drive_service = None
    
    def authenticate(self, use_service_account: bool = False) -> bool:
        """
        Authenticate with the Google Docs API using either OAuth2 or service account.
        
        Args:
            use_service_account: Whether to use service account authentication instead of OAuth
            
        Returns:
            bool: True if authentication was successful, False otherwise
        """
        try:
            if use_service_account and self.credentials_path:
                self.credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path, scopes=self.SCOPES)
            else:
                token_path = 'token.json'
                if os.path.exists(token_path):
                    self.credentials = Credentials.from_authorized_user_info(
                        json.loads(open(token_path).read()))
                
                # If credentials don't exist or are invalid, run the OAuth flow
                if not self.credentials or not self.credentials.valid:
                    if self.credentials and self.credentials.expired and self.credentials.refresh_token:
                        self.credentials.refresh(Request())
                    else:
                        if not self.credentials_path:
                            raise ValueError("No credentials path provided for OAuth flow")
                        
                        flow = InstalledAppFlow.from_client_secrets_file(
                            self.credentials_path, self.SCOPES)
                        self.credentials = flow.run_local_server(port=0)
                    
                    # Save credentials for future use
                    with open(token_path, 'w') as token:
                        token.write(self.credentials.to_json())
            
            # Build the services using the authenticated credentials
            self.service = build('docs', 'v1', credentials=self.credentials)
            self.drive_service = build('drive', 'v3', credentials=self.credentials)
            
            return True
            
        except Exception as e:
            print(f"Authentication error: {e}")
            return False
    
    def get_document(self, document_id: str) -> Dict[str, Any]:
        """
        Retrieve a document by its ID.
        
        Args:
            document_id: The Google Doc ID
            
        Returns:
            The document data as a dictionary
        """
        try:
            if not self.service:
                raise ValueError("Not authenticated. Call authenticate() first.")
                
            return self.service.documents().get(documentId=document_id).execute()
        except HttpError as error:
            print(f"Error retrieving document: {error}")
            return {}
    
    def get_document_content(self, document_id: str) -> str:
        """
        Extract the plain text content from a document.
        
        Args:
            document_id: The Google Doc ID
            
        Returns:
            The document content as plain text
        """
        doc = self.get_document(document_id)
        if not doc:
            return ""
        
        content = []
        for elem in doc.get('body', {}).get('content', []):
            if 'paragraph' in elem:
                for para_elem in elem['paragraph']['elements']:
                    if 'textRun' in para_elem:
                        content.append(para_elem['textRun']['content'])
        
        return ''.join(content)
    
    def update_document(self, document_id: str, requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Make changes to a document via the API.
        
        Args:
            document_id: The Google Doc ID
            requests: List of update requests as per Google Docs API format
            
        Returns:
            The response from the API
        """
        try:
            if not self.service:
                raise ValueError("Not authenticated. Call authenticate() first.")
                
            return self.service.documents().batchUpdate(
                documentId=document_id,
                body={'requests': requests}
            ).execute()
        except HttpError as error:
            print(f"Error updating document: {error}")
            return {}
    
    def watch_document(self, document_id: str, webhook_url: str) -> Dict[str, Any]:
        """
        Set up a webhook to monitor changes to a document.
        Note: This uses Drive API's watch method as Docs API doesn't have direct change notification.
        
        Args:
            document_id: The Google Doc ID
            webhook_url: URL to receive notifications
            
        Returns:
            The response from the API with the channel information
        """
        try:
            if not self.drive_service:
                raise ValueError("Not authenticated. Call authenticate() first.")
            
            # Create a notification channel
            channel_body = {
                'id': f'collabgpt-channel-{document_id}-{int(datetime.now().timestamp())}',
                'type': 'web_hook',
                'address': webhook_url,
            }
            
            response = self.drive_service.files().watch(
                fileId=document_id,
                body=channel_body
            ).execute()
            
            return response
            
        except HttpError as error:
            print(f"Error setting up document watch: {error}")
            return {}
    
    def stop_watching_document(self, channel_id: str, resource_id: str) -> bool:
        """
        Stop monitoring changes to a document.
        
        Args:
            channel_id: The channel ID returned by watch_document
            resource_id: The resource ID returned by watch_document
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.drive_service:
                raise ValueError("Not authenticated. Call authenticate() first.")
                
            self.drive_service.channels().stop(
                body={
                    'id': channel_id,
                    'resourceId': resource_id
                }
            ).execute()
            
            return True
            
        except HttpError as error:
            print(f"Error stopping document watch: {error}")
            return False
    
    def get_document_comments(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all comments from a document.
        
        Args:
            document_id: The Google Doc ID
            
        Returns:
            List of comment data
        """
        try:
            if not self.drive_service:
                raise ValueError("Not authenticated. Call authenticate() first.")
                
            # Use the Drive API to retrieve comments
            comments = []
            page_token = None
            
            while True:
                response = self.drive_service.comments().list(
                    fileId=document_id,
                    fields='comments,nextPageToken',
                    pageToken=page_token
                ).execute()
                
                comments.extend(response.get('comments', []))
                page_token = response.get('nextPageToken')
                
                if not page_token:
                    break
                    
            return comments
            
        except HttpError as error:
            print(f"Error retrieving comments: {error}")
            return []
    
    def add_comment(self, document_id: str, content: str, location: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a comment to a document at the specified location.
        
        Args:
            document_id: The Google Doc ID
            content: Comment text content
            location: Location descriptor (e.g., {'index': 5} for character position)
            
        Returns:
            Comment data if successful, empty dict otherwise
        """
        try:
            if not self.drive_service:
                raise ValueError("Not authenticated. Call authenticate() first.")
                
            return self.drive_service.comments().create(
                fileId=document_id,
                body={
                    'content': content,
                    'anchor': location
                }
            ).execute()
            
        except HttpError as error:
            print(f"Error adding comment: {error}")
            return {}
            
    def get_document_revision_history(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve the revision history of a document.
        
        Args:
            document_id: The Google Doc ID
            
        Returns:
            List of revision data
        """
        try:
            if not self.drive_service:
                raise ValueError("Not authenticated. Call authenticate() first.")
                
            revisions = []
            page_token = None
            
            while True:
                response = self.drive_service.revisions().list(
                    fileId=document_id,
                    fields='revisions(id,modifiedTime,lastModifyingUser),nextPageToken',
                    pageToken=page_token
                ).execute()
                
                revisions.extend(response.get('revisions', []))
                page_token = response.get('nextPageToken')
                
                if not page_token:
                    break
                    
            return revisions
            
        except HttpError as error:
            print(f"Error retrieving revision history: {error}")
            return []