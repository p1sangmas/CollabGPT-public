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
import time
import ssl
import http.client
import socket
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable, Tuple
from queue import Queue, Empty
import httplib2

from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.errors import HttpError
from googleapiclient.http import build_http

# Set longer timeout for API requests
socket.setdefaulttimeout(180)  # 3 minutes

class DocumentContentFetcher:
    """Helper class to fetch document content with continuous retry mechanism"""
    
    def __init__(self, google_docs_api, logger):
        self.google_docs_api = google_docs_api
        self.logger = logger
        self.content_queue = Queue()
        self.error_queue = Queue()
        self.active_fetches = {}  # document_id -> thread
        
    def fetch_content_async(self, document_id: str, max_retries: int = 5) -> Tuple[str, threading.Thread]:
        """
        Start an asynchronous fetch of document content with automatic retries
        
        Args:
            document_id: The Google Doc ID to fetch
            max_retries: Maximum number of retries
            
        Returns:
            A tuple with the request ID and the thread handling the request
        """
        # Create unique request ID for this fetch
        request_id = f"{document_id}_{int(time.time())}"
        
        # Create and start the fetch thread
        fetch_thread = threading.Thread(
            target=self._fetch_with_retries,
            args=(document_id, request_id, max_retries),
            daemon=True
        )
        
        self.active_fetches[request_id] = fetch_thread
        fetch_thread.start()
        
        return request_id, fetch_thread
        
    def _fetch_with_retries(self, document_id: str, request_id: str, max_retries: int) -> None:
        """
        Fetch document content with automatic retries
        
        Args:
            document_id: The Google Doc ID
            request_id: Unique identifier for this request
            max_retries: Maximum number of retries
        """
        retry_count = 0
        base_delay = 2  # Starting delay in seconds
        
        while retry_count <= max_retries:
            try:
                content = self.google_docs_api.get_document_content(document_id)
                # Put successful result in the queue
                self.content_queue.put((request_id, content))
                return
            except (ssl.SSLError, http.client.RemoteDisconnected, ConnectionError, 
                    TimeoutError, socket.timeout) as e:
                retry_count += 1
                
                if retry_count <= max_retries:
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** (retry_count - 1)) + (hash(document_id) % 2)
                    self.logger.warning(f"Content fetch failed (attempt {retry_count}/{max_retries}): {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    self.logger.error(f"Failed to fetch content for document {document_id} after {max_retries} attempts")
                    # Put the error in the queue
                    self.error_queue.put((request_id, str(e)))
                    return
        
    def get_content(self, request_id: str, timeout: int = 300) -> Optional[str]:
        """
        Get the fetched content or wait for it to complete
        
        Args:
            request_id: The request ID from fetch_content_async
            timeout: Maximum time to wait in seconds
            
        Returns:
            The document content or None if timeout or error
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check if content is available
            try:
                result_id, content = self.content_queue.get(block=False)
                if result_id == request_id:
                    return content
                else:
                    # Put it back for another consumer
                    self.content_queue.put((result_id, content))
            except Empty:
                pass
                
            # Check if error is available
            try:
                error_id, error = self.error_queue.get(block=False)
                if error_id == request_id:
                    self.logger.error(f"Error fetching content for request {request_id}: {error}")
                    return None
                else:
                    # Put it back for another consumer
                    self.error_queue.put((error_id, error))
            except Empty:
                pass
                
            # Check if thread is still alive
            if request_id in self.active_fetches and not self.active_fetches[request_id].is_alive():
                # Thread died without putting result in queue
                self.logger.error(f"Fetch thread for {request_id} terminated without result")
                if request_id in self.active_fetches:
                    del self.active_fetches[request_id]
                return None
                
            # Short sleep to avoid CPU burning
            time.sleep(0.1)
            
        # Timeout reached
        self.logger.error(f"Timeout waiting for document content (request: {request_id})")
        return None
        
    def cleanup(self, request_id: str) -> None:
        """
        Clean up resources for a completed request
        
        Args:
            request_id: The request ID to clean up
        """
        if request_id in self.active_fetches:
            # Don't wait for thread - daemon threads will be terminated on app exit
            del self.active_fetches[request_id]


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
        self._content_cache = {}  # document_id -> (timestamp, content)
        self._cache_ttl = 30  # seconds
    
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
                    try:
                        self.credentials = Credentials.from_authorized_user_info(
                            json.loads(open(token_path).read()))
                    except Exception as e:
                        print(f"Error loading existing token (will re-authenticate): {e}")
                        self.credentials = None
                
                # If credentials don't exist or are invalid, run the OAuth flow
                if not self.credentials or not self.credentials.valid:
                    if self.credentials and self.credentials.expired and self.credentials.refresh_token:
                        try:
                            self.credentials.refresh(Request())
                        except Exception as e:
                            print(f"Error refreshing token (will re-authenticate): {e}")
                            self.credentials = None
                            # Remove the invalid token file
                            if os.path.exists(token_path):
                                os.remove(token_path)
                                print("Removed invalid token file")
                    
                    if not self.credentials:
                        if not self.credentials_path:
                            raise ValueError("No credentials path provided for OAuth flow")
                        
                        print("Starting new OAuth flow")
                        flow = InstalledAppFlow.from_client_secrets_file(
                            self.credentials_path, self.SCOPES)
                        self.credentials = flow.run_local_server(port=0)
                    
                    # Save credentials for future use
                    with open(token_path, 'w') as token:
                        token.write(self.credentials.to_json())
                        print("New token saved successfully")
            
            # Build the services using the authenticated credentials
            # Configure the httplib2.Http object with extended timeout
            # But don't pass it directly to build()
            http = httplib2.Http(timeout=120)
            
            # Build the services with credentials only
            self.service = build('docs', 'v1', credentials=self.credentials)
            self.drive_service = build('drive', 'v3', credentials=self.credentials)
            
            # Then explicitly set the http client on the authorized http property
            self.service._http.http = http
            self.drive_service._http.http = http
            
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
    
    def get_document_content(self, document_id: str, use_cache: bool = True) -> str:
        """
        Extract the plain text content from a document.
        
        Args:
            document_id: The Google Doc ID
            use_cache: Whether to use cached content (if recent)
            
        Returns:
            The document content as plain text
        """
        # Check cache first if allowed
        if use_cache and document_id in self._content_cache:
            timestamp, content = self._content_cache[document_id]
            if time.time() - timestamp < self._cache_ttl:
                return content
                
        # Not in cache or cache expired, fetch from API
        doc = self.get_document(document_id)
        if not doc:
            return ""
        
        content = []
        for elem in doc.get('body', {}).get('content', []):
            if 'paragraph' in elem:
                for para_elem in elem['paragraph']['elements']:
                    if 'textRun' in para_elem:
                        content.append(para_elem['textRun']['content'])
        
        text_content = ''.join(content)
        
        # Update cache
        self._content_cache[document_id] = (time.time(), text_content)
        
        return text_content
        
    def get_document_content_async(self, fetcher, document_id: str) -> str:
        """
        Extract document content using the async fetcher with continuous retries
        
        Args:
            fetcher: DocumentContentFetcher instance
            document_id: The Google Doc ID
            
        Returns:
            Document content or empty string on failure
        """
        # Check cache first
        if document_id in self._content_cache:
            timestamp, content = self._content_cache[document_id]
            if time.time() - timestamp < self._cache_ttl:
                return content
        
        # Start async fetch
        request_id, _ = fetcher.fetch_content_async(document_id)
        
        # Wait for content with timeout
        content = fetcher.get_content(request_id)
        
        # Clean up
        fetcher.cleanup(request_id)
        
        # Update cache if we got content
        if content:
            self._content_cache[document_id] = (time.time(), content)
            
        return content or ""
    
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
    
    def update_document_content(self, document_id: str, content: str) -> bool:
        """
        Update the entire content of a document with the provided text.
        
        Args:
            document_id: The Google Doc ID
            content: The new document content as plain text
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.service:
                raise ValueError("Not authenticated. Call authenticate() first.")
            
            # A simpler approach that's less likely to cause range errors:
            # Instead of trying to delete all content and then insert new content,
            # we'll just use a single insertText with replaceAllText flag
                
            insert_request = {
                'insertText': {
                    'location': {
                        'index': 1  # Start at the beginning of the document
                    },
                    'text': content
                }
            }
            
            # First, get the document to check if it exists
            doc = self.get_document(document_id)
            if not doc:
                return False
                
            # If the document exists but is empty, just insert the content
            if not self.get_document_content(document_id).strip():
                result = self.update_document(document_id, [insert_request])
                return bool(result)
            
            # Otherwise, use replaceAllText to replace entire content
            replace_request = {
                'replaceAllText': {
                    'replaceText': content,
                    'containsText': {
                        'text': '.*',
                        'matchCase': False
                    }
                }
            }
            
            # Execute the request
            result = self.update_document(document_id, [replace_request])
            
            # If replace didn't work, fall back to a different approach
            if not result:
                # Try another approach - first delete content then insert new
                # Get current content and create a delete request that's more conservative
                current_content = self.get_document_content(document_id)
                
                # Create requests
                requests = []
                
                if current_content:
                    # Only delete if there's content
                    requests.append({
                        'deleteContentRange': {
                            'range': {
                                'startIndex': 1,
                                'endIndex': min(len(current_content), 1000000)  # Safer limit
                            }
                        }
                    })
                
                # Add the insert request
                requests.append(insert_request)
                
                # Execute the requests
                result = self.update_document(document_id, requests)
            
            return bool(result)
            
        except Exception as error:
            print(f"Error updating document content: {error}")
            return False
    
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
            
            # Generate a unique channel ID that includes the document ID for better tracking
            channel_id = f'collabgpt-channel-{document_id}-{int(datetime.now().timestamp())}'
            
            # Create a notification channel
            channel_body = {
                'id': channel_id,
                'type': 'web_hook',
                'address': webhook_url,
                # Important: Add a token for Google to include in notifications
                'token': document_id  # Using document_id as the token for verification
            }
            
            print(f"Setting up watch for document {document_id} with URL {webhook_url}")
            response = self.drive_service.files().watch(
                fileId=document_id,
                body=channel_body
            ).execute()
            
            print(f"Watch response: {response}")
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

    def list_recent_documents(self, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        List recently modified documents accessible to the user.
        
        Args:
            max_results: Maximum number of documents to return
            
        Returns:
            List of document metadata
        """
        try:
            if not self.drive_service:
                raise ValueError("Not authenticated. Call authenticate() first.")
                
            # Query files of type Google Docs
            results = self.drive_service.files().list(
                q="mimeType='application/vnd.google-apps.document'",
                orderBy="modifiedTime desc",
                pageSize=max_results,
                fields="files(id, name, modifiedTime, lastModifyingUser)"
            ).execute()
            
            return results.get('files', [])
            
        except HttpError as error:
            print(f"Error listing recent documents: {error}")
            return []