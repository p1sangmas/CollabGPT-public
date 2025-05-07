#!/usr/bin/env python
"""
Setup Webhook Subscriptions for Google Docs

This script sets up webhook subscriptions for Google Docs documents
to receive real-time notifications of changes.
"""

import os
import sys
import uuid
import json
from dotenv import load_dotenv
from src.api.google_docs import GoogleDocsAPI
from src.config import settings

def cleanup_webhook_subscription(document_id):
    """
    Remove an existing webhook subscription for a Google Docs document.
    
    Args:
        document_id: The Google Document ID
    """
    # Set up configuration
    load_dotenv()
    
    # Get credentials path from settings
    credentials_path = os.getenv('GOOGLE_CREDENTIALS_PATH') or 'credentials/google_credentials.json'
    use_service_account = os.getenv('GOOGLE_USE_SERVICE_ACCOUNT', 'false').lower() == 'true'
    
    # Check if subscription file exists
    subscription_file = f"data/webhook_subscription_{document_id}.txt"
    if not os.path.exists(subscription_file):
        print(f"No existing webhook subscription found for document: {document_id}")
        return False
    
    # Read subscription details
    channel_id = None
    resource_id = None
    with open(subscription_file, "r") as f:
        for line in f:
            if line.startswith("Channel ID:"):
                channel_id = line.split(":", 1)[1].strip()
            elif line.startswith("Resource ID:"):
                resource_id = line.split(":", 1)[1].strip()
    
    if not channel_id or not resource_id:
        print("Could not find channel ID or resource ID in subscription file")
        return False
    
    # Create API client and authenticate
    api = GoogleDocsAPI(credentials_path=credentials_path)
    if not api.authenticate(use_service_account=use_service_account):
        print("Failed to authenticate with Google API")
        return False
    
    # Ensure we have drive_service for watch functionality
    if not api.drive_service:
        print("Failed to initialize Google Drive API service")
        return False
    
    print(f"Removing webhook subscription for document: {document_id}")
    print(f"Channel ID: {channel_id}")
    print(f"Resource ID: {resource_id}")
    
    try:
        # Stop the watch on the document
        api.drive_service.channels().stop(
            body={
                'id': channel_id,
                'resourceId': resource_id
            }
        ).execute()
        
        print("Subscription successfully removed!")
        
        # Remove the subscription file
        os.remove(subscription_file)
        return True
        
    except Exception as e:
        print(f"Error removing webhook subscription: {str(e)}")
        return False

def setup_webhook_subscription(document_id, webhook_url, webhook_path="/webhook"):
    """
    Set up a webhook subscription for a Google Docs document.
    
    Args:
        document_id: The Google Document ID
        webhook_url: The external URL that Google will send notifications to
        webhook_path: The path on the webhook URL to send notifications to
    """
    # Set up configuration
    load_dotenv()
    
    # Get credentials path from settings
    credentials_path = os.getenv('GOOGLE_CREDENTIALS_PATH') or 'credentials/google_credentials.json'
    use_service_account = os.getenv('GOOGLE_USE_SERVICE_ACCOUNT', 'false').lower() == 'true'
    
    print(f"Using credentials from: {credentials_path}")
    
    # Create API client and authenticate
    api = GoogleDocsAPI(credentials_path=credentials_path)
    if not api.authenticate(use_service_account=use_service_account):
        print("Failed to authenticate with Google API")
        sys.exit(1)
    
    # Ensure we have drive_service for watch functionality
    if not api.drive_service:
        print("Failed to initialize Google Drive API service")
        sys.exit(1)
    
    # Ensure trailing slash on webhook URL
    if not webhook_url.endswith('/'):
        webhook_url += '/'
    
    # Remove leading slash from webhook path
    if webhook_path.startswith('/'):
        webhook_path = webhook_path[1:]
    
    # Create a unique channel ID for this subscription
    channel_id = str(uuid.uuid4())
    
    # Create the full notification URL
    notification_url = f"{webhook_url}{webhook_path}"
    
    print(f"Setting up webhook subscription for document: {document_id}")
    print(f"Notification URL: {notification_url}")
    print(f"Channel ID: {channel_id}")
    
    try:
        # Set up the watch on the document
        response = api.drive_service.files().watch(
            fileId=document_id,
            body={
                'id': channel_id,
                'type': 'web_hook',
                'address': notification_url,
                'token': os.getenv('WEBHOOK_SECRET_KEY', 'default_secret_key')
            }
        ).execute()
        
        print("\nSubscription successfully created!")
        print(f"Resource ID: {response.get('resourceId')}")
        print(f"Expiration: {response.get('expiration')}")
        
        # Register document with application
        if hasattr(settings, 'save_monitored_document'):
            try:
                # Try to get the document name
                doc_info = api.drive_service.files().get(fileId=document_id, fields="name").execute()
                doc_name = doc_info.get('name', f"Document {document_id}")
                
                # Save to monitored documents
                settings.save_monitored_document(document_id, doc_name, webhook_enabled=True)
                print(f"Document '{doc_name}' added to monitored documents")
            except Exception as e:
                print(f"Warning: Could not save to monitored documents: {str(e)}")
        
    except Exception as e:
        print(f"\nError setting up webhook subscription: {str(e)}")
        sys.exit(1)
        
    print("\nYour webhook is now set up to receive notifications when this document changes.")
    print("The subscription will expire after 1 week and will need to be renewed.")
    
    # Store subscription details for later reference
    with open(f"data/webhook_subscription_{document_id}.txt", "w") as f:
        f.write(f"Document ID: {document_id}\n")
        f.write(f"Channel ID: {channel_id}\n")
        f.write(f"Resource ID: {response.get('resourceId')}\n")
        f.write(f"Notification URL: {notification_url}\n")
        f.write(f"Expiration: {response.get('expiration')}\n")

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python setup_webhook.py <document_id> [--refresh]")
        sys.exit(1)
    
    # Get document ID from command line
    document_id = sys.argv[1]
    refresh_mode = "--refresh" in sys.argv
    
    # Load environment variables
    load_dotenv()
    webhook_url = os.getenv('WEBHOOK_EXTERNAL_URL')
    webhook_path = os.getenv('WEBHOOK_PATH', '/webhook')
    
    if not webhook_url:
        print("Error: WEBHOOK_EXTERNAL_URL not set in .env file")
        sys.exit(1)
    
    # If in refresh mode, clean up existing webhook first
    if refresh_mode or os.path.exists(f"data/webhook_subscription_{document_id}.txt"):
        print("Existing webhook subscription found.")
        if refresh_mode or input("Do you want to refresh the webhook with the current URL? (y/n): ").lower() == 'y':
            cleanup_webhook_subscription(document_id)
        else:
            print("Keeping existing webhook subscription.")
            sys.exit(0)
    
    # Set up the webhook subscription
    setup_webhook_subscription(document_id, webhook_url, webhook_path)

if __name__ == "__main__":
    main()