#!/usr/bin/env python3
"""
Script to add a Google Docs document to monitoring by CollabGPT.
"""

from dotenv import load_dotenv
load_dotenv()

from src.app import CollabGPT

app = CollabGPT()
app.initialize()
app.google_docs_api.authenticate()

# Replace with your actual document ID
doc_id = input("Enter your Google Doc ID: ")
result = app.process_document(doc_id)

print(f"Document analyzed and added to monitoring: {doc_id}")
print(f"Analysis results: {result}")

# Set up a webhook for real-time monitoring if enabled
if app.webhook_handler and input("Set up real-time monitoring via webhook? (y/n): ").lower() == 'y':
    success = app._setup_document_webhook(doc_id)
    if success:
        print(f"Webhook successfully set up for document: {doc_id}")
    else:
        print(f"Failed to set up webhook for document: {doc_id}")