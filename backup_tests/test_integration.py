#!/usr/bin/env python3
"""
Test script for CollabGPT Google Docs integration.
This script helps verify that the basic authentication and document operations work properly.
"""

import os
import sys
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.api.google_docs import GoogleDocsAPI
from src.services.document_analyzer import DocumentAnalyzer
from src.utils import logger

def test_google_docs_auth():
    """Test authentication with Google Docs API."""
    print("Testing Google Docs API authentication...")
    
    # Create GoogleDocsAPI instance
    credentials_path = os.environ.get('GOOGLE_CREDENTIALS_PATH') or 'credentials/google_credentials.json'
    use_service_account = os.environ.get('GOOGLE_USE_SERVICE_ACCOUNT', 'false').lower() == 'true'
    
    api = GoogleDocsAPI(credentials_path=credentials_path)
    
    # Attempt authentication
    success = api.authenticate(use_service_account=use_service_account)
    
    if success:
        print("✅ Authentication successful!")
    else:
        print("❌ Authentication failed. Check your credentials.")
        return None
        
    return api

def test_document_operations(api):
    """Test basic document operations."""
    if not api:
        return
        
    print("\nEnter a Google Doc ID to test (or leave blank to skip):")
    doc_id = input("> ").strip()
    
    if not doc_id:
        print("Skipping document operations test.")
        return
        
    print(f"Testing document operations with Doc ID: {doc_id}")
    
    # Get document
    print("Retrieving document...")
    document = api.get_document(doc_id)
    
    if not document:
        print("❌ Failed to retrieve document. Check the document ID and permissions.")
        return
        
    # Print document title
    title = document.get('title', 'Untitled')
    print(f"✅ Successfully retrieved document: '{title}'")
    
    # Get document content
    print("Extracting document content...")
    content = api.get_document_content(doc_id)
    
    if not content:
        print("❌ Failed to extract document content.")
        return
        
    # Print content preview
    preview = content[:100] + "..." if len(content) > 100 else content
    print(f"✅ Successfully extracted content preview: '{preview}'")
    
    # Test document analysis
    print("\nTesting document analysis...")
    analyzer = DocumentAnalyzer()
    analysis = analyzer.analyze_document(doc_id, content)
    
    # Print summary
    summary = analysis.get('summary', 'No summary generated')
    print(f"Document summary: '{summary}'")
    
    # Print key phrases
    key_phrases = analysis.get('key_phrases', [])[:5]
    print(f"Key phrases: {', '.join(key_phrases)}")
    
    return document, content, analysis

def test_document_comments(api, doc_id):
    """Test retrieving document comments."""
    if not api or not doc_id:
        return
        
    print("\nTesting comment retrieval...")
    comments = api.get_document_comments(doc_id)
    
    if not comments:
        print("No comments found or failed to retrieve comments.")
        return
        
    print(f"✅ Retrieved {len(comments)} comments.")
    for i, comment in enumerate(comments[:3], 1):
        content = comment.get('content', 'No content')
        author = comment.get('author', {}).get('displayName', 'Unknown')
        print(f"  Comment {i}: '{content[:50]}...' by {author}")
        
    return comments

def save_test_results(doc_id, document, content, analysis):
    """Save test results to a file for debugging."""
    if not document or not content or not analysis:
        return
        
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save document info
    with open(f'data/test_document_{doc_id}.json', 'w') as f:
        json.dump({
            'document': document,
            'analysis': analysis
        }, f, indent=2)
        
    # Save content
    with open(f'data/test_document_{doc_id}_content.txt', 'w') as f:
        f.write(content)
        
    print(f"\nTest results saved to data/test_document_{doc_id}.json")

def main():
    """Run the integration test."""
    print("CollabGPT Integration Test")
    print("=========================\n")
    
    # Test authentication
    api = test_google_docs_auth()
    
    if not api:
        return 1
        
    # Test document operations
    result = test_document_operations(api)
    
    if result:
        document, content, analysis = result
        doc_id = document.get('documentId')
        
        # Test comments
        test_document_comments(api, doc_id)
        
        # Save results
        save_test_results(doc_id, document, content, analysis)
    
    print("\nIntegration test completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())