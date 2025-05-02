#!/usr/bin/env python3
"""
Phase 1 Comprehensive Testing Script for CollabGPT

This script tests all Phase 1 features including:
1. Core Infrastructure:
   - Authentication with Google Docs API
   - Document change detection via webhooks
   - Document access and modification capabilities
   - Real-time monitoring infrastructure
   - Data storage for document histories and user preferences

2. Basic AI Capabilities:
   - Document content analysis
   - Simple summarization of document changes
   - Baseline RAG system for contextual understanding
   - Integration with chosen open source LLM
   - Prompt engineering templates for different agent tasks

Usage:
    python test_phase1.py -d <document_id> [-v]

Arguments:
    -d, --document_id: The Google Docs ID to test with
    -v, --verbose: Enable verbose logging
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
import logging
from typing import Dict, Any, List

# Add the project root to the path so we can import the app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.app import CollabGPT
from src.api.google_docs import GoogleDocsAPI
from src.services.document_analyzer import DocumentAnalyzer
from src.models.rag_system import RAGSystem
from src.models.llm_interface import LLMInterface
from src.config import settings


class Phase1Tester:
    """Test handler for Phase 1 functionality."""
    
    def __init__(self, document_id: str, verbose: bool = False):
        """
        Initialize the Phase 1 tester.
        
        Args:
            document_id: The Google Docs ID to test with
            verbose: Enable verbose logging
        """
        self.document_id = document_id
        self.verbose = verbose
        self.setup_logging()
        
        # Initialize app
        self.app = CollabGPT()
        if not self.app.initialize():
            self.logger.error("Failed to initialize CollabGPT application")
            sys.exit(1)
        
        # Start the application but skip webhook server for tests
        self.app.start = lambda: True
        self.app.running = True
        
        # Explicitly authenticate with Google Docs API
        self.logger.info("Authenticating with Google Docs API...")
        if not self.app.google_docs_api.authenticate(
            use_service_account=settings.GOOGLE_API.get('use_service_account', False)
        ):
            self.logger.error("Failed to authenticate with Google Docs API")
            sys.exit(1)
        
        # Store original content
        self.original_content = self.app.google_docs_api.get_document_content(document_id)
        if not self.original_content:
            self.logger.error(f"Could not retrieve content for document {document_id}")
            sys.exit(1)
            
        self.logger.info(f"Initialized Phase 1 tester for document: {document_id}")
        self.logger.info(f"Document title: {self.get_document_title()}")
        
    def setup_logging(self):
        """Set up logging for the tester."""
        self.logger = logging.getLogger("phase1_tester")
        self.logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        
        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(ch)
        
    def get_document_title(self) -> str:
        """Get the title of the test document."""
        doc = self.app.google_docs_api.get_document(self.document_id)
        return doc.get('title', 'Unknown Document')
    
    def run_all_tests(self):
        """Run all Phase 1 feature tests."""
        self.logger.info("Starting Phase 1 feature tests...")
        
        # Core Infrastructure Tests
        self.test_google_docs_api()
        self.test_document_operations()
        self.test_change_detection()
        self.test_data_storage()
        
        # Basic AI Capabilities Tests
        self.test_document_analysis()
        self.test_change_summarization()
        self.test_rag_system()
        self.test_llm_integration()
        self.test_prompt_templates()
        
        # Clean up
        self.cleanup()
        
        self.logger.info("All Phase 1 tests completed!")
        
    def test_google_docs_api(self):
        """Test authentication with Google Docs API."""
        self.logger.info("\n" + "="*50)
        self.logger.info("Test 1: Google Docs API Authentication")
        self.logger.info("="*50)
        
        # Test authentication
        api = self.app.google_docs_api
        
        # Check if API is authenticated
        if hasattr(api, 'service') and api.service:
            self.logger.info("✓ PASS: Authentication with Google Docs API successful")
        else:
            self.logger.error("✗ FAIL: Authentication with Google Docs API failed")
            
        # Try to list recent files to verify access
        try:
            files = api.list_recent_documents(max_results=5)
            if files:
                self.logger.info(f"Retrieved {len(files)} recent documents")
                for file in files[:3]:
                    self.logger.info(f"  - {file.get('name', 'Unnamed')} ({file.get('id', 'No ID')})")
                self.logger.info("✓ PASS: Document listing successful")
            else:
                self.logger.warning("⚠ WARN: No recent documents found or could not list documents")
        except Exception as e:
            self.logger.error(f"✗ FAIL: Error listing documents: {str(e)}")
        
    def test_document_operations(self):
        """Test document access and modification capabilities."""
        self.logger.info("\n" + "="*50)
        self.logger.info("Test 2: Document Operations")
        self.logger.info("="*50)
        
        api = self.app.google_docs_api
        
        # Test document retrieval
        self.logger.info("Testing document retrieval...")
        document = api.get_document(self.document_id)
        if document and 'title' in document:
            self.logger.info(f"✓ PASS: Successfully retrieved document '{document['title']}'")
        else:
            self.logger.error("✗ FAIL: Failed to retrieve document")
            
        # Test content extraction
        self.logger.info("Testing content extraction...")
        content = api.get_document_content(self.document_id)
        if content:
            preview = content[:50] + "..." if len(content) > 50 else content
            self.logger.info(f"✓ PASS: Successfully extracted content: '{preview}'")
        else:
            self.logger.error("✗ FAIL: Failed to extract document content")
            
        # Test document modification
        self.logger.info("Testing document modification...")
        test_text = f"\n\nTest modification at {datetime.now().isoformat()}"
        modified_content = content + test_text
        
        result = api.update_document_content(self.document_id, modified_content)
        if result:
            self.logger.info("✓ PASS: Document modification successful")
        else:
            self.logger.error("✗ FAIL: Document modification failed")
            
        # Verify modification - add a small delay to ensure changes are processed
        time.sleep(1)
        updated_content = api.get_document_content(self.document_id)
        if updated_content and test_text in updated_content:
            self.logger.info("✓ PASS: Verified modification in document content")
        else:
            self.logger.error(f"✗ FAIL: Could not verify modification. Expected to find: '{test_text}'")
            
        # Revert to original content
        api.update_document_content(self.document_id, content)
        self.logger.info("Reverted document to original content")
        
        # Test comment operations if available
        self.logger.info("Testing comment operations...")
        try:
            comments = api.get_document_comments(self.document_id)
            if comments is not None:
                self.logger.info(f"Retrieved {len(comments)} comments")
                self.logger.info("✓ PASS: Comment retrieval successful")
            else:
                self.logger.warning("⚠ WARN: No comments found or comment retrieval not supported")
        except Exception as e:
            self.logger.warning(f"⚠ WARN: Error retrieving comments: {str(e)}")
        
    def test_change_detection(self):
        """Test document change detection system."""
        self.logger.info("\n" + "="*50)
        self.logger.info("Test 3: Change Detection")
        self.logger.info("="*50)
        
        # Get original content
        original_content = self.app.google_docs_api.get_document_content(self.document_id)
        
        # Make a change to the document
        self.logger.info("Making a change to test detection...")
        test_change = f"\n\nChange detection test at {datetime.now().isoformat()}"
        modified_content = original_content + test_change
        
        # Update the document
        update_result = self.app.google_docs_api.update_document_content(
            self.document_id, modified_content
        )
        
        if not update_result:
            self.logger.error("✗ FAIL: Could not update document for change detection test")
            return
            
        # Simulate change detection through direct method call
        self.logger.info("Simulating change detection...")
        changes = self.app.process_document_changes(
            self.document_id, 
            original_content, 
            modified_content,
            user_id="test_user"
        )
        
        # Check change detection results
        if changes and 'changes' in changes:
            self.logger.info("✓ PASS: Change detection working")
            
            # Check what changes were detected
            diff = changes['changes'].get('diff', {})
            if diff:
                self.logger.info(f"Detected {len(diff.get('added', []))} additions and {len(diff.get('removed', []))} removals")
            
            # Check if there's a summary
            if 'summary' in changes['changes']:
                self.logger.info(f"Change summary: {changes['changes']['summary']}")
                self.logger.info("✓ PASS: Change summarization working")
            else:
                self.logger.warning("⚠ WARN: No change summary generated")
        else:
            self.logger.error("✗ FAIL: Change detection not working properly")
            
        # Revert document to original state
        self.app.google_docs_api.update_document_content(self.document_id, original_content)
        self.logger.info("Reverted document to original state")
        
    def test_data_storage(self):
        """Test data storage for document histories and user preferences."""
        self.logger.info("\n" + "="*50)
        self.logger.info("Test 4: Data Storage")
        self.logger.info("="*50)
        
        # Test document history storage
        self.logger.info("Testing document history storage...")
        
        # Create a test entry in document history
        now = datetime.now().isoformat()
        test_history_entry = {
            'document_id': self.document_id,
            'timestamp': now,
            'user_id': 'test_user',
            'action': 'edit',
            'metadata': {'test': True}
        }
        
        # Check if the application has a method to store document history
        if hasattr(self.app, 'store_document_history') and callable(self.app.store_document_history):
            result = self.app.store_document_history(test_history_entry)
            if result:
                self.logger.info("✓ PASS: Successfully stored document history entry")
            else:
                self.logger.warning("⚠ WARN: Could not store document history entry")
        else:
            # Try to access underlying storage directly if available
            self.logger.info("Direct document history storage method not available, checking data directory...")
            
            # Check if data directory exists
            if os.path.isdir('data'):
                self.logger.info("✓ PASS: Data directory exists for storage")
                
                # Look for document-related files
                doc_files = [f for f in os.listdir('data') if self.document_id in f]
                if doc_files:
                    self.logger.info(f"Found {len(doc_files)} files related to the test document")
                    for file in doc_files[:3]:
                        self.logger.info(f"  - {file}")
                    self.logger.info("✓ PASS: Document storage files found")
                else:
                    self.logger.warning("⚠ WARN: No document storage files found")
            else:
                self.logger.warning("⚠ WARN: Data directory not found")
        
        # Test monitored documents storage
        self.logger.info("Testing monitored documents storage...")
        if os.path.exists('data/monitored_documents.json'):
            try:
                with open('data/monitored_documents.json', 'r') as f:
                    monitored_docs = json.load(f)
                    
                self.logger.info(f"Found {len(monitored_docs)} monitored documents")
                self.logger.info("✓ PASS: Monitored documents storage working")
            except Exception as e:
                self.logger.warning(f"⚠ WARN: Error reading monitored documents: {str(e)}")
        else:
            self.logger.warning("⚠ WARN: No monitored documents file found")
            
    def test_document_analysis(self):
        """Test document content analysis."""
        self.logger.info("\n" + "="*50)
        self.logger.info("Test 5: Document Content Analysis")
        self.logger.info("="*50)
        
        # Get document content
        content = self.app.google_docs_api.get_document_content(self.document_id)
        if not content:
            self.logger.error("✗ FAIL: Could not get document content for analysis")
            return
            
        # Check if we have a document analyzer 
        analyzer = DocumentAnalyzer() if not hasattr(self.app, 'document_analyzer') else self.app.document_analyzer
        
        # Test document analysis
        self.logger.info("Performing document analysis...")
        analysis = analyzer.analyze_document(self.document_id, content)
        
        if analysis:
            self.logger.info("✓ PASS: Document analysis successful")
            
            # Check analysis components
            if 'structure' in analysis:
                self.logger.info(f"Document structure detected with {len(analysis['structure'])} sections")
                
            if 'summary' in analysis:
                self.logger.info(f"Document summary: {analysis['summary'][:100]}...")
                
            if 'key_phrases' in analysis:
                key_phrases = analysis['key_phrases'][:5] if len(analysis['key_phrases']) > 5 else analysis['key_phrases']
                self.logger.info(f"Key phrases: {', '.join(key_phrases)}")
                
            if 'topics' in analysis:
                topics = analysis['topics'][:5] if len(analysis['topics']) > 5 else analysis['topics']
                self.logger.info(f"Topics: {', '.join(topics)}")
        else:
            self.logger.error("✗ FAIL: Document analysis failed")
            
    def test_change_summarization(self):
        """Test simple summarization of document changes."""
        self.logger.info("\n" + "="*50)
        self.logger.info("Test 6: Change Summarization")
        self.logger.info("="*50)
        
        # Get original content
        original_content = self.app.google_docs_api.get_document_content(self.document_id)
        
        # Make a significant change to test summarization
        self.logger.info("Making changes to test summarization...")
        
        # Create a more substantial change that would be worth summarizing
        new_paragraph = "\n\n# New Test Section\n\nThis is a new section added to the document to test the change summarization capabilities. It contains multiple sentences to give the summarizer something substantial to work with. This section discusses important information that would typically be included in a document like this."
        
        modified_content = original_content + new_paragraph
        
        # Update the document
        update_result = self.app.google_docs_api.update_document_content(
            self.document_id, modified_content
        )
        
        if not update_result:
            self.logger.error("✗ FAIL: Could not update document for summarization test")
            return
            
        # Process the change to generate a summary
        self.logger.info("Generating change summary...")
        change_analysis = self.app.process_document_changes(
            self.document_id, 
            original_content, 
            modified_content,
            user_id="test_user"
        )
        
        # Check if a summary was generated
        if change_analysis and 'changes' in change_analysis and 'summary' in change_analysis['changes']:
            summary = change_analysis['changes']['summary']
            self.logger.info(f"Change summary: {summary}")
            self.logger.info("✓ PASS: Change summarization working")
        else:
            self.logger.warning("⚠ WARN: No change summary generated")
            
        # Revert document to original state
        self.app.google_docs_api.update_document_content(self.document_id, original_content)
        self.logger.info("Reverted document to original state")
        
    def test_rag_system(self):
        """Test baseline RAG system for contextual understanding."""
        self.logger.info("\n" + "="*50)
        self.logger.info("Test 7: Baseline RAG System")
        self.logger.info("="*50)
        
        # Get document content
        content = self.app.google_docs_api.get_document_content(self.document_id)
        
        # Check if we have a RAG system
        if not hasattr(self.app, 'rag_system'):
            self.logger.warning("⚠ WARN: RAG system not directly accessible from app object")
            
            # Create a test RAG system for testing
            self.logger.info("Creating a test RAG system...")
            rag_system = RAGSystem()
        else:
            rag_system = self.app.rag_system
            
        # Test document processing in RAG
        self.logger.info("Processing document with RAG system...")
        
        try:
            # Create metadata for the document
            metadata = {
                "title": self.get_document_title(),
                "document_id": self.document_id,
                "timestamp": datetime.now().isoformat()
            }
            
            # Process the document
            result = rag_system.process_document(self.document_id, content, metadata)
            
            if result:
                self.logger.info("✓ PASS: Document successfully processed by RAG system")
            else:
                self.logger.warning("⚠ WARN: RAG system did not confirm document processing")
                
            # Test retrieval
            self.logger.info("Testing context retrieval...")
            query = "main purpose of the document"
            
            context = rag_system.get_relevant_context(query, doc_id=self.document_id)
            
            if context:
                self.logger.info("✓ PASS: Successfully retrieved context from RAG system")
                self.logger.info(f"Context (truncated): {context[:150]}...")
            else:
                self.logger.warning("⚠ WARN: No context retrieved from RAG system")
                
        except Exception as e:
            self.logger.error(f"✗ FAIL: Error testing RAG system: {str(e)}")
            
    def test_llm_integration(self):
        """Test integration with chosen open source LLM."""
        self.logger.info("\n" + "="*50)
        self.logger.info("Test 8: LLM Integration")
        self.logger.info("="*50)
        
        # Check if LLM interface is available
        if not hasattr(self.app, 'llm_interface'):
            self.logger.warning("⚠ WARN: LLM interface not directly accessible from app object")
            
            # Create a test LLM interface
            self.logger.info("Creating a test LLM interface...")
            llm = LLMInterface()
        else:
            llm = self.app.llm_interface
            
        # Check if LLM is available
        if not llm.is_available():
            self.logger.warning("⚠ WARN: LLM is not available, skipping detailed tests")
            return
            
        self.logger.info("✓ PASS: LLM interface is available")
        
        # Test basic query
        self.logger.info("Testing basic LLM query...")
        try:
            response = llm.generate_text("Summarize what a collaborative document editor is in one sentence.")
            
            if response and response.text:
                self.logger.info(f"LLM response: {response.text}")
                self.logger.info("✓ PASS: Successfully generated text with LLM")
            else:
                self.logger.warning("⚠ WARN: LLM did not generate a response")
                
        except Exception as e:
            self.logger.error(f"✗ FAIL: Error generating text with LLM: {str(e)}")
            
        # Test with document context if we have a RAG system
        if hasattr(self.app, 'rag_system'):
            self.logger.info("Testing LLM with document context...")
            
            content = self.app.google_docs_api.get_document_content(self.document_id)
            context = self.app.rag_system.get_relevant_context("document main points", doc_id=self.document_id)
            
            if not context:
                self.logger.warning("⚠ WARN: Could not retrieve context from RAG system for LLM test")
                return
                
            # Try to generate a summary using the RAG context
            try:
                prompt = f"Based on the following context from a document, provide a brief summary:\n\n{context}"
                response = llm.generate_text(prompt)
                
                if response and response.text:
                    self.logger.info(f"LLM context-based response: {response.text[:150]}...")
                    self.logger.info("✓ PASS: Successfully generated context-based response with LLM")
                else:
                    self.logger.warning("⚠ WARN: LLM did not generate a context-based response")
                    
            except Exception as e:
                self.logger.error(f"✗ FAIL: Error generating context-based response: {str(e)}")
                
    def test_prompt_templates(self):
        """Test prompt engineering templates for different agent tasks."""
        self.logger.info("\n" + "="*50)
        self.logger.info("Test 9: Prompt Templates")
        self.logger.info("="*50)
        
        # Check if templates directory exists
        templates_dir = 'templates'
        if not os.path.isdir(templates_dir):
            self.logger.warning(f"⚠ WARN: Templates directory '{templates_dir}' not found")
            return
            
        # Check if template files exist
        template_files = os.listdir(templates_dir)
        
        if not template_files:
            self.logger.warning("⚠ WARN: No template files found in templates directory")
            return
            
        self.logger.info(f"Found {len(template_files)} template files:")
        for template_file in template_files:
            self.logger.info(f"  - {template_file}")
            
        # Check specific templates we expect to see
        expected_templates = ['summarize_changes.txt', 'summarize_document.txt', 
                              'suggest_edits.txt', 'resolve_conflict.txt']
        
        found_templates = []
        missing_templates = []
        
        for template in expected_templates:
            if template in template_files:
                found_templates.append(template)
            else:
                missing_templates.append(template)
                
        if found_templates:
            self.logger.info(f"✓ PASS: Found {len(found_templates)} expected templates")
        
        if missing_templates:
            self.logger.warning(f"⚠ WARN: Missing {len(missing_templates)} expected templates: {', '.join(missing_templates)}")
            
        # Test loading a template
        self.logger.info("Testing template loading...")
        if found_templates:
            template_path = os.path.join(templates_dir, found_templates[0])
            try:
                with open(template_path, 'r') as f:
                    template_content = f.read()
                    
                if template_content:
                    self.logger.info(f"Successfully loaded template: {found_templates[0]}")
                    self.logger.info(f"Template preview: {template_content[:100]}...")
                    self.logger.info("✓ PASS: Template loading working")
                else:
                    self.logger.warning(f"⚠ WARN: Template file {found_templates[0]} is empty")
            except Exception as e:
                self.logger.error(f"✗ FAIL: Error loading template: {str(e)}")
                
    def cleanup(self):
        """Clean up after tests."""
        self.logger.info("\n" + "="*50)
        self.logger.info("Cleaning up after tests")
        self.logger.info("="*50)
        
        # Restore original content
        if self.original_content:
            self.logger.info("Restoring original document content...")
            self.app.google_docs_api.update_document_content(
                self.document_id, self.original_content
            )
            
        # Stop the app
        if self.app.running:
            self.app.stop()
            
        self.logger.info("Cleanup complete")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Test Phase 1 features of CollabGPT")
    parser.add_argument("-d", "--document_id", required=True, help="Google Docs document ID")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Run tests
    tester = Phase1Tester(args.document_id, args.verbose)
    tester.run_all_tests()


if __name__ == "__main__":
    main()