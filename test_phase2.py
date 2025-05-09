#!/usr/bin/env python3
"""
Phase 2 Integration Test Script for CollabGPT

This script tests all Phase 2 features including:
1. Real-time change monitoring with minimal latency
2. Intelligent change summaries with categorization by importance
3. Basic conflict detection in concurrent editing scenarios
4. User activity tracking for contextual awareness
5. Comment analysis and organization
6. Enhanced RAG system with document history context
7. Sophisticated prompt chaining for complex tasks
8. Agent reasoning capabilities for edit suggestions
9. Context windows incorporating document structure
10. Feedback loops for suggestion improvement

Usage:
    python test_phase2.py -d <document_id> [-v]

Arguments:
    -d, --document_id: The Google Docs ID to test with
    -v, --verbose: Enable verbose logging
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
import random
import logging
from typing import Dict, Any, List


# Custom JSON encoder to handle datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


# Add the project root to the path so we can import the app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.app import CollabGPT
from src.api.google_docs import GoogleDocsAPI
from src.services.conflict_detector import ConflictType
from src.services.activity_tracker import ActivityType
from src.models.rag_system import RAGSystem
from src.models.context_window import ContextWindowManager
from src.models.llm_interface import LLMInterface, PromptChain
from src.services.edit_suggestion.edit_suggestion_system import EditSuggestionSystem, SuggestionType
from src.services.feedback_loop_system import FeedbackLoopSystem, FeedbackEntry
from src.config import settings


class Phase2Tester:
    """Test handler for Phase 2 functionality."""
    
    def __init__(self, document_id: str, verbose: bool = False):
        """
        Initialize the Phase 2 tester.
        
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
            
        self.logger.info(f"Initialized Phase 2 tester for document: {document_id}")
        self.logger.info(f"Document title: {self.get_document_title()}")
        
    def setup_logging(self):
        """Set up logging for the tester."""
        self.logger = logging.getLogger("phase2_tester")
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
        """Run all Phase 2 feature tests."""
        self.logger.info("Starting Phase 2 feature tests...")
        
        # Test 1: Real-time change monitoring
        self.test_real_time_monitoring()
        
        # Test 2: Intelligent change summaries
        self.test_change_summaries()
        
        # Test 3: Conflict detection
        self.test_conflict_detection()
        
        # Test 4: User activity tracking
        self.test_user_activity_tracking()
        
        # Test 5: Comment analysis 
        self.test_comment_analysis()
        
        # Test 6: Enhanced RAG with document history
        self.test_enhanced_rag_history()
        
        # Test 7: Sophisticated prompt chaining
        self.test_prompt_chaining()
        
        # Test 8: Agent reasoning for edit suggestions
        self.test_edit_suggestions()
        
        # Test 9: Document context windows
        self.test_context_windows()
        
        # Test 10: Feedback loops
        self.test_feedback_loops()
        
        # Clean up
        self.cleanup()
        
        self.logger.info("All Phase 2 tests completed!")
        
    def test_real_time_monitoring(self):
        """Test real-time change monitoring with latency measurement."""
        self.logger.info("\n" + "="*50)
        self.logger.info("Test 1: Real-time Change Monitoring")
        self.logger.info("="*50)
        
        # Make a change to the document
        original_content = self.app.google_docs_api.get_document_content(self.document_id)
        
        # Record start time
        start_time = time.time()
        
        # Make a test change - add timestamp to ensure content is different
        test_change = f"\n\nTest Change at {datetime.now().isoformat()}"
        new_content = original_content + test_change
        
        self.logger.info("Making a change to the document...")
        update_result = self.app.google_docs_api.update_document_content(
            self.document_id, new_content
        )
        
        if not update_result:
            self.logger.error("Failed to update document content")
            return
            
        # Manually process the change since we're not using webhooks
        self.logger.info("Processing document change...")
        change_analysis = self.app.process_document_changes(
            self.document_id, 
            original_content, 
            new_content,
            user_id="test_user"
        )
        
        # Calculate latency
        latency = time.time() - start_time
        
        # Log results
        self.logger.info(f"Change processing latency: {latency:.2f} seconds")
        if latency < 2.0:
            self.logger.info("✓ PASS: Real-time monitoring meets latency target (<2s)")
        else:
            self.logger.warning("⚠ WARN: Real-time monitoring latency exceeds target")
            
        # Check change analysis
        if change_analysis and 'changes' in change_analysis:
            self.logger.info("✓ PASS: Change analysis produced results")
            if self.verbose:
                self.logger.debug(f"Change analysis: {json.dumps(change_analysis, indent=2, cls=DateTimeEncoder)}")
        else:
            self.logger.error("✗ FAIL: Change analysis did not produce results")
            
        # Clean up - revert changes
        self.app.google_docs_api.update_document_content(
            self.document_id, original_content
        )
        self.logger.info("Test document reverted to original state")
        
    def test_change_summaries(self):
        """Test intelligent change summaries with categorization by importance."""
        self.logger.info("\n" + "="*50)
        self.logger.info("Test 2: Intelligent Change Summaries")
        self.logger.info("="*50)
        
        # Get current content
        original_content = self.app.google_docs_api.get_document_content(self.document_id)
        
        # Make a significant change that would be categorized as important
        new_content = original_content + "\n\n# IMPORTANT NEW SECTION\n\nThis is a critical addition to the document that contains key information about project deadlines and priorities. The team needs to review this section as soon as possible."
        
        self.logger.info("Making a significant change to the document...")
        update_result = self.app.google_docs_api.update_document_content(
            self.document_id, new_content
        )
        
        if not update_result:
            self.logger.error("Failed to update document content")
            return
            
        # Process the change
        self.logger.info("Processing document change with importance categorization...")
        change_analysis = self.app.process_document_changes(
            self.document_id, 
            original_content, 
            new_content,
            user_id="test_user"
        )
        
        # Check importance categorization
        if change_analysis and 'changes' in change_analysis and 'importance' in change_analysis['changes']:
            importance = change_analysis['changes']['importance']
            self.logger.info(f"Change importance: {importance.get('level', 'UNKNOWN')}")
            self.logger.info(f"Importance reasons: {importance.get('reasons', 'Not provided')}")
            self.logger.info("✓ PASS: Change importance categorization working")
        else:
            self.logger.error("✗ FAIL: Change importance categorization not working")
            
        # Check AI summary generation
        if 'ai_change_summary' in change_analysis:
            self.logger.info("✓ PASS: AI change summary generated")
            self.logger.info(f"Summary: {change_analysis['ai_change_summary']}")
        else:
            # This is a warning, not an error, since LLM might not be available
            self.logger.warning("⚠ WARN: AI change summary not generated (LLM may not be configured)")
            
        # Clean up - revert changes
        self.app.google_docs_api.update_document_content(
            self.document_id, original_content
        )
        self.logger.info("Test document reverted to original state")
        
    def test_conflict_detection(self):
        """Test basic conflict detection in concurrent editing scenarios."""
        self.logger.info("\n" + "="*50)
        self.logger.info("Test 3: Conflict Detection")
        self.logger.info("="*50)
        
        # Get current content
        original_content = self.app.google_docs_api.get_document_content(self.document_id)
        
        # Simulate two users editing the same section
        self.logger.info("Simulating concurrent edits by two users...")
        
        # User 1 edits
        user1_content = original_content + "\n\nUser 1's changes to the document."
        self.app.process_document_changes(
            self.document_id, 
            original_content, 
            user1_content,
            user_id="user1"
        )
        
        # User 2 edits the same document based on original content (simulating concurrent edit)
        user2_content = original_content + "\n\nUser 2's conflicting changes to the document."
        change_analysis = self.app.process_document_changes(
            self.document_id, 
            original_content, 
            user2_content,
            user_id="user2"
        )
        
        # Check for detected conflicts
        if change_analysis and 'conflicts' in change_analysis:
            conflicts = change_analysis['conflicts']
            self.logger.info(f"Detected {len(conflicts)} conflicts")
            for i, conflict in enumerate(conflicts):
                self.logger.info(f"Conflict {i+1}: {conflict['type']} - {conflict['description']}")
            if self.verbose:
                self.logger.debug(f"Conflicts: {json.dumps(conflicts, indent=2, cls=DateTimeEncoder)}")
            self.logger.info("✓ PASS: Conflict detection working")
        else:
            self.logger.warning("⚠ WARN: No conflicts detected in concurrent editing scenario")
            
        # Get conflicts directly from conflict detector
        conflicts = self.app.conflict_detector.get_conflicts(self.document_id)
        if conflicts:
            self.logger.info(f"Conflict detector has {len(conflicts)} active conflicts")
            self.logger.info("✓ PASS: Conflict detector storing conflicts")
        else:
            self.logger.warning("⚠ WARN: Conflict detector has no stored conflicts")
            
        # Clean up - revert changes and clear conflicts
        self.app.google_docs_api.update_document_content(
            self.document_id, original_content
        )
        self.app.conflict_detector.reset_conflicts(self.document_id)
        self.logger.info("Test document reverted to original state and conflicts cleared")
        
    def test_user_activity_tracking(self):
        """Test user activity tracking for contextual awareness."""
        self.logger.info("\n" + "="*50)
        self.logger.info("Test 4: User Activity Tracking")
        self.logger.info("="*50)
        
        # Get current content
        original_content = self.app.google_docs_api.get_document_content(self.document_id)
        
        # Simulate multiple users with different activities
        users = ["user1", "user2", "user3"]
        activities = [ActivityType.VIEW, ActivityType.EDIT, ActivityType.COMMENT]
        
        self.logger.info("Simulating activities from multiple users...")
        
        # Track a variety of activities
        for i in range(10):
            user = random.choice(users)
            activity = random.choice(activities)
            
            # Generate section names
            sections = ["Introduction", "Methodology", "Results", "Conclusion"]
            active_sections = random.sample(sections, k=random.randint(1, 3))
            
            # Track the activity (without content_length which is not supported)
            metadata = {}
            if activity == ActivityType.EDIT:
                # Add content length as metadata instead of parameter
                metadata['content_length'] = random.randint(10, 100)
                
            self.app.activity_tracker.track_activity(
                user_id=user,
                document_id=self.document_id,
                activity_type=activity,
                sections=active_sections,
                metadata=metadata
            )
            
            self.logger.info(f"Tracked {activity.name} activity for {user} in sections {', '.join(active_sections)}")
            
        # Get activity report
        activity_data = self.app.get_document_activity(self.document_id, hours=24)
        
        # Check the activity data
        if activity_data and 'active_users' in activity_data:
            active_users = activity_data['active_users']
            self.logger.info(f"Found {len(active_users)} active users in the document")
            for user_data in active_users:
                user_id = user_data.get('user_id')
                self.logger.info(f"User {user_id}: {user_data.get('total_activities', 0)} activities")
                if 'focused_sections' in user_data:
                    self.logger.info(f"  Focused sections: {', '.join(user_data['focused_sections'])}")
            
            self.logger.info("✓ PASS: User activity tracking working")
        else:
            self.logger.error("✗ FAIL: User activity tracking not working")
            
    def test_comment_analysis(self):
        """Test comment analysis and organization."""
        try:
            self.logger.info("\n" + "="*50)
            self.logger.info("Test 5: Comment Analysis and Organization")
            self.logger.info("="*50)
            
            # Create sample comment data directly for testing
            comments = [
                {
                    'id': 'comment1',
                    'author': {'id': 'user1', 'name': 'Test User 1'},
                    'content': 'This section needs more examples.',
                    'resolved': False,
                    'created_time': (datetime.now() - timedelta(hours=2)).isoformat()
                },
                {
                    'id': 'comment2',
                    'author': {'id': 'user2', 'name': 'Test User 2'},
                    'content': 'I have a question about this methodology?',
                    'resolved': False,
                    'created_time': (datetime.now() - timedelta(hours=1)).isoformat()
                },
                {
                    'id': 'comment3',
                    'author': {'id': 'user3', 'name': 'Test User 3'},
                    'content': 'Great work on this section!',
                    'resolved': True,
                    'created_time': datetime.now().isoformat()
                },
                {
                    'id': 'comment4',
                    'author': {'id': 'user1', 'name': 'Test User 1'},
                    'content': 'Please review this change.',
                    'resolved': False,
                    'created_time': datetime.now().isoformat()
                }
            ]
            
            self.logger.info("Processing simulated comments...")
            
            # Process comments directly through the comment analyzer instead of the app
            # Use a try-except block to catch any errors
            try:
                threads = self.app.comment_analyzer.process_comments(self.document_id, comments)
                self.logger.info(f"Processed {len(threads)} comment threads")
            except Exception as e:
                self.logger.error(f"Error processing comments: {str(e)}")
                # Don't fail the entire test, continue to check other functions
            
            # Get comment statistics - wrap in try-except to catch any errors
            try:
                comment_stats = self.app.comment_analyzer.get_comment_statistics(self.document_id)
                
                # Check comment analysis
                if comment_stats:
                    self.logger.info(f"Total comment threads: {comment_stats.get('total_threads', 0)}")
                    self.logger.info(f"Unresolved threads: {comment_stats.get('total_threads', 0) - comment_stats.get('resolved_threads', 0)}")
                    
                    if 'category_counts' in comment_stats:
                        categories = comment_stats['category_counts']
                        self.logger.info("Comment categories:")
                        for category, count in categories.items():
                            self.logger.info(f"  {category}: {count}")
                            
                    self.logger.info("✓ PASS: Comment analysis working")
                else:
                    self.logger.error("✗ FAIL: Comment analysis not working")
            except Exception as e:
                self.logger.error(f"Error getting comment statistics: {str(e)}")
            
            # Test thread resolution - wrap in try-except to catch any errors
            try:
                unresolved = self.app.comment_analyzer.get_unresolved_threads(self.document_id)
                self.logger.info(f"Found {len(unresolved)} unresolved comment threads")
                
                if unresolved:
                    thread_to_resolve = unresolved[0]
                    resolved = self.app.comment_analyzer.resolve_thread(
                        self.document_id, thread_to_resolve.thread_id, "test_user"
                    )
                    if resolved:
                        self.logger.info(f"Successfully resolved thread {thread_to_resolve.thread_id}")
                    else:
                        self.logger.warning(f"Failed to resolve thread {thread_to_resolve.thread_id}")
            except Exception as e:
                self.logger.error(f"Error during thread resolution: {str(e)}")
            
            # Cleanup - wrap in try-except to catch any errors
            try:
                # Clear all threads from the comment analyzer for this document
                if hasattr(self.app.comment_analyzer, 'threads') and self.document_id in self.app.comment_analyzer.threads:
                    self.app.comment_analyzer.threads[self.document_id] = []
                if hasattr(self.app.comment_analyzer, 'thread_categories') and self.document_id in self.app.comment_analyzer.thread_categories:
                    self.app.comment_analyzer.thread_categories[self.document_id] = {}
                
                self.logger.info("Comment analyzer data cleared")
            except Exception as e:
                self.logger.error(f"Error during cleanup: {str(e)}")
            
            self.logger.info("Comment analysis test completed")
        except Exception as e:
            # Catch any unexpected exceptions to prevent the entire test suite from failing
            self.logger.error(f"Unexpected error in comment analysis test: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        # Always return to ensure the test completes
        return
        
    def test_enhanced_rag_history(self):
        """Test enhanced RAG system with document history context."""
        self.logger.info("\n" + "="*50)
        self.logger.info("Test 6: Enhanced RAG with Document History")
        self.logger.info("="*50)
        
        # Get original content and create test RAG system
        original_content = self.app.google_docs_api.get_document_content(self.document_id)
        
        # Clear existing document from RAG if it exists
        if hasattr(self.app, 'rag_system') and self.app.rag_system:
            self.app.rag_system.vector_store.delete_chunks_by_doc_id(self.document_id)
        
        # Create a clean RAG system for testing
        rag_system = RAGSystem()
        
        # Process the original document
        self.logger.info("Adding document to RAG system...")
        metadata = {
            "title": self.get_document_title(),
            "version": 1,
            "last_updated": datetime.now().isoformat()
        }
        rag_system.process_document(self.document_id, original_content, metadata)
        
        # Make first round of changes to create history
        first_change = original_content + "\n\n# New Testing Section\n\nThis is version 1 of this section about RAG testing."
        self.logger.info("Making first change to create document history...")
        
        # Process first version
        metadata["version"] = 2
        metadata["last_updated"] = datetime.now().isoformat()
        rag_system.process_document(self.document_id, first_change, metadata)
        
        # Make second round of changes
        second_change = first_change.replace(
            "This is version 1 of this section about RAG testing.",
            "This is version 2 with more details about the enhanced RAG system and its implementation."
        )
        self.logger.info("Making second change to create document history...")
        
        # Process second version
        metadata["version"] = 3
        metadata["last_updated"] = datetime.now().isoformat()
        rag_system.process_document(self.document_id, second_change, metadata)
        
        # Track some activity for this document
        rag_system.track_activity(
            doc_id=self.document_id,
            user_id="user1", 
            activity_type="edit",
            section_id="New Testing Section"
        )
        
        rag_system.track_activity(
            doc_id=self.document_id,
            user_id="user2", 
            activity_type="view",
            section_id="New Testing Section"
        )
        
        # Test document context window retrieval
        context = rag_system.get_document_context_window(
            doc_id=self.document_id,
            focus_section="New Testing Section",
            window_size=2
        )
        
        if context:
            self.logger.info("Retrieved document context window:")
            self.logger.info(context[:500] + "..." if len(context) > 500 else context)
            self.logger.info("✓ PASS: RAG document context window working")
        else:
            self.logger.error("✗ FAIL: RAG document context window not working")
            
        # Test relevant context with history
        query = "testing RAG system"
        context = rag_system.get_relevant_context(
            query, 
            doc_id=self.document_id, 
            include_history=True,
            include_user_activity=True
        )
        
        if context:
            self.logger.info("Retrieved context with history:")
            self.logger.info(context[:500] + "..." if len(context) > 500 else context)
            
            # Check if history is included
            if "Version" in context and "edit" in context:
                self.logger.info("✓ PASS: RAG context includes document history and activity")
            else:
                self.logger.warning("⚠ WARN: RAG context may be missing history or activity info")
        else:
            self.logger.error("✗ FAIL: RAG context retrieval not working")
        
        # Clean up - update document with original content
        self.app.google_docs_api.update_document_content(self.document_id, original_content)
        self.logger.info("Test document reverted to original state")
    
    def test_prompt_chaining(self):
        """Test sophisticated prompt chaining for complex tasks."""
        self.logger.info("\n" + "="*50)
        self.logger.info("Test 7: Sophisticated Prompt Chaining")
        self.logger.info("="*50)
        
        # Skip test if LLM interface is not available
        if not hasattr(self.app, 'llm_interface') or not self.app.llm_interface or not self.app.llm_interface.is_available():
            self.logger.warning("⚠ WARN: Skipping prompt chaining test (LLM interface not available)")
            return
            
        try:
            # Create a test document analysis chain
            self.logger.info("Testing prompt chain for document analysis...")
            
            llm = self.app.llm_interface
            chain = llm.create_chain(name="test_analysis_chain")
            
            # Get original content to analyze
            original_content = self.app.google_docs_api.get_document_content(self.document_id)
            
            # Step 1: Extract key topics
            chain.add_step(
                "extract_key_points",
                name="extract_topics",
                max_tokens=300,
                temperature=0.2
            )
            
            # Step 2: Analyze writing style based on key points
            chain.add_step(
                "analyze_tone",
                name="analyze_style",
                max_tokens=300,
                temperature=0.3,
                input_mapping={
                    "content": "extract_topics.text"
                }
            )
            
            # Step 3: Generate recommendations based on both previous steps
            chain.add_step(
                "Based on the key points and writing style analysis, provide 3 specific recommendations to improve this document.\n\n"
                "Key points: {extract_topics.text}\n\n"
                "Style analysis: {analyze_style.text}\n\n"
                "Recommendations:",
                name="recommendations",
                max_tokens=500,
                temperature=0.7,
                input_mapping={
                    "extract_topics.text": "extract_topics.text",
                    "analyze_style.text": "analyze_style.text"
                }
            )
            
            # Execute the chain
            results = chain.execute(content=original_content)
            
            # Check results
            if results["success"]:
                self.logger.info("Prompt chain executed successfully")
                
                # Display results from each step
                if self.verbose:
                    self.logger.debug(f"Step 1 - Key Topics: {results['steps'][0]['result'].text[:150]}...")
                    self.logger.debug(f"Step 2 - Style Analysis: {results['steps'][1]['result'].text[:150]}...")
                
                # Display final recommendations
                if 'final_result' in results and results['final_result']:
                    self.logger.info("Generated recommendations:")
                    self.logger.info(results['final_result'].text)
                    self.logger.info("✓ PASS: Prompt chaining works")
                else:
                    self.logger.error("✗ FAIL: Prompt chain did not produce final result")
            else:
                self.logger.error(f"✗ FAIL: Prompt chain execution failed: {results}")
                
        except Exception as e:
            self.logger.error(f"Error during prompt chaining test: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def test_edit_suggestions(self):
        """Test agent reasoning capabilities for edit suggestions."""
        self.logger.info("\n" + "="*50)
        self.logger.info("Test 8: Agent Reasoning for Edit Suggestions")
        self.logger.info("="*50)
        
        # Skip test if LLM interface is not available
        if not hasattr(self.app, 'llm_interface') or not self.app.llm_interface or not self.app.llm_interface.is_available():
            self.logger.warning("⚠ WARN: Skipping edit suggestions test (LLM interface not available)")
            return
            
        # Get original content and create test RAG system
        original_content = self.app.google_docs_api.get_document_content(self.document_id)
        
        if not hasattr(self.app, 'rag_system') or not self.app.rag_system:
            self.logger.warning("⚠ WARN: RAG system not available")
            rag_system = RAGSystem()
        else:
            rag_system = self.app.rag_system
            
        # Process document to make sure it's in the RAG system
        metadata = {
            "title": self.get_document_title(),
            "version": 1,
            "last_updated": datetime.now().isoformat()
        }
        rag_system.process_document(self.document_id, original_content, metadata)
        
        try:
            # Create a test suggestion system
            suggestion_system = EditSuggestionSystem(
                llm_interface=self.app.llm_interface,
                rag_system=rag_system
            )
            
            # Generate suggestions
            self.logger.info("Generating edit suggestions...")
            suggestions = suggestion_system.generate_suggestions(
                doc_id=self.document_id,
                max_suggestions=2
            )
            
            # Check results
            if suggestions:
                self.logger.info(f"Generated {len(suggestions)} edit suggestions")
                
                for i, suggestion in enumerate(suggestions):
                    self.logger.info(f"\nSuggestion {i+1}:")
                    self.logger.info(f"Section: {suggestion.section_title}")
                    self.logger.info(f"Type: {suggestion.suggestion_type}")
                    self.logger.info(f"Confidence: {suggestion.confidence:.2f}")
                    self.logger.info(f"Suggestion: {suggestion.suggestion[:150]}...")
                    self.logger.info(f"Reasoning: {suggestion.reasoning[:150]}...")
                
                self.logger.info("✓ PASS: Edit suggestion system works")
            else:
                self.logger.warning("⚠ WARN: No suggestions generated (document may be too short or perfect)")
                
        except Exception as e:
            self.logger.error(f"Error during edit suggestions test: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
    def test_context_windows(self):
        """Test context windows incorporating document structure."""
        self.logger.info("\n" + "="*50)
        self.logger.info("Test 9: Context Windows with Document Structure")
        self.logger.info("="*50)
        
        # Get original content and create test RAG system
        original_content = self.app.google_docs_api.get_document_content(self.document_id)
        
        if not hasattr(self.app, 'rag_system') or not self.app.rag_system:
            self.logger.warning("⚠ WARN: RAG system not available")
            rag_system = RAGSystem()
        else:
            rag_system = self.app.rag_system
            
        # Process document to make sure it's in the RAG system
        metadata = {
            "title": self.get_document_title(),
            "version": 1,
            "last_updated": datetime.now().isoformat()
        }
        rag_system.process_document(self.document_id, original_content, metadata)
        
        try:
            # Create a context window manager
            context_manager = ContextWindowManager(rag_system)
            
            # Create a focused window
            self.logger.info("Creating focused context window...")
            focused_window = context_manager.create_focused_window(
                doc_id=self.document_id,
                include_metadata=True,
                include_history=True
            )
            
            if focused_window:
                self.logger.info(f"Created focused window for document '{focused_window.title}'")
                self.logger.info(f"Focus section: {focused_window.focus_section}")
                self.logger.info(f"Window metadata: {list(focused_window.metadata.keys())}")
                self.logger.info(f"Window content length: {len(focused_window.content)} chars")
                
                formatted_text = focused_window.to_text()
                self.logger.info(f"Formatted window text sample: {formatted_text[:200]}...")
                
                self.logger.info("✓ PASS: Focused context window creation works")
            else:
                self.logger.error("✗ FAIL: Failed to create focused context window")
                
            # Create a document map
            self.logger.info("\nCreating document map...")
            doc_map = context_manager.create_document_map(self.document_id)
            
            if doc_map:
                self.logger.info(f"Created document map: {doc_map.title}")
                self.logger.info(f"Map metadata: {list(doc_map.metadata.keys())}")
                self.logger.info(f"Map content sample: {doc_map.content[:200]}...")
                
                self.logger.info("✓ PASS: Document map creation works")
            else:
                self.logger.error("✗ FAIL: Failed to create document map")
                
            # Create a query-focused window
            self.logger.info("\nCreating query-focused context window...")
            query = "testing document"
            query_window = context_manager.create_query_focused_window(
                query=query,
                doc_id=self.document_id,
                max_sections=3
            )
            
            if query_window:
                self.logger.info(f"Created query window: {query_window.title}")
                self.logger.info(f"Query metadata: {list(query_window.metadata.keys())}")
                self.logger.info(f"Query content sample: {query_window.content[:200]}...")
                
                self.logger.info("✓ PASS: Query-focused window creation works")
            else:
                self.logger.warning("⚠ WARN: Query-focused window returned no results")
                
        except Exception as e:
            self.logger.error(f"Error during context windows test: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def test_feedback_loops(self):
        """Test feedback loops for suggestion improvement."""
        self.logger.info("\n" + "="*50)
        self.logger.info("Test 10: Feedback Loops for Suggestion Improvement")
        self.logger.info("="*50)
        
        # Skip test if LLM interface is not available
        if not hasattr(self.app, 'llm_interface') or not self.app.llm_interface or not self.app.llm_interface.is_available():
            self.logger.warning("⚠ WARN: Skipping feedback loops test (LLM interface not available)")
            return
        
        try:
            # Create a test suggestion and feedback system
            import tempfile
            
            # Create a temporary database file for testing
            with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
                temp_db_path = temp_db.name
            
            feedback_system = FeedbackLoopSystem(
                llm_interface=self.app.llm_interface,
                feedback_db_path=temp_db_path
            )
            
            # Create test suggestions
            from src.services.edit_suggestion.edit_suggestion_system import EditSuggestion
            
            test_suggestions = [
                EditSuggestion(
                    section_id="section1",
                    section_title="Introduction",
                    original_text="This is the introduction to the document.",
                    suggestion="Add more context about the purpose of this document.",
                    suggestion_type="completeness",
                    reasoning="The introduction lacks a clear statement of purpose.",
                    confidence=0.85
                ),
                EditSuggestion(
                    section_id="section2",
                    section_title="Methodology",
                    original_text="The methodology follows standard practices.",
                    suggestion="Specify which exact methodologies were used and provide citations.",
                    suggestion_type="clarity",
                    reasoning="The statement is too vague and needs specificity.",
                    confidence=0.9
                ),
                EditSuggestion(
                    section_id="section3",
                    section_title="Results",
                    original_text="The results were positive.",
                    suggestion="Include quantitative metrics and statistical significance.",
                    suggestion_type="completeness",
                    reasoning="Quantitative results are needed for proper evaluation.",
                    confidence=0.95
                )
            ]
            
            # Record feedback for each suggestion (mix of accepted/rejected)
            self.logger.info("Recording simulated user feedback...")
            suggestion_ids = []
            
            # First suggestion - accepted with feedback
            suggestion_ids.append(
                feedback_system.record_feedback(
                    suggestion=test_suggestions[0],
                    accepted=True,
                    user_feedback="Good suggestion, made the change.",
                    user_id="user1"
                )
            )
            
            # Second suggestion - rejected with feedback
            suggestion_ids.append(
                feedback_system.record_feedback(
                    suggestion=test_suggestions[1],
                    accepted=False,
                    user_feedback="Too specific for this overview section.",
                    user_id="user2"
                )
            )
            
            # Third suggestion - accepted without feedback
            suggestion_ids.append(
                feedback_system.record_feedback(
                    suggestion=test_suggestions[2],
                    accepted=True,
                    user_id="user1"
                )
            )
            
            # Get feedback statistics
            stats = feedback_system.get_feedback_stats()
            
            if stats:
                self.logger.info(f"Feedback stats: {stats['total_suggestions']} total suggestions")
                self.logger.info(f"Accepted: {stats['accepted_count']}, Rejected: {stats['rejected_count']}")
                self.logger.info(f"Acceptance rate: {stats['acceptance_rate']:.2f}")
                
                if 'suggestion_types' in stats:
                    for stype, data in stats['suggestion_types'].items():
                        self.logger.info(f"Type '{stype}': {data['count']} suggestions, " +
                                        f"{data['acceptance_rate']*100:.0f}% accepted")
                
                self.logger.info("✓ PASS: Feedback statistics collection works")
            else:
                self.logger.error("✗ FAIL: Failed to retrieve feedback statistics")
            
            # Force pattern update (needs 10 entries normally, but we'll bypass the check)
            feedback_system._should_update_patterns = lambda: True
            patterns = feedback_system.update_patterns()
            
            self.logger.info(f"Extracted {len(patterns)} feedback patterns")
            
            # Test prompt improvement
            original_prompt = "Please suggest improvements for clarity in this section."
            improved_prompt = feedback_system.apply_feedback_learning(
                prompt=original_prompt,
                suggestion_type="clarity"
            )
            
            if improved_prompt != original_prompt:
                self.logger.info("Successfully applied feedback learning to prompt:")
                self.logger.info(improved_prompt)
                self.logger.info("✓ PASS: Feedback-based prompt improvement works")
            else:
                self.logger.info("No prompt improvements applied (expected with limited data)")
            
            # Clean up temporary database
            import os
            try:
                os.unlink(temp_db_path)
                self.logger.info("Cleaned up temporary feedback database")
            except Exception as e:
                self.logger.warning(f"Could not delete temporary database: {e}")
            
        except Exception as e:
            self.logger.error(f"Error during feedback loops test: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        
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
    parser = argparse.ArgumentParser(description="Test Phase 2 features of CollabGPT")
    parser.add_argument("-d", "--document_id", required=True, help="Google Docs document ID")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Run tests
    tester = Phase2Tester(args.document_id, args.verbose)
    tester.run_all_tests()


if __name__ == "__main__":
    main()