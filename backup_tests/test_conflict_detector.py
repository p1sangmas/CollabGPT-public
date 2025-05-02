#!/usr/bin/env python
"""
Test file for the ConflictDetector class in CollabGPT.

This test file verifies that the ConflictDetector correctly identifies and
manages different types of conflicts in collaborative document editing.
"""

import unittest
import time
import json
from src.services.conflict_detector import ConflictDetector, ConflictType
from src.utils import logger

# Set up logging for tests
logger = logger.get_logger("test_conflict_detector")


class TestConflictDetector(unittest.TestCase):
    """Test cases for the ConflictDetector class."""

    def setUp(self):
        """Set up test environment before each test."""
        # Initialize conflict detector with a small window for testing
        self.conflict_detector = ConflictDetector(conflict_window_seconds=10)
        
        # Mock document and user IDs
        self.document_id = "test_doc_001"
        self.user1_id = "user1"
        self.user2_id = "user2"
        self.user3_id = "user3"
        
        # Sample document contents
        self.doc_content_original = """# Document Title

## Introduction
This is a test document for conflict detection.
It has multiple sections to test different scenarios.

## Section 1
This section contains important content.
We will use it to test content overlap conflicts.

## Section 2
This is another section with different content.
This will be used for testing sequential conflicts.

## Section 3
The final section contains concluding remarks.
We will use this to test structural conflicts.
"""

    def tearDown(self):
        """Clean up after each test."""
        self.conflict_detector = None

    def test_record_edit(self):
        """Test that edits are properly recorded."""
        # User 1 makes an edit
        modified_content = self.doc_content_original.replace(
            "This section contains important content.",
            "This section contains very important content that was updated."
        )
        
        edit_id = self.conflict_detector.record_edit(
            document_id=self.document_id,
            user_id=self.user1_id,
            content_before=self.doc_content_original,
            content_after=modified_content,
            affected_sections=["Section 1"]
        )
        
        # Verify the edit was recorded
        self.assertIn(self.document_id, self.conflict_detector.edits)
        self.assertIn(self.user1_id, self.conflict_detector.edits[self.document_id])
        self.assertEqual(len(self.conflict_detector.edits[self.document_id][self.user1_id]), 1)
        self.assertEqual(
            self.conflict_detector.edits[self.document_id][self.user1_id][0].edit_id, 
            edit_id
        )
        
        print(f"✓ Edit recording test passed: Edit ID {edit_id} recorded successfully")
        
    def test_content_overlap_conflict(self):
        """Test detection of content overlap conflicts."""
        # User 1 makes an edit to Section 1
        user1_modified = self.doc_content_original.replace(
            "This section contains important content.",
            "This section contains important content that user1 modified."
        )
        
        self.conflict_detector.record_edit(
            document_id=self.document_id,
            user_id=self.user1_id,
            content_before=self.doc_content_original,
            content_after=user1_modified,
            affected_sections=["Section 1"]
        )
        
        # User 2 makes a different edit to the same section
        user2_modified = self.doc_content_original.replace(
            "This section contains important content.",
            "This section contains crucial information that user2 changed."
        )
        
        self.conflict_detector.record_edit(
            document_id=self.document_id,
            user_id=self.user2_id,
            content_before=self.doc_content_original,
            content_after=user2_modified,
            affected_sections=["Section 1"]
        )
        
        # Check for conflicts
        conflicts = self.conflict_detector.get_conflicts(self.document_id)
        
        # Verify content overlap conflict is detected
        self.assertTrue(any(c.conflict_type == ConflictType.CONTENT_OVERLAP for c in conflicts))
        
        # Print conflict details
        overlap_conflicts = [c for c in conflicts if c.conflict_type == ConflictType.CONTENT_OVERLAP]
        for conflict in overlap_conflicts:
            print(f"\n✓ Content overlap conflict detected:")
            print(f"  Conflict ID: {conflict.conflict_id}")
            print(f"  Severity: {conflict.severity}")
            print(f"  Description: {conflict.description}")
            print(f"  Resolution: {conflict.suggested_resolution}")
        
    def test_sequential_conflict(self):
        """Test detection of sequential conflicts."""
        # User 1 makes an edit
        user1_modified = self.doc_content_original.replace(
            "This is another section with different content.",
            "This is another section with updated content by user1."
        )
        
        # Record user1 edit
        edit1_id = self.conflict_detector.record_edit(
            document_id=self.document_id,
            user_id=self.user1_id,
            content_before=self.doc_content_original,
            content_after=user1_modified,
            affected_sections=["Section 2"]
        )
        
        # Small delay to ensure the edits have different timestamps
        time.sleep(0.1)
        
        # User 2 makes an edit based on the original content (not seeing user1's changes)
        user2_modified = self.doc_content_original.replace(
            "This is another section with different content.",
            "This section has been completely rewritten by user2."
        )
        
        # Record user2 edit
        edit2_id = self.conflict_detector.record_edit(
            document_id=self.document_id,
            user_id=self.user2_id,
            content_before=self.doc_content_original,
            content_after=user2_modified,
            affected_sections=["Section 2"]
        )
        
        # Check for conflicts
        conflicts = self.conflict_detector.get_conflicts(self.document_id)
        
        # Verify sequential conflict is detected
        self.assertTrue(any(c.conflict_type == ConflictType.SEQUENTIAL for c in conflicts))
        
        # Print conflict details
        sequential_conflicts = [c for c in conflicts if c.conflict_type == ConflictType.SEQUENTIAL]
        for conflict in sequential_conflicts:
            print(f"\n✓ Sequential conflict detected:")
            print(f"  Conflict ID: {conflict.conflict_id}")
            print(f"  Severity: {conflict.severity}")
            print(f"  Description: {conflict.description}")
            print(f"  Resolution: {conflict.suggested_resolution}")
    
    def test_structural_conflict(self):
        """Test detection of structural conflicts."""
        # User 1 reorders sections
        user1_modified = """# Document Title

## Introduction
This is a test document for conflict detection.
It has multiple sections to test different scenarios.

## Section 3
The final section contains concluding remarks.
We will use this to test structural conflicts.

## Section 1
This section contains important content.
We will use it to test content overlap conflicts.

## Section 2
This is another section with different content.
This will be used for testing sequential conflicts.
"""
        
        # Record user1 edit (reordering sections)
        self.conflict_detector.record_edit(
            document_id=self.document_id,
            user_id=self.user1_id,
            content_before=self.doc_content_original,
            content_after=user1_modified,
            affected_sections=["Section 1", "Section 2", "Section 3"]
        )
        
        # User 2 adds a new section
        user2_modified = """# Document Title

## Introduction
This is a test document for conflict detection.
It has multiple sections to test different scenarios.

## Section 1
This section contains important content.
We will use it to test content overlap conflicts.

## Section 2
This is another section with different content.
This will be used for testing sequential conflicts.

## Section 3
The final section contains concluding remarks.
We will use this to test structural conflicts.

## New Section 4
This is a completely new section added by user2.
It contains additional information for the document.
"""
        
        # Record user2 edit (adding a section)
        self.conflict_detector.record_edit(
            document_id=self.document_id,
            user_id=self.user2_id,
            content_before=self.doc_content_original,
            content_after=user2_modified,
            affected_sections=["New Section 4"]
        )
        
        # Check for conflicts
        conflicts = self.conflict_detector.get_conflicts(self.document_id)
        
        # Verify structural conflict is detected
        self.assertTrue(any(c.conflict_type == ConflictType.STRUCTURAL for c in conflicts))
        
        # Print conflict details
        structural_conflicts = [c for c in conflicts if c.conflict_type == ConflictType.STRUCTURAL]
        for conflict in structural_conflicts:
            print(f"\n✓ Structural conflict detected:")
            print(f"  Conflict ID: {conflict.conflict_id}")
            print(f"  Severity: {conflict.severity}")
            print(f"  Description: {conflict.description}")
            print(f"  Resolution: {conflict.suggested_resolution}")
    
    def test_conflict_resolution(self):
        """Test marking conflicts as resolved."""
        # Create a conflict situation
        user1_modified = self.doc_content_original.replace(
            "This section contains important content.",
            "This section contains important content that user1 modified."
        )
        
        self.conflict_detector.record_edit(
            document_id=self.document_id,
            user_id=self.user1_id,
            content_before=self.doc_content_original,
            content_after=user1_modified,
            affected_sections=["Section 1"]
        )
        
        user2_modified = self.doc_content_original.replace(
            "This section contains important content.",
            "This section contains crucial information that user2 changed."
        )
        
        self.conflict_detector.record_edit(
            document_id=self.document_id,
            user_id=self.user2_id,
            content_before=self.doc_content_original,
            content_after=user2_modified,
            affected_sections=["Section 1"]
        )
        
        # Get the conflict
        conflicts = self.conflict_detector.get_conflicts(self.document_id)
        self.assertTrue(len(conflicts) > 0)
        
        conflict_id = conflicts[0].conflict_id
        
        # Mark the conflict as resolved
        resolution_success = self.conflict_detector.mark_conflict_resolved(
            conflict_id=conflict_id,
            resolution_method="Manual merge of both users' changes",
            resolved_by="admin"
        )
        
        # Verify resolution was successful
        self.assertTrue(resolution_success)
        
        # Verify conflict is now marked as resolved
        resolved_conflicts = self.conflict_detector.get_conflicts(self.document_id, resolved=True)
        unresolved_conflicts = self.conflict_detector.get_conflicts(self.document_id, resolved=False)
        
        self.assertTrue(any(c.conflict_id == conflict_id for c in resolved_conflicts))
        self.assertFalse(any(c.conflict_id == conflict_id for c in unresolved_conflicts))
        
        print(f"\n✓ Conflict resolution test passed:")
        print(f"  Conflict {conflict_id} successfully marked as resolved")
    
    def test_edit_history(self):
        """Test retrieving edit history for a document."""
        # Create a series of edits
        for i in range(3):
            modified_content = self.doc_content_original.replace(
                "This is a test document for conflict detection.",
                f"This is a test document for conflict detection (edit {i+1})."
            )
            
            self.conflict_detector.record_edit(
                document_id=self.document_id,
                user_id=self.user1_id,
                content_before=self.doc_content_original,
                content_after=modified_content,
                affected_sections=["Introduction"]
            )
            
            time.sleep(0.1)  # Ensure different timestamps
        
        # Get edit history
        history = self.conflict_detector.get_edit_history(self.document_id)
        
        # Verify history has the correct number of edits
        self.assertEqual(len(history), 3)
        
        # Verify history is in chronological order (most recent first)
        self.assertTrue(history[0]['timestamp'] > history[1]['timestamp'])
        self.assertTrue(history[1]['timestamp'] > history[2]['timestamp'])
        
        print(f"\n✓ Edit history test passed:")
        print(f"  Retrieved {len(history)} edits in correct order")
        
        # Pretty print the history
        print("\nEdit History:")
        for i, edit in enumerate(history):
            print(f"  {i+1}. Edit {edit['edit_id']} by {edit['user_id']} at {edit['time']}")


def run_all_tests():
    """Run all conflict detector tests with detailed output."""
    print("\n==== CONFLICT DETECTOR TESTS ====\n")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


def run_test_by_name(test_name):
    """Run a specific test by name."""
    print(f"\n==== RUNNING TEST: {test_name} ====\n")
    suite = unittest.TestSuite()
    suite.addTest(TestConflictDetector(test_name))
    runner = unittest.TextTestRunner()
    runner.run(suite)


if __name__ == "__main__":
    run_all_tests()
    
    # Optional: Run individual tests
    # Uncomment any of these to run specific tests
    # run_test_by_name('test_record_edit')
    # run_test_by_name('test_content_overlap_conflict')
    # run_test_by_name('test_sequential_conflict')
    # run_test_by_name('test_structural_conflict')
    # run_test_by_name('test_conflict_resolution')
    # run_test_by_name('test_edit_history')