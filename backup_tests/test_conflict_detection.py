"""
Test script for conflict detection functionality in CollabGPT.

This script tests the ConflictDetector class to ensure it properly detects
and manages conflicts in collaborative editing scenarios.
"""

import sys
import os
import time
from pathlib import Path

# Add the parent directory to sys.path to import the app modules
sys.path.append(str(Path(__file__).parent))

from src.services.conflict_detector import ConflictDetector, ConflictType
from src.utils.logger import get_logger

# Set up logging
logger = get_logger("test_conflict_detection")

def test_content_overlap_conflict():
    """Test detection of content overlap conflicts."""
    detector = ConflictDetector(conflict_window_seconds=10)
    document_id = "test_doc_001"
    
    # Initial content
    original_content = """
# Project Overview
This document outlines the project scope and deliverables.

## Introduction
The project aims to develop a collaborative AI system.

## Timeline
The project will be completed in 6 months.

## Budget
The budget for this project is $100,000.
"""
    
    # User 1 edits the Introduction section
    user1_edit = original_content.replace(
        "## Introduction\nThe project aims to develop a collaborative AI system.",
        "## Introduction\nThe project aims to develop a collaborative AI system with real-time capabilities."
    )
    
    # User 2 edits the same Introduction section differently
    user2_edit = original_content.replace(
        "## Introduction\nThe project aims to develop a collaborative AI system.",
        "## Introduction\nThe project aims to develop a team collaboration system powered by artificial intelligence."
    )
    
    # Record the edits
    logger.info("Recording User 1 edit")
    detector.record_edit(
        document_id=document_id,
        user_id="user1",
        content_before=original_content,
        content_after=user1_edit
    )
    
    # Add a small delay to simulate sequential edits
    time.sleep(1)
    
    logger.info("Recording User 2 edit")
    detector.record_edit(
        document_id=document_id,
        user_id="user2",
        content_before=original_content,
        content_after=user2_edit
    )
    
    # Check for conflicts
    conflicts = detector.get_conflicts(document_id)
    
    logger.info(f"Detected {len(conflicts)} conflicts")
    for conflict in conflicts:
        logger.info(f"Conflict {conflict.conflict_id}: {conflict.conflict_type.name} - Severity: {conflict.severity}")
        logger.info(f"Description: {conflict.description}")
        logger.info(f"Suggested resolution: {conflict.suggested_resolution}")
        
    assert len(conflicts) > 0, "Expected at least one content overlap conflict"
    assert any(c.conflict_type == ConflictType.CONTENT_OVERLAP for c in conflicts), "Expected CONTENT_OVERLAP conflict"
    
    return conflicts

def test_structural_conflict():
    """Test detection of structural conflicts."""
    detector = ConflictDetector(conflict_window_seconds=10)
    document_id = "test_doc_002"
    
    # Initial content
    original_content = """
# Project Overview
This document outlines the project scope and deliverables.

## Introduction
The project aims to develop a collaborative AI system.

## Timeline
The project will be completed in 6 months.

## Budget
The budget for this project is $100,000.
"""
    
    # User 1 rearranges sections (moves Timeline before Introduction)
    user1_edit = """
# Project Overview
This document outlines the project scope and deliverables.

## Timeline
The project will be completed in 6 months.

## Introduction
The project aims to develop a collaborative AI system.

## Budget
The budget for this project is $100,000.
"""
    
    # User 2 adds a new section and removes the Budget section
    user2_edit = """
# Project Overview
This document outlines the project scope and deliverables.

## Introduction
The project aims to develop a collaborative AI system.

## Timeline
The project will be completed in 6 months.

## Deliverables
The project will deliver a web-based application.
"""
    
    # Record the edits
    logger.info("Recording User 1 edit (rearranging sections)")
    detector.record_edit(
        document_id=document_id,
        user_id="user1",
        content_before=original_content,
        content_after=user1_edit
    )
    
    # Add a small delay to simulate sequential edits
    time.sleep(1)
    
    logger.info("Recording User 2 edit (adding/removing sections)")
    detector.record_edit(
        document_id=document_id,
        user_id="user2",
        content_before=original_content,
        content_after=user2_edit
    )
    
    # Check for conflicts
    conflicts = detector.get_conflicts(document_id)
    
    logger.info(f"Detected {len(conflicts)} conflicts")
    for conflict in conflicts:
        logger.info(f"Conflict {conflict.conflict_id}: {conflict.conflict_type.name} - Severity: {conflict.severity}")
        logger.info(f"Description: {conflict.description}")
        logger.info(f"Suggested resolution: {conflict.suggested_resolution}")
        
    assert len(conflicts) > 0, "Expected at least one structural conflict"
    assert any(c.conflict_type == ConflictType.STRUCTURAL for c in conflicts), "Expected STRUCTURAL conflict"
    
    return conflicts

def test_sequential_conflict():
    """Test detection of sequential editing conflicts."""
    detector = ConflictDetector(conflict_window_seconds=10)
    document_id = "test_doc_003"
    
    # Initial content
    original_content = """
# Meeting Notes
Date: May 1, 2025
Attendees: Alice, Bob, Charlie

## Discussion Points
1. Project timeline needs adjustment
2. Budget review is scheduled for next week
3. New requirements from the client
"""
    
    # User 1 updates the content
    user1_edit = """
# Meeting Notes
Date: May 1, 2025
Attendees: Alice, Bob, Charlie, David

## Discussion Points
1. Project timeline extended by 2 weeks
2. Budget review is scheduled for next week
3. New requirements from the client
4. Team capacity needs to be increased
"""
    
    # User 2 makes changes without seeing User 1's updates
    user2_edit = """
# Meeting Notes
Date: May 1, 2025
Attendees: Alice, Bob, Charlie

## Discussion Points
1. Project timeline needs adjustment
2. Budget review rescheduled to May 15
3. New requirements from the client have been documented
"""
    
    # Record the edits
    logger.info("Recording User 1 edit")
    detector.record_edit(
        document_id=document_id,
        user_id="user1",
        content_before=original_content,
        content_after=user1_edit
    )
    
    # Add a small delay to simulate sequential edits
    time.sleep(1)
    
    logger.info("Recording User 2 edit (based on original content)")
    detector.record_edit(
        document_id=document_id,
        user_id="user2",
        content_before=original_content,  # Note: based on original, not on user1's edit
        content_after=user2_edit
    )
    
    # Check for conflicts
    conflicts = detector.get_conflicts(document_id)
    
    logger.info(f"Detected {len(conflicts)} conflicts")
    for conflict in conflicts:
        logger.info(f"Conflict {conflict.conflict_id}: {conflict.conflict_type.name} - Severity: {conflict.severity}")
        logger.info(f"Description: {conflict.description}")
        logger.info(f"Suggested resolution: {conflict.suggested_resolution}")
        
    assert len(conflicts) > 0, "Expected at least one sequential conflict"
    assert any(c.conflict_type == ConflictType.SEQUENTIAL for c in conflicts), "Expected SEQUENTIAL conflict"
    
    return conflicts

def test_conflict_resolution():
    """Test marking conflicts as resolved."""
    detector = ConflictDetector(conflict_window_seconds=10)
    document_id = "test_doc_004"
    
    # Create a simple conflict
    original_content = "This is a test document."
    user1_edit = "This is a modified test document."
    user2_edit = "This is a test document with changes."
    
    detector.record_edit(document_id, "user1", original_content, user1_edit)
    detector.record_edit(document_id, "user2", original_content, user2_edit)
    
    # Get the conflicts
    conflicts = detector.get_conflicts(document_id)
    assert len(conflicts) > 0, "Expected at least one conflict"
    
    # Choose the first conflict to resolve
    conflict_id = conflicts[0].conflict_id
    
    # Mark it as resolved
    logger.info(f"Marking conflict {conflict_id} as resolved")
    result = detector.mark_conflict_resolved(conflict_id, "Manual merge by editor")
    assert result, "Expected successful conflict resolution"
    
    # Verify it's no longer in the unresolved conflicts
    unresolved = detector.get_conflicts(document_id)
    resolved = detector.get_conflicts(document_id, resolved=True)
    
    logger.info(f"Unresolved conflicts: {len(unresolved)}")
    logger.info(f"Total conflicts (including resolved): {len(resolved)}")
    
    assert len(unresolved) < len(conflicts), "Expected fewer unresolved conflicts"
    assert len(resolved) == len(conflicts), "Expected same total number of conflicts"
    
    # Verify the specific conflict is marked as resolved
    found = False
    for conflict in resolved:
        if conflict.conflict_id == conflict_id:
            assert conflict.resolved, "Expected conflict to be marked as resolved"
            assert conflict.resolution_method == "Manual merge by editor", "Expected correct resolution method"
            found = True
            break
            
    assert found, "Could not find the resolved conflict"
    
    return resolved

def run_all_tests():
    """Run all conflict detection tests."""
    logger.info("Starting ConflictDetector tests...")
    
    try:
        # Test content overlap conflicts
        logger.info("\n=== Testing Content Overlap Conflicts ===")
        content_conflicts = test_content_overlap_conflict()
        
        # Test structural conflicts
        logger.info("\n=== Testing Structural Conflicts ===")
        structural_conflicts = test_structural_conflict()
        
        # Test sequential conflicts
        logger.info("\n=== Testing Sequential Conflicts ===")
        sequential_conflicts = test_sequential_conflict()
        
        # Test conflict resolution
        logger.info("\n=== Testing Conflict Resolution ===")
        resolved_conflicts = test_conflict_resolution()
        
        # Print summary
        logger.info("\n=== Test Summary ===")
        logger.info(f"Content overlap conflicts detected: {len(content_conflicts)}")
        logger.info(f"Structural conflicts detected: {len(structural_conflicts)}")
        logger.info(f"Sequential conflicts detected: {len(sequential_conflicts)}")
        logger.info(f"Conflicts successfully resolved in resolution test: {len(resolved_conflicts)}")
        
        logger.info("All conflict detection tests completed successfully!")
        return True
        
    except AssertionError as e:
        logger.error(f"Test failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during testing: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)