"""
Conflict detection module for CollabGPT.

This module identifies and manages editing conflicts in collaborative documents,
helping users resolve overlapping changes and maintain document integrity.
"""

import re
import uuid
import time
import difflib
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum, auto
from dataclasses import dataclass, field
import hashlib
from datetime import datetime

from ..utils import logger


class ConflictType(Enum):
    """Types of document editing conflicts."""
    CONTENT_OVERLAP = auto()  # Two users edited the same content
    SEQUENTIAL = auto()       # Edit based on outdated version
    STRUCTURAL = auto()       # Document structure changes (section reordering, etc.)
    FORMATTING = auto()       # Formatting conflicts (style, layout)
    REFERENCE = auto()        # References/links affected by other edits


@dataclass
class EditRecord:
    """Record of a document edit."""
    edit_id: str
    document_id: str
    user_id: str
    timestamp: float
    content_before_hash: str
    content_after_hash: str
    content_before: str
    content_after: str
    affected_sections: List[str] = field(default_factory=list)


@dataclass
class Conflict:
    """Represents a detected conflict between edits."""
    conflict_id: str
    document_id: str
    conflict_type: ConflictType
    severity: int  # 1-5 scale, 5 being most severe
    edits: List[EditRecord] = field(default_factory=list)
    description: str = ""
    suggested_resolution: str = ""
    affected_sections: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolved_at: Optional[float] = None
    resolved_by: Optional[str] = None
    resolution_method: Optional[str] = None


class ConflictDetector:
    """
    Detects and manages conflicts in collaborative document editing.
    """
    
    def __init__(self, conflict_window_seconds: int = 60):
        """
        Initialize the conflict detector.
        
        Args:
            conflict_window_seconds: Time window in seconds for considering 
                                    concurrent edits for conflict detection
        """
        self.logger = logger.get_logger("conflict_detector")
        self.conflict_window_seconds = conflict_window_seconds
        
        # Storage for edit records and conflicts
        self.edits: Dict[str, Dict[str, List[EditRecord]]] = {}  # document_id -> user_id -> [edits]
        self.conflicts: Dict[str, List[Conflict]] = {}  # document_id -> [conflicts]
        
        self.logger.info(f"Conflict detector initialized with {conflict_window_seconds}s window")
    
    def record_edit(self, document_id: str, user_id: str, 
                  content_before: str, content_after: str,
                  affected_sections: Optional[List[str]] = None) -> str:
        """
        Record a document edit and check for conflicts.
        
        Args:
            document_id: The document identifier
            user_id: The user who made the edit
            content_before: Document content before the edit
            content_after: Document content after the edit
            affected_sections: List of section names affected by the edit
            
        Returns:
            The edit_id of the recorded edit
        """
        # Create edit record
        edit_id = str(uuid.uuid4())
        
        # Generate content hashes
        before_hash = self._hash_content(content_before)
        after_hash = self._hash_content(content_after)
        
        # Infer affected sections if not provided
        if affected_sections is None:
            affected_sections = self._infer_affected_sections(content_before, content_after)
        
        # Create and store the edit record
        edit = EditRecord(
            edit_id=edit_id,
            document_id=document_id,
            user_id=user_id,
            timestamp=time.time(),
            content_before_hash=before_hash,
            content_after_hash=after_hash,
            content_before=content_before,
            content_after=content_after,
            affected_sections=affected_sections
        )
        
        # Initialize document entry if needed
        if document_id not in self.edits:
            self.edits[document_id] = {}
            
        # Initialize user entry if needed
        if user_id not in self.edits[document_id]:
            self.edits[document_id][user_id] = []
            
        # Add the edit record
        self.edits[document_id][user_id].append(edit)
        
        self.logger.info(f"Recorded edit {edit_id} by user {user_id} for document {document_id}")
        self.logger.info(f"Edit affected sections: {affected_sections}")
        
        # Check for conflicts
        self._detect_conflicts(document_id, edit)
        
        return edit_id
    
    def get_conflicts(self, document_id: str, resolved: bool = False) -> List[Conflict]:
        """
        Get conflicts for a document.
        
        Args:
            document_id: The document identifier
            resolved: Whether to include resolved conflicts (True) or 
                      only unresolved conflicts (False, default)
            
        Returns:
            List of detected conflicts
        """
        if document_id not in self.conflicts:
            return []
            
        conflicts = self.conflicts[document_id]
        
        # Filter based on resolved status
        if not resolved:
            conflicts = [c for c in conflicts if not c.resolved]
            
        return conflicts
    
    def mark_conflict_resolved(self, conflict_id: str, 
                             resolution_method: str, 
                             resolved_by: str = "system") -> bool:
        """
        Mark a conflict as resolved.
        
        Args:
            conflict_id: The conflict identifier
            resolution_method: Description of how the conflict was resolved
            resolved_by: User ID or system identifier of who resolved it
            
        Returns:
            True if successfully marked as resolved, False otherwise
        """
        # Find the conflict
        for document_id, conflicts in self.conflicts.items():
            for conflict in conflicts:
                if conflict.conflict_id == conflict_id:
                    conflict.resolved = True
                    conflict.resolved_at = time.time()
                    conflict.resolved_by = resolved_by
                    conflict.resolution_method = resolution_method
                    
                    self.logger.info(f"Marked conflict {conflict_id} as resolved by {resolved_by}")
                    self.logger.info(f"Resolution method: {resolution_method}")
                    
                    return True
        
        self.logger.warning(f"Could not find conflict {conflict_id} to mark as resolved")
        return False
    
    def get_edit_history(self, document_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get the edit history for a document.
        
        Args:
            document_id: The document identifier
            limit: Maximum number of edits to return
            
        Returns:
            List of edit records in chronological order
        """
        if document_id not in self.edits:
            return []
            
        # Gather all edits from all users
        all_edits = []
        for user_edits in self.edits[document_id].values():
            all_edits.extend(user_edits)
            
        # Sort by timestamp (most recent first)
        all_edits.sort(key=lambda e: e.timestamp, reverse=True)
        
        # Apply limit
        all_edits = all_edits[:limit]
        
        # Convert to dictionaries
        return [
            {
                'edit_id': edit.edit_id,
                'user_id': edit.user_id,
                'timestamp': edit.timestamp,
                'affected_sections': edit.affected_sections,
                'time': datetime.fromtimestamp(edit.timestamp).strftime('%Y-%m-%d %H:%M:%S')
            }
            for edit in all_edits
        ]
    
    def _detect_conflicts(self, document_id: str, current_edit: EditRecord) -> None:
        """
        Detect conflicts with the current edit.
        
        Args:
            document_id: The document identifier
            current_edit: The edit to check for conflicts
        """
        # Ensure we have a conflict list for this document
        if document_id not in self.conflicts:
            self.conflicts[document_id] = []
            
        current_time = time.time()
        conflict_window_start = current_time - self.conflict_window_seconds
        current_user = current_edit.user_id
        
        # Get all recent edits from other users
        other_recent_edits = []
        for user_id, user_edits in self.edits[document_id].items():
            if user_id != current_user:
                for edit in user_edits:
                    if edit.timestamp >= conflict_window_start:
                        other_recent_edits.append(edit)
        
        if not other_recent_edits:
            self.logger.info(f"No other recent edits found for conflict detection")
            return
            
        self.logger.info(f"Checking for conflicts against {len(other_recent_edits)} recent edits")
        
        # Check for each type of conflict
        
        # 1. Content overlap conflicts
        content_conflicts = self._detect_content_overlap_conflicts(current_edit, other_recent_edits)
        
        # 2. Sequential conflicts
        sequential_conflicts = self._detect_sequential_conflicts(current_edit, other_recent_edits)
        
        # 3. Structural conflicts
        structural_conflicts = self._detect_structural_conflicts(current_edit, other_recent_edits)
        
        # Add detected conflicts to the list
        all_conflicts = content_conflicts + sequential_conflicts + structural_conflicts
        
        if all_conflicts:
            self.logger.warning(f"Detected {len(all_conflicts)} conflicts for document {document_id}")
            for conflict in all_conflicts:
                self.conflicts[document_id].append(conflict)
        else:
            self.logger.info(f"No conflicts detected for document {document_id}")
    
    def _detect_content_overlap_conflicts(self, current_edit: EditRecord, 
                                       other_edits: List[EditRecord]) -> List[Conflict]:
        """
        Detect content overlap conflicts where multiple users edited the same content.
        
        Args:
            current_edit: The current edit
            other_edits: List of other recent edits to check against
            
        Returns:
            List of detected content overlap conflicts
        """
        conflicts = []
        document_id = current_edit.document_id
        
        # Get diff between current edit's before and after content
        current_diff = self._get_content_diff(current_edit.content_before, current_edit.content_after)
        
        for other_edit in other_edits:
            # Get diff for the other edit
            other_diff = self._get_content_diff(other_edit.content_before, other_edit.content_after)
            
            # Check for overlapping sections
            overlap = self._find_diff_overlap(current_diff, other_diff)
            
            if overlap:
                self.logger.info(f"Found content overlap between edits from {current_edit.user_id} and {other_edit.user_id}")
                
                # Check if we already have a conflict for these edits
                conflict_exists = False
                for conflict in self.conflicts.get(document_id, []):
                    if (conflict.conflict_type == ConflictType.CONTENT_OVERLAP and
                        any(e.edit_id == current_edit.edit_id for e in conflict.edits) and
                        any(e.edit_id == other_edit.edit_id for e in conflict.edits)):
                        conflict_exists = True
                        break
                
                if not conflict_exists:
                    # Create a new conflict
                    overlap_text = "\n".join(overlap)
                    overlap_preview = overlap_text[:100] + ("..." if len(overlap_text) > 100 else "")
                    
                    # Find affected sections
                    affected_sections = list(set(current_edit.affected_sections) & 
                                             set(other_edit.affected_sections))
                    
                    # Determine severity based on overlap size and affected sections
                    severity = min(5, 2 + len(affected_sections))
                    
                    conflict = Conflict(
                        conflict_id=str(uuid.uuid4()),
                        document_id=document_id,
                        conflict_type=ConflictType.CONTENT_OVERLAP,
                        severity=severity,
                        edits=[current_edit, other_edit],
                        description=f"Content overlap conflict between edits by {current_edit.user_id} "
                                    f"and {other_edit.user_id} in {', '.join(affected_sections) if affected_sections else 'unknown section'}.",
                        suggested_resolution=f"Review both changes and merge manually. The overlapping content includes: \"{overlap_preview}\"",
                        affected_sections=affected_sections
                    )
                    
                    conflicts.append(conflict)
        
        return conflicts
    
    def _detect_sequential_conflicts(self, current_edit: EditRecord, 
                                  other_edits: List[EditRecord]) -> List[Conflict]:
        """
        Detect sequential conflicts where an edit was based on outdated content.
        
        Args:
            current_edit: The current edit
            other_edits: List of other recent edits to check against
            
        Returns:
            List of detected sequential conflicts
        """
        conflicts = []
        document_id = current_edit.document_id
        
        # Check if current edit is based on outdated content
        for other_edit in other_edits:
            # If the other edit happened after current edit's base content was fetched
            if (current_edit.content_before_hash == other_edit.content_before_hash and
                current_edit.content_after_hash != other_edit.content_after_hash):
                
                # This might be a sequential conflict - both edits started from the same base
                self.logger.info(f"Possible sequential conflict: {current_edit.user_id} and {other_edit.user_id} "
                            f"edited from the same base content")
                
                # Check if we already have a conflict for these edits
                conflict_exists = False
                for conflict in self.conflicts.get(document_id, []):
                    if (conflict.conflict_type == ConflictType.SEQUENTIAL and
                        any(e.edit_id == current_edit.edit_id for e in conflict.edits) and
                        any(e.edit_id == other_edit.edit_id for e in conflict.edits)):
                        conflict_exists = True
                        break
                
                if not conflict_exists:
                    # Determine affected sections
                    affected_sections = list(set(current_edit.affected_sections) | 
                                              set(other_edit.affected_sections))
                    
                    # Sequential conflicts can be less severe if different sections
                    common_sections = set(current_edit.affected_sections) & set(other_edit.affected_sections)
                    severity = 4 if common_sections else 2
                    
                    conflict = Conflict(
                        conflict_id=str(uuid.uuid4()),
                        document_id=document_id,
                        conflict_type=ConflictType.SEQUENTIAL,
                        severity=severity,
                        edits=[current_edit, other_edit],
                        description=f"Sequential conflict: {current_edit.user_id} made changes without "
                                    f"seeing {other_edit.user_id}'s prior edits to the same content.",
                        suggested_resolution=f"Review both changes and create a new version that incorporates both edits.",
                        affected_sections=affected_sections
                    )
                    
                    conflicts.append(conflict)
        
        return conflicts
    
    def _detect_structural_conflicts(self, current_edit: EditRecord, 
                                  other_edits: List[EditRecord]) -> List[Conflict]:
        """
        Detect structural conflicts like section reordering or content reorganization.
        
        Args:
            current_edit: The current edit
            other_edits: List of other recent edits to check against
            
        Returns:
            List of detected structural conflicts
        """
        conflicts = []
        document_id = current_edit.document_id
        
        # Extract sections from before and after content for current edit
        current_before_sections = self._extract_sections(current_edit.content_before)
        current_after_sections = self._extract_sections(current_edit.content_after)
        
        # Check if sections were reordered, added, or removed
        current_structural_change = self._detect_structural_change(
            current_before_sections, current_after_sections
        )
        
        if not current_structural_change:
            # No structural change in current edit
            return []
            
        for other_edit in other_edits:
            # Extract sections for the other edit
            other_before_sections = self._extract_sections(other_edit.content_before)
            other_after_sections = self._extract_sections(other_edit.content_after)
            
            # Check if other edit also made structural changes
            other_structural_change = self._detect_structural_change(
                other_before_sections, other_after_sections
            )
            
            if other_structural_change:
                self.logger.info(f"Possible structural conflict: Both {current_edit.user_id} and {other_edit.user_id} "
                            f"made structural changes")
                
                # Check if the structural changes conflict with each other
                conflicting_changes = self._check_conflicting_structural_changes(
                    current_before_sections, current_after_sections,
                    other_before_sections, other_after_sections
                )
                
                if conflicting_changes:
                    # Check if we already have a conflict for these edits
                    conflict_exists = False
                    for conflict in self.conflicts.get(document_id, []):
                        if (conflict.conflict_type == ConflictType.STRUCTURAL and
                            any(e.edit_id == current_edit.edit_id for e in conflict.edits) and
                            any(e.edit_id == other_edit.edit_id for e in conflict.edits)):
                            conflict_exists = True
                            break
                    
                    if not conflict_exists:
                        # Structural conflicts tend to be severe
                        severity = 5 if len(conflicting_changes) > 2 else 4
                        
                        # Create conflict description based on the changes
                        change_description = ""
                        if "reordered" in conflicting_changes:
                            change_description += "reordered sections"
                        if "added" in conflicting_changes:
                            change_description += ", added sections" if change_description else "added sections"
                        if "removed" in conflicting_changes:
                            change_description += ", removed sections" if change_description else "removed sections"
                        
                        # Get affected sections from both edits
                        affected_sections = list(set(current_edit.affected_sections) | 
                                                 set(other_edit.affected_sections))
                        
                        conflict = Conflict(
                            conflict_id=str(uuid.uuid4()),
                            document_id=document_id,
                            conflict_type=ConflictType.STRUCTURAL,
                            severity=severity,
                            edits=[current_edit, other_edit],
                            description=f"Structural conflict: Both {current_edit.user_id} and {other_edit.user_id} "
                                        f"{change_description} simultaneously.",
                            suggested_resolution=f"Review both versions and manually merge the structural changes. "
                                                f"This requires careful attention to document organization.",
                            affected_sections=affected_sections
                        )
                        
                        conflicts.append(conflict)
        
        return conflicts
    
    def _hash_content(self, content: str) -> str:
        """
        Generate a hash of the content for quick comparisons.
        
        Args:
            content: The content to hash
            
        Returns:
            SHA-256 hash of the content
        """
        if not content:
            return ""
            
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _get_content_diff(self, before: str, after: str) -> List[str]:
        """
        Get the diff between before and after content.
        
        Args:
            before: The content before changes
            after: The content after changes
            
        Returns:
            List of lines that were changed
        """
        before_lines = before.splitlines()
        after_lines = after.splitlines()
        
        # Get the diff
        diff = difflib.unified_diff(before_lines, after_lines, lineterm='')
        
        # Skip the headers (first 3 lines)
        diff_lines = list(diff)[3:]
        
        # Extract only added/removed lines
        changed_lines = [line for line in diff_lines if line.startswith('+') or line.startswith('-')]
        
        return changed_lines
    
    def _find_diff_overlap(self, diff1: List[str], diff2: List[str]) -> List[str]:
        """
        Find overlapping changes in two diffs.
        
        Args:
            diff1: First diff
            diff2: Second diff
            
        Returns:
            List of overlapping lines
        """
        # Extract the affected lines (without +/- prefix)
        lines1 = [line[1:].strip() for line in diff1]
        lines2 = [line[1:].strip() for line in diff2]
        
        # Find overlapping lines
        overlap = []
        for line in lines1:
            if line and any(self._line_similarity(line, l2) > 0.7 for l2 in lines2):
                overlap.append(line)
                
        return overlap
    
    def _line_similarity(self, line1: str, line2: str) -> float:
        """
        Calculate similarity between two lines.
        
        Args:
            line1: First line
            line2: Second line
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not line1 or not line2:
            return 0.0
            
        # Use difflib's SequenceMatcher for similarity
        return difflib.SequenceMatcher(None, line1, line2).ratio()
    
    def _infer_affected_sections(self, content_before: str, content_after: str) -> List[str]:
        """
        Infer which sections were affected by changes.
        
        Args:
            content_before: Content before changes
            content_after: Content after changes
            
        Returns:
            List of affected section names
        """
        # Extract sections
        before_sections = self._extract_sections(content_before)
        after_sections = self._extract_sections(content_after)
        
        # Identify different or changed sections
        affected = set()
        
        # Added or removed sections
        before_titles = set(s['title'] for s in before_sections)
        after_titles = set(s['title'] for s in after_sections)
        
        # Sections that were added or removed
        affected.update(before_titles ^ after_titles)
        
        # Sections with content changes
        for before_section in before_sections:
            title = before_section['title']
            if title in after_titles:
                # Find matching section in after
                after_section = next((s for s in after_sections if s['title'] == title), None)
                if after_section:
                    # Check if content is different
                    if before_section['content'] != after_section['content']:
                        affected.add(title)
        
        return list(affected)
    
    def _extract_sections(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract sections from document content.
        
        Args:
            content: Document content
            
        Returns:
            List of section dictionaries with title and content
        """
        if not content:
            return []
            
        # Simple section extraction based on markdown headings
        # This can be extended to handle other formats
        
        # Split content into lines
        lines = content.splitlines()
        
        sections = []
        current_section = None
        current_content = []
        
        for line in lines:
            # Check if line is a heading
            heading_match = re.match(r'^(#+)\s+(.+)$', line)
            
            if heading_match:
                # New heading found
                if current_section:
                    # Save previous section
                    sections.append({
                        'title': current_section,
                        'content': '\n'.join(current_content)
                    })
                
                # Start new section
                current_section = heading_match.group(2).strip()
                current_content = [line]
            elif current_section:
                # Continue adding to current section
                current_content.append(line)
            else:
                # Content before first heading
                current_content.append(line)
        
        # Add the last section if there is one
        if current_section:
            sections.append({
                'title': current_section,
                'content': '\n'.join(current_content)
            })
        elif current_content:
            # Document without headings, treat as one section
            sections.append({
                'title': 'Document',
                'content': '\n'.join(current_content)
            })
        
        return sections
    
    def _detect_structural_change(self, before_sections: List[Dict[str, Any]], 
                               after_sections: List[Dict[str, Any]]) -> bool:
        """
        Detect if there were structural changes (section reordering, adding, removing).
        
        Args:
            before_sections: Sections before changes
            after_sections: Sections after changes
            
        Returns:
            True if structural changes were detected, False otherwise
        """
        before_titles = [s['title'] for s in before_sections]
        after_titles = [s['title'] for s in after_sections]
        
        # Check if sections were added or removed
        if set(before_titles) != set(after_titles):
            return True
            
        # Check if sections were reordered
        if before_titles != after_titles:
            return True
            
        return False
    
    def _check_conflicting_structural_changes(self, 
                                          current_before: List[Dict[str, Any]], 
                                          current_after: List[Dict[str, Any]],
                                          other_before: List[Dict[str, Any]], 
                                          other_after: List[Dict[str, Any]]) -> Set[str]:
        """
        Check if two sets of structural changes conflict with each other.
        
        Args:
            current_before: Sections before current edit
            current_after: Sections after current edit
            other_before: Sections before other edit
            other_after: Sections after other edit
            
        Returns:
            Set of conflict types ("reordered", "added", "removed")
        """
        conflicts = set()
        
        # Get section titles
        current_before_titles = [s['title'] for s in current_before]
        current_after_titles = [s['title'] for s in current_after]
        other_before_titles = [s['title'] for s in other_before]
        other_after_titles = [s['title'] for s in other_after]
        
        # Check for reordering conflicts
        current_reordered = current_before_titles != current_after_titles
        other_reordered = other_before_titles != other_after_titles
        
        if current_reordered and other_reordered:
            conflicts.add("reordered")
        
        # Check for added sections conflicts
        current_added = set(current_after_titles) - set(current_before_titles)
        other_added = set(other_after_titles) - set(other_before_titles)
        
        if current_added and other_added:
            # Check if different sections were added in each edit
            if current_added != other_added:
                conflicts.add("added")
        
        # Check for removed sections conflicts
        current_removed = set(current_before_titles) - set(current_after_titles)
        other_removed = set(other_before_titles) - set(other_after_titles)
        
        if current_removed and other_removed:
            # Check if different sections were removed in each edit
            if current_removed != other_removed:
                conflicts.add("removed")
                
        # Special case: one edit added a section that another edit removed
        if (current_added and other_removed and current_added & other_removed) or \
           (current_removed and other_added and current_removed & other_added):
            conflicts.add("added")
            conflicts.add("removed")
        
        return conflicts

    def reset_conflicts(self, document_id: str) -> bool:
        """
        Clear all conflicts for a document.
        
        Args:
            document_id: The document identifier
            
        Returns:
            True if conflicts were cleared, False if document had no conflicts
        """
        if document_id in self.conflicts:
            conflict_count = len(self.conflicts[document_id])
            self.conflicts[document_id] = []
            self.logger.info(f"Cleared {conflict_count} conflicts for document {document_id}")
            return True
        
        self.logger.info(f"No conflicts to clear for document {document_id}")
        return False