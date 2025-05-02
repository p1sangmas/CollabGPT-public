"""
Comment Analyzer for CollabGPT.

This module provides functionality for analyzing, organizing, and managing
comments in collaborative documents.
"""

import re
import uuid
import threading
from typing import Dict, List, Any, Set, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict

from ..utils.performance import measure_latency, get_performance_monitor
from ..utils import logger

class CommentCategory(Enum):
    """Categories for document comments."""
    QUESTION = 1        # Questions about content
    SUGGESTION = 2      # Suggestions for changes
    CORRECTION = 3      # Error corrections
    CLARIFICATION = 4   # Requests for clarification
    APPROVAL = 5        # Approvals or acknowledgments
    PRIORITY = 6        # Comments marked as priority/important
    GENERAL = 7         # General comments
    
    @classmethod
    def get_description(cls, category):
        """Get a description of the comment category."""
        descriptions = {
            cls.QUESTION: "Questions about content",
            cls.SUGGESTION: "Suggestions for changes",
            cls.CORRECTION: "Error corrections",
            cls.CLARIFICATION: "Requests for clarification",
            cls.APPROVAL: "Approvals or acknowledgments",
            cls.PRIORITY: "Priority/important comments",
            cls.GENERAL: "General comments"
        }
        return descriptions.get(category, "Unknown category")


class CommentThread:
    """Represents a thread of comments in a document."""
    
    def __init__(self, thread_id: str, document_id: str, section: str = None):
        """
        Initialize a comment thread.
        
        Args:
            thread_id: Unique identifier for the thread
            document_id: Document identifier
            section: Document section the thread belongs to
        """
        self.thread_id = thread_id
        self.document_id = document_id
        self.section = section
        self.comments = []
        self.categories = set()
        self.participants = set()
        self.created_at = None
        self.last_activity = None
        self.resolved = False
        self.resolved_at = None
        self.resolved_by = None
        self.quoted_text = None
    
    def add_comment(self, comment: Dict[str, Any]) -> None:
        """
        Add a comment to the thread.
        
        Args:
            comment: Comment data
        """
        self.comments.append(comment)
        self.participants.add(comment['author']['id'])
        
        # Get timestamp - handle both 'created' and 'created_time' keys
        timestamp = comment.get('created', comment.get('created_time'))
        
        # If it's a string (ISO format from created_time), convert to timestamp
        if isinstance(timestamp, str):
            try:
                # Parse ISO format string to datetime then to timestamp
                timestamp = datetime.fromisoformat(timestamp).timestamp()
            except ValueError:
                # Fallback if parsing fails
                timestamp = datetime.now().timestamp()
        
        # Update timestamps
        comment_time = datetime.fromtimestamp(timestamp)
        if self.created_at is None or comment_time < self.created_at:
            self.created_at = comment_time
        
        if self.last_activity is None or comment_time > self.last_activity:
            self.last_activity = comment_time
        
        # Set quoted text from first comment if not already set
        if self.quoted_text is None and 'quotedText' in comment:
            self.quoted_text = comment['quotedText']
    
    def resolve(self, user_id: str) -> None:
        """
        Mark the thread as resolved.
        
        Args:
            user_id: User who resolved the thread
        """
        self.resolved = True
        self.resolved_at = datetime.now()
        self.resolved_by = user_id
    
    def get_comment_count(self) -> int:
        """Get the number of comments in the thread."""
        return len(self.comments)
    
    def get_latest_comment(self) -> Optional[Dict[str, Any]]:
        """Get the most recent comment in the thread."""
        if not self.comments:
            return None
            
        # Handle both 'created' and 'created_time' keys for flexibility
        return max(self.comments, key=lambda c: c.get('created', c.get('created_time', 0)))
    
    def get_thread_summary(self) -> Dict[str, Any]:
        """Get a summary of the thread."""
        return {
            'thread_id': self.thread_id,
            'document_id': self.document_id,
            'section': self.section,
            'comment_count': self.get_comment_count(),
            'participants': list(self.participants),
            'categories': [cat.name for cat in self.categories],
            'created_at': self.created_at,
            'last_activity': self.last_activity,
            'resolved': self.resolved,
            'resolved_at': self.resolved_at,
            'resolved_by': self.resolved_by,
            'quoted_text': self.quoted_text
        }


class CommentAnalyzer:
    """
    Analyzes and organizes document comments.
    """
    
    def __init__(self):
        """Initialize the comment analyzer."""
        self.threads = {}  # document_id -> list of CommentThread objects
        self.thread_categories = {}  # document_id -> category -> list of thread_ids
        self.resolved_threads = {}  # document_id -> list of CommentThread objects
        self.logger = logger.get_logger("comment_analyzer")
        self.lock = threading.Lock()
        self.performance_monitor = get_performance_monitor()
    
    def process_comments(self, document_id: str, comments: List[Dict[str, Any]]) -> List[CommentThread]:
        """
        Process a list of comments from a document.
        
        Args:
            document_id: The document identifier
            comments: List of comment data
            
        Returns:
            List of CommentThread objects
        """
        with measure_latency("process_comments", self.performance_monitor):
            # Sort comments by creation time for proper thread construction
            # Handle both 'created' and 'created_time' keys for flexibility
            sorted_comments = sorted(comments, key=lambda c: c.get('created', c.get('created_time')))
            
            # Try to acquire the lock with a timeout to prevent deadlocks
            lock_acquired = self.lock.acquire(timeout=5)  # 5 second timeout
            try:
                if not lock_acquired:
                    self.logger.warning("Could not acquire lock for process_comments - returning empty list")
                    return []
                
                # Initialize document entries if not exist
                if document_id not in self.threads:
                    self.threads[document_id] = []
                    self.thread_categories[document_id] = defaultdict(list)
                    self.resolved_threads[document_id] = []
                
                # Group comments into threads
                threads = self._group_comments_into_threads(document_id, sorted_comments)
                
                # Categorize threads
                for thread in threads:
                    self._categorize_thread(thread)
                
                # Update thread category lookup
                self._update_thread_categories(document_id)
                
                return threads
            finally:
                # Always release the lock if we acquired it
                if lock_acquired:
                    self.lock.release()
    
    def _group_comments_into_threads(self, document_id: str, 
                                    comments: List[Dict[str, Any]]) -> List[CommentThread]:
        """
        Group comments into threads based on reply structure.
        
        Args:
            document_id: The document identifier
            comments: List of comment data sorted by creation time
            
        Returns:
            List of CommentThread objects
        """
        # Map comment IDs to threads
        comment_to_thread = {}
        threads = []
        
        for comment in comments:
            thread_id = None
            
            # Check if this is a reply to an existing comment
            if 'replyTo' in comment and comment['replyTo'] in comment_to_thread:
                # Add to existing thread
                thread_id = comment_to_thread[comment['replyTo']]
            else:
                # Start new thread
                thread_id = str(uuid.uuid4())
                thread = CommentThread(thread_id, document_id)
                self.threads[document_id].append(thread)
                threads.append(thread)
            
            # Add comment to thread
            for thread in self.threads[document_id]:
                if thread.thread_id == thread_id:
                    thread.add_comment(comment)
                    comment_to_thread[comment['id']] = thread_id
                    break
        
        return threads
    
    def _categorize_thread(self, thread: CommentThread) -> None:
        """
        Categorize a comment thread based on its content.
        
        Args:
            thread: The CommentThread to categorize
        """
        if not thread.comments:
            thread.categories.add(CommentCategory.GENERAL)
            return
        
        # Analyze first comment to determine primary category
        first_comment = thread.comments[0]
        content = first_comment.get('content', '').lower()
        
        # Look for specific patterns
        patterns = {
            CommentCategory.QUESTION: [r'\?$', r'what', r'why', r'how', r'when', r'where', r'who', r'which'],
            CommentCategory.SUGGESTION: [r'suggest', r'recommend', r'perhaps', r'maybe', r'could', r'would be better'],
            CommentCategory.CORRECTION: [r'incorrect', r'error', r'typo', r'mistake', r'wrong', r'fix', r'change to'],
            CommentCategory.CLARIFICATION: [r'clarify', r'unclear', r'confusing', r'explain', r'what does this mean'],
            CommentCategory.APPROVAL: [r'approve', r'lgtm', r'looks good', r'agree', r'sounds good', r'\+1']
        }
        
        # Check for priority markers
        if '#priority' in content or '#important' in content or '#urgent' in content:
            thread.categories.add(CommentCategory.PRIORITY)
        
        # Check for other categories
        for category, patterns_list in patterns.items():
            for pattern in patterns_list:
                if re.search(pattern, content):
                    thread.categories.add(category)
                    break
        
        # If no specific category was found, use GENERAL
        if not thread.categories or (len(thread.categories) == 1 and CommentCategory.PRIORITY in thread.categories):
            thread.categories.add(CommentCategory.GENERAL)
    
    def _update_thread_categories(self, document_id: str) -> None:
        """
        Update the thread category lookup for a document.
        
        Args:
            document_id: The document identifier
        """
        # Clear existing category mappings
        self.thread_categories[document_id] = defaultdict(list)
        
        # Add each thread to its categories
        for thread in self.threads[document_id]:
            for category in thread.categories:
                self.thread_categories[document_id][category].append(thread.thread_id)
    
    def get_threads(self, document_id: str, resolved: bool = None) -> List[CommentThread]:
        """
        Get comment threads for a document.
        
        Args:
            document_id: The document identifier
            resolved: Filter by resolved status
            
        Returns:
            List of CommentThread objects
        """
        # Try to acquire the lock with a timeout to prevent deadlocks
        lock_acquired = self.lock.acquire(timeout=5)  # 5 second timeout
        try:
            if not lock_acquired:
                self.logger.warning("Could not acquire lock for get_threads - returning empty list")
                return []
                
            if document_id not in self.threads:
                return []
            
            if resolved is None:
                return self.threads[document_id]
            else:
                return [t for t in self.threads[document_id] if t.resolved == resolved]
        finally:
            # Always release the lock if we acquired it
            if lock_acquired:
                self.lock.release()
    
    def get_threads_by_category(self, document_id: str, 
                             category: CommentCategory) -> List[CommentThread]:
        """
        Get threads with a specific category.
        
        Args:
            document_id: The document identifier
            category: The comment category
            
        Returns:
            List of CommentThread objects
        """
        # Try to acquire the lock with a timeout to prevent deadlocks
        lock_acquired = self.lock.acquire(timeout=5)  # 5 second timeout
        try:
            if not lock_acquired:
                self.logger.warning("Could not acquire lock for get_threads_by_category - returning empty list")
                return []
                
            if document_id not in self.thread_categories or category not in self.thread_categories[document_id]:
                return []
            
            thread_ids = self.thread_categories[document_id][category]
            return [t for t in self.threads[document_id] if t.thread_id in thread_ids]
        finally:
            # Always release the lock if we acquired it
            if lock_acquired:
                self.lock.release()
    
    def get_threads_by_section(self, document_id: str, section: str) -> List[CommentThread]:
        """
        Get threads for a specific document section.
        
        Args:
            document_id: The document identifier
            section: The section name
            
        Returns:
            List of CommentThread objects
        """
        # Try to acquire the lock with a timeout to prevent deadlocks
        lock_acquired = self.lock.acquire(timeout=5)  # 5 second timeout
        try:
            if not lock_acquired:
                self.logger.warning("Could not acquire lock for get_threads_by_section - returning empty list")
                return []
                
            if document_id not in self.threads:
                return []
            
            return [t for t in self.threads[document_id] if t.section == section]
        finally:
            # Always release the lock if we acquired it
            if lock_acquired:
                self.lock.release()
    
    def get_comment_statistics(self, document_id: str) -> Dict[str, Any]:
        """
        Get statistics about comments in a document.
        
        Args:
            document_id: The document identifier
            
        Returns:
            Comment statistics
        """
        # Try to acquire the lock with a timeout to prevent deadlocks
        lock_acquired = self.lock.acquire(timeout=5)  # 5 second timeout
        try:
            if not lock_acquired:
                self.logger.warning("Could not acquire lock for get_comment_statistics - returning empty stats")
                return {
                    'total_threads': 0,
                    'total_comments': 0,
                    'resolved_threads': 0,
                    'category_counts': {},
                    'user_activity': {}
                }
                
            if document_id not in self.threads:
                return {
                    'total_threads': 0,
                    'total_comments': 0,
                    'resolved_threads': 0,
                    'category_counts': {},
                    'user_activity': {}
                }
            
            threads = self.threads[document_id]
            
            # Count total comments
            total_comments = sum(thread.get_comment_count() for thread in threads)
            
            # Count threads by category
            category_counts = defaultdict(int)
            for thread in threads:
                for category in thread.categories:
                    category_counts[category.name] += 1
            
            # Track user activity
            user_activity = defaultdict(lambda: {'total_comments': 0, 'threads_created': 0, 'threads_participated': 0})
            
            for thread in threads:
                # Track thread creator
                if thread.comments:
                    creator_id = thread.comments[0]['author']['id']
                    user_activity[creator_id]['threads_created'] += 1
                
                # Track participation
                for user_id in thread.participants:
                    user_activity[user_id]['threads_participated'] += 1
                
                # Count comments by user
                for comment in thread.comments:
                    user_id = comment['author']['id']
                    user_activity[user_id]['total_comments'] += 1
            
            return {
                'total_threads': len(threads),
                'total_comments': total_comments,
                'resolved_threads': sum(1 for t in threads if t.resolved),
                'category_counts': dict(category_counts),
                'user_activity': dict(user_activity)
            }
        finally:
            # Always release the lock if we acquired it
            if lock_acquired:
                self.lock.release()
    
    def resolve_thread(self, document_id: str, thread_id: str, user_id: str) -> bool:
        """
        Mark a thread as resolved.
        
        Args:
            document_id: The document identifier
            thread_id: The thread identifier
            user_id: User who resolved the thread
            
        Returns:
            True if thread was found and resolved, False otherwise
        """
        # Try to acquire the lock with a timeout to prevent deadlocks
        lock_acquired = self.lock.acquire(timeout=5)  # 5 second timeout
        try:
            if not lock_acquired:
                self.logger.warning("Could not acquire lock for resolve_thread - returning False")
                return False
                
            if document_id not in self.threads:
                return False
            
            for thread in self.threads[document_id]:
                if thread.thread_id == thread_id:
                    if not thread.resolved:
                        thread.resolve(user_id)
                        self.resolved_threads[document_id].append(thread)
                    return True
            
            return False
        finally:
            # Always release the lock if we acquired it
            if lock_acquired:
                self.lock.release()
    
    def get_user_comment_activity(self, document_id: str, user_id: str) -> Dict[str, Any]:
        """
        Get a summary of a user's comment activity in a document.
        
        Args:
            document_id: The document identifier
            user_id: The user identifier
            
        Returns:
            User comment activity summary
        """
        # Try to acquire the lock with a timeout to prevent deadlocks
        lock_acquired = self.lock.acquire(timeout=5)  # 5 second timeout
        try:
            if not lock_acquired:
                self.logger.warning("Could not acquire lock for get_user_comment_activity - returning empty activity")
                return {
                    'total_comments': 0,
                    'threads_created': 0,
                    'threads_participated': 0,
                    'recent_comments': []
                }
                
            if document_id not in self.threads:
                return {
                    'total_comments': 0,
                    'threads_created': 0,
                    'threads_participated': 0,
                    'recent_comments': []
                }
            
            threads = self.threads[document_id]
            
            # Count comments, created threads, and participation
            total_comments = 0
            threads_created = 0
            participated_threads = []
            recent_comments = []
            
            # Last week cutoff for recent comments
            cutoff = datetime.now() - timedelta(days=7)
            
            for thread in threads:
                is_creator = thread.comments and thread.comments[0]['author']['id'] == user_id
                if is_creator:
                    threads_created += 1
                
                user_in_thread = False
                for comment in thread.comments:
                    if comment['author']['id'] == user_id:
                        total_comments += 1
                        user_in_thread = True
                        
                        # Get timestamp - handle both 'created' and 'created_time' keys
                        timestamp = comment.get('created', comment.get('created_time'))
                        
                        # If it's a string (ISO format from created_time), convert to timestamp
                        if isinstance(timestamp, str):
                            try:
                                # Parse ISO format string to datetime
                                timestamp = datetime.fromisoformat(timestamp).timestamp()
                            except ValueError:
                                # Fallback if parsing fails
                                timestamp = datetime.now().timestamp()
                                
                        comment_time = datetime.fromtimestamp(timestamp)
                        
                        # Check if recent
                        if comment_time >= cutoff:
                            recent_comments.append({
                                'thread_id': thread.thread_id,
                                'comment_id': comment['id'],
                                'content': comment['content'],
                                'created': comment_time,
                                'quoted_text': thread.quoted_text
                            })
                
                if user_in_thread:
                    participated_threads.append(thread.thread_id)
            
            # Sort recent comments by time (newest first)
            recent_comments.sort(key=lambda c: c['created'], reverse=True)
            
            return {
                'total_comments': total_comments,
                'threads_created': threads_created,
                'threads_participated': len(participated_threads),
                'participated_thread_ids': participated_threads,
                'recent_comments': recent_comments[:10]  # Limit to 10 most recent
            }
        finally:
            # Always release the lock if we acquired it
            if lock_acquired:
                self.lock.release()
    
    def cleanup_old_resolved_threads(self, max_age_days: int = 30) -> int:
        """
        Remove old resolved threads.
        
        Args:
            max_age_days: Maximum age in days to keep
            
        Returns:
            Number of threads removed
        """
        # Try to acquire the lock with a timeout to prevent deadlocks
        lock_acquired = self.lock.acquire(timeout=5)  # 5 second timeout
        try:
            if not lock_acquired:
                self.logger.warning("Could not acquire lock for cleanup_old_resolved_threads - returning 0")
                return 0
                
            cutoff = datetime.now() - timedelta(days=max_age_days)
            removed_count = 0
            
            # Clean up old resolved threads
            for document_id in list(self.resolved_threads.keys()):
                original_count = len(self.resolved_threads[document_id])
                self.resolved_threads[document_id] = [
                    thread for thread in self.resolved_threads[document_id]
                    if not thread.resolved or thread.resolved_at >= cutoff
                ]
                removed_count += original_count - len(self.resolved_threads[document_id])
                
                # Remove empty documents
                if not self.resolved_threads[document_id]:
                    del self.resolved_threads[document_id]
            
            return removed_count
        finally:
            # Always release the lock if we acquired it
            if lock_acquired:
                self.lock.release()
    
    def get_unresolved_threads(self, document_id: str) -> List[CommentThread]:
        """
        Get unresolved comment threads for a document.
        
        Args:
            document_id: The document identifier
            
        Returns:
            List of unresolved CommentThread objects
        """
        # This method should not acquire its own lock and then call get_threads
        # which also acquires a lock - that would cause a deadlock.
        # Instead, call get_threads directly which already has safe lock handling.
        return self.get_threads(document_id, resolved=False)