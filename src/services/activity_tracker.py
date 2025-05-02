"""
Activity Tracker for CollabGPT.

This module tracks user activity in documents, providing insights into 
user behavior and document usage patterns.
"""

import uuid
import threading
from typing import Dict, List, Any, Set, Optional
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict

from ..utils.performance import measure_latency, get_performance_monitor
from ..utils import logger

class ActivityType(Enum):
    """Types of user activities in documents."""
    VIEW = 1        # User viewed the document
    EDIT = 2        # User edited the document
    COMMENT = 3     # User added a comment
    RESOLVE = 4     # User resolved a comment or conflict
    SHARE = 5       # User shared the document
    DOWNLOAD = 6    # User downloaded/exported the document
    
    @classmethod
    def get_description(cls, activity_type):
        """Get a description of the activity type."""
        descriptions = {
            cls.VIEW: "Viewed document",
            cls.EDIT: "Edited document",
            cls.COMMENT: "Commented on document",
            cls.RESOLVE: "Resolved comment or conflict",
            cls.SHARE: "Shared document",
            cls.DOWNLOAD: "Downloaded/exported document"
        }
        return descriptions.get(activity_type, "Unknown activity")


class UserActivityTracker:
    """
    Tracks user activity in documents for contextual awareness.
    """
    
    def __init__(self, activity_retention_days: int = 30):
        """
        Initialize the activity tracker.
        
        Args:
            activity_retention_days: Number of days to retain activity data
        """
        self.activity_retention_days = activity_retention_days
        self.activities = {}  # document_id -> list of activities
        self.user_documents = defaultdict(set)  # user_id -> set of document_ids
        self.logger = logger.get_logger("activity_tracker")
        self.lock = threading.Lock()
        self.performance_monitor = get_performance_monitor()
    
    def track_activity(self, user_id: str, document_id: str, activity_type: ActivityType, 
                      metadata: Dict[str, Any] = None, sections: List[str] = None,
                      timestamp: Optional[datetime] = None) -> str:
        """
        Record a user activity.
        
        Args:
            user_id: The user identifier
            document_id: The document identifier
            activity_type: Type of activity
            metadata: Additional activity data
            sections: Affected document sections
            timestamp: Activity timestamp (defaults to now)
            
        Returns:
            Activity ID
        """
        with measure_latency("track_activity", self.performance_monitor):
            activity_id = str(uuid.uuid4())
            timestamp = timestamp or datetime.now()
            
            activity = {
                'id': activity_id,
                'user_id': user_id,
                'document_id': document_id,
                'activity_type': activity_type,
                'timestamp': timestamp,
                'metadata': metadata or {},
                'sections': sections or []
            }
            
            with self.lock:
                # Initialize document entry if not exists
                if document_id not in self.activities:
                    self.activities[document_id] = []
                
                # Add activity to history
                self.activities[document_id].append(activity)
                
                # Track which documents the user has accessed
                self.user_documents[user_id].add(document_id)
            
            return activity_id
    
    def get_user_activities(self, user_id: str, document_id: Optional[str] = None, 
                           hours: int = 24, activity_type: Optional[ActivityType] = None) -> List[Dict[str, Any]]:
        """
        Get activities for a specific user.
        
        Args:
            user_id: The user identifier
            document_id: Optional document filter
            hours: Time window in hours
            activity_type: Optional activity type filter
            
        Returns:
            List of activity records
        """
        with self.lock:
            result = []
            cutoff = datetime.now() - timedelta(hours=hours)
            
            # If document_id is specified, only look at that document
            if document_id:
                if document_id not in self.activities:
                    return []
                
                documents = [document_id]
            else:
                # Otherwise look at all documents the user has accessed
                documents = self.user_documents.get(user_id, set())
            
            # Collect activities across specified documents
            for doc_id in documents:
                if doc_id not in self.activities:
                    continue
                    
                for activity in self.activities[doc_id]:
                    if (activity['user_id'] == user_id and
                        activity['timestamp'] >= cutoff and
                        (activity_type is None or activity['activity_type'] == activity_type)):
                        result.append(activity)
            
            # Sort by timestamp (newest first)
            return sorted(result, key=lambda a: a['timestamp'], reverse=True)
    
    def get_document_activities(self, document_id: str, hours: int = 24, 
                              activity_type: Optional[ActivityType] = None) -> List[Dict[str, Any]]:
        """
        Get all activities for a specific document.
        
        Args:
            document_id: The document identifier
            hours: Time window in hours
            activity_type: Optional activity type filter
            
        Returns:
            List of activity records
        """
        with self.lock:
            if document_id not in self.activities:
                return []
                
            cutoff = datetime.now() - timedelta(hours=hours)
            
            filtered_activities = [
                activity for activity in self.activities[document_id]
                if activity['timestamp'] >= cutoff and
                (activity_type is None or activity['activity_type'] == activity_type)
            ]
            
            # Sort by timestamp (newest first)
            return sorted(filtered_activities, key=lambda a: a['timestamp'], reverse=True)
    
    def get_active_users(self, document_id: str, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get a summary of active users for a document.
        
        Args:
            document_id: The document identifier
            hours: Time window in hours
            
        Returns:
            List of user activity summaries
        """
        with measure_latency("get_active_users", self.performance_monitor):
            activities = self.get_document_activities(document_id, hours)
            
            if not activities:
                return []
                
            # Group by user
            user_activities = defaultdict(list)
            for activity in activities:
                user_activities[activity['user_id']].append(activity)
            
            # Create user summaries
            users = []
            for user_id, acts in user_activities.items():
                # Get unique activity types
                activity_types = {act['activity_type'] for act in acts}
                
                # Collect focused sections
                section_counts = defaultdict(int)
                for act in acts:
                    for section in act.get('sections', []):
                        section_counts[section] += 1
                
                # Get top sections (most frequently interacted with)
                focused_sections = [s for s, _ in sorted(section_counts.items(), 
                                                        key=lambda x: x[1], reverse=True)]
                
                # Find first and last activity
                first_activity = min(acts, key=lambda a: a['timestamp'])['timestamp']
                last_activity = max(acts, key=lambda a: a['timestamp'])['timestamp']
                
                users.append({
                    'user_id': user_id,
                    'total_activities': len(acts),
                    'activity_types': activity_types,
                    'focused_sections': focused_sections,
                    'first_activity': first_activity,
                    'last_activity': last_activity
                })
            
            # Sort by activity count (most active first)
            return sorted(users, key=lambda u: u['total_activities'], reverse=True)
    
    def get_activity_timeline(self, document_id: str, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get a chronological timeline of activities for a document.
        
        Args:
            document_id: The document identifier
            hours: Time window in hours
            
        Returns:
            List of activities in chronological order
        """
        activities = self.get_document_activities(document_id, hours)
        
        # Sort by timestamp (oldest first for timeline view)
        return sorted(activities, key=lambda a: a['timestamp'])
    
    def get_section_activity(self, document_id: str, section: str, 
                           hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get activities for a specific document section.
        
        Args:
            document_id: The document identifier
            section: The section name
            hours: Time window in hours
            
        Returns:
            List of activities for the section
        """
        with self.lock:
            if document_id not in self.activities:
                return []
                
            cutoff = datetime.now() - timedelta(hours=hours)
            
            section_activities = [
                activity for activity in self.activities[document_id]
                if activity['timestamp'] >= cutoff and
                section in activity.get('sections', [])
            ]
            
            # Sort by timestamp (newest first)
            return sorted(section_activities, key=lambda a: a['timestamp'], reverse=True)
    
    def get_section_editors(self, document_id: str, section: str, 
                          hours: int = 24) -> List[str]:
        """
        Get users who have edited a specific section.
        
        Args:
            document_id: The document identifier
            section: The section name
            hours: Time window in hours
            
        Returns:
            List of user IDs who edited the section
        """
        section_activities = self.get_section_activity(document_id, section, hours)
        
        # Get users who performed EDIT activities
        editors = {
            activity['user_id'] for activity in section_activities
            if activity['activity_type'] == ActivityType.EDIT
        }
        
        return list(editors)
    
    def get_user_activity_stats(self, user_id: str, 
                              days: int = 7) -> Dict[str, Any]:
        """
        Get activity statistics for a user.
        
        Args:
            user_id: The user identifier
            days: Time window in days
            
        Returns:
            Activity statistics
        """
        activities = self.get_user_activities(user_id, hours=days*24)
        
        if not activities:
            return {
                'user_id': user_id,
                'total_activities': 0,
                'active_documents': 0,
                'activity_breakdown': {},
                'activity_trend': []
            }
        
        # Count documents
        documents = {act['document_id'] for act in activities}
        
        # Activity type breakdown
        activity_breakdown = defaultdict(int)
        for activity in activities:
            activity_type = activity['activity_type'].name
            activity_breakdown[activity_type] += 1
        
        # Activity trend by day
        now = datetime.now()
        day_counts = defaultdict(int)
        
        for activity in activities:
            # Get day difference (0 = today, 1 = yesterday, etc.)
            day_diff = (now.date() - activity['timestamp'].date()).days
            if day_diff < days:  # Only include days within the window
                day_counts[day_diff] += 1
        
        # Convert to list of day counts
        trend = [day_counts.get(i, 0) for i in range(days)]
        trend.reverse()  # Oldest first
        
        return {
            'user_id': user_id,
            'total_activities': len(activities),
            'active_documents': len(documents),
            'activity_breakdown': dict(activity_breakdown),
            'activity_trend': trend
        }
    
    def cleanup_old_activities(self) -> int:
        """
        Remove activities older than the retention period.
        
        Returns:
            Number of activities removed
        """
        with self.lock:
            cutoff = datetime.now() - timedelta(days=self.activity_retention_days)
            removed_count = 0
            
            # Clean up old activities
            for document_id in list(self.activities.keys()):
                original_count = len(self.activities[document_id])
                self.activities[document_id] = [
                    activity for activity in self.activities[document_id]
                    if activity['timestamp'] >= cutoff
                ]
                removed_count += original_count - len(self.activities[document_id])
                
                # Remove empty documents
                if not self.activities[document_id]:
                    del self.activities[document_id]
            
            # Rebuild user_documents based on remaining activities
            self.user_documents = defaultdict(set)
            for document_id, activities in self.activities.items():
                for activity in activities:
                    self.user_documents[activity['user_id']].add(document_id)
            
            return removed_count