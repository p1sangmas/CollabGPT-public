"""
Feedback Loop System for CollabGPT.

This module implements a feedback loop system that collects and processes user feedback
on suggestions to improve the quality of future suggestions.
"""

from typing import Dict, List, Any, Optional, Tuple
import json
import os
from datetime import datetime
import re
from pathlib import Path
import sqlite3

from ..utils import logger
from ..config import settings
from ..models.llm_interface import LLMInterface
from .edit_suggestion.edit_suggestion_system import EditSuggestion


class FeedbackEntry:
    """Represents a user feedback entry on a suggestion."""
    
    def __init__(self, 
                 suggestion_id: str,
                 original_suggestion: EditSuggestion,
                 accepted: bool,
                 user_feedback: Optional[str] = None,
                 user_id: Optional[str] = None,
                 timestamp: Optional[datetime] = None):
        """
        Initialize feedback entry.
        
        Args:
            suggestion_id: Unique identifier for the suggestion
            original_suggestion: The original suggestion object
            accepted: Whether the suggestion was accepted
            user_feedback: Optional feedback comment from user
            user_id: Optional user identifier
            timestamp: When feedback was provided (defaults to now)
        """
        self.suggestion_id = suggestion_id
        self.original_suggestion = original_suggestion
        self.accepted = accepted
        self.user_feedback = user_feedback
        self.user_id = user_id
        self.timestamp = timestamp or datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "suggestion_id": self.suggestion_id,
            "original_suggestion": self.original_suggestion.to_dict(),
            "accepted": self.accepted,
            "user_feedback": self.user_feedback,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeedbackEntry':
        """Create from dictionary."""
        return cls(
            suggestion_id=data["suggestion_id"],
            original_suggestion=EditSuggestion.from_dict(data["original_suggestion"]),
            accepted=data["accepted"],
            user_feedback=data.get("user_feedback"),
            user_id=data.get("user_id"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else None
        )


class FeedbackPattern:
    """Represents a pattern extracted from feedback data."""
    
    def __init__(self, 
                 pattern_type: str,
                 pattern: str,
                 section_types: List[str],
                 success_rate: float,
                 sample_count: int,
                 examples: List[Dict[str, Any]]):
        """
        Initialize a feedback pattern.
        
        Args:
            pattern_type: Type of pattern (e.g., 'acceptance', 'rejection')
            pattern: The pattern description
            section_types: Types of document sections this applies to
            success_rate: Rate of success for this pattern
            sample_count: Number of samples used to identify pattern
            examples: List of example suggestions for this pattern
        """
        self.pattern_type = pattern_type
        self.pattern = pattern
        self.section_types = section_types
        self.success_rate = success_rate
        self.sample_count = sample_count
        self.examples = examples
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pattern_type": self.pattern_type,
            "pattern": self.pattern,
            "section_types": self.section_types,
            "success_rate": self.success_rate,
            "sample_count": self.sample_count,
            "examples": self.examples
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeedbackPattern':
        """Create from dictionary."""
        return cls(
            pattern_type=data["pattern_type"],
            pattern=data["pattern"],
            section_types=data["section_types"],
            success_rate=data["success_rate"],
            sample_count=data["sample_count"],
            examples=data["examples"]
        )


class FeedbackLoopSystem:
    """
    System for collecting and processing user feedback on suggestions.
    
    This system implements a feedback loop that improves the quality of
    suggestions over time based on user feedback.
    """
    
    def __init__(self, llm_interface: LLMInterface, feedback_db_path: Optional[str] = None):
        """
        Initialize the feedback loop system.
        
        Args:
            llm_interface: LLM interface for pattern analysis
            feedback_db_path: Path to feedback database (creates default if None)
        """
        self.llm = llm_interface
        self.logger = logger.get_logger("feedback_loop_system")
        
        # Set up database path
        if feedback_db_path:
            self.db_path = feedback_db_path
        else:
            self.db_path = os.path.join(settings.DATA_DIR, "feedback.db")
            
        # Initialize database
        self._init_database()
        
        # Cache for extracted patterns
        self.patterns: List[FeedbackPattern] = []
        self._load_patterns()
    
    def record_feedback(self, 
                       suggestion: EditSuggestion, 
                       accepted: bool, 
                       user_feedback: Optional[str] = None,
                       user_id: Optional[str] = None) -> str:
        """
        Record user feedback on a suggestion.
        
        Args:
            suggestion: The suggestion that received feedback
            accepted: Whether the suggestion was accepted
            user_feedback: Optional feedback comment from user
            user_id: Optional user identifier
            
        Returns:
            Suggestion ID
        """
        # Generate a unique ID for the suggestion if needed
        suggestion_id = f"suggestion_{suggestion.section_id}_{int(datetime.now().timestamp())}"
        
        # Create feedback entry
        entry = FeedbackEntry(
            suggestion_id=suggestion_id,
            original_suggestion=suggestion,
            accepted=accepted,
            user_feedback=user_feedback,
            user_id=user_id
        )
        
        # Store in database
        self._store_feedback(entry)
        
        self.logger.info(
            f"Recorded feedback for suggestion {suggestion_id}: " +
            f"{'accepted' if accepted else 'rejected'}" +
            (f", feedback: {user_feedback}" if user_feedback else "")
        )
        
        # Update patterns if we have enough new feedback
        if self._should_update_patterns():
            self.update_patterns()
            
        return suggestion_id
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """
        Get statistics about collected feedback.
        
        Returns:
            Dictionary of feedback statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {
            "total_suggestions": 0,
            "accepted_count": 0,
            "rejected_count": 0,
            "acceptance_rate": 0.0,
            "suggestion_types": {},
            "feedback_patterns": len(self.patterns),
            "recent_feedback": []
        }
        
        # Get total counts
        cursor.execute("SELECT COUNT(*) FROM feedback")
        stats["total_suggestions"] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM feedback WHERE accepted = 1")
        stats["accepted_count"] = cursor.fetchone()[0]
        
        stats["rejected_count"] = stats["total_suggestions"] - stats["accepted_count"]
        
        if stats["total_suggestions"] > 0:
            stats["acceptance_rate"] = stats["accepted_count"] / stats["total_suggestions"]
        
        # Get counts by suggestion type
        cursor.execute(
            "SELECT suggestion_type, COUNT(*), SUM(accepted) " +
            "FROM feedback GROUP BY suggestion_type"
        )
        for suggestion_type, count, accepted_count in cursor.fetchall():
            stats["suggestion_types"][suggestion_type] = {
                "count": count,
                "accepted": accepted_count,
                "acceptance_rate": accepted_count / count if count > 0 else 0
            }
        
        # Get recent feedback (last 10 entries)
        cursor.execute(
            "SELECT suggestion_id, accepted, timestamp, user_feedback " +
            "FROM feedback ORDER BY timestamp DESC LIMIT 10"
        )
        for suggestion_id, accepted, timestamp, user_feedback in cursor.fetchall():
            stats["recent_feedback"].append({
                "suggestion_id": suggestion_id,
                "accepted": bool(accepted),
                "timestamp": timestamp,
                "user_feedback": user_feedback
            })
        
        conn.close()
        return stats
    
    def update_patterns(self) -> List[FeedbackPattern]:
        """
        Update feedback patterns based on collected feedback.
        
        Returns:
            List of extracted patterns
        """
        self.logger.info("Updating feedback patterns")
        
        # Get all feedback data
        feedback_entries = self._load_all_feedback()
        
        if len(feedback_entries) < 10:  # Need at least 10 entries for pattern analysis
            self.logger.info(f"Not enough feedback entries for pattern analysis: {len(feedback_entries)}")
            return self.patterns
            
        # Group by suggestion type
        by_type = {}
        for entry in feedback_entries:
            suggestion_type = entry.original_suggestion.suggestion_type
            if suggestion_type not in by_type:
                by_type[suggestion_type] = []
            by_type[suggestion_type].append(entry)
            
        # Extract patterns from each type with sufficient samples
        new_patterns = []
        for suggestion_type, entries in by_type.items():
            if len(entries) >= 5:  # Need at least 5 entries per type
                type_patterns = self._extract_patterns_for_type(suggestion_type, entries)
                new_patterns.extend(type_patterns)
                
        # Add patterns from acceptance vs. rejection analysis
        accepted = [e for e in feedback_entries if e.accepted]
        rejected = [e for e in feedback_entries if not e.accepted]
        
        if len(accepted) >= 5 and len(rejected) >= 5:
            acceptance_patterns = self._analyze_acceptance_patterns(accepted, rejected)
            new_patterns.extend(acceptance_patterns)
            
        # Update patterns
        self.patterns = new_patterns
        
        # Save patterns to disk
        self._save_patterns(new_patterns)
        
        self.logger.info(f"Updated {len(new_patterns)} feedback patterns")
        return new_patterns
    
    def get_improvement_prompt(self, 
                              suggestion_type: str, 
                              section_type: Optional[str] = None) -> str:
        """
        Get a prompt improvement based on feedback patterns.
        
        Args:
            suggestion_type: Type of suggestion to improve
            section_type: Optional type of document section
            
        Returns:
            Prompt improvement text
        """
        # Filter patterns by type and section
        relevant_patterns = []
        for pattern in self.patterns:
            if pattern.pattern_type == "acceptance" and pattern.success_rate >= 0.7:
                if suggestion_type in pattern.section_types:
                    if not section_type or section_type in pattern.section_types:
                        relevant_patterns.append(pattern)
        
        if not relevant_patterns:
            return ""
            
        # Build improvement prompt
        prompt_improvements = [
            f"Based on feedback patterns, consider these successful approaches for {suggestion_type} suggestions:"
        ]
        
        for pattern in relevant_patterns:
            prompt_improvements.append(f"- {pattern.pattern} (success rate: {pattern.success_rate:.0%})")
            
            # Add an example if available
            if pattern.examples:
                example = pattern.examples[0]  # Take first example
                if "text" in example:
                    prompt_improvements.append(f"  Example: \"{example['text']}\"")
                    
        return "\n".join(prompt_improvements)
    
    def apply_feedback_learning(self, 
                               prompt: str, 
                               suggestion_type: str,
                               section_type: Optional[str] = None) -> str:
        """
        Apply learned feedback patterns to improve a prompt.
        
        Args:
            prompt: Original prompt
            suggestion_type: Type of suggestion
            section_type: Optional type of document section
            
        Returns:
            Improved prompt
        """
        # Get improvement hints
        improvement = self.get_improvement_prompt(suggestion_type, section_type)
        
        if not improvement:
            return prompt
            
        # Append improvement to prompt
        return f"{prompt}\n\n{improvement}"
    
    def _init_database(self) -> None:
        """Initialize the feedback database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create feedback table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            suggestion_id TEXT PRIMARY KEY,
            section_id TEXT,
            section_title TEXT,
            suggestion_type TEXT,
            original_text TEXT,
            suggestion_text TEXT,
            reasoning TEXT,
            confidence REAL,
            accepted INTEGER,
            user_feedback TEXT,
            user_id TEXT,
            timestamp TEXT,
            metadata TEXT
        )
        ''')
        
        # Create patterns table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS patterns (
            pattern_id TEXT PRIMARY KEY,
            pattern_type TEXT,
            pattern TEXT,
            section_types TEXT,
            success_rate REAL,
            sample_count INTEGER,
            examples TEXT,
            timestamp TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def _store_feedback(self, entry: FeedbackEntry) -> None:
        """Store feedback entry in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            '''
            INSERT OR REPLACE INTO feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                entry.suggestion_id,
                entry.original_suggestion.section_id,
                entry.original_suggestion.section_title,
                entry.original_suggestion.suggestion_type,
                entry.original_suggestion.original_text,
                entry.original_suggestion.suggestion,
                entry.original_suggestion.reasoning,
                entry.original_suggestion.confidence,
                1 if entry.accepted else 0,
                entry.user_feedback,
                entry.user_id,
                entry.timestamp.isoformat(),
                json.dumps(entry.original_suggestion.metadata)
            )
        )
        
        conn.commit()
        conn.close()
    
    def _load_all_feedback(self) -> List[FeedbackEntry]:
        """Load all feedback entries from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM feedback ORDER BY timestamp"
        )
        
        entries = []
        for row in cursor.fetchall():
            suggestion_id = row[0]
            section_id = row[1]
            section_title = row[2]
            suggestion_type = row[3]
            original_text = row[4]
            suggestion_text = row[5]
            reasoning = row[6]
            confidence = row[7]
            accepted = bool(row[8])
            user_feedback = row[9]
            user_id = row[10]
            timestamp = datetime.fromisoformat(row[11])
            metadata = json.loads(row[12]) if row[12] else {}
            
            # Recreate the original suggestion
            suggestion = EditSuggestion(
                section_id=section_id,
                section_title=section_title,
                original_text=original_text,
                suggestion=suggestion_text,
                suggestion_type=suggestion_type,
                reasoning=reasoning,
                confidence=confidence,
                metadata=metadata
            )
            
            entry = FeedbackEntry(
                suggestion_id=suggestion_id,
                original_suggestion=suggestion,
                accepted=accepted,
                user_feedback=user_feedback,
                user_id=user_id,
                timestamp=timestamp
            )
            
            entries.append(entry)
            
        conn.close()
        return entries
    
    def _should_update_patterns(self) -> bool:
        """Check if patterns should be updated based on new feedback."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get total feedback count
        cursor.execute("SELECT COUNT(*) FROM feedback")
        total_count = cursor.fetchone()[0]
        
        # Get pattern count
        cursor.execute("SELECT COUNT(*) FROM patterns")
        pattern_count = cursor.fetchone()[0]
        
        conn.close()
        
        # Update if we have no patterns or significant new feedback
        if pattern_count == 0:
            return total_count >= 10
        else:
            # Update if feedback count increased by at least 20% since last pattern extraction
            return total_count >= pattern_count * 1.2
    
    def _extract_patterns_for_type(self, suggestion_type: str, 
                                  entries: List[FeedbackEntry]) -> List[FeedbackPattern]:
        """
        Extract patterns for a specific suggestion type.
        
        Args:
            suggestion_type: Type of suggestion
            entries: List of feedback entries for this type
            
        Returns:
            List of extracted patterns
        """
        patterns = []
        
        # Split into accepted and rejected
        accepted = [e for e in entries if e.accepted]
        rejected = [e for e in entries if not e.accepted]
        
        # Skip if not enough data
        if len(accepted) < 3 or len(rejected) < 3:
            return patterns
            
        # Use LLM to analyze patterns in accepted suggestions
        if len(accepted) >= 5:
            # Create a summary of accepted suggestions for analysis
            examples = []
            for entry in accepted[:10]:  # Limit to 10 examples
                examples.append({
                    "suggestion": entry.original_suggestion.suggestion,
                    "reasoning": entry.original_suggestion.reasoning,
                    "feedback": entry.user_feedback
                })
            
            # Create prompt for pattern analysis
            prompt = f"""
            Analyze the following {len(examples)} ACCEPTED suggestions of type "{suggestion_type}" to identify patterns that make them successful:
            
            {json.dumps(examples, indent=2)}
            
            Identify 1-3 specific patterns that appear to lead to acceptance of these suggestions.
            Provide a clear and concise description of each pattern.
            
            Format your response as a JSON array where each object has:
            - "pattern": a clear description of the pattern
            - "confidence": a number from 0.0 to 1.0 indicating your confidence
            
            Output only the JSON array, nothing else.
            """
            
            # Get response from LLM
            response = self.llm.generate(prompt, max_tokens=800, temperature=0.2)
            
            if response.success:
                try:
                    # Extract JSON array from response
                    pattern_match = re.search(r'\[\s*\{.*\}\s*\]', response.text, re.DOTALL)
                    if pattern_match:
                        pattern_json = pattern_match.group(0)
                        extracted_patterns = json.loads(pattern_json)
                        
                        # Create pattern objects
                        for p in extracted_patterns:
                            if "pattern" in p:
                                # Create a pattern for successful suggestions
                                pattern = FeedbackPattern(
                                    pattern_type="acceptance",
                                    pattern=p["pattern"],
                                    section_types=[suggestion_type],
                                    success_rate=p.get("confidence", 0.7),
                                    sample_count=len(accepted),
                                    examples=[{
                                        "text": e.original_suggestion.suggestion,
                                        "reasoning": e.original_suggestion.reasoning
                                    } for e in accepted[:3]]  # Include up to 3 examples
                                )
                                patterns.append(pattern)
                except Exception as e:
                    self.logger.error(f"Error parsing pattern analysis response: {e}")
        
        # Similarly analyze rejected suggestions to identify patterns to avoid
        if len(rejected) >= 5:
            # Create a summary of rejected suggestions
            examples = []
            for entry in rejected[:10]:
                examples.append({
                    "suggestion": entry.original_suggestion.suggestion,
                    "reasoning": entry.original_suggestion.reasoning,
                    "feedback": entry.user_feedback
                })
            
            prompt = f"""
            Analyze the following {len(examples)} REJECTED suggestions of type "{suggestion_type}" to identify patterns that led to rejection:
            
            {json.dumps(examples, indent=2)}
            
            Identify 1-3 specific patterns that appear to lead to rejection of these suggestions.
            Provide a clear and concise description of each pattern to avoid.
            
            Format your response as a JSON array where each object has:
            - "pattern_to_avoid": a clear description of the pattern that should be avoided
            - "confidence": a number from 0.0 to 1.0 indicating your confidence
            
            Output only the JSON array, nothing else.
            """
            
            response = self.llm.generate(prompt, max_tokens=800, temperature=0.2)
            
            if response.success:
                try:
                    pattern_match = re.search(r'\[\s*\{.*\}\s*\]', response.text, re.DOTALL)
                    if pattern_match:
                        pattern_json = pattern_match.group(0)
                        extracted_patterns = json.loads(pattern_json)
                        
                        for p in extracted_patterns:
                            if "pattern_to_avoid" in p:
                                # Create a pattern for rejected suggestions
                                pattern = FeedbackPattern(
                                    pattern_type="rejection",
                                    pattern="AVOID: " + p["pattern_to_avoid"],
                                    section_types=[suggestion_type],
                                    success_rate=p.get("confidence", 0.7),
                                    sample_count=len(rejected),
                                    examples=[{
                                        "text": e.original_suggestion.suggestion,
                                        "reasoning": e.original_suggestion.reasoning
                                    } for e in rejected[:3]]
                                )
                                patterns.append(pattern)
                except Exception as e:
                    self.logger.error(f"Error parsing rejection pattern analysis: {e}")
        
        return patterns
    
    def _analyze_acceptance_patterns(self, 
                                    accepted: List[FeedbackEntry],
                                    rejected: List[FeedbackEntry]) -> List[FeedbackPattern]:
        """
        Analyze patterns that differentiate accepted vs rejected suggestions.
        
        Args:
            accepted: List of accepted feedback entries
            rejected: List of rejected feedback entries
            
        Returns:
            List of extracted patterns
        """
        patterns = []
        
        # Limit to manageable number of examples
        accepted_sample = accepted[:15]
        rejected_sample = rejected[:15]
        
        # Prepare examples for comparison
        accepted_examples = [
            {
                "suggestion": e.original_suggestion.suggestion,
                "type": e.original_suggestion.suggestion_type,
                "confidence": e.original_suggestion.confidence,
                "feedback": e.user_feedback
            } for e in accepted_sample
        ]
        
        rejected_examples = [
            {
                "suggestion": e.original_suggestion.suggestion,
                "type": e.original_suggestion.suggestion_type,
                "confidence": e.original_suggestion.confidence,
                "feedback": e.user_feedback
            } for e in rejected_sample
        ]
        
        # Create prompt for comparative analysis
        prompt = f"""
        Compare these ACCEPTED suggestions:
        {json.dumps(accepted_examples, indent=2)}
        
        with these REJECTED suggestions:
        {json.dumps(rejected_examples, indent=2)}
        
        Identify key differences between accepted and rejected suggestions. What patterns make a suggestion more likely to be accepted?
        
        Format your response as a JSON array where each object has:
        - "pattern": a clear description of a pattern that leads to acceptance
        - "applies_to_types": an array of suggestion types this pattern applies to
        - "confidence": a number from 0.0 to 1.0 indicating your confidence
        
        Output only the JSON array, nothing else.
        """
        
        response = self.llm.generate(prompt, max_tokens=1000, temperature=0.3)
        
        if response.success:
            try:
                pattern_match = re.search(r'\[\s*\{.*\}\s*\]', response.text, re.DOTALL)
                if pattern_match:
                    pattern_json = pattern_match.group(0)
                    extracted_patterns = json.loads(pattern_json)
                    
                    for p in extracted_patterns:
                        if "pattern" in p and "applies_to_types" in p:
                            pattern = FeedbackPattern(
                                pattern_type="comparative",
                                pattern=p["pattern"],
                                section_types=p["applies_to_types"],
                                success_rate=p.get("confidence", 0.7),
                                sample_count=len(accepted_sample) + len(rejected_sample),
                                examples=[{
                                    "text": e.original_suggestion.suggestion,
                                    "type": e.original_suggestion.suggestion_type
                                } for e in accepted_sample[:3]]
                            )
                            patterns.append(pattern)
            except Exception as e:
                self.logger.error(f"Error parsing comparative pattern analysis: {e}")
        
        return patterns
    
    def _save_patterns(self, patterns: List[FeedbackPattern]) -> None:
        """Save extracted patterns to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # First clear existing patterns
        cursor.execute("DELETE FROM patterns")
        
        # Insert new patterns
        for i, pattern in enumerate(patterns):
            pattern_id = f"pattern_{i}_{int(datetime.now().timestamp())}"
            cursor.execute(
                '''
                INSERT INTO patterns VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    pattern_id,
                    pattern.pattern_type,
                    pattern.pattern,
                    json.dumps(pattern.section_types),
                    pattern.success_rate,
                    pattern.sample_count,
                    json.dumps(pattern.examples),
                    datetime.now().isoformat()
                )
            )
        
        conn.commit()
        conn.close()
    
    def _load_patterns(self) -> None:
        """Load patterns from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM patterns")
        
        patterns = []
        for row in cursor.fetchall():
            pattern_type = row[1]
            pattern_text = row[2]
            section_types = json.loads(row[3])
            success_rate = row[4]
            sample_count = row[5]
            examples = json.loads(row[6])
            
            pattern = FeedbackPattern(
                pattern_type=pattern_type,
                pattern=pattern_text,
                section_types=section_types,
                success_rate=success_rate,
                sample_count=sample_count,
                examples=examples
            )
            
            patterns.append(pattern)
            
        conn.close()
        self.patterns = patterns