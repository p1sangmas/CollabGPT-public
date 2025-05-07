"""
Edit Suggestion System for CollabGPT.

This module implements intelligent edit suggestion capabilities with advanced
reasoning to provide contextually-aware suggestions for document improvements.
"""

from typing import Dict, List, Any, Optional, Tuple
import re
from datetime import datetime
import json

from ...utils import logger
from ...models.llm_interface import LLMInterface
from ...models.rag_system import RAGSystem
from ..document_analyzer import DocumentAnalyzer


class SuggestionType:
    """Types of suggestions that can be made."""
    CLARITY = "clarity"
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    ACCURACY = "accuracy"
    STRUCTURE = "structure"
    STYLE = "style"


class EditSuggestion:
    """Represents a suggested edit for a document section."""
    
    def __init__(self, 
                 section_id: str, 
                 section_title: str,
                 original_text: str,
                 suggestion: str,
                 suggestion_type: str,
                 reasoning: str,
                 confidence: float = 0.0,
                 metadata: Dict[str, Any] = None):
        """
        Initialize an edit suggestion.
        
        Args:
            section_id: Identifier for the section
            section_title: Title of the section
            original_text: Original text that is being improved
            suggestion: The suggested edit
            suggestion_type: Type of suggestion (from SuggestionType)
            reasoning: Explanation of why this suggestion is made
            confidence: Confidence score (0.0 to 1.0)
            metadata: Additional metadata about the suggestion
        """
        self.section_id = section_id
        self.section_title = section_title
        self.original_text = original_text
        self.suggestion = suggestion
        self.suggestion_type = suggestion_type
        self.reasoning = reasoning
        self.confidence = confidence
        self.timestamp = datetime.now()
        self.metadata = metadata or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "section_id": self.section_id,
            "section_title": self.section_title,
            "original_text": self.original_text,
            "suggestion": self.suggestion,
            "suggestion_type": self.suggestion_type,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EditSuggestion':
        """Create from dictionary."""
        suggestion = cls(
            section_id=data["section_id"],
            section_title=data["section_title"],
            original_text=data["original_text"],
            suggestion=data["suggestion"],
            suggestion_type=data["suggestion_type"],
            reasoning=data["reasoning"],
            confidence=data["confidence"],
            metadata=data.get("metadata", {})
        )
        
        if "timestamp" in data:
            suggestion.timestamp = datetime.fromisoformat(data["timestamp"])
            
        return suggestion


class EditSuggestionSystem:
    """
    System for generating intelligent document edit suggestions.
    
    Uses advanced reasoning to suggest contextually-aware improvements
    based on document history, user activity, and content analysis.
    """
    
    def __init__(self, llm_interface: LLMInterface, rag_system: RAGSystem):
        """
        Initialize the suggestion system.
        
        Args:
            llm_interface: LLM interface for generating suggestions
            rag_system: RAG system for document context
        """
        self.llm = llm_interface
        self.rag = rag_system
        self.analyzer = DocumentAnalyzer()
        self.logger = logger.get_logger("edit_suggestion_system")
        
        # Store feedback on suggestions to improve over time
        self.suggestion_feedback = {}
        
    def generate_suggestions(self, 
                            doc_id: str, 
                            section_id: Optional[str] = None,
                            max_suggestions: int = 3) -> List[EditSuggestion]:
        """
        Generate edit suggestions for a document or section.
        
        Args:
            doc_id: Document identifier
            section_id: Optional section identifier (if None, analyzes whole document)
            max_suggestions: Maximum number of suggestions to generate
            
        Returns:
            List of EditSuggestion objects
        """
        self.logger.info(f"Generating suggestions for document {doc_id}, section {section_id}")
        
        # Get document chunks from RAG system
        chunks = self.rag.vector_store.get_chunks_by_doc_id(doc_id)
        
        if not chunks:
            self.logger.warning(f"No content found for document {doc_id}")
            return []
        
        suggestions = []
        
        # If specific section is requested, find that chunk
        if section_id:
            target_chunks = [chunk for chunk in chunks 
                           if chunk.chunk_id == section_id or 
                              chunk.metadata.get('section') == section_id]
            
            if not target_chunks:
                self.logger.warning(f"Section {section_id} not found in document {doc_id}")
                return []
        else:
            # Analyze all chunks for potential improvements
            # Sort chunks to maintain document order
            target_chunks = sorted(chunks, 
                                 key=lambda c: c.metadata.get('section_index', 0) 
                                 if 'section_index' in c.metadata else 9999)
            
            # Limit to most important chunks based on edit history and content
            if len(target_chunks) > max_suggestions:
                target_chunks = self._identify_improvement_candidates(doc_id, target_chunks, max_suggestions)
        
        # Generate suggestions for each target chunk
        for chunk in target_chunks:
            section_title = chunk.metadata.get('section', f"Section {chunk.chunk_id}")
            
            # Get historical context for this section
            section_history = self._get_section_history(chunk)
            
            # Use a chain of prompts for sophisticated reasoning
            suggestion = self._generate_suggestion_with_reasoning(doc_id, chunk, section_history)
            
            if suggestion:
                suggestions.append(suggestion)
                
            # Don't exceed maximum suggestions
            if len(suggestions) >= max_suggestions:
                break
                
        return suggestions
    
    def _generate_suggestion_with_reasoning(self, 
                                           doc_id: str, 
                                           chunk, 
                                           section_history: str) -> Optional[EditSuggestion]:
        """
        Generate an edit suggestion using a chain of reasoning steps.
        
        Args:
            doc_id: Document identifier
            chunk: Document chunk to generate suggestion for
            section_history: History context for the section
            
        Returns:
            EditSuggestion object if generation successful, None otherwise
        """
        section_title = chunk.metadata.get('section', f"Section {chunk.chunk_id}")
        section_content = chunk.text
        
        # Create a prompt chain for sophisticated reasoning
        chain = self.llm.create_chain(name=f"edit_suggestion_{chunk.chunk_id}")
        
        # Step 1: Analyze the section for issues and improvement areas
        chain.add_step(
            "Step 1: Analyze the following document section for potential issues, gaps, inconsistencies, "
            "or opportunities for improvement. Consider clarity, completeness, consistency, and structure.\n\n"
            "Document section: {section_title}\n"
            "Content:\n{section_content}\n\n"
            "Document History Context:\n{section_history}\n\n"
            "Analysis:",
            name="analyze_issues",
            max_tokens=500,
            temperature=0.3
        )
        
        # Step 2: Identify specific improvement types
        chain.add_step(
            "Step 2: Based on the analysis, identify the most important improvement type for this section. "
            "Select one of the following: clarity, completeness, consistency, accuracy, structure, style.\n\n"
            "Previous analysis:\n{analyze_issues.text}\n\n"
            "Most important improvement type (one word only):",
            name="identify_type",
            max_tokens=50,
            temperature=0.2,
            input_mapping={
                "analyze_issues.text": "analyze_issues.text"
            }
        )
        
        # Step 3: Generate a specific suggestion
        chain.add_step(
            "Step 3: Based on the analysis and improvement type, generate a specific, actionable suggestion "
            "for improving the document section. Provide the exact text or explanation of what should be changed.\n\n"
            "Analysis: {analyze_issues.text}\n"
            "Improvement type: {identify_type.text}\n"
            "Original content: {section_content}\n\n"
            "Specific suggestion:",
            name="generate_suggestion",
            max_tokens=500,
            temperature=0.7,
            input_mapping={
                "analyze_issues.text": "analyze_issues.text",
                "identify_type.text": "identify_type.text"
            }
        )
        
        # Step 4: Provide reasoning for the suggestion
        chain.add_step(
            "Step 4: Explain the reasoning behind your suggestion. Why would this improve the document? "
            "How does it address the identified issues?\n\n"
            "Analysis: {analyze_issues.text}\n"
            "Improvement type: {identify_type.text}\n"
            "Suggestion: {generate_suggestion.text}\n\n"
            "Reasoning:",
            name="explain_reasoning",
            max_tokens=300,
            temperature=0.4,
            input_mapping={
                "analyze_issues.text": "analyze_issues.text",
                "identify_type.text": "identify_type.text",
                "generate_suggestion.text": "generate_suggestion.text"
            }
        )
        
        # Step 5: Estimate confidence
        chain.add_step(
            "Step 5: On a scale from 0.0 to 1.0, estimate your confidence in this suggestion. "
            "Consider how certain you are that this suggestion would improve the document, based on "
            "the available context and the clarity of the issues identified.\n\n"
            "Suggestion: {generate_suggestion.text}\n"
            "Reasoning: {explain_reasoning.text}\n\n"
            "Confidence (just the number between 0.0 and 1.0):",
            name="estimate_confidence",
            max_tokens=50,
            temperature=0.1,
            input_mapping={
                "generate_suggestion.text": "generate_suggestion.text",
                "explain_reasoning.text": "explain_reasoning.text"
            }
        )
        
        # Execute the chain
        result = chain.execute(
            section_title=section_title,
            section_content=section_content,
            section_history=section_history
        )
        
        # Check if chain completed successfully
        if not result["success"]:
            self.logger.error(f"Failed to generate suggestion for {section_title}: {result}")
            return None
        
        # Extract information from chain results
        suggestion_text = result["steps"][2]["result"].text
        suggestion_type = result["steps"][1]["result"].text.strip().lower()
        reasoning = result["steps"][3]["result"].text
        
        # Parse confidence score
        try:
            confidence_text = result["steps"][4]["result"].text.strip()
            confidence_match = re.search(r'(\d+\.\d+|\d+)', confidence_text)
            if confidence_match:
                confidence = float(confidence_match.group(1))
                # Ensure within range
                confidence = max(0.0, min(1.0, confidence))
            else:
                confidence = 0.5  # Default
        except Exception as e:
            self.logger.warning(f"Failed to parse confidence score: {e}")
            confidence = 0.5  # Default
        
        # Map to standard suggestion types
        if suggestion_type not in [SuggestionType.CLARITY, SuggestionType.COMPLETENESS,
                                SuggestionType.CONSISTENCY, SuggestionType.ACCURACY,
                                SuggestionType.STRUCTURE, SuggestionType.STYLE]:
            # Default to closest match
            suggestion_type = self._map_to_suggestion_type(suggestion_type)
        
        # Create suggestion object
        return EditSuggestion(
            section_id=chunk.chunk_id,
            section_title=section_title,
            original_text=section_content,
            suggestion=suggestion_text,
            suggestion_type=suggestion_type,
            reasoning=reasoning,
            confidence=confidence,
            metadata={
                "document_id": doc_id,
                "version": chunk.version,
                "analysis": result["steps"][0]["result"].text
            }
        )
    
    def _map_to_suggestion_type(self, text: str) -> str:
        """Map free text to a standard suggestion type."""
        text = text.lower()
        
        if any(word in text for word in ["clear", "clarity", "understand", "readable"]):
            return SuggestionType.CLARITY
        elif any(word in text for word in ["complete", "missing", "gap", "add"]):
            return SuggestionType.COMPLETENESS
        elif any(word in text for word in ["consistent", "align", "match"]):
            return SuggestionType.CONSISTENCY
        elif any(word in text for word in ["accurate", "correct", "fact", "true"]):
            return SuggestionType.ACCURACY
        elif any(word in text for word in ["structure", "organize", "flow", "format"]):
            return SuggestionType.STRUCTURE
        elif any(word in text for word in ["style", "tone", "voice", "wording"]):
            return SuggestionType.STYLE
        else:
            return SuggestionType.COMPLETENESS  # Default
    
    def _get_section_history(self, chunk) -> str:
        """Get formatted history context for a section."""
        if not chunk.previous_versions:
            return "No previous versions found for this section."
            
        history_parts = [f"Edit History for '{chunk.metadata.get('section', 'Unknown Section')}':"]
        
        # Add information about each previous version, most recent first
        for i, version in enumerate(reversed(chunk.previous_versions)):
            version_number = chunk.version - i - 1
            timestamp = version.get("timestamp", "Unknown time")
            
            # Format timestamp for readability if it's a string
            if isinstance(timestamp, str):
                try:
                    dt = datetime.fromisoformat(timestamp)
                    timestamp = dt.strftime("%Y-%m-%d %H:%M")
                except (ValueError, TypeError):
                    pass
                    
            history_parts.append(f"Version {version_number} ({timestamp}):")
            
            # Include snippet of the previous version
            text = version.get("text", "")
            if text:
                history_parts.append(text[:300] + ("..." if len(text) > 300 else ""))
                history_parts.append("")  # Empty line between versions
            
        return "\n".join(history_parts)
    
    def _identify_improvement_candidates(self, 
                                        doc_id: str, 
                                        chunks: List[Any], 
                                        max_count: int) -> List[Any]:
        """
        Identify the chunks that would benefit most from suggestions.
        
        Args:
            doc_id: Document identifier
            chunks: List of document chunks
            max_count: Maximum number of chunks to return
            
        Returns:
            List of chunks prioritized for improvement
        """
        # Calculate an "improvement potential" score for each chunk
        scored_chunks = []
        
        for chunk in chunks:
            score = 0
            section_title = chunk.metadata.get('section', f"Section {chunk.chunk_id}")
            
            # More complex sections might need more help
            score += len(chunk.text.split()) / 200  # Length factor
            
            # Recently edited sections might still need refinement
            if chunk.version > 1:
                score += min(chunk.version, 5) * 0.5  # Version factor
                
            # Sections with more users working on them might need clarity
            if doc_id in self.rag.document_activity:
                activity = self.rag.document_activity[doc_id]
                section_id = chunk.chunk_id
                
                # If this section has been frequently edited
                if section_id in activity.get("edit_frequency", {}):
                    score += activity["edit_frequency"][section_id] * 0.3
                    
                # If this section has edit history with multiple users
                if section_id in activity.get("section_history", {}):
                    users = set(item["user_id"] for item in activity["section_history"][section_id])
                    score += len(users) * 0.5
            
            # Quality heuristics - more potential for improvement if these are present
            quality_issues = self._detect_quality_issues(chunk.text)
            score += len(quality_issues) * 0.7
            
            scored_chunks.append((chunk, score, quality_issues))
            
        # Sort by score (descending) and return top chunks
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        self.logger.info(f"Prioritized sections for improvement: " + 
                       ", ".join([f"{c.metadata.get('section', c.chunk_id)}({s:.1f})" 
                                 for c, s, _ in scored_chunks[:max_count]]))
        
        return [chunk for chunk, _, _ in scored_chunks[:max_count]]
    
    def _detect_quality_issues(self, text: str) -> List[str]:
        """
        Detect potential quality issues in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of detected issues
        """
        issues = []
        
        # Check for very short paragraphs (might indicate incomplete thoughts)
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        if any(len(p.split()) < 15 for p in paragraphs if p):
            issues.append("contains_short_paragraphs")
            
        # Check for inconsistent formatting
        if re.search(r'\b[a-z]+[A-Z]', text):  # camelCase detection
            issues.append("mixed_case_styles")
            
        # Check for potential jargon or complex terminology
        complex_terms = re.findall(r'\b[A-Z][a-z]*(?:[A-Z][a-z]*)+\b', text)  # CamelCase terms
        if len(complex_terms) > 3:
            issues.append("technical_jargon")
            
        # Check for potential ambiguity (sentences with hedge words)
        hedge_words = ['may', 'might', 'could', 'possibly', 'perhaps', 'sometimes', 
                      'often', 'usually', 'generally', 'typically']
        if any(f" {word} " in f" {text.lower()} " for word in hedge_words):
            issues.append("contains_ambiguity")
            
        # Check for very long sentences (might indicate complexity)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if any(len(sent.split()) > 40 for sent in sentences if sent):
            issues.append("contains_long_sentences")
            
        return issues
    
    def record_suggestion_feedback(self, 
                                  suggestion_id: str, 
                                  accepted: bool, 
                                  user_feedback: Optional[str] = None) -> None:
        """
        Record feedback on a suggestion to improve future suggestions.
        
        Args:
            suggestion_id: Identifier for the suggestion
            accepted: Whether the suggestion was accepted
            user_feedback: Optional user feedback text
        """
        self.suggestion_feedback[suggestion_id] = {
            "timestamp": datetime.now().isoformat(),
            "accepted": accepted,
            "feedback": user_feedback
        }
        
        self.logger.info(f"Recorded feedback for suggestion {suggestion_id}: " +
                       f"{'accepted' if accepted else 'rejected'}" +
                       (f", feedback: {user_feedback}" if user_feedback else ""))