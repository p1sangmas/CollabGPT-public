"""
Enhanced RAG (Retrieval Augmented Generation) system for CollabGPT.

This module implements an advanced RAG system that provides rich contextual information
about documents with history tracking to enhance AI-generated summaries and suggestions.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
import os
import json
import re
from datetime import datetime, timedelta
import math
from pathlib import Path

from ..utils import logger
from ..config import settings

# Import context window if available
try:
    from .context_window import ContextWindowManager, ContextWindow
    CONTEXT_WINDOWS_AVAILABLE = True
except ImportError:
    CONTEXT_WINDOWS_AVAILABLE = False


class DocumentChunk:
    """Represents a chunk of document content for retrieval purposes."""
    
    def __init__(self, 
                 doc_id: str,
                 chunk_id: str,
                 text: str,
                 metadata: Dict[str, Any] = None,
                 embedding: List[float] = None,
                 version: int = 1,   # Added version tracking
                 previous_versions: List[Dict[str, Any]] = None):  # Added version history
        """
        Initialize a document chunk.
        
        Args:
            doc_id: The document identifier
            chunk_id: The chunk identifier within the document
            text: The text content of the chunk
            metadata: Additional information about the chunk
            embedding: Vector embedding of the chunk (if available)
            version: Version number of this chunk
            previous_versions: List of previous versions of this chunk's content
        """
        self.doc_id = doc_id
        self.chunk_id = chunk_id
        self.text = text
        self.metadata = metadata or {}
        self.embedding = embedding
        self.last_updated = datetime.now()
        self.version = version
        self.previous_versions = previous_versions or []

    def add_version(self, previous_text: str, metadata: Dict[str, Any] = None) -> None:
        """
        Add a previous version of this chunk to the history.
        
        Args:
            previous_text: The text content of the previous version
            metadata: Additional metadata for the previous version
        """
        version_data = {
            "text": previous_text,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.previous_versions.append(version_data)
        self.version += 1


class SimpleVectorStore:
    """
    A basic vector store implementation for document chunks.
    
    In a production environment, this would be replaced with a proper
    vector database like Pinecone, Weaviate, or Chroma.
    """
    
    def __init__(self, persist_dir: str = None):
        """
        Initialize the vector store.
        
        Args:
            persist_dir: Directory to persist vector data (optional)
        """
        self.chunks: Dict[str, DocumentChunk] = {}
        self.persist_dir = persist_dir
        
        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)
            self._load_persisted_data()
    
    def add_chunk(self, chunk: DocumentChunk) -> str:
        """
        Add a chunk to the vector store.
        
        Args:
            chunk: The document chunk to add
            
        Returns:
            The chunk ID
        """
        # Generate a composite ID if not already part of the chunk
        chunk_id = f"{chunk.doc_id}:{chunk.chunk_id}"
        
        # Store the chunk
        self.chunks[chunk_id] = chunk
        
        # Persist if configured
        if self.persist_dir:
            self._persist_chunk(chunk)
            
        return chunk_id
    
    def add_chunks(self, chunks: List[DocumentChunk]) -> List[str]:
        """
        Add multiple chunks to the vector store.
        
        Args:
            chunks: List of document chunks to add
            
        Returns:
            List of chunk IDs
        """
        return [self.add_chunk(chunk) for chunk in chunks]
    
    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """
        Retrieve a chunk by ID.
        
        Args:
            chunk_id: The ID of the chunk to retrieve
            
        Returns:
            The document chunk if found, None otherwise
        """
        return self.chunks.get(chunk_id)
    
    def get_chunks_by_doc_id(self, doc_id: str) -> List[DocumentChunk]:
        """
        Retrieve all chunks for a specific document.
        
        Args:
            doc_id: The document identifier
            
        Returns:
            List of document chunks for the document
        """
        return [chunk for chunk_id, chunk in self.chunks.items() 
                if chunk.doc_id == doc_id]
    
    def delete_chunks_by_doc_id(self, doc_id: str) -> int:
        """
        Delete all chunks for a specific document.
        
        Args:
            doc_id: The document identifier
            
        Returns:
            Number of chunks deleted
        """
        to_delete = [chunk_id for chunk_id, chunk in self.chunks.items() 
                    if chunk.doc_id == doc_id]
        
        for chunk_id in to_delete:
            del self.chunks[chunk_id]
            
            # Remove persisted data if configured
            if self.persist_dir:
                chunk_file = os.path.join(self.persist_dir, f"{chunk_id}.json")
                if os.path.exists(chunk_file):
                    os.remove(chunk_file)
        
        return len(to_delete)
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for chunks relevant to a query using basic keyword matching.
        
        In a production RAG system, this would use vector similarity search.
        
        Args:
            query: The search query
            top_k: Maximum number of results to return
            
        Returns:
            List of (chunk, relevance_score) tuples
        """
        # Simple keyword-based relevance scoring
        query_terms = re.findall(r'\w+', query.lower())
        
        results = []
        for chunk in self.chunks.values():
            text_lower = chunk.text.lower()
            
            # Count occurrences of query terms in the chunk
            score = sum(text_lower.count(term) for term in query_terms)
            
            # Normalize by chunk length to avoid favoring longer chunks
            if score > 0:
                normalized_score = score / (len(chunk.text.split()) + 1)
                results.append((chunk, normalized_score))
        
        # Sort by score and return top-k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _persist_chunk(self, chunk: DocumentChunk) -> None:
        """
        Persist a chunk to disk.
        
        Args:
            chunk: The document chunk to persist
        """
        chunk_id = f"{chunk.doc_id}:{chunk.chunk_id}"
        chunk_data = {
            "doc_id": chunk.doc_id,
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "metadata": chunk.metadata,
            "last_updated": chunk.last_updated.isoformat(),
            "version": chunk.version,  # Persist version
            "previous_versions": chunk.previous_versions  # Persist version history
        }
        
        # Don't store empty embeddings
        if chunk.embedding:
            chunk_data["embedding"] = chunk.embedding
            
        chunk_file = os.path.join(self.persist_dir, f"{chunk_id}.json")
        with open(chunk_file, 'w') as f:
            json.dump(chunk_data, f)
    
    def _load_persisted_data(self) -> None:
        """Load all persisted chunks from disk."""
        if not os.path.exists(self.persist_dir):
            return
            
        for filename in os.listdir(self.persist_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.persist_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        
                    # Create chunk from persisted data
                    chunk = DocumentChunk(
                        doc_id=data["doc_id"],
                        chunk_id=data["chunk_id"],
                        text=data["text"],
                        metadata=data.get("metadata", {}),
                        embedding=data.get("embedding"),
                        version=data.get("version", 1),  # Load version
                        previous_versions=data.get("previous_versions", [])  # Load version history
                    )
                    
                    # Parse the last_updated timestamp
                    if "last_updated" in data:
                        try:
                            chunk.last_updated = datetime.fromisoformat(data["last_updated"])
                        except (ValueError, TypeError):
                            chunk.last_updated = datetime.now()
                    
                    # Store the chunk in memory
                    chunk_id = f"{chunk.doc_id}:{chunk.chunk_id}"
                    self.chunks[chunk_id] = chunk
                    
                except Exception as e:
                    logger.error(f"Error loading chunk from {filepath}: {e}")


class RAGSystem:
    """
    Retrieval Augmented Generation system for document context.
    """
    
    def __init__(self, vector_store: SimpleVectorStore = None):
        """
        Initialize the RAG system.
        
        Args:
            vector_store: Vector store to use (creates one if None)
        """
        if vector_store:
            self.vector_store = vector_store
        else:
            # Create a vector store persisted in the data directory
            persist_dir = os.path.join(settings.DATA_DIR, "rag_data")
            self.vector_store = SimpleVectorStore(persist_dir)
            
        self.logger = logger.get_logger("rag_system")
        
        # Track document collaboration metrics
        self.document_activity = {}  # doc_id -> activity metrics
        
    def track_activity(self, doc_id: str, user_id: str, activity_type: str, 
                      section_id: str = None, timestamp: datetime = None):
        """
        Track user activity on documents to build contextual awareness
        
        Args:
            doc_id: Document identifier
            user_id: User identifier
            activity_type: Type of activity (edit, comment, view, etc.)
            section_id: Optional section identifier
            timestamp: Activity timestamp (defaults to now)
        """
        if doc_id not in self.document_activity:
            self.document_activity[doc_id] = {
                "last_updated": datetime.now(),
                "edit_frequency": {},  # section -> frequency
                "user_activity": {},   # user_id -> activity count
                "section_history": {}  # section -> history of changes
            }
            
        activity = self.document_activity[doc_id]
        activity["last_updated"] = timestamp or datetime.now()
        
        # Update user activity
        if user_id not in activity["user_activity"]:
            activity["user_activity"][user_id] = {}
        
        if activity_type not in activity["user_activity"][user_id]:
            activity["user_activity"][user_id][activity_type] = 0
            
        activity["user_activity"][user_id][activity_type] += 1
        
        # Update section edit frequency if applicable
        if section_id and activity_type == "edit":
            if section_id not in activity["edit_frequency"]:
                activity["edit_frequency"][section_id] = 0
            activity["edit_frequency"][section_id] += 1
            
            # Track section history
            if section_id not in activity["section_history"]:
                activity["section_history"][section_id] = []
                
            activity["section_history"][section_id].append({
                "user_id": user_id,
                "timestamp": timestamp or datetime.now(),
                "activity_type": activity_type
            })
            
    def process_document(self, doc_id: str, content: str, 
                         metadata: Dict[str, Any] = None) -> List[str]:
        """
        Process a document and store its chunks for retrieval.
        
        Args:
            doc_id: The document identifier
            content: The full document content
            metadata: Additional document metadata
            
        Returns:
            List of chunk IDs created
        """
        # Save existing chunks before removing them to preserve history
        existing_chunks = self.vector_store.get_chunks_by_doc_id(doc_id)
        
        # Extract content from existing chunks by section/chunk ID for version tracking
        existing_content_by_id = {}
        for chunk in existing_chunks:
            existing_content_by_id[chunk.chunk_id] = {
                "text": chunk.text,
                "metadata": chunk.metadata,
                "version": chunk.version,
                "previous_versions": chunk.previous_versions
            }
        
        # Split the new document into chunks
        new_chunks = self._chunk_document(doc_id, content, metadata)
        
        # For each new chunk, check if it exists in previous version and add history if so
        for chunk in new_chunks:
            if chunk.chunk_id in existing_content_by_id:
                existing_data = existing_content_by_id[chunk.chunk_id]
                
                # Only add version history if content has changed
                if chunk.text != existing_data["text"]:
                    # Add previous content as a version
                    chunk.previous_versions = existing_data["previous_versions"]
                    chunk.add_version(
                        existing_data["text"], 
                        existing_data["metadata"]
                    )
                    chunk.version = existing_data["version"] + 1
                else:
                    # Content hasn't changed, keep existing versions
                    chunk.version = existing_data["version"]
                    chunk.previous_versions = existing_data["previous_versions"]
                    
                self.logger.info(
                    f"Updated chunk {chunk.chunk_id} in document {doc_id}, " +
                    f"now at version {chunk.version}"
                )
        
        # Now remove old chunks
        self.vector_store.delete_chunks_by_doc_id(doc_id)
        
        # Add the new/updated chunks to the vector store
        chunk_ids = self.vector_store.add_chunks(new_chunks)
        
        self.logger.info(f"Processed document {doc_id}: created/updated {len(new_chunks)} chunks")
        
        return chunk_ids
    
    def get_relevant_context(self, query: str, doc_id: str = None, 
                             max_chunks: int = 3, include_history: bool = True,
                             include_user_activity: bool = True) -> str:
        """
        Get relevant document context based on a query.
        
        Args:
            query: The query to find relevant context for
            doc_id: Optional document ID to limit context to a specific document
            max_chunks: Maximum number of chunks to include
            include_history: Whether to include historical versions in context
            include_user_activity: Whether to include user activity patterns
            
        Returns:
            Combined text from relevant chunks with enhanced context
        """
        # First, perform a search
        results = self.vector_store.search(query, max_chunks * 2)  # Get more results initially
        
        # Filter by document ID if specified
        if doc_id:
            results = [(chunk, score) for chunk, score in results 
                      if chunk.doc_id == doc_id]
        
        # Extract and combine the text from the chunks
        if not results:
            return ""
            
        context_parts = []
        sections_included = set()  # Track which sections we've included
        
        # First pass: include the most relevant chunks
        for chunk, score in results[:max_chunks]:
            # Add metadata about the chunk
            header = f"Document: {chunk.metadata.get('title', chunk.doc_id)}"
            if 'section' in chunk.metadata:
                header += f" | Section: {chunk.metadata['section']}"
                sections_included.add(chunk.metadata.get('section'))
            
            # Include version information if available
            if chunk.version > 1:
                header += f" | Version: {chunk.version}"
                
            # Add the chunk text with its header
            context_parts.append(f"{header}\n{chunk.text}")
            
            # Add historical context if requested and available
            if include_history and chunk.previous_versions:
                history_parts = self._format_chunk_history(chunk, max_versions=2)
                if history_parts:
                    context_parts.append(history_parts)
        
        # Add document-level activity context if available and requested
        if include_user_activity and doc_id and doc_id in self.document_activity:
            activity_context = self._format_activity_context(doc_id, sections_included)
            if activity_context:
                context_parts.append(activity_context)
            
        return "\n\n---\n\n".join(context_parts)
    
    def _format_chunk_history(self, chunk: DocumentChunk, max_versions: int = 2) -> str:
        """
        Format the history of a chunk for context inclusion.
        
        Args:
            chunk: The document chunk
            max_versions: Maximum number of previous versions to include
            
        Returns:
            Formatted history text
        """
        if not chunk.previous_versions:
            return ""
            
        # Only include the most recent versions up to max_versions
        recent_versions = chunk.previous_versions[-max_versions:]
        
        history_parts = [f"Historical changes for section '{chunk.metadata.get('section', 'Unknown Section')}':"]
        
        for i, version in enumerate(recent_versions):
            version_number = chunk.version - len(recent_versions) + i
            timestamp = version.get("timestamp", "Unknown time")
            
            # Format timestamp for readability if it's a string
            if isinstance(timestamp, str):
                try:
                    dt = datetime.fromisoformat(timestamp)
                    timestamp = dt.strftime("%Y-%m-%d %H:%M")
                except (ValueError, TypeError):
                    pass
                    
            history_parts.append(f"Version {version_number} ({timestamp}):")
            
            # Include a snippet of the previous version
            text = version.get("text", "")
            if len(text) > 200:
                text = text[:200] + "..."
            history_parts.append(text)
            
        return "\n".join(history_parts)
    
    def _format_activity_context(self, doc_id: str, sections_included: Set[str]) -> str:
        """
        Format document activity context for inclusion in responses
        
        Args:
            doc_id: Document identifier
            sections_included: Set of section names already included in context
            
        Returns:
            Formatted activity context
        """
        activity = self.document_activity.get(doc_id)
        if not activity:
            return ""
            
        parts = ["Document Activity Context:"]
        
        # Add last update time
        parts.append(f"Last updated: {activity['last_updated'].strftime('%Y-%m-%d %H:%M')}")
        
        # Add most active users (up to 3)
        user_activity = activity.get("user_activity", {})
        if user_activity:
            # Count total actions per user
            user_totals = {}
            for user_id, actions in user_activity.items():
                user_totals[user_id] = sum(actions.values())
                
            # Get top users
            top_users = sorted(user_totals.items(), key=lambda x: x[1], reverse=True)[:3]
            if top_users:
                parts.append("Most active users:")
                for user_id, count in top_users:
                    parts.append(f"- {user_id}: {count} total actions")
        
        # Add frequently edited sections not already in context
        edit_frequency = activity.get("edit_frequency", {})
        if edit_frequency:
            frequent_sections = sorted(edit_frequency.items(), key=lambda x: x[1], reverse=True)
            frequent_sections = [s for s in frequent_sections if s[0] not in sections_included][:2]
            
            if frequent_sections:
                parts.append("Other frequently edited sections:")
                for section_id, count in frequent_sections:
                    parts.append(f"- {section_id}: {count} edits")
        
        return "\n".join(parts)

    def get_document_context_window(self, doc_id: str, 
                                   focus_section: str = None,
                                   window_size: int = 3) -> str:
        """
        Get a contextual window of document content centered around a section
        
        Args:
            doc_id: Document identifier
            focus_section: Section to focus on (if None, use most active section)
            window_size: Number of sections to include on each side of focus
            
        Returns:
            Combined text from the context window
        """
        chunks = self.vector_store.get_chunks_by_doc_id(doc_id)
        if not chunks:
            return ""
            
        # Sort chunks by section index if available
        chunks.sort(key=lambda c: c.metadata.get('section_index', 0))
        
        # If no focus section provided, use most active section from tracking
        if not focus_section and doc_id in self.document_activity:
            edit_frequency = self.document_activity[doc_id].get("edit_frequency", {})
            if edit_frequency:
                focus_section = max(edit_frequency.items(), key=lambda x: x[1])[0]
        
        # Find the focus chunk
        focus_index = 0
        for i, chunk in enumerate(chunks):
            if (focus_section and 
                chunk.metadata.get('section') == focus_section) or (
                not focus_section and i == len(chunks) // 2):
                focus_index = i
                break
                
        # Calculate window boundaries
        start_index = max(0, focus_index - window_size)
        end_index = min(len(chunks) - 1, focus_index + window_size)
        
        # Build context from chunks in window
        context_parts = []
        
        # Add document title
        doc_title = chunks[0].metadata.get('title', doc_id)
        context_parts.append(f"Document: {doc_title}")
        
        # Add context window
        for i in range(start_index, end_index + 1):
            chunk = chunks[i]
            
            # Mark the focus section
            if i == focus_index:
                section_header = f"[FOCUS] Section: {chunk.metadata.get('section', f'Section {i}')}"
            else:
                section_header = f"Section: {chunk.metadata.get('section', f'Section {i}')}"
                
            context_parts.append(f"{section_header}\n{chunk.text}")
            
        return "\n\n---\n\n".join(context_parts)
        
    def get_document_history(self, doc_id: str, section_title: str = None) -> Dict[str, Any]:
        """
        Get the change history for a document or specific section.
        
        Args:
            doc_id: The document identifier
            section_title: Optional section title to filter changes
            
        Returns:
            Dictionary with history information
        """
        chunks = self.vector_store.get_chunks_by_doc_id(doc_id)
        
        if not chunks:
            return {"error": f"No document found with ID: {doc_id}"}
            
        history = {
            "document_id": doc_id,
            "sections": []
        }
        
        for chunk in chunks:
            # Skip if we're looking for a specific section and this isn't it
            if section_title and chunk.metadata.get('section') != section_title:
                continue
                
            # Only include chunks with history
            if chunk.version > 1 and chunk.previous_versions:
                section_history = {
                    "section": chunk.metadata.get('section', 'Unknown Section'),
                    "current_version": chunk.version,
                    "current_content": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                    "changes": []
                }
                
                # Add information about each previous version
                for i, version in enumerate(chunk.previous_versions):
                    version_number = i + 1
                    timestamp = version.get("timestamp", "Unknown time")
                    
                    # Format timestamp for readability if it's a string
                    if isinstance(timestamp, str):
                        try:
                            dt = datetime.fromisoformat(timestamp)
                            timestamp = dt.strftime("%Y-%m-%d %H:%M")
                        except (ValueError, TypeError):
                            pass
                    
                    version_info = {
                        "version": version_number,
                        "timestamp": timestamp,
                        "content_snippet": version.get("text", "")[:200] + "..." 
                                          if len(version.get("text", "")) > 200 
                                          else version.get("text", "")
                    }
                    section_history["changes"].append(version_info)
                
                history["sections"].append(section_history)
        
        return history
    
    def analyze_document_structure(self, doc_id: str) -> Dict[str, Any]:
        """
        Analyze document structure to provide metadata for RAG context
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Dictionary with structural metadata
        """
        chunks = self.vector_store.get_chunks_by_doc_id(doc_id)
        if not chunks:
            return {"error": f"No document found with ID: {doc_id}"}
            
        # Extract section metadata
        sections = []
        for chunk in chunks:
            if 'section' in chunk.metadata:
                sections.append({
                    "section": chunk.metadata['section'],
                    "index": chunk.metadata.get('section_index', 0),
                    "word_count": len(chunk.text.split()),
                    "last_updated": chunk.last_updated.isoformat()
                })
        
        # Sort sections by index
        sections.sort(key=lambda s: s["index"])
        
        # Calculate basic document statistics
        total_words = sum(section["word_count"] for section in sections)
        
        return {
            "document_id": doc_id,
            "section_count": len(sections),
            "total_words": total_words,
            "average_section_length": total_words / len(sections) if sections else 0,
            "sections": sections
        }
    
    def _chunk_document(self, doc_id: str, content: str, 
                        metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
        """
        Split a document into chunks for retrieval.
        
        Args:
            doc_id: The document identifier
            content: The full document content
            metadata: Additional document metadata
            
        Returns:
            List of document chunks
        """
        if not content:
            return []
            
        chunks = []
        metadata = metadata or {}
        
        # Try to split by sections if they exist
        sections = self._identify_sections(content)
        
        if sections:
            # Create a chunk for each section
            for i, section in enumerate(sections):
                section_metadata = metadata.copy()
                section_metadata.update({
                    'section': section['title'],
                    'section_index': i,
                })
                
                chunk = DocumentChunk(
                    doc_id=doc_id,
                    chunk_id=f"section_{i}",
                    text=section['content'],
                    metadata=section_metadata,
                )
                chunks.append(chunk)
        else:
            # Fall back to paragraph-based chunking
            paragraphs = [p for p in content.split('\n\n') if p.strip()]
            
            # Group paragraphs into chunks of reasonable size
            chunk_size = 3  # paragraphs per chunk
            for i in range(0, len(paragraphs), chunk_size):
                chunk_text = '\n\n'.join(paragraphs[i:i+chunk_size])
                
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_index'] = i // chunk_size
                
                chunk = DocumentChunk(
                    doc_id=doc_id,
                    chunk_id=f"chunk_{i // chunk_size}",
                    text=chunk_text,
                    metadata=chunk_metadata,
                )
                chunks.append(chunk)
        
        return chunks
    
    def _identify_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Identify document sections based on headings.
        
        Args:
            text: The document text
            
        Returns:
            List of section information
        """
        # Use heuristics to identify section headings (similar to DocumentAnalyzer)
        lines = text.split('\n')
        sections = []
        current_section = None
        current_content = []
        
        heading_patterns = [
            r'^#{1,6}\s+(.+)$',  # Markdown headings
            r'^([A-Z][A-Za-z\s]+)$',  # All caps or title case, standalone
            r'^([IVX]+\.\s+.+)$',  # Roman numeral headings
            r'^(\d+\.\s+.+)$'  # Numbered headings
        ]
        
        for i, line in enumerate(lines):
            line = line.strip()
            is_heading = False
            
            for pattern in heading_patterns:
                match = re.match(pattern, line)
                if match:
                    # Save previous section if exists
                    if current_section:
                        sections.append({
                            'title': current_section,
                            'content': '\n'.join(current_content),
                            'level': 1,  # Simplified level detection
                            'position': len(sections)
                        })
                    
                    # Start new section
                    current_section = match.group(1)
                    current_content = []
                    is_heading = True
                    break
            
            if not is_heading and line:
                current_content.append(line)
        
        # Add the final section
        if current_section:
            sections.append({
                'title': current_section,
                'content': '\n'.join(current_content),
                'level': 1,
                'position': len(sections)
            })
        elif current_content:  # Content without headings
            sections.append({
                'title': 'Untitled Section',
                'content': '\n'.join(current_content),
                'level': 1,
                'position': 0
            })
            
        return sections