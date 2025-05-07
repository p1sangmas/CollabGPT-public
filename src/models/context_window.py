"""
Context Windows for document structure.

This module provides intelligent context windows that organize document content
into structured contexts for more effective AI reasoning.
"""

from typing import Dict, List, Any, Optional, Tuple
import re
from dataclasses import dataclass
from datetime import datetime

from ..utils import logger
from .rag_system import RAGSystem


@dataclass
class ContextWindow:
    """
    Represents a context window with structured document information.
    
    A context window is a carefully selected subset of document content
    that provides the most relevant context for a specific operation.
    """
    document_id: str
    title: str
    focus_section: Optional[str]
    content: str
    metadata: Dict[str, Any]
    
    def add_section(self, section_title: str, section_content: str) -> None:
        """Add a section to the context window."""
        self.content += f"\n\n## {section_title}\n{section_content}"
        
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the context window."""
        self.metadata[key] = value
        
    def to_text(self) -> str:
        """Convert the context window to formatted text."""
        parts = [f"# {self.title}"]
        
        # Add metadata section if it exists
        if self.metadata:
            meta_parts = ["## Document Metadata"]
            for key, value in self.metadata.items():
                if isinstance(value, dict):
                    meta_parts.append(f"### {key}")
                    for k, v in value.items():
                        meta_parts.append(f"- {k}: {v}")
                else:
                    meta_parts.append(f"- {key}: {value}")
            parts.append("\n".join(meta_parts))
            
        # Add main content
        parts.append(self.content)
        
        return "\n\n".join(parts)
        

class ContextWindowManager:
    """
    Manages context windows for document operations.
    
    This class creates and manages intelligent context windows that incorporate
    document structure, focus points, and relevance metrics.
    """
    
    def __init__(self, rag_system: RAGSystem):
        """
        Initialize the context window manager.
        
        Args:
            rag_system: RAG system to use for document content
        """
        self.rag = rag_system
        self.logger = logger.get_logger("context_window_manager")
        
    def create_focused_window(self, 
                             doc_id: str, 
                             focus_section: Optional[str] = None,
                             query: Optional[str] = None,
                             include_metadata: bool = True,
                             include_history: bool = True,
                             window_size: int = 3) -> ContextWindow:
        """
        Create a context window focused on a specific section or query.
        
        Args:
            doc_id: Document identifier
            focus_section: Optional section to focus on
            query: Optional query to find relevant content (if no focus_section)
            include_metadata: Whether to include document metadata
            include_history: Whether to include history information
            window_size: Number of sections to include on each side of focus
            
        Returns:
            A ContextWindow object
        """
        # Get document chunks
        chunks = self.rag.vector_store.get_chunks_by_doc_id(doc_id)
        
        if not chunks:
            self.logger.warning(f"No content found for document {doc_id}")
            return ContextWindow(
                document_id=doc_id,
                title="Empty Document",
                focus_section=None,
                content="",
                metadata={"error": "Document not found or empty"}
            )
            
        # Sort chunks by section index
        chunks = sorted(chunks, 
                       key=lambda c: c.metadata.get('section_index', 0) 
                       if 'section_index' in c.metadata else 9999)
        
        # Get document title from first chunk if available
        doc_title = chunks[0].metadata.get('title', f"Document {doc_id}")
        
        # If we have a query but no focus section, find the most relevant section
        if query and not focus_section:
            results = self.rag.vector_store.search(query, 1)
            if results:
                focus_chunk, _ = results[0]
                if focus_chunk.doc_id == doc_id:
                    focus_section = focus_chunk.metadata.get('section')
                    self.logger.info(f"Found focus section '{focus_section}' for query: {query}")
        
        # If we still don't have a focus section but have activity data, use most active section
        if not focus_section and doc_id in self.rag.document_activity:
            edit_frequency = self.rag.document_activity[doc_id].get("edit_frequency", {})
            if edit_frequency:
                section_id = max(edit_frequency.items(), key=lambda x: x[1])[0]
                # Find the chunk with this section ID
                for chunk in chunks:
                    if chunk.chunk_id == section_id:
                        focus_section = chunk.metadata.get('section')
                        break
                        
        # If we still don't have a focus, use the middle section
        if not focus_section and chunks:
            middle_index = len(chunks) // 2
            focus_chunk = chunks[middle_index]
            focus_section = focus_chunk.metadata.get('section', f"Section {focus_chunk.chunk_id}")
        
        # Find the index of the focus section
        focus_index = 0
        for i, chunk in enumerate(chunks):
            if chunk.metadata.get('section') == focus_section:
                focus_index = i
                break
        
        # Calculate window boundaries
        start_index = max(0, focus_index - window_size)
        end_index = min(len(chunks) - 1, focus_index + window_size)
        
        # Create the context window
        context_window = ContextWindow(
            document_id=doc_id,
            title=doc_title,
            focus_section=focus_section,
            content="",
            metadata={}
        )
        
        # Add metadata if requested
        if include_metadata:
            # Add basic document info
            context_window.add_metadata("document_id", doc_id)
            context_window.add_metadata("total_sections", len(chunks))
            context_window.add_metadata("focus_section", focus_section)
            
            # Add document structure metadata 
            structure = self.rag.analyze_document_structure(doc_id)
            if not isinstance(structure, dict) or "error" not in structure:
                context_window.add_metadata("document_structure", {
                    "section_count": structure.get("section_count", 0),
                    "total_words": structure.get("total_words", 0),
                    "average_section_length": round(structure.get("average_section_length", 0), 1)
                })
            
            # Add activity data if available
            if doc_id in self.rag.document_activity:
                activity = self.rag.document_activity[doc_id]
                if "user_activity" in activity:
                    user_count = len(activity["user_activity"])
                    context_window.add_metadata("collaboration", {
                        "user_count": user_count,
                        "last_updated": activity.get("last_updated", datetime.now()).strftime("%Y-%m-%d %H:%M")
                    })
        
        # Add sections within the window to the context
        for i in range(start_index, end_index + 1):
            chunk = chunks[i]
            section_title = chunk.metadata.get('section', f"Section {i+1}")
            
            # Mark the focus section
            if i == focus_index:
                formatted_title = f"{section_title} [FOCUS]"
            else:
                formatted_title = section_title
            
            # Add the section content
            context_window.add_section(formatted_title, chunk.text)
            
            # Add history for this section if requested
            if include_history and chunk.version > 1 and chunk.previous_versions:
                history_text = self._format_section_history(chunk)
                if history_text:
                    context_window.add_section(f"{section_title} - History", history_text)
        
        return context_window
    
    def create_document_map(self, doc_id: str) -> ContextWindow:
        """
        Create a high-level map of the document structure.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            A ContextWindow with the document map
        """
        chunks = self.rag.vector_store.get_chunks_by_doc_id(doc_id)
        
        if not chunks:
            self.logger.warning(f"No content found for document {doc_id}")
            return ContextWindow(
                document_id=doc_id,
                title="Empty Document",
                focus_section=None,
                content="",
                metadata={"error": "Document not found or empty"}
            )
            
        # Sort chunks by section index
        chunks = sorted(chunks, 
                       key=lambda c: c.metadata.get('section_index', 0) 
                       if 'section_index' in c.metadata else 9999)
        
        # Get document title from first chunk if available
        doc_title = chunks[0].metadata.get('title', f"Document {doc_id}")
        
        # Create the context window
        context_window = ContextWindow(
            document_id=doc_id,
            title=f"Document Map: {doc_title}",
            focus_section=None,
            content="",
            metadata={
                "document_id": doc_id,
                "total_sections": len(chunks)
            }
        )
        
        # Add document structure analysis
        structure = self.rag.analyze_document_structure(doc_id)
        if not isinstance(structure, dict) or "error" not in structure:
            context_window.add_metadata("structure", {
                "section_count": structure.get("section_count", 0),
                "total_words": structure.get("total_words", 0),
                "average_section_length": round(structure.get("average_section_length", 0), 1)
            })
        
        # Add activity data if available
        if doc_id in self.rag.document_activity:
            activity = self.rag.document_activity[doc_id]
            if "edit_frequency" in activity:
                # Find most edited sections
                top_sections = sorted(
                    activity["edit_frequency"].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:3]
                
                if top_sections:
                    context_window.add_metadata("most_active_sections", {
                        section_id: count for section_id, count in top_sections
                    })
        
        # Add outline of all sections with brief summaries
        outline_parts = ["## Document Outline"]
        
        for i, chunk in enumerate(chunks):
            section_title = chunk.metadata.get('section', f"Section {i+1}")
            # Create a very brief summary (first sentence or first 100 chars)
            text = chunk.text
            summary = text.split('.')[0] + '.' if '.' in text else text[:100] + '...'
            
            # Add section version if available
            version_info = f" (v{chunk.version})" if chunk.version > 1 else ""
            
            outline_parts.append(f"### {section_title}{version_info}")
            outline_parts.append(summary)
            
            # Add a note about edit activity if this is an active section
            if (doc_id in self.rag.document_activity and 
                chunk.chunk_id in self.rag.document_activity[doc_id].get("edit_frequency", {})):
                edit_count = self.rag.document_activity[doc_id]["edit_frequency"][chunk.chunk_id]
                if edit_count > 1:
                    outline_parts.append(f"*This section has been edited {edit_count} times*")
        
        context_window.content = "\n\n".join(outline_parts)
        return context_window
    
    def create_query_focused_window(self, 
                                   query: str,
                                   doc_id: str = None,
                                   max_sections: int = 5,
                                   include_metadata: bool = True) -> ContextWindow:
        """
        Create a context window focused on answering a specific query.
        
        Args:
            query: The search query
            doc_id: Optional document ID to limit results to
            max_sections: Maximum number of sections to include
            include_metadata: Whether to include document metadata
            
        Returns:
            A ContextWindow with relevant content for the query
        """
        # Search for relevant chunks
        results = self.rag.vector_store.search(query, top_k=max_sections * 2)
        
        # Filter by document if specified
        if doc_id:
            results = [(chunk, score) for chunk, score in results if chunk.doc_id == doc_id]
            
        if not results:
            self.logger.warning(f"No relevant content found for query: {query}")
            return ContextWindow(
                document_id=doc_id or "multiple",
                title=f"Query Results: {query}",
                focus_section=None,
                content="No relevant content found.",
                metadata={"query": query}
            )
        
        # If results span multiple documents and no doc_id was specified,
        # group chunks by document and sort documents by relevance
        if not doc_id:
            # Group chunks by document
            docs = {}
            for chunk, score in results:
                if chunk.doc_id not in docs:
                    docs[chunk.doc_id] = {
                        "chunks": [],
                        "total_score": 0,
                        "title": chunk.metadata.get('title', f"Document {chunk.doc_id}")
                    }
                docs[chunk.doc_id]["chunks"].append((chunk, score))
                docs[chunk.doc_id]["total_score"] += score
                
            # Sort documents by relevance score
            sorted_docs = sorted(docs.items(), key=lambda x: x[1]["total_score"], reverse=True)
            
            # Take most relevant sections from most relevant documents
            selected_chunks = []
            for doc_id, doc_info in sorted_docs:
                # Sort chunks within this document by relevance
                sorted_chunks = sorted(doc_info["chunks"], key=lambda x: x[1], reverse=True)
                # Add top chunks from this document
                remaining = max_sections - len(selected_chunks)
                if remaining <= 0:
                    break
                selected_chunks.extend([c for c, _ in sorted_chunks[:remaining]])
                
            title = f"Multi-Document Query Results: {query}"
        else:
            # Single document mode - just take top chunks
            sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
            selected_chunks = [chunk for chunk, _ in sorted_results[:max_sections]]
            
            # Get document title
            if selected_chunks:
                title = selected_chunks[0].metadata.get('title', f"Document {doc_id}")
                title = f"Query Results in {title}: {query}"
            else:
                title = f"Query Results: {query}"
        
        # Create the context window
        context_window = ContextWindow(
            document_id=doc_id or "multiple",
            title=title,
            focus_section=None,
            content="",
            metadata={"query": query}
        )
        
        # Add metadata if requested
        if include_metadata:
            context_window.add_metadata("result_count", len(selected_chunks))
            
            if doc_id:
                # Add document-specific metadata
                structure = self.rag.analyze_document_structure(doc_id)
                if not isinstance(structure, dict) or "error" not in structure:
                    context_window.add_metadata("document_structure", {
                        "section_count": structure.get("section_count", 0),
                        "total_words": structure.get("total_words", 0)
                    })
        
        # Add content from each selected chunk
        for i, chunk in enumerate(selected_chunks):
            # Get section information
            section_title = chunk.metadata.get('section', f"Section {chunk.chunk_id}")
            doc_title = chunk.metadata.get('title', f"Document {chunk.doc_id}")
            
            # For multi-document mode, include document title in section header
            if not doc_id:
                section_header = f"{doc_title} - {section_title}"
            else:
                section_header = section_title
                
            # Add the section content
            context_window.add_section(section_header, chunk.text)
        
        return context_window
    
    def _format_section_history(self, chunk, max_versions: int = 2) -> str:
        """Format section history for inclusion in a context window."""
        if not chunk.previous_versions:
            return ""
            
        # Only include the most recent versions up to max_versions
        recent_versions = chunk.previous_versions[-max_versions:]
        
        history_parts = []
        
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
            
            # Include content snippet
            text = version.get("text", "")
            if text:
                if len(text) > 300:
                    history_parts.append(text[:300] + "...")
                else:
                    history_parts.append(text)
            
        return "\n\n".join(history_parts)