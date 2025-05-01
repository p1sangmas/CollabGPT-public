"""
Basic RAG (Retrieval Augmented Generation) system for CollabGPT.

This module implements a simple RAG system that provides contextual information
about documents to enhance AI-generated summaries and suggestions.
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


class DocumentChunk:
    """Represents a chunk of document content for retrieval purposes."""
    
    def __init__(self, 
                 doc_id: str,
                 chunk_id: str,
                 text: str,
                 metadata: Dict[str, Any] = None,
                 embedding: List[float] = None):
        """
        Initialize a document chunk.
        
        Args:
            doc_id: The document identifier
            chunk_id: The chunk identifier within the document
            text: The text content of the chunk
            metadata: Additional information about the chunk
            embedding: Vector embedding of the chunk (if available)
        """
        self.doc_id = doc_id
        self.chunk_id = chunk_id
        self.text = text
        self.metadata = metadata or {}
        self.embedding = embedding
        self.last_updated = datetime.now()


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
                        embedding=data.get("embedding")
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
        # First, remove any existing chunks for this document
        self.vector_store.delete_chunks_by_doc_id(doc_id)
        
        # Split the document into chunks
        chunks = self._chunk_document(doc_id, content, metadata)
        
        # Add chunks to the vector store
        chunk_ids = self.vector_store.add_chunks(chunks)
        
        self.logger.info(f"Processed document {doc_id}: created {len(chunks)} chunks")
        
        return chunk_ids
    
    def get_relevant_context(self, query: str, doc_id: str = None, 
                             max_chunks: int = 3) -> str:
        """
        Get relevant document context based on a query.
        
        Args:
            query: The query to find relevant context for
            doc_id: Optional document ID to limit context to a specific document
            max_chunks: Maximum number of chunks to include
            
        Returns:
            Combined text from relevant chunks
        """
        # First, perform a search
        results = self.vector_store.search(query, max_chunks)
        
        # Filter by document ID if specified
        if doc_id:
            results = [(chunk, score) for chunk, score in results 
                       if chunk.doc_id == doc_id]
        
        # Extract and combine the text from the chunks
        if not results:
            return ""
            
        context_parts = []
        
        for chunk, score in results:
            # Add metadata about the chunk
            header = f"Document: {chunk.metadata.get('title', chunk.doc_id)}"
            if 'section' in chunk.metadata:
                header += f" | Section: {chunk.metadata['section']}"
            
            # Add the chunk text with its header
            context_parts.append(f"{header}\n{chunk.text}")
            
        return "\n\n---\n\n".join(context_parts)
    
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