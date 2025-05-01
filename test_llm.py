#!/usr/bin/env python3
"""
Test script for CollabGPT LLM integration.
This script helps verify that the language model integration works properly.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.models.llm_interface import LLMInterface
from src.models.rag_system import RAGSystem, DocumentChunk
from src.utils import logger

def test_llm_interface():
    """Test basic LLM functionality."""
    print("Testing LLM Interface...")
    
    # Create LLM interface
    llm = LLMInterface()
    
    print(f"LLM mode: {llm.mode}")
    
    # Test basic prompt
    print("\nTesting basic prompt generation...")
    test_prompt = "Summarize the benefits of collaborative document editing."
    
    response = llm.generate(test_prompt)
    
    if response.success:
        print(f"✅ Successfully generated response:")
        print(f"---\n{response.text}\n---")
    else:
        print(f"❌ Generation failed: {response.error}")
        
    # Test template-based prompt
    print("\nTesting template-based prompt...")
    
    # First check if templates exist
    template_name = "summarize_document"
    template = llm.get_template(template_name)
    
    if template:
        print(f"✅ Found template: {template_name}")
        
        # Test generating with template
        response = llm.generate_with_template(
            template_name,
            document_content="This is a test document about collaborative editing. "
                           "It discusses the benefits of real-time collaboration "
                           "and how AI can enhance the experience."
        )
        
        if response.success:
            print(f"✅ Successfully generated template-based response:")
            print(f"---\n{response.text}\n---")
        else:
            print(f"❌ Template-based generation failed: {response.error}")
    else:
        print(f"❌ Template not found: {template_name}")
        
    return llm

def test_rag_system():
    """Test basic RAG functionality."""
    print("\nTesting RAG System...")
    
    # Create RAG system
    rag = RAGSystem()
    
    # Create test document
    doc_id = "test_doc_001"
    content = """
    # Document Collaboration Best Practices
    
    ## Introduction
    This document outlines best practices for team collaboration on shared documents.
    
    ## Real-time Editing
    Real-time editing allows multiple team members to work on a document simultaneously.
    This improves efficiency and reduces version control issues.
    
    ## Comment Systems
    Using comments effectively helps maintain clear communication about specific parts
    of a document without altering the main content.
    
    ## Version History
    Always maintain a clear version history to track changes and revert if necessary.
    """
    
    # Process document
    print("Processing test document...")
    chunk_ids = rag.process_document(
        doc_id=doc_id,
        content=content,
        metadata={"title": "Document Collaboration Best Practices"}
    )
    
    print(f"✅ Document processed into {len(chunk_ids)} chunks")
    
    # Test retrieval
    print("\nTesting context retrieval...")
    query = "best practices for comments in documents"
    context = rag.get_relevant_context(query, doc_id=doc_id)
    
    if context:
        print(f"✅ Successfully retrieved context:")
        print(f"---\n{context[:200]}...\n---")
    else:
        print("❌ Failed to retrieve relevant context")
        
    return rag

def test_llm_with_rag(llm, rag):
    """Test combining LLM with RAG."""
    if not llm or not rag:
        return
        
    print("\nTesting LLM with RAG integration...")
    
    # Define a query
    query = "suggestions for improving document collaboration"
    
    # Get relevant context
    context = rag.get_relevant_context(query)
    
    # Create prompt with context
    prompt = f"""
    Please provide suggestions for improving document collaboration based on the following context:
    
    {context}
    
    Suggestions:
    """
    
    # Generate response
    response = llm.generate(prompt)
    
    if response.success:
        print(f"✅ Successfully generated RAG-enhanced response:")
        print(f"---\n{response.text}\n---")
    else:
        print(f"❌ RAG-enhanced generation failed: {response.error}")

def main():
    """Run the LLM and RAG tests."""
    print("CollabGPT LLM and RAG Test")
    print("==========================\n")
    
    # Test LLM interface
    llm = test_llm_interface()
    
    # Test RAG system
    rag = test_rag_system()
    
    # Test combined LLM+RAG
    if llm and rag:
        test_llm_with_rag(llm, rag)
    
    print("\nLLM and RAG testing completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())