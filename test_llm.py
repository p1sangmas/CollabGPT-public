#!/usr/bin/env python3
"""
Test script for LLM integration with OpenRouter.

This script validates that the LLM interface can successfully
connect to OpenRouter and generate responses using the DeepSeek model.
"""

import sys
import os
from pathlib import Path
import time

# Add root directory to path to allow importing the CollabGPT package
root_dir = Path(__file__).parent
sys.path.append(str(root_dir))

from dotenv import load_dotenv
load_dotenv()

from src.models.llm_interface import LLMInterface
from src.utils import logger

# Configure logging
logger = logger.get_logger("test_llm")
logger.info("Testing LLM integration with OpenRouter")

def test_basic_prompt():
    """Test a basic prompt to verify LLM is working."""
    llm = LLMInterface()
    
    # Basic test prompt
    prompt = "You are CollabGPT, an AI assistant for document collaboration. Please write a brief paragraph explaining your purpose."
    
    logger.info(f"Sending test prompt to model: {llm.model_path}")
    response = llm.generate(prompt, max_tokens=200)
    
    if response.success:
        logger.info("LLM response received successfully!")
        logger.info(f"Response: {response.text}")
        logger.info(f"Metadata: {response.metadata}")
        return True
    else:
        logger.error(f"LLM response failed: {response.error}")
        return False

def test_template_prompt():
    """Test a template-based prompt."""
    llm = LLMInterface()
    
    # Sample document content for testing
    document_content = """
    # CollabGPT Project Overview
    
    CollabGPT is an AI agent designed for real-time team collaboration in document editing environments. 
    It aims to become an AI teammate that joins collaborative documents, summarizes changes, 
    suggests edits, and helps resolve conflicts.
    
    ## Features
    - Real-time document monitoring
    - Change summarization
    - Edit suggestions
    - Conflict resolution
    """
    
    logger.info("Testing document summarization template")
    response = llm.generate_with_template(
        "summarize_document",
        document_content=document_content,
        max_tokens=200
    )
    
    if response.success:
        logger.info("Template response received successfully!")
        logger.info(f"Summary: {response.text}")
        return True
    else:
        logger.error(f"Template response failed: {response.error}")
        return False

if __name__ == "__main__":
    # Run the tests
    basic_test_success = test_basic_prompt()
    template_test_success = test_template_prompt()
    
    # Report results
    if basic_test_success and template_test_success:
        logger.info("✅ All LLM tests passed successfully!")
        logger.info("The LLM integration with OpenRouter is working correctly.")
        sys.exit(0)
    else:
        logger.error("❌ Some LLM tests failed.")
        tests_failed = []
        if not basic_test_success:
            tests_failed.append("Basic prompt test")
        if not template_test_success:
            tests_failed.append("Template prompt test")
        logger.error(f"Failed tests: {', '.join(tests_failed)}")
        sys.exit(1)