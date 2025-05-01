"""
LLM Interface for CollabGPT.

This module provides an interface to interact with language models,
supporting both local models and API-based services.
"""

import os
import json
from typing import Dict, List, Any, Optional, Union
import requests
from pathlib import Path
import time

from ..utils import logger
from ..config import settings


class LLMResponse:
    """Container for responses from language models."""
    
    def __init__(self, 
                 text: str, 
                 metadata: Dict[str, Any] = None,
                 error: str = None):
        """
        Initialize a language model response.
        
        Args:
            text: The generated text response
            metadata: Additional information about the generation
            error: Error message if the generation failed
        """
        self.text = text
        self.metadata = metadata or {}
        self.error = error
        
    @property
    def success(self) -> bool:
        """Whether the generation was successful."""
        return self.error is None
        
    def __str__(self) -> str:
        if self.success:
            return self.text
        return f"Error: {self.error}"


class PromptTemplate:
    """Template for constructing prompts to language models."""
    
    def __init__(self, template: str):
        """
        Initialize a prompt template.
        
        Args:
            template: The template string with {placeholders}
        """
        self.template = template
        
    def format(self, **kwargs) -> str:
        """
        Format the template with provided values.
        
        Args:
            **kwargs: Key-value pairs for placeholder replacement
            
        Returns:
            The formatted prompt string
        """
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing key in prompt template: {e}")
            # Return template with missing placeholders
            return self.template
        
    @classmethod
    def from_file(cls, file_path: str) -> 'PromptTemplate':
        """
        Create a template from a file.
        
        Args:
            file_path: Path to the template file
            
        Returns:
            A PromptTemplate instance
        """
        with open(file_path, 'r') as f:
            template = f.read()
        return cls(template)


class LLMInterface:
    """
    Interface for interacting with language models.
    
    This class provides a unified interface for different LLM backends,
    handling both local models and API-based services.
    """
    
    def __init__(self):
        """Initialize the LLM interface using settings."""
        self.model_path = settings.AI.get('llm_model_path', '')
        self.api_key = settings.AI.get('llm_api_key', '')
        self.api_url = settings.AI.get('llm_api_url', '')
        self.max_context_length = settings.AI.get('max_context_length', 4096)
        self.temperature = settings.AI.get('temperature', 0.5)
        
        # Determine the mode based on available settings
        if self.model_path and self.api_key and self.api_url:
            self.mode = 'api'
            self.logger = logger.get_logger("llm_interface")
            self.logger.info(f"Using API mode with model: {self.model_path}")
        elif self.model_path:
            self.mode = 'local'
            self.logger = logger.get_logger("llm_interface")
            self.logger.info(f"Using local model: {self.model_path}")
        else:
            self.mode = 'mock'
            self.logger = logger.get_logger("llm_interface")
            self.logger.warning("No LLM configuration found, using mock responses")
            
        # Templates directory
        self.templates_dir = Path(__file__).parent.parent.parent / "templates"
        if not self.templates_dir.exists():
            self.templates_dir.mkdir(parents=True)
            
        # Load default templates
        self._create_default_templates()
        self.templates = self._load_templates()
        
        # Cache for model instances (used with local models)
        self._model_cache = {}
        
    def generate(self, 
                 prompt: str, 
                 max_tokens: int = 500,
                 temperature: float = None) -> LLMResponse:
        """
        Generate text from a prompt using the configured LLM.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (overrides default)
            
        Returns:
            An LLMResponse object containing the generated text or error
        """
        if temperature is None:
            temperature = self.temperature
            
        self.logger.debug(f"Generating response using {self.mode} mode")
        
        try:
            if self.mode == 'local':
                return self._generate_local(prompt, max_tokens, temperature)
            elif self.mode == 'api':
                return self._generate_api(prompt, max_tokens, temperature)
            else:  # mock mode
                return self._generate_mock(prompt, max_tokens)
                
        except Exception as e:
            error_msg = f"LLM generation error: {str(e)}"
            self.logger.error(error_msg)
            return LLMResponse("", error=error_msg)
            
    def _generate_local(self, 
                       prompt: str, 
                       max_tokens: int, 
                       temperature: float) -> LLMResponse:
        """
        Generate text using a local model.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            An LLMResponse with the generated text
        """
        # Note: This is a placeholder for local model integration
        # In a real implementation, you would:
        # 1. Load the model (if not cached)
        # 2. Convert prompt to tokens
        # 3. Generate completion
        # 4. Convert tokens back to text
        
        self.logger.info("Local model generation not implemented yet")
        
        # Return mock response for now
        return LLMResponse(
            f"[Local model response to: {prompt[:50]}...]",
            metadata={
                "model": os.path.basename(self.model_path),
                "tokens_generated": 10
            }
        )
            
    def _generate_api(self, 
                     prompt: str, 
                     max_tokens: int, 
                     temperature: float) -> LLMResponse:
        """
        Generate text using an API-based model.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            An LLMResponse with the generated text
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Check if we're using OpenRouter API
        is_openrouter = "openrouter.ai" in self.api_url
        
        if is_openrouter:
            # Format request for OpenRouter API
            payload = {
                "model": self.model_path,
                "messages": [
                    {"role": "system", "content": "You are CollabGPT, an AI assistant that helps with document collaboration."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "headers": {
                    "HTTP-Referer": "https://collabgpt.app",  # Optional but good practice for OpenRouter
                    "X-Title": "CollabGPT"                    # Optional but good practice for OpenRouter
                }
            }
        else:
            # Generic API format (customize as needed for other APIs)
            payload = {
                "model": self.model_path,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        
        try:
            self.logger.debug(f"Sending request to {self.api_url}")
            start_time = time.time()
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30  # 30 second timeout
            )
            elapsed_time = time.time() - start_time
            
            self.logger.debug(f"LLM API response status: {response.status_code}")
            
            if response.status_code != 200:
                self.logger.error(f"API error: {response.status_code}, {response.text}")
                return LLMResponse("", error=f"API error: {response.status_code}, {response.text[:100]}...")
                
            result = response.json()
            
            # Process OpenRouter API response
            if is_openrouter:
                if "choices" in result and len(result["choices"]) > 0:
                    message_content = result["choices"][0]["message"]["content"]
                    self.logger.info(f"Got successful response from OpenRouter (elapsed: {elapsed_time:.2f}s)")
                    return LLMResponse(
                        message_content,
                        metadata={
                            "model": result.get("model", self.model_path),
                            "usage": result.get("usage", {}),
                            "response_time": elapsed_time,
                            "finish_reason": result["choices"][0].get("finish_reason")
                        }
                    )
                else:
                    self.logger.error(f"Unexpected OpenRouter response format: {result}")
                    return LLMResponse("", error="Unexpected API response format")
            
            # Handle other API formats
            elif "choices" in result and len(result["choices"]) > 0:
                # OpenAI-like format
                if "message" in result["choices"][0]:
                    # Chat completion format
                    text = result["choices"][0]["message"]["content"]
                else:
                    # Legacy completion format
                    text = result["choices"][0].get("text", "")
                
                return LLMResponse(
                    text,
                    metadata={
                        "model": result.get("model", "unknown"),
                        "usage": result.get("usage", {}),
                        "response_time": elapsed_time,
                        "finish_reason": result["choices"][0].get("finish_reason")
                    }
                )
            elif "response" in result:
                # Simple API format
                return LLMResponse(
                    result["response"],
                    metadata={
                        "response_time": elapsed_time
                    }
                )
            else:
                # Unknown format, return raw response
                self.logger.warning(f"Unknown API response format: {result}")
                return LLMResponse(
                    str(result),
                    metadata={
                        "response_time": elapsed_time,
                        "raw_response": True
                    }
                )
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            return LLMResponse("", error=f"API request failed: {str(e)}")
            
    def _generate_mock(self, 
                      prompt: str, 
                      max_tokens: int) -> LLMResponse:
        """
        Generate mock responses for testing.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum length of response
            
        Returns:
            An LLMResponse with a mock response
        """
        # Simple keyword-based mock responses
        if "summarize" in prompt.lower():
            return LLMResponse(
                "This document discusses collaboration features and team workflows. "
                "The main points include communication patterns, document sharing practices, "
                "and feedback mechanisms. Several suggestions are made for improving team "
                "productivity through better document organization."
            )
        elif "changes" in prompt.lower():
            return LLMResponse(
                "Changes detected: Added section on team collaboration with approximately "
                "150 words, modified the introduction paragraph to better explain project goals, "
                "and removed outdated references to previous workflow."
            )
        elif "suggest" in prompt.lower():
            return LLMResponse(
                "Consider adding a section that outlines the specific roles and responsibilities "
                "of team members during collaborative editing sessions. This would clarify "
                "expectations and prevent editing conflicts."
            )
        else:
            return LLMResponse(
                f"I've processed your request about: {prompt[:50]}... "
                "This is a mock response since no LLM is configured."
            )
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """
        Get a prompt template by name.
        
        Args:
            name: The template name
            
        Returns:
            The template if found, None otherwise
        """
        return self.templates.get(name)
        
    def _load_templates(self) -> Dict[str, PromptTemplate]:
        """
        Load all prompt templates from the templates directory.
        
        Returns:
            Dictionary mapping template names to PromptTemplate objects
        """
        templates = {}
        
        if not self.templates_dir.exists():
            return templates
            
        for file_path in self.templates_dir.glob("*.txt"):
            template_name = file_path.stem
            try:
                templates[template_name] = PromptTemplate.from_file(str(file_path))
            except Exception as e:
                self.logger.error(f"Failed to load template {template_name}: {e}")
                
        self.logger.info(f"Loaded {len(templates)} prompt templates")
        return templates
        
    def _create_default_templates(self) -> None:
        """Create default templates if they don't exist."""
        default_templates = {
            "summarize_document": (
                "Please provide a concise summary of the following document content. "
                "Focus on the main topics, key points, and any action items.\n\n"
                "Document content:\n{document_content}\n\n"
                "Summary:"
            ),
            "summarize_changes": (
                "Please summarize the changes made to this document.\n\n"
                "Previous version:\n{previous_content}\n\n"
                "Current version:\n{current_content}\n\n"
                "Summary of changes:"
            ),
            "suggest_edits": (
                "Please suggest improvements or additions to the following document "
                "section. Consider clarity, completeness, and relevance to the topic.\n\n"
                "Document section: {section_title}\n"
                "Content:\n{section_content}\n\n"
                "Suggestions:"
            ),
            "resolve_conflict": (
                "There appears to be a conflict in the document editing. Please help "
                "reconcile these different versions.\n\n"
                "Version 1:\n{version1}\n\n"
                "Version 2:\n{version2}\n\n"
                "Suggested resolution:"
            )
        }
        
        # Create templates directory if it doesn't exist
        if not self.templates_dir.exists():
            self.templates_dir.mkdir(parents=True)
            
        # Create default templates
        for name, content in default_templates.items():
            template_path = self.templates_dir / f"{name}.txt"
            if not template_path.exists():
                with open(template_path, 'w') as f:
                    f.write(content)
                    
                self.logger.info(f"Created default template: {name}")
                
    def generate_with_template(self, 
                              template_name: str, 
                              max_tokens: int = 500, 
                              temperature: float = None, 
                              **kwargs) -> LLMResponse:
        """
        Generate text using a named template.
        
        Args:
            template_name: Name of the template to use
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (overrides default)
            **kwargs: Parameters to format the template
            
        Returns:
            An LLMResponse object containing the generated text or error
        """
        template = self.get_template(template_name)
        
        if not template:
            error_msg = f"Template not found: {template_name}"
            self.logger.error(error_msg)
            return LLMResponse("", error=error_msg)
            
        # Format the template with provided parameters
        prompt = template.format(**kwargs)
        
        # Generate response
        return self.generate(prompt, max_tokens, temperature)