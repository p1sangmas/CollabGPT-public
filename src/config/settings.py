"""
Configuration settings for CollabGPT.

This module contains all configuration parameters for the application,
including API credentials, feature flags, and operational settings.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'data'
CREDENTIALS_DIR = BASE_DIR / 'credentials'

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
CREDENTIALS_DIR.mkdir(exist_ok=True)

# Google API settings
GOOGLE_API = {
    'credentials_path': os.environ.get('GOOGLE_CREDENTIALS_PATH') or str(CREDENTIALS_DIR / 'google_credentials.json'),
    'use_service_account': os.environ.get('GOOGLE_USE_SERVICE_ACCOUNT', 'false').lower() == 'true',
    'token_path': str(DATA_DIR / 'token.json'),
    'api_scopes': [
        'https://www.googleapis.com/auth/documents',
        'https://www.googleapis.com/auth/drive',
        'https://www.googleapis.com/auth/drive.file',
    ],
}

# Webhook settings
WEBHOOK = {
    'host': os.environ.get('WEBHOOK_HOST', 'localhost'),
    'port': int(os.environ.get('WEBHOOK_PORT', 8000)),
    'path': os.environ.get('WEBHOOK_PATH', '/webhook'),
    'secret_key': os.environ.get('WEBHOOK_SECRET_KEY', ''),
    'external_url': os.environ.get('WEBHOOK_EXTERNAL_URL', ''),  # For public webhooks
}

# Document processing settings
DOCUMENT = {
    'max_summary_sentences': int(os.environ.get('MAX_SUMMARY_SENTENCES', 3)),
    'significant_change_threshold': int(os.environ.get('SIGNIFICANT_CHANGE_THRESHOLD', 5)),  # words
    'default_language': os.environ.get('DEFAULT_LANGUAGE', 'english'),
    'history_retention_days': int(os.environ.get('HISTORY_RETENTION_DAYS', 30)),
    'cache_enabled': os.environ.get('CACHE_ENABLED', 'true').lower() == 'true',
}

# AI processing settings
AI = {
    'rag_enabled': os.environ.get('RAG_ENABLED', 'true').lower() == 'true',
    'llm_model_path': os.environ.get('LLM_MODEL_PATH', ''),  # Local model path
    'llm_api_key': os.environ.get('LLM_API_KEY', ''),  # For cloud-based LLMs
    'llm_api_url': os.environ.get('LLM_API_URL', ''),
    'max_context_length': int(os.environ.get('MAX_CONTEXT_LENGTH', 4096)),
    'temperature': float(os.environ.get('LLM_TEMPERATURE', 0.5)),
}

# Logging settings
LOGGING = {
    'level': os.environ.get('LOG_LEVEL', 'INFO'),
    'file_path': os.environ.get('LOG_FILE') or str(DATA_DIR / 'collabgpt.log'),
    'max_file_size_mb': int(os.environ.get('LOG_MAX_FILE_SIZE_MB', 10)),
    'backup_count': int(os.environ.get('LOG_BACKUP_COUNT', 5)),
}

# Feature flags
FEATURES = {
    'real_time_monitoring': os.environ.get('FEATURE_REAL_TIME_MONITORING', 'true').lower() == 'true',
    'comment_analysis': os.environ.get('FEATURE_COMMENT_ANALYSIS', 'true').lower() == 'true',
    'edit_suggestions': os.environ.get('FEATURE_EDIT_SUGGESTIONS', 'false').lower() == 'true',
    'conflict_detection': os.environ.get('FEATURE_CONFLICT_DETECTION', 'true').lower() == 'true',
}

# Monitored documents
def get_monitored_documents() -> List[Dict[str, Any]]:
    """
    Get the list of documents that should be monitored.
    
    Returns:
        List of document configurations
    """
    docs_file = DATA_DIR / 'monitored_documents.json'
    if docs_file.exists():
        with open(docs_file, 'r') as f:
            return json.load(f)
    return []

def save_monitored_document(document_id: str, name: str, webhook_enabled: bool = True) -> None:
    """
    Add a document to the monitoring list.
    
    Args:
        document_id: The Google Doc ID
        name: A friendly name for the document
        webhook_enabled: Whether to set up a webhook for this document
    """
    docs = get_monitored_documents()
    
    # Check if document is already in the list
    for doc in docs:
        if doc.get('id') == document_id:
            doc.update({
                'name': name,
                'webhook_enabled': webhook_enabled,
                'last_updated': str(datetime.now())
            })
            break
    else:
        # Document not found, add it
        from datetime import datetime
        docs.append({
            'id': document_id,
            'name': name,
            'webhook_enabled': webhook_enabled,
            'added': str(datetime.now()),
            'last_updated': str(datetime.now())
        })
    
    # Save the updated list
    docs_file = DATA_DIR / 'monitored_documents.json'
    with open(docs_file, 'w') as f:
        json.dump(docs, f, indent=2)

def remove_monitored_document(document_id: str) -> bool:
    """
    Remove a document from the monitoring list.
    
    Args:
        document_id: The Google Doc ID
        
    Returns:
        True if document was removed, False if not found
    """
    docs = get_monitored_documents()
    initial_count = len(docs)
    
    docs = [doc for doc in docs if doc.get('id') != document_id]
    
    if len(docs) < initial_count:
        docs_file = DATA_DIR / 'monitored_documents.json'
        with open(docs_file, 'w') as f:
            json.dump(docs, f, indent=2)
        return True
    
    return False

# Load environment-specific settings if available
env = os.environ.get('COLLABGPT_ENV', 'development')
env_settings_file = BASE_DIR / f'config/{env}_settings.py'

if env_settings_file.exists():
    print(f"Loading environment-specific settings for: {env}")
    # This is a simple approach - in a production app you might use importlib
    with open(env_settings_file, 'r') as f:
        exec(f.read())