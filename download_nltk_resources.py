#!/usr/bin/env python3
"""
Download required NLTK resources for CollabGPT.

This script ensures all necessary NLTK resources are available.
"""

import nltk
import ssl
import sys

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def download_resources():
    """Download all required NLTK resources."""
    print("Downloading required NLTK resources...")
    
    resources = [
        'punkt',  # For sentence tokenization
        'stopwords',  # For stopword filtering
        'averaged_perceptron_tagger',  # For POS tagging
        'wordnet',  # For word definitions and synonyms
    ]
    
    for resource in resources:
        try:
            print(f"Downloading {resource}...")
            nltk.download(resource)
            print(f"✓ Successfully downloaded {resource}")
        except Exception as e:
            print(f"✕ Failed to download {resource}: {e}")
            return False
    
    print("\nAll required NLTK resources have been downloaded successfully!")
    return True

if __name__ == "__main__":
    if download_resources():
        sys.exit(0)
    else:
        sys.exit(1)