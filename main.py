#!/usr/bin/env python3
"""
Entry point script for CollabGPT.
"""

import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.app import main

if __name__ == "__main__":
    sys.exit(main())