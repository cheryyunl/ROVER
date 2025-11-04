#!/usr/bin/env python3
"""
Configuration file for ROVER evaluation
"""

import os

# OpenAI Configuration
# Users should set these environment variables or modify this file
OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY", 
    ""
)

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-4o"
)

# Legacy Azure support (deprecated)
AZURE_API_KEY = os.getenv(
    "AZURE_OPENAI_API_KEY", 
    ""
)

AZURE_ENDPOINT = os.getenv(
    "AZURE_OPENAI_ENDPOINT",
    "https://dil-nu-openai-eastus.openai.azure.com"
)

AZURE_DEPLOYMENT_NAME = os.getenv(
    "AZURE_OPENAI_DEPLOYMENT",
    "gpt-4o"
)

AZURE_API_VERSION = os.getenv(
    "AZURE_OPENAI_API_VERSION", 
    "2024-08-01-preview"
)

# ROVER data paths
ROVER_GEN_DIR = os.getenv(
    "ROVER_GEN_DIR",
    "/Users/cheryunl/Documents/eval/gen_banana"
)

# Evaluation settings
MAX_RETRIES = int(os.getenv(
    "MAX_RETRIES",
    "3"
))

# Instructions for users
def print_config_help():
    """Print help for configuring OpenAI"""
    print(f"""
=== OpenAI Configuration ===

Option 1 - Environment Variables (Recommended):
export OPENAI_API_KEY="your-api-key"
export OPENAI_MODEL="gpt-4o"

Option 2 - Edit config.py:
Modify the default values in config.py

Current Configuration:
- API Key: {'Set' if OPENAI_API_KEY else 'Not Set'}
- Model: {OPENAI_MODEL}
- ROVER-GEN Directory: {ROVER_GEN_DIR}
- Max Retries: {MAX_RETRIES}

Legacy Azure Support (Deprecated):
- Azure API Key: {'Set' if AZURE_API_KEY else 'Not Set'}
- Azure Endpoint: {AZURE_ENDPOINT}
- Azure Deployment: {AZURE_DEPLOYMENT_NAME}
    """)

if __name__ == "__main__":
    print_config_help()
