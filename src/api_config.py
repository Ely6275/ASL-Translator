"""
api_config.py

Anthropic API key configuration.

HOW TO SET YOUR API KEY:
  Option 1 (recommended): Set an environment variable so the key
  never appears in your code or git history:

      Windows PowerShell:
          $env:ANTHROPIC_API_KEY = "sk-ant-..."

      Mac/Linux:
          export ANTHROPIC_API_KEY="sk-ant-..."

  Option 2: Paste your key directly into the ANTHROPIC_API_KEY
  variable below. Do NOT commit this file to a public repo if you
  do this.

Get your API key at: https://console.anthropic.com/
"""

import os

# Reads from environment variable first, falls back to the hardcoded value.
# Replace the empty string with your key only if you are NOT using env vars.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


def get_headers():
    """Returns the required headers for the Anthropic API."""
    if not ANTHROPIC_API_KEY:
        raise ValueError(
            "No Anthropic API key found.\n\n"
            "Set the ANTHROPIC_API_KEY environment variable or paste "
            "your key into src/api_config.py.\n\n"
            "Get your key at: https://console.anthropic.com/"
        )
    return {
        "Content-Type":    "application/json",
        "x-api-key":       ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
    }
