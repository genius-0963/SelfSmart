#!/usr/bin/env python3
"""Simple test script to verify basic setup"""

import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

try:
    import schedule
    print("✅ schedule module imported successfully")
except ImportError as e:
    print(f"❌ schedule module not found: {e}")

try:
    from pydantic_settings import BaseSettings
    print("✅ pydantic_settings module imported successfully")
except ImportError as e:
    print(f"❌ pydantic_settings module not found: {e}")

try:
    import aiohttp
    print("✅ aiohttp module imported successfully")
except ImportError as e:
    print(f"❌ aiohttp module not found: {e}")

print("\n✅ Basic imports working!")
