"""
tests/conftest.py — Shared pytest configuration for the IG pipeline tests.
Adds the repo root to sys.path so imports work regardless of how pytest is invoked.
"""
import os
import sys
# FIX: Force JAX to use CPU backend to prevent XLA compiler crashes
os.environ["JAX_PLATFORMS"] = "cpu"

# Add repo root to path so 'import ig_pipeline' etc. work
repo_root = os.path.join(os.path.dirname(__file__), '..')
# Add repo root to path so 'import ig_pipeline' etc. work
#repo_root = os.path.join(os.path.dirname(__file__), '..')
if repo_root not in sys.path:
    sys.path.insert(0, os.path.abspath(repo_root))
