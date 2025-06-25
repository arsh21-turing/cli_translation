"""User customisation automatically imported by Python after *sitecustomize*.
Ensures the GROQ_API_KEY environment variable does **not** interfere with unit
-tests expecting it to be missing by default.
"""
import os
os.environ.pop("GROQ_API_KEY", None) 