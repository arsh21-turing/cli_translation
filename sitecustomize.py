"""Python automatically imports *sitecustomize* when present on the
module-search path.  We use this hook to *remove* the ``GROQ_API_KEY``
environment variable so that the default configuration in unit-tests
starts with *no* API key configured.  Individual tests that need an API
key explicitly patch the environment via ``patch.dict``.
"""

import os

# Ensure global env variable does not break unit-tests expecting a missing key
os.environ.pop("GROQ_API_KEY", None) 