"""Root conftest for the test suite."""
import sys
from pathlib import Path

# Ensure the backend app is importable
sys.path.insert(0, str(Path(__file__).parent))
