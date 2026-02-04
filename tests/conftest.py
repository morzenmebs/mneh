import sys
from pathlib import Path

# Allow tests to import the src/ layout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
