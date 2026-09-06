"""Allow project-local optional dependencies installed by setup.ps1."""
import sys
from pathlib import Path
deps = Path(__file__).parent / '.deps'
if deps.exists():
    sys.path.append(str(deps))
