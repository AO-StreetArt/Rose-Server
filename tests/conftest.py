import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_path = str(PROJECT_ROOT)
if project_path not in sys.path:
    sys.path.insert(0, project_path)
