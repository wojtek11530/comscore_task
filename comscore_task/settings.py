"""The file with paths to folders used in the project."""

from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.resolve()
STORAGE_DIR = PROJECT_DIR / "storage"
CHECKPOINTS_DIR = STORAGE_DIR / "checkpoints"
