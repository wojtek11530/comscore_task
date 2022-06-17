"""The file with paths to folders used in the project."""

from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.resolve()
STORAGE_DIR = PROJECT_DIR / "storage"
MODELS_DIR = STORAGE_DIR / "models"
DATASETS_DIR = STORAGE_DIR / "data"
CHECKPOINTS_DIR = STORAGE_DIR / "checkpoints"

class_mapping = {
    'FB': 0,
    'TW': 1
}
reverse_class_mapping = {v: k for k, v in class_mapping.items()}
