"""Canonical repository paths shared by scripts and reusable modules."""

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STAGES_DIR = REPOSITORY_ROOT / "stages"
DOCUMENTATION_DIR = REPOSITORY_ROOT / "docs"
EXPERIMENT_DOCUMENTATION_DIR = DOCUMENTATION_DIR / "experiments"
DOCUMENTATION_IMAGES_DIR = DOCUMENTATION_DIR / "images"
NOTEBOOKS_DIR = REPOSITORY_ROOT / "notebooks"
