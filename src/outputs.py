"""Shared validation and model-specific output directories."""

from pathlib import Path
import re


def validate_model_name(name: str) -> str:
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
        raise ValueError("Use 1–64 letters, digits, underscores or hyphens; start with a letter or digit.")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                *(f"LPT{i}" for i in range(1, 10))}
    if name.upper() in reserved:
        raise ValueError("That model name is reserved by the operating system.")
    return name


def model_output_directory(output_dir: Path, model_name: str) -> Path:
    directory = Path(output_dir) / validate_model_name(model_name)
    if directory.is_symlink():
        raise ValueError(f"Model output directory must not be a symbolic link: {directory}")
    return directory
