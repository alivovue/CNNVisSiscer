"""Validation-based training and complete, reproducible model bundles."""

from dataclasses import dataclass
import json
from pathlib import Path
import re
import warnings

import joblib
import numpy as np

from src.preprocessing.preprocess import FEATURES, PROJECT_ROOT, PreparedData, positive_integer
from src.train.model import build_model


MODEL_DIR = PROJECT_ROOT / "output" / "model"
SEED = 42


@dataclass
class LoadedModel:
    model: object
    metadata: dict
    feature_scaler: object
    target_scaler: object
    history: dict | None


def validate_model_name(name: str) -> str:
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
        raise ValueError("Use 1–64 letters, digits, underscores or hyphens; start with a letter or digit.")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                *(f"LPT{i}" for i in range(1, 10))}
    if name.upper() in reserved:
        raise ValueError("That model name is reserved by the operating system.")
    return name


def artifact_paths(name: str, model_dir: Path = MODEL_DIR) -> dict[str, Path]:
    validate_model_name(name)
    directory = Path(model_dir)
    return {"model": directory / f"{name}.keras",
            "metadata": directory / f"{name}_metadata.json",
            "feature_scaler": directory / f"{name}_feature_scaler.joblib",
            "target_scaler": directory / f"{name}_target_scaler.joblib",
            "history": directory / f"{name}_history.json"}


def ensure_name_available(name: str, model_dir: Path = MODEL_DIR) -> None:
    paths = artifact_paths(name, model_dir)
    if any(path.exists() or path.is_symlink() for path in paths.values()):
        raise FileExistsError(f"Model '{name}' already has saved artifacts. Choose a different name.")


def train_model(model, data: PreparedData, epochs: int = 50, batch_size: int = 32,
                verbose: int = 1) -> tuple[dict, dict]:
    positive_integer(epochs, "Epochs")
    positive_integer(batch_size, "Batch size")
    import tensorflow as tf

    stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                               restore_best_weights=True)
    history = model.fit(
        data.train.X, data.train.y,
        validation_data=(data.validation.X, data.validation.y),
        epochs=epochs, batch_size=batch_size, shuffle=False, verbose=verbose,
        callbacks=[stopping, tf.keras.callbacks.TerminateOnNaN()],
    ).history
    history = {key: [float(value) for value in values] for key, values in history.items()}
    if not history.get("val_loss") or not history.get("loss"):
        raise ValueError("Training did not produce training and validation losses.")
    if not all(np.isfinite(values).all() for values in history.values()):
        raise ValueError("Training produced nonfinite losses or metrics; the model was not saved.")
    best = int(np.argmin(history["val_loss"]))
    summary = {"max_epochs": epochs, "completed_epochs": len(history["loss"]),
               "best_validation_epoch": best + 1, "best_validation_loss": history["val_loss"][best],
               "batch_size": batch_size, "early_stopping_patience": 10,
               "early_stopping_triggered": bool(stopping.stopped_epoch > 0)}
    return history, summary


def save_bundle(model, data: PreparedData, name: str, kernel_size: int, filters: int,
                history: dict, summary: dict, model_dir: Path = MODEL_DIR, seed: int = SEED) -> dict:
    """Reserve every filename exclusively and roll back partial writes on failure."""
    ensure_name_available(name, model_dir)
    paths = artifact_paths(name, model_dir)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    reserved = []
    metadata = {**data.metadata, **summary, "schema_version": 1, "model_name": name,
                "kernel_size": kernel_size, "filters": filters, "random_seed": seed,
                "scaler_paths": {"feature": paths["feature_scaler"].name,
                                 "target": paths["target_scaler"].name},
                "history_path": paths["history"].name}
    try:
        for path in paths.values():
            with path.open("xb"):
                pass
            reserved.append(path)
        model.save(paths["model"])
        joblib.dump(data.feature_scaler, paths["feature_scaler"])
        joblib.dump(data.target_scaler, paths["target_scaler"])
        paths["history"].write_text(json.dumps(history, indent=2, allow_nan=False) + "\n")
        # Metadata is the final completion marker for a usable bundle.
        paths["metadata"].write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    except BaseException:
        for path in reserved:
            path.unlink(missing_ok=True)
        raise
    return metadata


def train_and_save(data: PreparedData, name: str, kernel_size: int = 3, filters: int = 32,
                   epochs: int = 50, batch_size: int = 32, model_dir: Path = MODEL_DIR,
                   seed: int = SEED, verbose: int = 1) -> LoadedModel:
    ensure_name_available(name, model_dir)
    model = build_model(data.metadata["window_size"], kernel_size, filters,
                        len(data.metadata["features"]), seed)
    history, summary = train_model(model, data, epochs, batch_size, verbose)
    metadata = save_bundle(model, data, name, kernel_size, filters, history, summary, model_dir, seed)
    return LoadedModel(model, metadata, data.feature_scaler, data.target_scaler, history)


def list_models(model_dir: Path = MODEL_DIR) -> list[str]:
    return sorted(path.stem for path in Path(model_dir).glob("*.keras") if path.is_file())


def load_bundle(name: str, model_dir: Path = MODEL_DIR) -> LoadedModel:
    paths = artifact_paths(name, model_dir)
    for kind in ("model", "metadata", "feature_scaler", "target_scaler"):
        if not paths[kind].is_file():
            raise FileNotFoundError(f"Missing {kind.replace('_', ' ')} file: {paths[kind]}")
    try:
        metadata = json.loads(paths["metadata"].read_text())
    except (ValueError, OSError) as exc:
        raise ValueError(f"Cannot read model metadata: {exc}") from exc
    required = {"schema_version", "model_name", "dataset_path", "forecasting_mode", "features", "target",
                "window_size", "kernel_size", "filters", "batch_size", "max_epochs", "completed_epochs",
                "best_validation_epoch", "best_validation_loss", "random_seed", "boundaries", "scaling",
                "scaler_paths", "history_path", "dataset_quality"}
    if not isinstance(metadata, dict) or required.difference(metadata):
        raise ValueError("Model metadata is incomplete or malformed.")
    if metadata["schema_version"] != 1 or metadata["model_name"] != name:
        raise ValueError("Unsupported metadata version or mismatched model name.")
    mode = metadata["forecasting_mode"]
    if mode not in FEATURES or metadata["features"] != FEATURES[mode] or metadata["target"] != "cnt":
        raise ValueError("Saved feature configuration is invalid.")
    for key in ("window_size", "kernel_size", "filters", "batch_size", "max_epochs", "completed_epochs",
                "best_validation_epoch"):
        positive_integer(metadata[key], key)
    if (metadata["kernel_size"] > metadata["window_size"]
            or metadata["window_size"] >= 13_000
            or not 1 <= metadata["best_validation_epoch"] <= metadata["completed_epochs"] <= metadata["max_epochs"]):
        raise ValueError("Saved architecture or training configuration is inconsistent.")
    if metadata["scaler_paths"] != {"feature": paths["feature_scaler"].name,
                                    "target": paths["target_scaler"].name}:
        raise ValueError("Scaler paths do not match this model bundle.")
    if metadata["history_path"] != paths["history"].name:
        raise ValueError("History path does not match this model bundle.")
    feature_scaler = joblib.load(paths["feature_scaler"])
    target_scaler = joblib.load(paths["target_scaler"])
    import tensorflow as tf

    model = tf.keras.models.load_model(paths["model"], compile=False)
    if tuple(model.input_shape[1:]) != (metadata["window_size"], len(metadata["features"])) or model.output_shape[-1] != 1:
        raise ValueError("Saved model input/output dimensions disagree with its metadata.")
    convolution = model.get_layer("convolution")
    if convolution.kernel_size != (metadata["kernel_size"],) or convolution.filters != metadata["filters"]:
        raise ValueError("Saved convolution architecture disagrees with its metadata.")
    history = None
    if paths["history"].is_file():
        try:
            history = json.loads(paths["history"].read_text())
            if (not isinstance(history, dict) or not history.get("loss") or not history.get("val_loss")
                    or len(history["loss"]) != len(history["val_loss"])
                    or len(history["loss"]) != metadata["completed_epochs"]
                    or not all(np.isfinite(history[key]).all() for key in ("loss", "val_loss"))):
                raise ValueError("Invalid loss sequences")
        except (ValueError, TypeError, OSError) as exc:
            warnings.warn(f"Training history is unavailable; skipping loss graph: {exc}", stacklevel=2)
            history = None
    return LoadedModel(model, metadata, feature_scaler, target_scaler, history)
