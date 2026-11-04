"""Chronological, gap-aware preparation of one-hour-ahead forecasting data."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Callable
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.outputs import model_output_directory


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "hour.csv"
PRE_RESULT_DIR = PROJECT_ROOT / "output" / "pre_result"
TRAIN_END = 13_000
VALIDATION_END = 15_000
FEATURES = {"univariate": ["cnt"], "multivariate": ["cnt", "temp", "hum", "windspeed"]}
TARGET = "cnt"


@dataclass
class WindowSplit:
    X: np.ndarray
    y: np.ndarray
    timestamps: pd.DatetimeIndex
    target_indices: np.ndarray


@dataclass
class PreparedData:
    frame: pd.DataFrame
    train: WindowSplit
    validation: WindowSplit
    test: WindowSplit
    feature_scaler: MinMaxScaler
    target_scaler: MinMaxScaler
    metadata: dict


def positive_integer(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{label} must be a positive integer.")
    return int(value)


def load_dataset(path: Path = DATASET_PATH) -> tuple[pd.DataFrame, dict]:
    """Validate without dropping observations, filling counts, or altering the CSV."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}. Place the UCI hourly CSV at data/hour.csv.")
    # Hash the exact bytes that are parsed, so saved boundaries cannot silently be
    # reused against changed data (including changes outside the selected features).
    from io import BytesIO

    contents = path.read_bytes()
    frame = pd.read_csv(BytesIO(contents))
    required = {"dteday", "hr", *FEATURES["multivariate"]}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}.")
    if frame.empty:
        raise ValueError("Dataset contains no observations.")
    nulls = frame.isna().sum()
    if nulls.any():
        raise ValueError(f"Missing values cannot be safely filled: {nulls[nulls > 0].to_dict()}.")
    # Every UCI column except the date is numeric, including unused attributes.
    for column in frame.columns.difference(["dteday"]):
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Column '{column}' contains nonnumeric attributes.") from exc
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Column '{column}' contains nonfinite attributes.")
    if ((frame.hr < 0) | (frame.hr > 23) | (frame.hr % 1 != 0)).any():
        raise ValueError("Hour values must be integers from 0 to 23.")
    for column in ("cnt", "casual", "registered"):
        if column in frame and (frame[column] < 0).any():
            raise ValueError(f"Negative rental counts in '{column}' are invalid.")
    if (frame.cnt % 1 != 0).any():
        raise ValueError("Rental counts must be whole numbers.")
    for column in ("temp", "hum", "windspeed", "atemp"):
        if column in frame and not frame[column].between(0, 1).all():
            raise ValueError(f"Normalized attribute '{column}' must be between 0 and 1.")
    try:
        dates = pd.to_datetime(frame.dteday, format="%Y-%m-%d", errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("Column 'dteday' must contain valid YYYY-MM-DD dates.") from exc
    frame["timestamp"] = dates + pd.to_timedelta(frame.hr, unit="h")
    if frame.timestamp.duplicated().any():
        duplicates = frame.loc[frame.timestamp.duplicated(), "timestamp"].head(3).astype(str).tolist()
        raise ValueError(f"Duplicate hourly timestamps cannot be resolved safely: {duplicates}.")
    was_sorted = bool(frame.timestamp.is_monotonic_increasing)
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    differences = frame.timestamp.diff().dropna()
    gap_sizes = differences[differences > pd.Timedelta(hours=1)]
    quality = {
        "rows": len(frame),
        "missing_values": 0,
        "duplicate_timestamps": 0,
        "input_was_sorted": was_sorted,
        "hourly_gaps": len(gap_sizes),
        "missing_hours": int((gap_sizes / pd.Timedelta(hours=1) - 1).sum()),
        "first_timestamp": frame.timestamp.iloc[0].isoformat(),
        "last_timestamp": frame.timestamp.iloc[-1].isoformat(),
        "sha256": hashlib.sha256(contents).hexdigest(),
    }
    return frame, quality


def split_boundaries(frame: pd.DataFrame) -> dict:
    if len(frame) < VALIDATION_END + 1:
        raise ValueError(f"At least 15,001 usable rows are required; found {len(frame):,}.")
    boundaries = {}
    for name, start, stop in (("training", 0, TRAIN_END), ("validation", TRAIN_END, VALIDATION_END),
                              ("testing", VALIDATION_END, len(frame))):
        boundaries[name] = {
            "start": start, "stop": stop, "rows": stop - start,
            "first_timestamp": frame.timestamp.iloc[start].isoformat(),
            "last_timestamp": frame.timestamp.iloc[stop - 1].isoformat(),
        }
    return boundaries


def generate_windows(frame: pd.DataFrame, scaled_features: np.ndarray, scaled_target: np.ndarray,
                     window_size: int) -> tuple[WindowSplit, WindowSplit, WindowSplit, dict]:
    """Assign by target row; earlier split context is permitted, future context is not."""
    positive_integer(window_size, "Window size")
    if window_size >= TRAIN_END:
        raise ValueError("Window size must be below 13,000 to leave training targets.")
    if len(scaled_features) != len(frame) or len(scaled_target) != len(frame):
        raise ValueError("Scaled features and targets must match the dataset row count.")
    # A prefix sum tests the window's w input hours AND the step to its target.
    timestamps = pd.DatetimeIndex(frame.timestamp)
    discontinuities = np.r_[0, np.diff(timestamps.asi8) != pd.Timedelta(hours=1).value]
    prefix = np.cumsum(discontinuities)
    indices = np.arange(window_size, len(frame))
    continuous = prefix[indices] == prefix[indices - window_size]
    valid = indices[continuous]
    skipped = {}
    splits = []
    for name, start, stop in (("training", 0, TRAIN_END), ("validation", TRAIN_END, VALIDATION_END),
                              ("testing", VALIDATION_END, len(frame))):
        targets = valid[(valid >= start) & (valid < stop)]
        candidates = indices[(indices >= start) & (indices < stop)]
        skipped[name] = int(len(candidates) - len(targets))
        if not len(targets):
            raise ValueError(f"No continuous {window_size}-hour windows in the {name} period. "
                             "Choose a smaller window or resolve dataset timestamp gaps.")
        X = np.stack([scaled_features[i - window_size:i] for i in targets]).astype(np.float32)
        y = np.asarray(scaled_target).reshape(-1)[targets].astype(np.float32)
        splits.append(WindowSplit(X, y, timestamps[targets], targets))
    return *splits, skipped


def prepare_data(path: Path = DATASET_PATH, mode: str = "univariate", window_size: int = 24,
                 *, feature_scaler: MinMaxScaler | None = None,
                 target_scaler: MinMaxScaler | None = None, saved_metadata: dict | None = None,
                 report: Callable[[str], None] = print) -> PreparedData:
    if mode not in FEATURES:
        raise ValueError("Forecasting mode must be 'univariate' or 'multivariate'.")
    positive_integer(window_size, "Window size")
    report("Loading dataset and validating numeric values and hourly timestamps...")
    frame, quality = load_dataset(path)
    boundaries = split_boundaries(frame)
    features = FEATURES[mode].copy()
    if saved_metadata is not None:
        if (saved_metadata.get("forecasting_mode") != mode or saved_metadata.get("features") != features
                or saved_metadata.get("target") != TARGET or saved_metadata.get("window_size") != window_size):
            raise ValueError("Saved preprocessing settings do not match the requested configuration.")
        if saved_metadata.get("dataset_quality", {}).get("sha256") != quality["sha256"]:
            raise ValueError("Dataset has changed since training (SHA-256 mismatch); original test rows are required.")
        if saved_metadata.get("boundaries") != boundaries:
            raise ValueError("Saved chronological split boundaries do not match the dataset.")
        if saved_metadata.get("scaling") != scaling_configuration():
            raise ValueError("Unsupported saved scaling configuration.")
        if feature_scaler is None or target_scaler is None:
            raise ValueError("Loading requires both saved scalers; fitting replacements is forbidden.")
    if (feature_scaler is None) != (target_scaler is None):
        raise ValueError("Feature and target scalers must be supplied together.")
    values = frame[features].to_numpy(dtype=np.float64)
    targets = frame[[TARGET]].to_numpy(dtype=np.float64)
    if feature_scaler is None:
        report("Fitting feature and target MinMaxScalers on the first 13,000 rows only...")
        feature_scaler = MinMaxScaler().fit(values[:TRAIN_END])
        target_scaler = MinMaxScaler().fit(targets[:TRAIN_END])
    else:
        for scaler, count in ((feature_scaler, len(features)), (target_scaler, 1)):
            if (not isinstance(scaler, MinMaxScaler) or getattr(scaler, "n_features_in_", None) != count
                    or getattr(scaler, "n_samples_seen_", None) != TRAIN_END
                    or scaler.clip or tuple(scaler.feature_range) != (0, 1)):
                raise ValueError("Saved scaler is incompatible with training-only MinMax scaling.")
        report("Reusing saved scalers without fitting...")
    scaled_features = feature_scaler.transform(values)
    scaled_target = target_scaler.transform(targets)
    train, validation, test, skipped = generate_windows(frame, scaled_features, scaled_target, window_size)
    metadata = {
        "dataset_path": str(Path(path).resolve()), "forecasting_mode": mode,
        "features": features, "target": TARGET, "window_size": window_size,
        "prediction_horizon_hours": 1, "boundaries": boundaries, "dataset_quality": quality,
        "scaling": scaling_configuration(), "skipped_gap_windows": skipped,
        "samples": {"training": len(train.y), "validation": len(validation.y), "testing": len(test.y)},
    }
    report(f"Dataset: {len(frame):,} rows; {quality['hourly_gaps']} gaps / {quality['missing_hours']} missing hours.")
    for name in boundaries:
        report(f"{name.capitalize()}: {boundaries[name]['rows']:,} rows, "
               f"{metadata['samples'][name]:,} windows, {skipped[name]:,} windows skipped across gaps.")
    return PreparedData(frame, train, validation, test, feature_scaler, target_scaler, metadata)


def scaling_configuration() -> dict:
    return {"type": "MinMaxScaler", "feature_range": [0, 1], "clip": False,
            "fit_start": 0, "fit_stop": TRAIN_END}


def save_preprocessed_data(data: PreparedData, model_name: str,
                           output_dir: Path = PRE_RESULT_DIR) -> dict[str, Path]:
    """Keep inspected rows, scaled values, and the exact windows supplied to the CNN."""
    directory = model_output_directory(output_dir, model_name)
    directory.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:8]
    prefix = f"{model_name}_{run_id}"
    paths = {"cleaned": directory / f"{prefix}_cleaned.csv",
             "scaled": directory / f"{prefix}_scaled.csv",
             "windows": directory / f"{prefix}_windows.npz",
             "metadata": directory / f"{prefix}_metadata.json"}
    labels = np.empty(len(data.frame), dtype="U10")
    for name, boundary in data.metadata["boundaries"].items():
        labels[boundary["start"]:boundary["stop"]] = name
    cleaned = data.frame.copy()
    cleaned.insert(0, "row_index", np.arange(len(cleaned)))
    cleaned.insert(1, "split", labels)
    features = data.metadata["features"]
    target = data.metadata["target"]
    scaled = pd.DataFrame(data.feature_scaler.transform(data.frame[features].to_numpy(dtype=np.float64))
                          .astype(np.float32), columns=features)
    scaled.insert(0, "timestamp", data.frame.timestamp.to_numpy())
    scaled.insert(0, "split", labels)
    scaled.insert(0, "row_index", np.arange(len(scaled)))
    scaled[f"target_{target}"] = data.target_scaler.transform(
        data.frame[[target]].to_numpy(dtype=np.float64)).reshape(-1).astype(np.float32)
    arrays = {}
    for name, split in (("train", data.train), ("val", data.validation), ("test", data.test)):
        arrays[f"X_{name}"] = split.X
        arrays[f"y_{name}"] = split.y
        arrays[f"timestamps_{name}"] = split.timestamps.to_numpy(dtype="datetime64[ns]")
        arrays[f"target_indices_{name}"] = split.target_indices
    metadata = {**data.metadata, "model_name": model_name, "preprocessing_run": run_id,
                "files": {key: path.name for key, path in paths.items()}}
    written = []
    try:
        for key, frame in (("cleaned", cleaned), ("scaled", scaled)):
            with paths[key].open("x") as handle:
                written.append(paths[key])
                frame.to_csv(handle, index=False, date_format="%Y-%m-%d %H:%M:%S")
        with paths["windows"].open("xb") as handle:
            written.append(paths["windows"])
            np.savez_compressed(handle, **arrays)
        with paths["metadata"].open("x") as handle:
            written.append(paths["metadata"])
            json.dump(metadata, handle, indent=2, allow_nan=False)
            handle.write("\n")
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    return paths
