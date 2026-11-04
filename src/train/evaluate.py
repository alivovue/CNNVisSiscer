"""Final held-out evaluation in original bike-rental count units."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from src.preprocessing.preprocess import PROJECT_ROOT, PreparedData, positive_integer
from src.outputs import model_output_directory, validate_model_name


TEST_OUTPUT_DIR = PROJECT_ROOT / "output" / "test_output"
NUMERICAL_DIR = PROJECT_ROOT / "output" / "numerical"


@dataclass
class EvaluationResult:
    run_id: str
    predictions: pd.DataFrame
    metrics: dict
    predictions_path: Path
    metrics_path: Path
    numerical_paths: dict[str, Path] = field(default_factory=dict)


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if not len(actual) or actual.shape != predicted.shape:
        raise ValueError("Actual and predicted values must have matching, nonempty shapes.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Evaluation values must be finite.")
    error = predicted - actual
    mse = float(np.mean(error ** 2))
    return {"MAE": float(np.mean(np.abs(error))), "MSE": mse, "RMSE": float(np.sqrt(mse))}


def evaluate_model(model, data: PreparedData, metadata: dict, output_dir: Path = TEST_OUTPUT_DIR,
                   batch_size: int | None = None, *, numerical_dir: Path = NUMERICAL_DIR,
                   history: dict | None = None) -> EvaluationResult:
    name = validate_model_name(metadata["model_name"])
    batch_size = positive_integer(metadata["batch_size"] if batch_size is None else batch_size, "Batch size")
    # Direct inference in bounded batches avoids building a separate tf.data
    # threadpool and explicitly forces inference mode. No optimizer or fitting.
    chunks = []
    for start in range(0, len(data.test.y), batch_size):
        output = np.asarray(model(data.test.X[start:start + batch_size], training=False))
        expected = min(batch_size, len(data.test.y) - start)
        if output.shape != (expected, 1):
            raise ValueError(f"Expected predictions with shape ({expected}, 1); got {output.shape}.")
        chunks.append(output)
    if not chunks:
        raise ValueError("No test windows are available for evaluation.")
    scaled = np.concatenate(chunks)
    predicted = data.target_scaler.inverse_transform(scaled.astype(np.float64)).reshape(-1)
    # Original observed counts avoid inverse-scaling roundoff in the ground truth.
    actual = data.frame.iloc[data.test.target_indices]["cnt"].to_numpy(dtype=np.float64)
    scores = calculate_metrics(actual, predicted)
    boundaries = data.metadata["boundaries"]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:8]
    metrics = {"model_name": name, "evaluation_run": run_id,
               "forecasting_mode": metadata["forecasting_mode"], "window_size": metadata["window_size"],
               "kernel_size": metadata["kernel_size"], "training_rows": boundaries["training"]["rows"],
               "validation_rows": boundaries["validation"]["rows"], "testing_rows": boundaries["testing"]["rows"],
               "test_samples": len(actual), "error_definition": "predicted - actual", **scores}
    predictions = pd.DataFrame({"timestamp": data.test.timestamps, "actual": actual,
                                "predicted": predicted, "error": predicted - actual})
    output_dir = model_output_directory(output_dir, name)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{name}_{run_id}_predictions.csv"
    metrics_path = output_dir / f"{name}_{run_id}_metrics.json"
    result = EvaluationResult(run_id, predictions, metrics, csv_path, metrics_path)
    written = []
    try:
        with csv_path.open("x") as handle:
            written.append(csv_path)
            predictions.to_csv(handle, index=False, date_format="%Y-%m-%d %H:%M:%S")
        with metrics_path.open("x") as handle:
            written.append(metrics_path)
            json.dump(metrics, handle, indent=2, allow_nan=False)
            handle.write("\n")
        result.numerical_paths = save_numerical_results(result, history, numerical_dir)
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    return result


def calculate_error_statistics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Descriptive statistics over observed test targets, using population variance."""
    scores = calculate_metrics(actual, predicted)
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    error = predicted - actual
    absolute = np.abs(error)
    centered_actual_sum = float(np.sum((actual - actual.mean()) ** 2))
    actual_absolute_sum = float(np.sum(np.abs(actual)))
    statistics = {
        "samples": len(actual), **scores,
        "mean_error_bias": float(error.mean()),
        "error_variance": float(error.var(ddof=0)),
        "error_std": float(error.std(ddof=0)),
        "error_min": float(error.min()), "error_max": float(error.max()),
        "error_median": float(np.median(error)),
        "error_p05": float(np.percentile(error, 5)), "error_p95": float(np.percentile(error, 95)),
        "absolute_error_median": float(np.median(absolute)),
        "absolute_error_p90": float(np.percentile(absolute, 90)),
        "absolute_error_p95": float(np.percentile(absolute, 95)),
        "absolute_error_p99": float(np.percentile(absolute, 99)),
        "absolute_error_max": float(absolute.max()),
        "overprediction_count": int(np.count_nonzero(error > 0)),
        "underprediction_count": int(np.count_nonzero(error < 0)),
        "exact_prediction_count": int(np.count_nonzero(error == 0)),
        "actual_mean": float(actual.mean()), "actual_std": float(actual.std(ddof=0)),
        "actual_min": float(actual.min()), "actual_max": float(actual.max()),
        "predicted_mean": float(predicted.mean()), "predicted_std": float(predicted.std(ddof=0)),
        "predicted_min": float(predicted.min()), "predicted_max": float(predicted.max()),
        "actual_total": float(actual.sum()), "predicted_total": float(predicted.sum()),
        "total_error": float(error.sum()),
        "R2": float(1 - np.sum(error ** 2) / centered_actual_sum) if centered_actual_sum > 0 else None,
        "WAPE_percent": float(100 * absolute.sum() / actual_absolute_sum) if actual_absolute_sum > 0 else None,
    }
    return statistics


def _group_error_statistics(errors: pd.DataFrame, group: pd.Series, label: str) -> pd.DataFrame:
    grouped = errors.groupby(group.rename(label), sort=True)
    table = grouped.agg(
        samples=("error", "size"), actual_mean=("actual", "mean"), predicted_mean=("predicted", "mean"),
        actual_total=("actual", "sum"), predicted_total=("predicted", "sum"),
        mean_error_bias=("error", "mean"), MAE=("absolute_error", "mean"), MSE=("squared_error", "mean"),
        error_min=("error", "min"), error_max=("error", "max"),
    )
    table["RMSE"] = np.sqrt(table.MSE)
    table["error_variance"] = grouped.error.var(ddof=0)
    table["error_std"] = grouped.error.std(ddof=0)
    return table.reset_index()


def save_numerical_results(evaluation: EvaluationResult, history: dict | None = None,
                           output_dir: Path = NUMERICAL_DIR) -> dict[str, Path]:
    """Export numerical tables without inference, fitting, or changing original results."""
    name = validate_model_name(evaluation.metrics["model_name"])
    validate_model_name(evaluation.run_id)
    required = {"timestamp", "actual", "predicted"}
    if not required.issubset(evaluation.predictions.columns):
        raise ValueError("Numerical reports require timestamp, actual, and predicted columns.")
    errors = evaluation.predictions[["timestamp", "actual", "predicted"]].copy()
    errors["timestamp"] = pd.to_datetime(errors.timestamp, errors="raise")
    if errors.timestamp.isna().any() or errors.timestamp.duplicated().any():
        raise ValueError("Numerical reports require valid, unique target timestamps.")
    errors = errors.sort_values("timestamp").reset_index(drop=True)
    errors[["actual", "predicted"]] = errors[["actual", "predicted"]].astype(np.float64)
    statistics = calculate_error_statistics(errors.actual, errors.predicted)
    if evaluation.metrics.get("test_samples", len(errors)) != len(errors):
        raise ValueError("Prediction row count disagrees with the saved test sample count.")
    errors["error"] = errors.predicted - errors.actual
    errors["absolute_error"] = errors.error.abs()
    errors["squared_error"] = errors.error ** 2
    errors["deviation_from_mean_error"] = errors.error - statistics["mean_error_bias"]
    tables = {
        "prediction_errors": errors,
        "error_statistics": pd.DataFrame({"statistic": statistics.keys(), "value": statistics.values()}),
        "daily_errors": _group_error_statistics(errors, errors.timestamp.dt.normalize(), "date"),
        "hourly_errors": _group_error_statistics(errors, errors.timestamp.dt.hour, "hour_of_day"),
    }
    if history:
        if not isinstance(history, dict) or not history.get("loss") or not history.get("val_loss"):
            raise ValueError("Training history must contain loss and val_loss sequences.")
        losses = np.asarray(history["val_loss"], dtype=float)
        if any(np.asarray(values).ndim != 1 or len(values) != len(losses)
               or not np.isfinite(np.asarray(values, dtype=float)).all() for values in history.values()):
            raise ValueError("Training history must contain equally sized, finite numeric sequences.")
        training = pd.DataFrame(history)
        training.insert(0, "epoch", np.arange(1, len(training) + 1))
        training["is_best_validation_epoch"] = training.epoch == int(np.argmin(losses)) + 1
        tables["training_history"] = training
    summary = {
        **evaluation.metrics, "evaluation_run": evaluation.run_id,
        "first_test_timestamp": errors.timestamp.iloc[0].isoformat(),
        "last_test_timestamp": errors.timestamp.iloc[-1].isoformat(),
        "error_statistics": statistics,
        "definitions": {
            "error": "predicted - actual; positive means overprediction",
            "mean_error_bias": "Arithmetic mean of signed prediction errors",
            "deviation_from_mean_error": "Each signed error minus the mean signed error",
            "error_variance": "Mean squared deviation from mean error; population variance (ddof=0)",
            "error_std": "Square root of population error variance (ddof=0)",
            "absolute_error_p95": "95th percentile of absolute errors using linear interpolation",
            "R2": "1 - sum(squared errors) / sum((actual - mean(actual))**2); null for constant actuals",
            "WAPE_percent": "100 * sum(absolute errors) / sum(abs(actual)); null for an all-zero actual series",
            "exact_prediction_count": "Number of predictions with exactly zero error; no rounding or tolerance",
            "units": "Rental counts, except MSE/variance (squared counts), R2 (unitless), WAPE (percent), and counts of samples",
            "aggregation": "Only observed eligible test targets; absent hours are not filled. Totals cover these targets only.",
            "hourly_errors": "Grouped by hour of day (0-23) across all test dates, not consecutive hourly bins",
            "training_history": "loss/val_loss are MSE in scaled target units; mae/val_mae are scaled MAE",
        },
    }
    directory = model_output_directory(output_dir, name)
    directory.mkdir(parents=True, exist_ok=True)
    prefix = f"{name}_{evaluation.run_id}"
    paths = {key: directory / f"{prefix}_{key}.csv" for key in tables}
    paths["summary"] = directory / f"{prefix}_summary.json"
    written = []
    try:
        for key, table in tables.items():
            with paths[key].open("x") as handle:
                written.append(paths[key])
                table.to_csv(handle, index=False, date_format="%Y-%m-%d %H:%M:%S")
        with paths["summary"].open("x") as handle:
            written.append(paths["summary"])
            json.dump(summary, handle, indent=2, allow_nan=False)
            handle.write("\n")
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    return paths
