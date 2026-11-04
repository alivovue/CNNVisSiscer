"""Final held-out evaluation in original bike-rental count units."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from src.preprocessing.preprocess import PROJECT_ROOT, PreparedData, positive_integer
from src.outputs import model_output_directory, validate_model_name


TEST_OUTPUT_DIR = PROJECT_ROOT / "output" / "test_output"


@dataclass
class EvaluationResult:
    run_id: str
    predictions: pd.DataFrame
    metrics: dict
    predictions_path: Path
    metrics_path: Path


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
                   batch_size: int | None = None) -> EvaluationResult:
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
    written = []
    try:
        with csv_path.open("x") as handle:
            written.append(csv_path)
            predictions.to_csv(handle, index=False, date_format="%Y-%m-%d %H:%M:%S")
        with metrics_path.open("x") as handle:
            written.append(metrics_path)
            json.dump(metrics, handle, indent=2, allow_nan=False)
            handle.write("\n")
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    return EvaluationResult(run_id, predictions, metrics, csv_path, metrics_path)
