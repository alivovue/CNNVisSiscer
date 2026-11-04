"""Publication-sized, noninteractive PNG figures for each evaluation run."""

from pathlib import Path

import numpy as np

from src.preprocessing.preprocess import PROJECT_ROOT, PreparedData
from src.train.evaluate import EvaluationResult


VISUALIZATION_DIR = PROJECT_ROOT / "output" / "visualization"


def generate_visualizations(data: PreparedData, evaluation: EvaluationResult, metadata: dict,
                            history: dict | None = None, output_dir: Path = VISUALIZATION_DIR,
                            report=print) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{metadata['model_name']}_{evaluation.run_id}"
    saved = []

    def save(kind, title, ylabel, draw, dates=True):
        path = output_dir / f"{prefix}_{kind}.png"
        fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
        try:
            draw(ax)
            ax.set(title=title, xlabel="Timestamp" if dates else "Epoch", ylabel=ylabel)
            ax.grid(alpha=0.25)
            ax.legend(loc="best")
            if dates:
                locator = AutoDateLocator()
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
            with path.open("xb") as handle:
                fig.savefig(handle, format="png", dpi=300)
            saved.append(path)
            report(f"Saved {path}")
        finally:
            plt.close(fig)

    def original(ax):
        # NaNs split the line at absent hours, so the chart does not imply observations.
        series = data.frame.set_index("timestamp")["cnt"].asfreq("h")
        ax.plot(series.index, series.values, color="#2563eb", linewidth=0.65, label="Observed rentals")
        for name, color in (("validation", "#d97706"), ("testing", "#dc2626")):
            boundary = data.metadata["boundaries"][name]["start"]
            ax.axvline(data.frame.timestamp.iloc[boundary], color=color, linestyle="--",
                       label=f"{name.capitalize()} starts (row {boundary + 1:,})")

    save("original_series", "Bike rentals and chronological split boundaries", "Bike rentals", original)
    if history and history.get("loss") and history.get("val_loss"):
        def losses(ax):
            epochs = np.arange(1, len(history["loss"]) + 1)
            ax.plot(epochs, history["loss"], label="Training loss")
            ax.plot(epochs, history["val_loss"], label="Validation loss")
            best = metadata["best_validation_epoch"]
            ax.axvline(best, color="#16a34a", linestyle="--", label=f"Best validation epoch: {best}")
        save("training_loss", "Training and validation loss", "MSE (scaled target)", losses, dates=False)
    else:
        report("Training history unavailable; skipping the training-loss graph.")

    # Reindex to preserve visible breaks wherever missing hours invalidated predictions.
    results = evaluation.predictions.set_index("timestamp").asfreq("h")

    def comparison(ax):
        ax.plot(results.index, results.actual, linewidth=0.9, label="Actual", color="#2563eb")
        ax.plot(results.index, results.predicted, linewidth=0.9, alpha=0.8, label="CNN prediction", color="#dc2626")
    save("actual_vs_predicted", "Next-hour forecasts on the final test period", "Bike rentals", comparison)

    def errors(ax):
        ax.plot(results.index, results.error, linewidth=0.8, color="#7c3aed", label="Prediction − actual")
        ax.axhline(0, color="black", linestyle="--", linewidth=0.9, label="Zero error")
    save("prediction_errors", "Test prediction errors over time", "Error (bike rentals)", errors)
    return saved
