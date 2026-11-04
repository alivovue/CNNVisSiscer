"""Run with ``python -m src.main``; all configuration is interactive."""

from pathlib import Path

from src.preprocessing.preprocess import DATASET_PATH, prepare_data, save_preprocessed_data
from src.train.evaluate import evaluate_model
from src.train.train import MODEL_DIR, ensure_name_available, list_models, load_bundle, train_and_save
from src.visualization.visualize import generate_visualizations


def banner(title: str) -> None:
    print(f"\n{'=' * 40}\n{title:^40}\n{'=' * 40}\n")


def prompt_choice(prompt: str, choices) -> str:
    while True:
        value = input(prompt).strip()
        if value in choices:
            return value
        print(f"Invalid choice. Enter one of: {', '.join(choices)}.")


def prompt_integer(label: str, default: int, maximum: int | None = None) -> int:
    while True:
        raw = input(f"Enter {label} (default: {default}): ").strip()
        try:
            value = int(raw) if raw else default
            if value < 1 or (maximum is not None and value > maximum):
                raise ValueError
            return value
        except ValueError:
            limit = f" from 1 to {maximum:,}" if maximum is not None else " greater than zero"
            print(f"Enter a whole number{limit}.")


def prompt_model_name(model_dir: Path = MODEL_DIR) -> str:
    while True:
        name = input("Enter a name for the model: ").strip()
        try:
            ensure_name_available(name, model_dir)
            return name
        except (ValueError, FileExistsError) as exc:
            print(exc)


def select_model(model_dir: Path = MODEL_DIR) -> str | None:
    names = list_models(model_dir)
    if not names:
        print(f"No saved CNN models found in {model_dir}.")
        return None
    print("Saved CNN models:")
    for number, name in enumerate(names, 1):
        print(f"[{number}] {name}")
    while True:
        choice = input("Select a model by name or number (Enter to return): ").strip()
        if not choice:
            return None
        if choice in names:
            return choice
        if choice.isascii() and choice.isdecimal() and 1 <= int(choice) <= len(names):
            return names[int(choice) - 1]
        print("Select a listed model name or number.")


def evaluate_and_visualize(bundle, data) -> None:
    banner("MODEL EVALUATION")
    print("Generating final test predictions and calculating metrics...")
    result = evaluate_model(bundle.model, data, bundle.metadata, history=bundle.history)
    for metric in ("MAE", "MSE", "RMSE"):
        print(f"{metric}: {result.metrics[metric]:.6f}")
    print(f"Test samples: {result.metrics['test_samples']:,}")
    print(f"Predictions saved to {result.predictions_path}")
    print(f"Metrics saved to {result.metrics_path}")
    print(f"Numerical reports saved to {result.numerical_paths['summary'].parent}")
    banner("VISUALIZATION")
    print("Would you like to generate visualizations?\n\n[1] Yes\n[2] No\n")
    if prompt_choice("Enter your choice: ", ("1", "2")) == "1":
        paths = generate_visualizations(data, result, bundle.metadata, bundle.history)
        print(f"Generated {len(paths)} visualizations.")
    banner("PROCESS COMPLETE")


def new_model_workflow() -> None:
    banner("FORECASTING CONFIGURATION")
    print("Select forecasting mode:\n\n[1] Univariate\n[2] Multivariate\n")
    mode = {"1": "univariate", "2": "multivariate"}[prompt_choice("Enter your choice: ", ("1", "2"))]
    window_size = prompt_integer("input window size", 24, maximum=12_999)
    kernel_size = prompt_integer("kernel size", min(3, window_size), maximum=window_size)
    filters = prompt_integer("number of filters", 32)
    epochs = prompt_integer("training epochs", 50)
    batch_size = prompt_integer("batch size", 32)
    name = prompt_model_name()
    banner("DATA PREPROCESSING")
    data = prepare_data(DATASET_PATH, mode, window_size)
    preprocessed = save_preprocessed_data(data, name)
    print(f"Preprocessed data saved to {preprocessed['metadata'].parent}")
    banner("CNN TRAINING")
    bundle = train_and_save(data, name, kernel_size, filters, epochs, batch_size)
    metadata = bundle.metadata
    if metadata["early_stopping_triggered"]:
        print("Early stopping triggered.")
    print(f"Completed epochs: {metadata['completed_epochs']}")
    print(f"Best validation epoch: {metadata['best_validation_epoch']}")
    print(f"Best validation loss: {metadata['best_validation_loss']:.8f}")
    print("Best validation weights restored; model, scalers, metadata and history saved.")
    evaluate_and_visualize(bundle, data)


def load_model_workflow() -> None:
    name = select_model()
    if name is None:
        return
    bundle = load_bundle(name)
    metadata = bundle.metadata
    print(f"Loaded {name}: {metadata['forecasting_mode']}, {metadata['window_size']}-hour window, "
          f"kernel {metadata['kernel_size']}, {metadata['filters']} filters.")
    banner("DATA PREPROCESSING")
    path = Path(metadata["dataset_path"])
    # Support moving the intact project to another machine, while fingerprint
    # validation still requires byte-identical observations and boundaries.
    if not path.is_file():
        path = DATASET_PATH
    data = prepare_data(path, metadata["forecasting_mode"], metadata["window_size"],
                        feature_scaler=bundle.feature_scaler, target_scaler=bundle.target_scaler,
                        saved_metadata=metadata)
    preprocessed = save_preprocessed_data(data, name)
    print(f"Preprocessed data saved to {preprocessed['metadata'].parent}")
    evaluate_and_visualize(bundle, data)


def main() -> None:
    while True:
        try:
            banner("CNN TIME-SERIES FORECASTING")
            print("Dataset: Bike Sharing Dataset\n")
            print("Select an operation:\n\n[1] Generate and train a new CNN model\n"
                  "[2] Load an existing CNN model\n[3] Exit\n")
            choice = prompt_choice("Enter your choice: ", ("1", "2", "3"))
            if choice == "3":
                print("Goodbye.")
                return
            if choice == "1":
                new_model_workflow()
            else:
                load_model_workflow()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            return
        except Exception as exc:
            print(f"\nOperation failed: {exc}\nReturning to the main menu.")


if __name__ == "__main__":
    main()
