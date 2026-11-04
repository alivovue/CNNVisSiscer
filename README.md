# CNNVisSiscer

An interactive TensorFlow/Keras CNN that predicts bike rentals **one hour ahead**
using the UCI Bike Sharing Dataset. Configure, train, save, load, evaluate, and
optionally plot models entirely through terminal prompts.

## Setup and run

Use Python 3.10–3.12 in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.main
```

Run from the project root. No command-line arguments are required. Data and output
paths are resolved relative to the project, so the modules also work when imported
from elsewhere. The application uses `data/hour.csv`, an unchanged copy of the
supplied `data/bike+sharing+dataset/hour.csv`. The original download and extracted
files are preserved. Dataset provenance and attribute definitions are in the
supplied `data/bike+sharing+dataset/Readme.txt` and on the
[UCI Bike Sharing page](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset).

The main menu provides:

1. **Generate and train a new CNN model.** Choose a mode, input window, convolution
   kernel, number of filters, maximum epochs, batch size, and a unique model name.
   Press Enter to use defaults: 24 hours, kernel 3, 32 filters, 50 epochs, batch 32.
2. **Load an existing CNN model.** Select a saved model by its name or list number.
   Its original configuration, scalers, and split boundaries are reused to evaluate
   the same final test period. There is no training or scaler fitting.
3. **Exit.** EOF and Ctrl+C also end the session cleanly.

Invalid values are explained and requested again. Windows must be positive and
below 13,000 observations to leave training samples. Kernels must fit inside the
window; a window shorter than three defaults to a matching kernel. Model names
contain 1–64 letters, digits, underscores, or hyphens and start with a letter or
digit. Existing or partial model bundles cannot be overwritten.

## Forecasting modes and architecture

| Mode | Historical input features | Target |
| --- | --- | --- |
| Univariate | `cnt` | Next-hour `cnt` |
| Multivariate | `cnt`, `temp`, `hum`, `windspeed` | Next-hour `cnt` |

Both modes use this architecture:

```text
Input (window_size, features)
→ Conv1D(filters, kernel_size, valid padding)
→ ReLU
→ Flatten
→ Dense(16, ReLU)
→ Dense(1, linear)
```

The optimizer is Adam, the loss is mean squared error, and training also records
mean absolute error. The random seed is 42 and is saved with each model. Hardware
and TensorFlow versions can affect exact training reproducibility. Input arrays
and scaled targets are NumPy `float32`; predictions are inverse-transformed into
original rental counts. Linear predictions are preserved without rounding or
clipping, including any negative predictions.

## Dataset validation and chronological splitting

The loader combines `dteday` and `hr`, sorts chronologically, and checks required
columns, missing values, duplicate timestamps, numeric and finite attributes,
valid hours, whole nonnegative rental counts, and normalized weather attributes.
Unsafe records cause an informative error; observations are never silently
discarded or filled. The CSV is never rewritten by preprocessing.

| Split | Zero-based slice | Rows in the supplied CSV | Use |
| --- | --- | --- | --- |
| Training | `df.iloc[:13000]` | 13,000 | Weight updates and scaler fitting |
| Validation | `df.iloc[13000:15000]` | 2,000 | Model selection and early stopping |
| Testing | `df.iloc[15000:]` | 2,379 | Final evaluation only |

At least 15,001 usable observations are required. The first 15,000 rows form the
development portion. Neither observations nor training samples are shuffled.
Separate feature and target `MinMaxScaler` instances are fitted on the **first
13,000 rows only**. Validation and test transformations may lie outside 0–1.
The scaling behavior follows the
[scikit-learn MinMaxScaler documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html).

Each window contains the `window_size` hourly observations strictly before its
target. Targets determine split membership. The first validation and test targets
can use context from the preceding split, so boundary context alone never removes
them. This is rolling one-step-ahead evaluation: later test forecasts may use
already observed earlier test counts. It is not a recursive multi-hour forecast.

The actual dataset has missing hours. A window is skipped whenever its inputs or
the one-hour step to its target cross a gap. Missing observations are never
invented as zeros. The CLI reports row counts, detected gaps, and usable/skipped
window counts per split. Consequently, usable sample counts differ from row
counts, and a boundary target is skipped when its context contains a real gap.
If any split has no usable windows, choose a smaller window; the application
reports an error instead of training or evaluating an empty split.

## Validation, saving, and loading

Training passes the validation split separately to `model.fit(..., shuffle=False)`.
Keras `EarlyStopping` monitors `val_loss` with patience 10 and restores the best
validation weights, including when the configured epoch limit is reached. See
[Keras EarlyStopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping).
The final test split never participates in weight updates, early stopping, or
hyperparameter selection.

The following files form a saved bundle in `output/model/<name>/`:

```text
<name>.keras
<name>_metadata.json
<name>_feature_scaler.joblib
<name>_target_scaler.joblib
<name>_history.json
```

The `.keras` file contains the architecture and weights, following the
[TensorFlow saving guide](https://www.tensorflow.org/guide/keras/serialization_and_saving).
Metadata records the model and dataset paths, mode, features, target, window,
kernel, filters, batch size, maximum/completed epochs, best validation epoch/loss,
random seed, exact row/timestamp boundaries, scaling settings, scaler filenames,
dataset quality, window counts, and SHA-256 fingerprint of the original CSV.
History stores all completed epochs' training and validation metrics. Failed
saves remove their partial files; existing models are never replaced.

Loading checks the architecture against metadata and requires the saved scalers.
Preprocessing verifies the CSV fingerprint and original split boundaries before
inference. A changed dataset is rejected rather than silently changing the test
period. If the project has moved, the local `data/hour.csv` is used when the old
absolute path is unavailable, and must have the same fingerprint. Missing model,
metadata, or scaler files are errors. Missing/unreadable history skips the loss
plot gracefully. Keep bundle files together and load trusted local bundles, since
Joblib scalers use Python object serialization. Models saved in the older flat
`output/model/` layout are also listed and loadable; new models use their own
folders. An existing model folder, even if incomplete, reserves its model name.

## Evaluation and output

After training/model selection, or after loading, inference runs with
`training=False` and does not change weights. MAE, MSE, and RMSE are calculated in
original rental-count units against the observed test targets. The application
prints the executed model's numeric results and saves:

- `output/test_output/<name>/<name>_<run>_predictions.csv` with
  `timestamp,actual,predicted,error`, where `error = predicted - actual`.
- `output/test_output/<name>/<name>_<run>_metrics.json` with model configuration, split
  row counts, test sample count, and MAE/MSE/RMSE.
- `output/visualization/<name>/<name>_<run>_*.png` when visualization is selected:
  original time series with split boundaries, training/validation loss with the
  best epoch, actual versus predicted test counts, and prediction error with a
  zero reference line.

Each evaluation uses a timestamp and random run identifier to avoid overwrites.
Plots use a noninteractive Matplotlib backend, readable 12 × 4.5 inch figures,
300 DPI PNG output, and visible breaks at missing hours. Figures are closed after
saving. Declining visualization preserves the model and evaluation files and
returns to the main menu. All generated application artifacts stay in the
`output/` directory and are excluded from Git.

Every output category groups files by model name. For a model named `testing`:

```text
output/
├── model/
│   └── testing/
│       ├── testing.keras
│       ├── testing_metadata.json
│       ├── testing_feature_scaler.joblib
│       ├── testing_target_scaler.joblib
│       └── testing_history.json
├── test_output/
│   └── testing/
│       ├── testing_<run>_predictions.csv
│       └── testing_<run>_metrics.json
├── visualization/
│   └── testing/
│       └── testing_<run>_*.png
└── pre_result/
    └── testing/
        ├── testing_<run>_cleaned.csv
        ├── testing_<run>_scaled.csv
        ├── testing_<run>_windows.npz
        └── testing_<run>_metadata.json
```

Preprocessing results are saved automatically before training and when loading a
model. `cleaned.csv` contains the validated, chronologically sorted observations,
timestamps, zero-based row indices, and split labels. `scaled.csv` contains the
selected scaled features and the separate `target_cnt` column with timestamps,
row indices, and split labels. The same training-fitted scalers are used without
refitting. The original `data/hour.csv` remains raw and unchanged.

`windows.npz` preserves the exact float32 `X_train`, `y_train`, `X_val`, `y_val`,
`X_test`, and `y_test` arrays. It also contains `timestamps_train`,
`timestamps_val`, `timestamps_test` and corresponding `target_indices_*` arrays.
Open it with `np.load(path, allow_pickle=False)`. The JSON records the model name,
features, scaling configuration, split boundaries, quality report, skipped gap
counts, sample counts, and exported filenames. Each preprocessing run gets a
unique identifier, so loading a model again preserves previous exports.

## Tests

```bash
python -m unittest -v test_output_layout
```

The standard-library unittest suite covers model-specific artifact paths,
collisions and rollback, saving/loading models, compatibility with the old flat
layout, model isolation, evaluation and visualization paths, preprocessing CSV
and NPZ round trips, repeated exports, and CLI workflow integration. Synthetic
fixtures are used only in unit tests. Tests use temporary directories and clean
up their artifacts. Normal execution always uses the actual CSV.
