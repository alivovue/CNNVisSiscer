"""Configurable one-dimensional CNN for a single next-hour prediction."""

from src.preprocessing.preprocess import positive_integer


def build_model(window_size: int = 24, kernel_size: int = 3, filters: int = 32,
                feature_count: int = 1, seed: int = 42):
    for value, name in ((window_size, "Window size"), (kernel_size, "Kernel size"),
                        (filters, "Filter count"), (feature_count, "Feature count")):
        positive_integer(value, name)
    if kernel_size > window_size:
        raise ValueError("Kernel size must not exceed the input window size.")
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window_size, feature_count)),
        tf.keras.layers.Conv1D(filters, kernel_size, name="convolution"),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="linear"),
    ], name="bike_rental_cnn")
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mean_squared_error",
                  metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")])
    return model
