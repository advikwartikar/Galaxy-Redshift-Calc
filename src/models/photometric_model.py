"""Photometric-feature MLP regressor."""

from ._tensorflow import get_tensorflow


def build_photometric_model(
    input_dim: int,
    hidden_units: tuple[int, ...] = (128, 64),
    dropout_rate: float = 0.2,
):
    """Build an MLP that regresses one continuous redshift value."""
    if input_dim <= 0 or not hidden_units or any(units <= 0 for units in hidden_units):
        raise ValueError("input_dim and every hidden layer size must be positive.")
    if not 0 <= dropout_rate < 1:
        raise ValueError("dropout_rate must be in [0, 1).")
    tf = get_tensorflow()
    inputs = tf.keras.Input(shape=(input_dim,), name="photometric_features")
    x = inputs
    for index, units in enumerate(hidden_units):
        x = tf.keras.layers.Dense(units, activation="relu", name=f"photometric_dense_{index + 1}")(x)
        if dropout_rate:
            x = tf.keras.layers.Dropout(dropout_rate)(x)
    output = tf.keras.layers.Dense(1, name="redshift")(x)
    return tf.keras.Model(inputs=inputs, outputs=output, name="photometric_mlp")

