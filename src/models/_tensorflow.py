"""Lazy TensorFlow import so data/evaluation utilities remain independently usable."""


def get_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required to build or load neural network models. "
            "Install the dependencies listed in requirements.txt."
        ) from exc
    return tf

