"""Evaluate available checkpoints on the shared held-out test set."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

from src.evaluation.metrics import save_comparison_report
from src.training.common import (
    image_arrays, load_config, load_matched_data, target_arrays,
)
from src.data.preprocessing import split_dataset


def evaluate(config_path: str = "configs/config.yaml") -> Path:
    config = load_config(config_path)
    frame, _, features, target, image_path_column = load_matched_data(config)
    preprocessing = config["preprocessing"]
    splits = split_dataset(
        frame,
        train_ratio=preprocessing["train_ratio"],
        validation_ratio=preprocessing["validation_ratio"],
        test_ratio=preprocessing["test_ratio"],
        random_seed=preprocessing["random_seed"],
    )
    test = splits[2]
    y_test = target_arrays(splits, target)[2]
    scaler_path = Path(config["models"]["photometric_scaler"])
    if not scaler_path.is_file():
        raise FileNotFoundError(
            f"Scaler not found: {scaler_path}. Train the photometric or hybrid model first."
        )
    processor = joblib.load(scaler_path)
    photo_test = processor.transform(test)
    image_test = image_arrays(
        (splits[2],), image_path_column, preprocessing["image_size"]
    )[0]
    results = {}
    for label, key, inputs in (
        ("Photometric-only", "photometric_checkpoint", photo_test),
        ("Image-only", "image_checkpoint", image_test),
        ("Hybrid", "multimodal_checkpoint", [image_test, photo_test]),
    ):
        checkpoint = Path(config["models"][key])
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Cannot compare all three models: {label} checkpoint is missing at "
                f"{checkpoint}. Train all model variants first."
            )
        model = tf.keras.models.load_model(checkpoint)
        prediction = np.asarray(model.predict(inputs, verbose=0)).reshape(-1)
        results[label] = (y_test, prediction)
    output_directory = Path("reports/metrics")
    comparison = save_comparison_report(
        results, output_directory, Path("reports/figures")
    )
    print(comparison.to_string(index=False))
    return output_directory / "model_comparison.csv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    print(f"Saved evaluation report: {evaluate(parser.parse_args().config)}")
