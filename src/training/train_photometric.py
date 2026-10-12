"""Train the photometric-only MLP: python -m src.training.train_photometric."""

import argparse
from pathlib import Path

import tensorflow as tf

from src.models.photometric_model import build_photometric_model
from src.training.common import load_config, load_matched_data, split_and_scale, target_arrays


def train(config_path: str = "configs/config.yaml") -> Path:
    config = load_config(config_path)
    tf.keras.utils.set_random_seed(config["preprocessing"]["random_seed"])
    frame, _, features, target, _ = load_matched_data(config)
    splits, processor = split_and_scale(frame, config, features)
    x_train, x_validation = (
        processor.transform(split) for split in splits[:2]
    )
    y_train, y_validation = target_arrays(splits[:2], target)
    model = build_photometric_model(len(features))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config["training"]["learning_rate"]),
        loss="mse",
        metrics=["mae"],
    )
    checkpoint = Path(config["models"]["photometric_checkpoint"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        batch_size=config["training"]["batch_size"],
        epochs=config["training"]["epochs"],
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint, monitor="val_loss", save_best_only=True
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            ),
        ],
    )
    return checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    print(f"Saved best photometric checkpoint: {train(parser.parse_args().config)}")
