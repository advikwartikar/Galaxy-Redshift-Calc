"""Train the hybrid model: python -m src.training.train_multimodal."""

import argparse
from pathlib import Path

import tensorflow as tf

from src.models.multimodal_model import build_multimodal_model
from src.training.common import (
    image_arrays, load_config, load_matched_data, split_and_scale, target_arrays,
)


def train(config_path: str = "configs/config.yaml") -> Path:
    config = load_config(config_path)
    tf.keras.utils.set_random_seed(config["preprocessing"]["random_seed"])
    frame, _, features, target, image_path_column = load_matched_data(config)
    splits, processor = split_and_scale(frame, config, features)
    size = config["preprocessing"]["image_size"]
    images_train, images_validation = image_arrays(splits[:2], image_path_column, size)
    photo_train, photo_validation = (
        processor.transform(split) for split in splits[:2]
    )
    y_train, y_validation = target_arrays(splits[:2], target)
    model = build_multimodal_model(
        photometric_input_dim=len(features),
        image_size=size,
        trainable_backbone=config["training"].get("fine_tune_backbone", False),
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config["training"]["learning_rate"]),
        loss="mse",
        metrics=["mae"],
    )
    checkpoint = Path(config["models"]["multimodal_checkpoint"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.fit(
        [images_train, photo_train],
        y_train,
        validation_data=([images_validation, photo_validation], y_validation),
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
    print(f"Saved best hybrid checkpoint: {train(parser.parse_args().config)}")
