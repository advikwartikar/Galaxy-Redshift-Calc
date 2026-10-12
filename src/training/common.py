"""Shared configuration and leakage-safe dataset preparation for training."""

from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import yaml

from src.data.dataset_merger import MatchReport, merge_datasets
from src.data.loader import load_image_metadata, load_photometric_data
from src.data.preprocessing import FeaturePreprocessor, clean_records, split_dataset


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f"Could not parse YAML config {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
    return config


def load_matched_data(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, MatchReport, list[str], str, str]:
    """Load the one vetted joined sample used by all model training commands."""
    data = config["data"]
    features = data["photometric_feature_columns"]
    target = data["target_column"]
    photometric_id = data["photometric_object_id_column"]
    image_id = data["image_object_id_column"]
    photometry = load_photometric_data(
        data["photometric_csv"], features, target, photometric_id
    )
    images = load_image_metadata(
        data["image_metadata_csv"],
        image_id,
        data["image_path_column"],
        require_files=True,
    )
    merged, report = merge_datasets(
        photometry,
        images,
        photometric_id,
        image_id,
        source_compatibility_verified=data.get("source_compatibility_verified", False),
    )
    print(
        "Exact-ID match report: "
        f"{report.matched_records}/{report.photometric_records} photometric records "
        f"({report.photometric_match_rate:.1%}); "
        f"{report.matched_records}/{report.image_records} image records "
        f"({report.image_match_rate:.1%})."
    )
    cleaned = clean_records(merged, features, target, photometric_id)
    return cleaned, report, list(features), target, data["image_path_column"]


def split_and_scale(
    frame: pd.DataFrame, config: dict[str, Any], feature_columns: list[str]
) -> tuple[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], FeaturePreprocessor]:
    preprocessing = config["preprocessing"]
    splits = split_dataset(
        frame,
        train_ratio=preprocessing["train_ratio"],
        validation_ratio=preprocessing["validation_ratio"],
        test_ratio=preprocessing["test_ratio"],
        random_seed=preprocessing["random_seed"],
    )
    processor = FeaturePreprocessor().fit(splits[0], feature_columns)
    scaler_path = Path(config["models"]["photometric_scaler"])
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(processor, scaler_path)
    return splits, processor


def image_arrays(
    frames: Sequence[pd.DataFrame],
    image_path_column: str,
    image_size: int,
) -> tuple[np.ndarray, ...]:
    """Load resized/preprocessed image batches for the baseline training skeleton."""
    from src.data.preprocessing import preprocess_image

    arrays = []
    for frame in frames:
        arrays.append(
            np.stack(
                [
                    preprocess_image(path, image_size)
                    for path in frame[image_path_column].astype(str)
                ]
            )
        )
    return tuple(arrays)


def target_arrays(
    frames: Sequence[pd.DataFrame], target_column: str
) -> tuple[np.ndarray, ...]:
    return tuple(
        frame[target_column].to_numpy(dtype=np.float32)
        for frame in frames
    )
