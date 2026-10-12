"""Checkpoint-backed inference with explicit missing-artifact errors."""

from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image

from src.data.preprocessing import FeaturePreprocessor, preprocess_image
from src.models._tensorflow import get_tensorflow
from src.training.common import load_config


MODEL_KEYS = {
    "Photometric-only": "photometric",
    "Image-only": "image",
    "Hybrid": "multimodal",
}


class RedshiftPredictor:
    """Load a selected trained model and validate its required input modality."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self._models: dict[str, object] = {}
        self._preprocessor: FeaturePreprocessor | None = None

    @classmethod
    def from_config(cls, config_path: str | Path = "configs/config.yaml"):
        return cls(load_config(config_path))

    def _checkpoint(self, model_name: str) -> Path:
        try:
            path = Path(self.config["models"][f"{MODEL_KEYS[model_name]}_checkpoint"])
        except KeyError as exc:
            raise ValueError(f"Unknown model selection: {model_name}") from exc
        if not path.is_file():
            raise FileNotFoundError(
                f"The {model_name} model has not been trained or its checkpoint is missing: "
                f"{path}. Run the corresponding training command first."
            )
        return path

    def _load_model(self, model_name: str):
        if model_name not in MODEL_KEYS:
            raise ValueError(f"Unknown model selection: {model_name}")
        if model_name not in self._models:
            self._models[model_name] = get_tensorflow().keras.models.load_model(
                self._checkpoint(model_name)
            )
        return self._models[model_name]

    def _load_preprocessor(self) -> FeaturePreprocessor:
        if self._preprocessor is None:
            import joblib

            scaler_path = Path(self.config["models"]["photometric_scaler"])
            if not scaler_path.is_file():
                raise FileNotFoundError(
                    f"Photometric scaler is missing: {scaler_path}. "
                    "Train the photometric or hybrid model first."
                )
            self._preprocessor = joblib.load(scaler_path)
            if not isinstance(self._preprocessor, FeaturePreprocessor):
                raise TypeError(f"Unexpected scaler artifact type: {type(self._preprocessor).__name__}")
        return self._preprocessor

    def predict(
        self,
        model_name: str,
        *,
        photometric_features: Mapping[str, float] | None = None,
        image: Image.Image | None = None,
    ) -> float:
        """Predict a redshift using only the modalities required by the selection."""
        if model_name not in MODEL_KEYS:
            raise ValueError(f"Unknown model selection: {model_name}")
        needs_photo = model_name in ("Photometric-only", "Hybrid")
        needs_image = model_name in ("Image-only", "Hybrid")
        if needs_photo and photometric_features is None:
            raise ValueError(f"{model_name} prediction requires photometric features.")
        if needs_image and image is None:
            raise ValueError(f"{model_name} prediction requires a galaxy image.")

        model = self._load_model(model_name)
        image_batch = None
        photo_batch = None
        size = int(self.config["preprocessing"]["image_size"])
        if needs_image:
            image_batch = np.expand_dims(preprocess_image(image, size), axis=0)
        if needs_photo:
            columns = self.config["data"]["photometric_feature_columns"]
            missing = [column for column in columns if column not in photometric_features]
            if missing:
                raise ValueError(f"Missing photometric feature(s): {', '.join(missing)}")
            try:
                row = {column: float(photometric_features[column]) for column in columns}
            except (TypeError, ValueError) as exc:
                raise ValueError("Photometric features must be finite numeric values.") from exc
            if not np.isfinite(list(row.values())).all():
                raise ValueError("Photometric features must be finite numeric values.")
            import pandas as pd

            photo_batch = self._load_preprocessor().transform(pd.DataFrame([row]))

        if model_name == "Photometric-only":
            prediction = model.predict(photo_batch, verbose=0)
        elif model_name == "Image-only":
            prediction = model.predict(image_batch, verbose=0)
        else:
            prediction = model.predict([image_batch, photo_batch], verbose=0)
        value = float(np.asarray(prediction).reshape(-1)[0])
        if not np.isfinite(value):
            raise ValueError("Model returned a non-finite redshift estimate.")
        return value

