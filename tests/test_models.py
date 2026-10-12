import numpy as np
import pytest

from src.inference.predictor import RedshiftPredictor
from src.models.image_model import build_image_model
from src.models.multimodal_model import build_multimodal_model
from src.models.photometric_model import build_photometric_model


def test_model_input_and_output_shapes_without_pretrained_download():
    pytest.importorskip("tensorflow")
    photo = build_photometric_model(input_dim=5)
    image = build_image_model(image_size=32, weights=None)
    hybrid = build_multimodal_model(
        photometric_input_dim=5, image_size=32, weights=None
    )
    assert photo.input_shape == (None, 5)
    assert photo.output_shape == (None, 1)
    assert image.input_shape == (None, 32, 32, 3)
    assert image.output_shape == (None, 1)
    assert hybrid.input_shape == [(None, 32, 32, 3), (None, 5)]
    assert hybrid.output_shape == (None, 1)
    assert photo(np.zeros((2, 5), dtype=np.float32)).shape == (2, 1)


def test_missing_model_checkpoint_is_reported(tmp_path):
    predictor = RedshiftPredictor(
        {
            "models": {
                "photometric_checkpoint": str(tmp_path / "missing.keras"),
                "image_checkpoint": str(tmp_path / "image.keras"),
                "multimodal_checkpoint": str(tmp_path / "hybrid.keras"),
                "photometric_scaler": str(tmp_path / "scaler.joblib"),
            },
            "preprocessing": {"image_size": 224},
            "data": {"photometric_feature_columns": ["u"]},
        }
    )
    with pytest.raises(FileNotFoundError, match="has not been trained"):
        predictor._load_model("Photometric-only")
