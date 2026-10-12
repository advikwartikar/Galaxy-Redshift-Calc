import pandas as pd
import pytest

from src.data.loader import load_image_metadata, load_photometric_data


def test_photometric_loader_validates_configured_columns(tmp_path):
    path = tmp_path / "catalogue.csv"
    pd.DataFrame({"g": [1.0], "z": [0.2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required column"):
        load_photometric_data(path, ["g", "r"], "z", "objid")


def test_image_metadata_loader_rejects_missing_files_when_requested(tmp_path):
    path = tmp_path / "images.csv"
    pd.DataFrame({"objid": [1], "image_path": ["not-here.jpg"]}).to_csv(
        path, index=False
    )
    with pytest.raises(FileNotFoundError, match="image file"):
        load_image_metadata(path, "objid", "image_path", require_files=True)


def test_image_metadata_loader_rejects_blank_paths(tmp_path):
    path = tmp_path / "images.csv"
    pd.DataFrame({"objid": [1], "image_path": [" "]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing or blank"):
        load_image_metadata(path, "objid", "image_path")

