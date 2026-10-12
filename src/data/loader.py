"""Validated loaders for photometric catalogues and image metadata."""

from pathlib import Path
from typing import Sequence

import pandas as pd


def _read_csv(path: str | Path, description: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"{description} CSV was not found: {csv_path}. "
            "Update the dataset path in configs/config.yaml."
        )
    try:
        return pd.read_csv(csv_path)
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not parse {description} CSV {csv_path}: {exc}") from exc


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], description: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{description} is missing required column(s): {', '.join(missing)}. "
            f"Available columns: {', '.join(map(str, frame.columns))}"
        )


def load_photometric_data(
    csv_path: str | Path,
    feature_columns: Sequence[str],
    target_column: str,
    object_id_column: str | None = None,
) -> pd.DataFrame:
    """Load a catalogue after validating configured features, target, and optional ID."""
    if not feature_columns:
        raise ValueError("At least one photometric feature column must be configured.")
    required = list(feature_columns) + [target_column]
    if object_id_column:
        required.append(object_id_column)
    frame = _read_csv(csv_path, "Photometric catalogue")
    _require_columns(frame, required, "Photometric catalogue")
    return frame


def load_image_metadata(
    csv_path: str | Path,
    object_id_column: str,
    image_path_column: str,
    require_files: bool = False,
) -> pd.DataFrame:
    """Load image-path metadata and optionally verify every referenced file exists."""
    frame = _read_csv(csv_path, "Image metadata")
    _require_columns(frame, [object_id_column, image_path_column], "Image metadata")
    paths = frame[image_path_column].astype("string").str.strip()
    if paths.isna().any() or paths.eq("").any():
        raise ValueError(
            f"Image metadata column '{image_path_column}' contains missing or blank paths."
        )
    frame = frame.copy()
    frame[image_path_column] = paths
    if require_files:
        missing = [value for value in paths if not Path(str(value)).is_file()]
        if missing:
            sample = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"{len(missing)} image file(s) referenced by '{image_path_column}' "
                f"do not exist; examples: {sample}"
            )
    return frame

