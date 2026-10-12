"""Leakage-safe tabular and image preprocessing helpers."""

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler


class FeaturePreprocessor:
    """Median imputation and standardization fitted exclusively on training data."""

    def __init__(self) -> None:
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.feature_columns: list[str] | None = None

    def fit(self, training_frame: pd.DataFrame, feature_columns: Sequence[str]) -> "FeaturePreprocessor":
        self.feature_columns = list(feature_columns)
        if not self.feature_columns:
            raise ValueError("At least one feature column is required.")
        _validate_numeric_frame(training_frame, self.feature_columns)
        numeric = _numeric_feature_frame(training_frame, self.feature_columns)
        imputed = self.imputer.fit_transform(numeric)
        self.scaler.fit(imputed)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.feature_columns is None:
            raise RuntimeError("FeaturePreprocessor must be fitted on training data first.")
        _validate_numeric_frame(frame, self.feature_columns, require_observed=False)
        numeric = _numeric_feature_frame(frame, self.feature_columns)
        values = self.imputer.transform(numeric)
        return self.scaler.transform(values).astype(np.float32)

    def fit_transform(
        self, training_frame: pd.DataFrame, feature_columns: Sequence[str]
    ) -> np.ndarray:
        return self.fit(training_frame, feature_columns).transform(training_frame)


def _validate_numeric_frame(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    require_observed: bool = True,
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature column(s): {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Cannot preprocess an empty dataset.")
    for column in columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        finite_or_missing = numeric.isna() | np.isfinite(numeric)
        if not finite_or_missing.all():
            raise ValueError(f"Feature '{column}' contains an invalid infinite value.")
        if require_observed and numeric.notna().sum() == 0:
            raise ValueError(f"Feature '{column}' has no valid numeric values.")


def _numeric_feature_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    numeric = frame[list(columns)].apply(pd.to_numeric, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan)


def clean_records(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    object_id_column: str | None = None,
) -> pd.DataFrame:
    """Coerce numeric fields, remove unusable targets, and deduplicate by object ID."""
    missing = [column for column in [*feature_columns, target_column] if column not in frame]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")
    cleaned = frame.copy()
    for column in [*feature_columns, target_column]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        cleaned.loc[~np.isfinite(cleaned[column]), column] = np.nan
    cleaned = cleaned.dropna(subset=[target_column])
    if object_id_column:
        if object_id_column not in cleaned.columns:
            raise ValueError(f"Missing object identifier column '{object_id_column}'.")
        cleaned = cleaned.dropna(subset=[object_id_column])
        cleaned = cleaned.drop_duplicates(subset=[object_id_column], keep="first")
    cleaned = cleaned.reset_index(drop=True)
    if cleaned.empty:
        raise ValueError("No valid records remain after cleaning missing/invalid targets.")
    return cleaned


def split_dataset(
    frame: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    group_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create deterministic, disjoint train/validation/test splits."""
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(value <= 0 for value in ratios) or not np.isclose(sum(ratios), 1.0):
        raise ValueError("Train, validation, and test ratios must be positive and sum to 1.")
    if len(frame) < 3:
        raise ValueError("At least three records are required for train/validation/test splits.")
    indices = np.arange(len(frame))
    if group_column is None:
        train_idx, remaining_idx = train_test_split(
            indices, test_size=validation_ratio + test_ratio,
            random_state=random_seed, shuffle=True,
        )
        relative_test = test_ratio / (validation_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            remaining_idx, test_size=relative_test, random_state=random_seed, shuffle=True
        )
    else:
        if group_column not in frame:
            raise ValueError(f"Missing group column '{group_column}'.")
        groups = frame[group_column].to_numpy()
        if pd.isna(groups).any():
            raise ValueError(f"Group column '{group_column}' contains missing values.")
        if pd.Series(groups).nunique() < 3:
            raise ValueError(
                f"At least three distinct groups are required to create three splits "
                f"using '{group_column}'."
            )
        first = GroupShuffleSplit(n_splits=1, test_size=validation_ratio + test_ratio,
                                  random_state=random_seed)
        train_pos, remaining_pos = next(first.split(indices, groups=groups))
        remaining_groups = groups[remaining_pos]
        relative_test = test_ratio / (validation_ratio + test_ratio)
        second = GroupShuffleSplit(n_splits=1, test_size=relative_test,
                                   random_state=random_seed)
        val_pos, test_pos = next(
            second.split(remaining_pos, groups=remaining_groups)
        )
        train_idx = indices[train_pos]
        val_idx = indices[remaining_pos[val_pos]]
        test_idx = indices[remaining_pos[test_pos]]
    return tuple(
        frame.iloc[index].reset_index(drop=True)
        for index in (train_idx, val_idx, test_idx)
    )


def preprocess_image(
    image: Image.Image | str | Path,
    image_size: int | tuple[int, int] = 224,
) -> np.ndarray:
    """Resize an image and apply TensorFlow's ResNet50 preprocessing function."""
    if isinstance(image_size, int):
        size = (image_size, image_size)
    else:
        size = image_size
    if len(size) != 2 or any(dimension <= 0 for dimension in size):
        raise ValueError("image_size must contain two positive dimensions.")
    try:
        if isinstance(image, (str, Path)):
            with Image.open(image) as opened:
                rgb = opened.convert("RGB").resize((size[1], size[0]))
        else:
            rgb = image.convert("RGB").resize((size[1], size[0]))
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Could not read galaxy image: {exc}") from exc
    pixels = np.asarray(rgb, dtype=np.float32)
    try:
        from tensorflow.keras.applications.resnet50 import preprocess_input
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required to apply ResNet50 preprocessing. "
            "Install dependencies from requirements.txt."
        ) from exc
    return preprocess_input(pixels)
