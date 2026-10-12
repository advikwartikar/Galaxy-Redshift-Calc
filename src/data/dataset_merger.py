"""Explicit exact-ID merging with match diagnostics and strict validation."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MatchReport:
    """Counts describing an exact identifier join."""

    photometric_records: int
    image_records: int
    matched_records: int
    photometric_match_rate: float
    image_match_rate: float


def _canonical_identifier(value: object) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    return text or None


def merge_datasets(
    photometric: pd.DataFrame,
    images: pd.DataFrame,
    photometric_id_column: str,
    image_id_column: str,
    *,
    source_compatibility_verified: bool = False,
) -> tuple[pd.DataFrame, MatchReport]:
    """Inner-join catalogues only after caller verifies identifier provenance.

    Exact identifier equality is not a substitute for astronomical
    cross-identification. The caller must confirm source, release, and identifier
    semantics, or provide a separately verified coordinate cross-match first.
    """
    if not source_compatibility_verified:
        raise ValueError(
            "Refusing to match datasets: verify catalogue provenance and identifier "
            "compatibility first, then explicitly set source_compatibility_verified=True."
        )
    for frame, column, label in (
        (photometric, photometric_id_column, "Photometric catalogue"),
        (images, image_id_column, "Image metadata"),
    ):
        if column not in frame.columns:
            raise ValueError(f"{label} is missing identifier column '{column}'.")

    left = photometric.copy()
    right = images.copy()
    left_key = "__photometric_join_id"
    right_key = "__image_join_id"
    if left_key in left.columns or right_key in right.columns:
        raise ValueError("Input data uses a reserved internal join-column name.")
    left[left_key] = left[photometric_id_column].map(_canonical_identifier)
    right[right_key] = right[image_id_column].map(_canonical_identifier)
    for frame, key, label in (
        (left, left_key, "Photometric catalogue"),
        (right, right_key, "Image metadata"),
    ):
        if frame[key].isna().any():
            raise ValueError(f"{label} contains missing or blank object identifiers.")
        duplicates = frame[key].duplicated(keep=False)
        if duplicates.any():
            count = int(duplicates.sum())
            raise ValueError(
                f"{label} contains {count} rows with duplicate object identifiers; "
                "resolve duplicates before matching."
            )

    merged = left.merge(
        right,
        left_on=left_key,
        right_on=right_key,
        how="inner",
        validate="one_to_one",
        suffixes=("_photometric", "_image"),
    )
    merged = merged.drop(columns=[left_key, right_key])
    matched = len(merged)
    report = MatchReport(
        photometric_records=len(left),
        image_records=len(right),
        matched_records=matched,
        photometric_match_rate=matched / len(left) if len(left) else 0.0,
        image_match_rate=matched / len(right) if len(right) else 0.0,
    )
    return merged, report

