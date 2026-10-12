import numpy as np
import pandas as pd
import pytest

from src.data.dataset_merger import merge_datasets
from src.data.preprocessing import FeaturePreprocessor, clean_records, split_dataset


def test_clean_records_coerces_invalid_features_and_removes_invalid_targets():
    frame = pd.DataFrame(
        {
            "objid": [1, 2, 3, 3],
            "flux": ["1.5", "bad", "3.5", "9.0"],
            "redshift": ["0.1", "0.2", "inf", "0.4"],
        }
    )
    cleaned = clean_records(frame, ["flux"], "redshift", "objid")
    assert cleaned["objid"].tolist() == [1, 2, 3]
    assert cleaned.loc[1, "flux"] != cleaned.loc[1, "flux"]


def test_feature_preprocessor_fits_training_only_and_imputes_missing_values():
    train = pd.DataFrame({"u": [1.0, np.nan, 3.0], "g": [10.0, 20.0, 30.0]})
    validation = pd.DataFrame({"u": [1000.0], "g": [40.0]})
    preprocessor = FeaturePreprocessor().fit(train, ["u", "g"])
    transformed = preprocessor.transform(validation)
    assert transformed.shape == (1, 2)
    assert transformed[0, 0] > 100
    assert np.isfinite(transformed).all()


def test_feature_preprocessor_imputes_a_feature_missing_from_entire_validation_batch():
    train = pd.DataFrame({"u": [1.0, 3.0], "g": [10.0, 30.0]})
    validation = pd.DataFrame({"u": [np.nan], "g": [20.0]})
    transformed = FeaturePreprocessor().fit(train, ["u", "g"]).transform(validation)
    assert transformed.shape == (1, 2)
    assert transformed[0, 0] == pytest.approx(0.0)


def test_missing_feature_column_has_helpful_error():
    preprocessor = FeaturePreprocessor()
    with pytest.raises(RuntimeError, match="fitted"):
        preprocessor.transform(pd.DataFrame({"u": [1.0]}))


def test_invalid_numeric_values_are_rejected_by_feature_preprocessor():
    with pytest.raises(ValueError, match="no valid numeric values"):
        FeaturePreprocessor().fit(pd.DataFrame({"u": ["bad", None]}), ["u"])


def test_splits_are_disjoint_and_reproducible():
    frame = pd.DataFrame({"x": range(100)})
    splits = split_dataset(frame, random_seed=12)
    repeated = split_dataset(frame, random_seed=12)
    assert [len(part) for part in splits] == [70, 15, 15]
    assert [part["x"].tolist() for part in splits] == [
        part["x"].tolist() for part in repeated
    ]
    assert not set(splits[0].x) & set(splits[1].x)
    assert not set(splits[0].x) & set(splits[2].x)
    assert not set(splits[1].x) & set(splits[2].x)


def test_split_ratios_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        split_dataset(pd.DataFrame({"x": range(10)}), 0.8, 0.1, 0.2)


def test_merge_requires_explicit_source_compatibility_verification():
    left = pd.DataFrame({"objid": [1], "redshift": [0.1]})
    right = pd.DataFrame({"object_id": [1], "image_path": ["galaxy.jpg"]})
    with pytest.raises(ValueError, match="verify catalogue provenance"):
        merge_datasets(left, right, "objid", "object_id")


def test_merge_reports_exact_matches_and_rejects_duplicate_ids():
    left = pd.DataFrame({"objid": [1, 2], "redshift": [0.1, 0.2]})
    right = pd.DataFrame({"object_id": ["1", "3"], "image_path": ["a.jpg", "b.jpg"]})
    merged, report = merge_datasets(
        left, right, "objid", "object_id", source_compatibility_verified=True
    )
    assert len(merged) == 1
    assert report.matched_records == 1
    assert report.photometric_match_rate == 0.5
    duplicate_images = pd.concat([right, right.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate object identifiers"):
        merge_datasets(
            left, duplicate_images, "objid", "object_id",
            source_compatibility_verified=True,
        )
