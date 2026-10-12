import numpy as np
import pytest

from src.evaluation.metrics import calculate_metrics


def test_metrics_match_known_values():
    result = calculate_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 4.0])
    assert result["MAE"] == pytest.approx(1 / 3)
    assert result["RMSE"] == pytest.approx(np.sqrt(1 / 3))
    assert result["R2"] == pytest.approx(0.5)


def test_metrics_reject_empty_mismatched_and_nonfinite_inputs():
    with pytest.raises(ValueError, match="empty"):
        calculate_metrics([], [])
    with pytest.raises(ValueError, match="At least two"):
        calculate_metrics([1.0], [1.0])
    with pytest.raises(ValueError, match="same number"):
        calculate_metrics([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="finite"):
        calculate_metrics([1.0], [np.nan])
