"""Reusable regression metrics and evaluation report generation."""

from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(
    actual: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Return MAE, RMSE, and R² after validating paired finite values."""
    y_true = np.asarray(actual, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        raise ValueError("Cannot calculate metrics for empty arrays.")
    if y_true.size < 2:
        raise ValueError("At least two paired values are required to calculate R².")
    if y_true.shape != y_pred.shape:
        raise ValueError("Actual and predicted values must have the same number of items.")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Actual and predicted values must all be finite.")
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def evaluate_model(
    model,
    inputs,
    actual: Sequence[float] | np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    """Predict using a Keras-compatible model and return metrics plus predictions."""
    predicted = np.asarray(model.predict(inputs, verbose=0)).reshape(-1)
    return calculate_metrics(actual, predicted), predicted


def save_comparison_report(
    results: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    output_directory: str | Path,
    figure_directory: str | Path | None = None,
) -> pd.DataFrame:
    """Save a metrics CSV and actual-vs-predicted/residual plots for each model."""
    if not results:
        raise ValueError("At least one model result is required.")
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    figures = Path(figure_directory) if figure_directory is not None else destination
    figures.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []
    for name, (actual, predicted) in results.items():
        metrics = calculate_metrics(actual, predicted)
        rows.append({"model": name, **metrics})
        _save_plots(name, np.asarray(actual), np.asarray(predicted), figures)
    comparison = pd.DataFrame(rows, columns=["model", "MAE", "RMSE", "R2"])
    comparison.to_csv(destination / "model_comparison.csv", index=False)
    return comparison


def _save_plots(name: str, actual: np.ndarray, predicted: np.ndarray, directory: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "model"
    residuals = actual - predicted
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(actual, predicted, alpha=0.6)
    lower = float(min(actual.min(), predicted.min()))
    upper = float(max(actual.max(), predicted.max()))
    axis.plot([lower, upper], [lower, upper], linestyle="--", color="black")
    axis.set(xlabel="Actual redshift", ylabel="Predicted redshift", title=f"{name}: actual vs predicted")
    figure.tight_layout()
    figure.savefig(directory / f"{safe_name}_actual_vs_predicted.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.scatter(predicted, residuals, alpha=0.6)
    axis.axhline(0, linestyle="--", color="black")
    axis.set(xlabel="Predicted redshift", ylabel="Residual (actual - predicted)", title=f"{name}: residuals")
    figure.tight_layout()
    figure.savefig(directory / f"{safe_name}_residuals.png", dpi=150)
    plt.close(figure)
