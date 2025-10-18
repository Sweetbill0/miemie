"""Command line interface for the short-term wind power forecasting baseline."""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import List

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_processing import (
        FeatureMatrix,
        build_short_term_features,
        fill_missing_values,
        load_short_term_data,
        save_csv,
        save_json,
    )
    from model import RidgeRegression, StandardScaler
else:
    from .data_processing import (
        FeatureMatrix,
        build_short_term_features,
        fill_missing_values,
        load_short_term_data,
        save_csv,
        save_json,
    )
    from .model import RidgeRegression, StandardScaler


def _create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-path",
        default="changzhan/短期预测/train.csv",
        help="Path to the training CSV file.",
    )
    parser.add_argument(
        "--history-steps",
        type=int,
        default=16,
        help="Number of past 15-minute observations to include in each feature window.",
    )
    parser.add_argument(
        "--forecast-steps",
        type=int,
        default=4,
        help="Number of future 15-minute targets to predict.",
    )
    parser.add_argument(
        "--validation-samples",
        type=int,
        default=24 * 4,
        help="Number of samples to reserve for validation (default: one day).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where artefacts such as predictions and model parameters are stored.",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=1e-2,
        help="L2 regularisation strength for the ridge regression baseline.",
    )
    return parser


def _train_models(
    feature_matrix: FeatureMatrix,
    validation_samples: int,
    regularization: float,
) -> dict:
    sample_count = len(feature_matrix.features)
    if validation_samples >= sample_count:
        raise ValueError("Validation set must be smaller than the total sample count.")
    train_end = sample_count - validation_samples
    train_features = feature_matrix.features[:train_end]
    validation_features = feature_matrix.features[train_end:]
    train_targets = feature_matrix.targets[:train_end]
    validation_targets = feature_matrix.targets[train_end:]
    timestamps = feature_matrix.timestamps

    scaler = StandardScaler()
    scaler.fit_list(train_features)
    transformed_train = _add_intercept(scaler.transform(train_features))
    transformed_validation = _add_intercept(scaler.transform(validation_features))

    horizons = len(train_targets[0]) if train_targets else 0
    models: List[RidgeRegression] = []
    validation_predictions: List[List[float]] = [
        [0.0 for _ in range(len(validation_features))] for _ in range(horizons)
    ]
    metrics = []
    for horizon_index in range(horizons):
        model = RidgeRegression(regularization=regularization)
        model.fit(
            transformed_train,
            [targets[horizon_index] for targets in train_targets],
        )
        models.append(model)
        predictions = model.predict(transformed_validation)
        validation_predictions[horizon_index] = predictions
        actuals = [targets[horizon_index] for targets in validation_targets]
        mae = _mean_absolute_error(predictions, actuals)
        rmse = _root_mean_squared_error(predictions, actuals)
        metrics.append({"mae": mae, "rmse": rmse})

    prediction_rows = _format_validation_predictions(
        timestamps[-len(validation_features) :],
        validation_targets,
        validation_predictions,
    )

    model_artifact = {
        "history_steps": feature_matrix.history_steps,
        "forecast_steps": feature_matrix.forecast_steps,
        "feature_names": feature_matrix.feature_names,
        "scaler_means": scaler.means,
        "scaler_stds": scaler.stds,
        "models": [model.coefficients for model in models],
        "validation_metrics": metrics,
    }
    return {
        "metrics": metrics,
        "prediction_rows": prediction_rows,
        "model_artifact": model_artifact,
        "scaler": scaler,
    }


def _add_intercept(rows: List[List[float]]) -> List[List[float]]:
    return [[1.0] + row for row in rows]


def _mean_absolute_error(predictions: List[float], actuals: List[float]) -> float:
    total = 0.0
    for pred, actual in zip(predictions, actuals):
        total += abs(pred - actual)
    return total / len(predictions) if predictions else float("nan")


def _root_mean_squared_error(predictions: List[float], actuals: List[float]) -> float:
    total = 0.0
    for pred, actual in zip(predictions, actuals):
        diff = pred - actual
        total += diff * diff
    return math.sqrt(total / len(predictions)) if predictions else float("nan")


def _format_validation_predictions(
    timestamps: List,
    validation_targets: List[List[float]],
    validation_predictions: List[List[float]],
) -> List[dict]:
    horizon_minutes = [15 * (index + 1) for index in range(len(validation_predictions))]
    rows: List[dict] = []
    for sample_idx, timestamp in enumerate(timestamps):
        row = {
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M"),
        }
        for horizon_index, minutes_ahead in enumerate(horizon_minutes):
            row[f"prediction_{minutes_ahead}m"] = round(
                validation_predictions[horizon_index][sample_idx], 4
            )
            row[f"actual_{minutes_ahead}m"] = round(
                validation_targets[sample_idx][horizon_index], 4
            )
        rows.append(row)
    return rows


def main() -> None:
    parser = _create_argument_parser()
    args = parser.parse_args()

    rows = load_short_term_data(args.train_path)
    fill_missing_values(rows)
    feature_matrix = build_short_term_features(
        rows,
        history_steps=args.history_steps,
        forecast_steps=args.forecast_steps,
    )

    artefacts = _train_models(
        feature_matrix,
        validation_samples=args.validation_samples,
        regularization=args.regularization,
    )

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "validation_metrics.json")
    predictions_path = os.path.join(output_dir, "validation_predictions.csv")
    model_path = os.path.join(output_dir, "ridge_regression_model.json")

    save_json({"metrics": artefacts["metrics"]}, metrics_path)
    if artefacts["prediction_rows"]:
        fieldnames = list(artefacts["prediction_rows"][0].keys())
        save_csv(artefacts["prediction_rows"], predictions_path, fieldnames)
    save_json(artefacts["model_artifact"], model_path)

    print("Validation metrics saved to", metrics_path)
    for horizon_index, metric in enumerate(artefacts["metrics"], start=1):
        print(
            f"Horizon +{15 * horizon_index} minutes -> "
            f"MAE: {metric['mae']:.3f}, RMSE: {metric['rmse']:.3f}"
        )


if __name__ == "__main__":
    main()
