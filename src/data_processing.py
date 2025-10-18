"""Data loading and feature engineering helpers for wind power forecasting."""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Sequence


@dataclass
class TimeSeriesRow:
    """Container for a single timestamped observation."""

    timestamp: datetime
    nwp_wind_speed: float
    tower_wind_speed: float
    power: float


def load_short_term_data(path: str) -> List[TimeSeriesRow]:
    """Load the short-term forecasting dataset from ``train.csv``."""

    rows: List[TimeSeriesRow] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            timestamp = datetime.strptime(raw_row["time"], "%Y/%m/%d %H:%M")
            nwp_wind_speed = float(raw_row["nwp_ws"] or 0.0)
            tower_wind_speed = float(raw_row["tower_ws"] or 0.0)
            power = float(raw_row["power"] or 0.0)
            rows.append(
                TimeSeriesRow(
                    timestamp=timestamp,
                    nwp_wind_speed=nwp_wind_speed,
                    tower_wind_speed=tower_wind_speed,
                    power=power,
                )
            )
    return rows


def fill_missing_values(rows: List[TimeSeriesRow]) -> None:
    """In-place forward fill for missing values that may be encoded as NaN."""

    last_nwp = rows[0].nwp_wind_speed if rows else 0.0
    last_tower = rows[0].tower_wind_speed if rows else 0.0
    last_power = rows[0].power if rows else 0.0
    for row in rows:
        if math.isnan(row.nwp_wind_speed):
            row.nwp_wind_speed = last_nwp
        else:
            last_nwp = row.nwp_wind_speed
        if math.isnan(row.tower_wind_speed):
            row.tower_wind_speed = last_tower
        else:
            last_tower = row.tower_wind_speed
        if math.isnan(row.power):
            row.power = last_power
        else:
            last_power = row.power


@dataclass
class FeatureMatrix:
    features: List[List[float]]
    targets: List[List[float]]
    timestamps: List[datetime]
    feature_names: List[str]
    history_steps: int
    forecast_steps: int


def build_short_term_features(
    rows: Sequence[TimeSeriesRow],
    history_steps: int = 16,
    forecast_steps: int = 4,
) -> FeatureMatrix:
    """Create sliding window features for the short-term prediction task."""

    if history_steps < 1:
        raise ValueError("history_steps must be positive")
    if forecast_steps < 1:
        raise ValueError("forecast_steps must be positive")
    if len(rows) <= history_steps + forecast_steps:
        raise ValueError("Not enough rows to build the requested windows.")

    feature_rows: List[List[float]] = []
    targets: List[List[float]] = []
    timestamps: List[datetime] = []
    feature_names: List[str] | None = None

    for idx in range(history_steps - 1, len(rows) - forecast_steps):
        history = rows[idx - history_steps + 1 : idx + 1]
        forecast_window = rows[idx + 1 : idx + 1 + forecast_steps]
        base_time = history[-1].timestamp
        feature_vector: List[float]
        feature_vector, names = _extract_features(history, base_time)
        if feature_names is None:
            feature_names = names
        targets.append([row.power for row in forecast_window])
        feature_rows.append(feature_vector)
        timestamps.append(base_time)

    if feature_names is None:
        raise RuntimeError("Failed to generate feature names.")

    return FeatureMatrix(
        features=feature_rows,
        targets=targets,
        timestamps=timestamps,
        feature_names=feature_names,
        history_steps=history_steps,
        forecast_steps=forecast_steps,
    )


def _extract_features(history: Sequence[TimeSeriesRow], base_time: datetime) -> tuple[List[float], List[str]]:
    """Convert a window of observations into a feature vector."""

    values: List[float] = []
    names: List[str] = []
    window_length = len(history)
    for offset, row in enumerate(history):
        lag = window_length - 1 - offset
        suffix = f"t-{lag}" if lag else "t"
        values.extend([row.nwp_wind_speed, row.tower_wind_speed, row.power])
        names.extend(
            [
                f"nwp_wind_speed_{suffix}",
                f"tower_wind_speed_{suffix}",
                f"power_{suffix}",
            ]
        )
        values.append(row.tower_wind_speed - row.nwp_wind_speed)
        names.append(f"wind_speed_gap_{suffix}")
    # Aggregate statistics across the history window.
    for label, series in (
        ("nwp_wind_speed", [row.nwp_wind_speed for row in history]),
        ("tower_wind_speed", [row.tower_wind_speed for row in history]),
        ("power", [row.power for row in history]),
    ):
        mean_value = sum(series) / len(series)
        max_value = max(series)
        min_value = min(series)
        values.extend([mean_value, max_value, min_value, max_value - min_value])
        names.extend(
            [
                f"{label}_mean",
                f"{label}_max",
                f"{label}_min",
                f"{label}_range",
            ]
        )
    # Temporal encoding.
    minute_of_day = base_time.hour * 60 + base_time.minute
    day_of_week = base_time.weekday()
    seasonal_rad = 2.0 * math.pi * minute_of_day / (24 * 60)
    weekly_rad = 2.0 * math.pi * day_of_week / 7.0
    values.extend(
        [
            minute_of_day,
            math.sin(seasonal_rad),
            math.cos(seasonal_rad),
            day_of_week,
            math.sin(weekly_rad),
            math.cos(weekly_rad),
        ]
    )
    names.extend(
        [
            "minute_of_day",
            "minute_of_day_sin",
            "minute_of_day_cos",
            "day_of_week",
            "day_of_week_sin",
            "day_of_week_cos",
        ]
    )
    return values, names


def save_json(data: Dict[str, object], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def save_csv(rows: Sequence[Dict[str, object]], path: str, fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
