"""Reproduce the ETT oil-temperature nowcast and forecast experiments.

Run from the repository root:
    python src/run_analysis.py --data-dir data --output-dir results

The evaluation uses rolling-origin forecasts. At every origin t, only observed
values available at t and deterministic calendar features for t+h are used.
The h=0 experiment is a same-time nowcast: load features at t are allowed, but
OT(t) is never used as an input. The h>0 experiments may use observed OT(t).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Keep the experiment sequential and avoid joblib's WMIC core-count probe on Windows.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2")
LOAD_COLUMNS = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL")
TARGET = "OT"
RANDOM_STATE = 42
SPLIT_TRAIN = 0.60
SPLIT_VALIDATION = 0.80


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_factories() -> dict[str, Any]:
    return {
        "Ridge(alpha=10)": lambda: make_pipeline(
            StandardScaler(), Ridge(alpha=10.0)
        ),
        "HistGradientBoostingRegressor": lambda: HistGradientBoostingRegressor(
            learning_rate=0.07,
            max_iter=180,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
    }


def load_dataset(data_dir: Path, name: str) -> tuple[pd.DataFrame, Path]:
    path = data_dir / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. See README.md for the source.")
    frame = pd.read_csv(path, parse_dates=["date"])
    required = ["date", *LOAD_COLUMNS, TARGET]
    missing_columns = sorted(set(required) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"{name} is missing required columns: {missing_columns}")
    return frame[required].copy(), path


def cadence_minutes(frame: pd.DataFrame) -> int:
    differences = frame["date"].diff().dropna()
    if differences.empty:
        raise ValueError("A dataset must contain at least two rows.")
    seconds = differences.dt.total_seconds().to_numpy()
    unique, counts = np.unique(seconds, return_counts=True)
    modal_seconds = int(unique[np.argmax(counts)])
    if modal_seconds not in (900, 3600):
        raise ValueError(f"Unexpected modal interval: {modal_seconds} seconds")
    return modal_seconds // 60


def data_quality_and_eda(frame: pd.DataFrame, path: Path, name: str) -> tuple[dict, pd.DataFrame]:
    cadence = cadence_minutes(frame)
    target = frame[TARGET].astype(float)
    daily_lag = 24 * (60 // cadence)
    weekly_lag = 7 * daily_lag
    time_diffs = frame["date"].diff().dropna().dt.total_seconds() / 60
    quality = {
        "dataset": name,
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "cadence_minutes": cadence,
        "start": frame["date"].iloc[0].isoformat(sep=" "),
        "end": frame["date"].iloc[-1].isoformat(sep=" "),
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_timestamps": int(frame["date"].duplicated().sum()),
        "non_modal_intervals": int((time_diffs != cadence).sum()),
        "sha256": sha256_file(path),
    }
    eda = {
        "dataset": name,
        "rows": int(len(frame)),
        "cadence_minutes": cadence,
        "mean": float(target.mean()),
        "std": float(target.std(ddof=1)),
        "min": float(target.min()),
        "q01": float(target.quantile(0.01)),
        "q05": float(target.quantile(0.05)),
        "q25": float(target.quantile(0.25)),
        "median": float(target.median()),
        "q75": float(target.quantile(0.75)),
        "q95": float(target.quantile(0.95)),
        "q99": float(target.quantile(0.99)),
        "max": float(target.max()),
        "skew": float(target.skew()),
        "acf_1step": float(target.autocorr(1)),
        "acf_1day": float(target.autocorr(daily_lag)),
        "acf_1week": float(target.autocorr(weekly_lag)),
        "corr_HUFL": float(frame["HUFL"].corr(target)),
        "corr_HULL": float(frame["HULL"].corr(target)),
        "corr_MUFL": float(frame["MUFL"].corr(target)),
        "corr_MULL": float(frame["MULL"].corr(target)),
        "corr_LUFL": float(frame["LUFL"].corr(target)),
        "corr_LULL": float(frame["LULL"].corr(target)),
    }
    return quality, eda


def build_examples(frame: pd.DataFrame, horizon_steps: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Build horizon-specific rows without using measurements after origin t."""
    values = frame[TARGET].to_numpy(dtype=float)
    dates = pd.DatetimeIndex(frame["date"])
    loads = frame[list(LOAD_COLUMNS)].to_numpy(dtype=float)
    cadence = cadence_minutes(frame)
    samples_per_day = 24 * (60 // cadence)
    samples_per_week = 7 * samples_per_day

    feature_names = [f"{column}_at_origin" for column in LOAD_COLUMNS]
    if horizon_steps > 0:
        feature_names.append("OT_at_origin")
    feature_names += ["OT_lag_1", "OT_lag_2", "OT_lag_1day", "OT_lag_1week", "OT_mean_24h_asof_origin"]
    feature_names += ["target_hour_sin", "target_hour_cos", "target_week_sin", "target_week_cos"]

    origin_indices = np.arange(samples_per_week, len(frame) - horizon_steps, dtype=int)
    target_indices = origin_indices + horizon_steps
    rows = []
    baseline_persistence = []
    baseline_daily = []
    baseline_weekly = []

    for origin, target_index in zip(origin_indices, target_indices):
        row = list(loads[origin])
        if horizon_steps > 0:
            row.append(values[origin])
            lag_1, lag_2 = values[origin - 1], values[origin - 2]
            roll = values[origin - samples_per_day + 1 : origin + 1]
        else:
            # At h=0 the target is OT(origin), so only earlier OT observations are inputs.
            lag_1, lag_2 = values[origin - 1], values[origin - 2]
            roll = values[origin - samples_per_day : origin]
        lag_day = values[origin - samples_per_day]
        lag_week = values[origin - samples_per_week]
        row.extend([lag_1, lag_2, lag_day, lag_week, float(np.mean(roll))])

        target_date = dates[target_index]
        hour_fraction = (target_date.hour * 60 + target_date.minute) / (24 * 60)
        week_fraction = (target_date.dayofweek * 24 * 60 + target_date.hour * 60 + target_date.minute) / (7 * 24 * 60)
        row.extend(
            [
                np.sin(2 * np.pi * hour_fraction),
                np.cos(2 * np.pi * hour_fraction),
                np.sin(2 * np.pi * week_fraction),
                np.cos(2 * np.pi * week_fraction),
            ]
        )
        rows.append(row)
        # At h=0, origin is also the target time. Use the last available OT,
        # not OT(origin), as the persistence baseline to avoid target leakage.
        baseline_persistence.append(values[origin - 1] if horizon_steps == 0 else values[origin])
        baseline_daily.append(values[target_index - samples_per_day])
        baseline_weekly.append(values[target_index - samples_per_week])

    X = pd.DataFrame(rows, columns=feature_names)
    y = values[target_indices]
    baselines = {
        "Persistence": np.asarray(baseline_persistence),
        "Daily seasonal naive": np.asarray(baseline_daily),
        "Weekly seasonal naive": np.asarray(baseline_weekly),
    }
    return X, y, target_indices, baselines


def scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_dataset(frame: pd.DataFrame, name: str) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    cadence = cadence_minutes(frame)
    steps_per_hour = 60 // cadence
    horizons = ((0, "0h nowcast"), (6 * steps_per_hour, "6h ahead"), (24 * steps_per_hour, "24h ahead"))
    all_metrics: list[dict] = []
    importances: list[dict] = []
    split_records: list[dict] = []
    prediction_preview: list[dict] = []
    factories = model_factories()

    for horizon_steps, horizon_label in horizons:
        X, y, target_indices, baselines = build_examples(frame, horizon_steps)
        n_rows = len(frame)
        train_end = int(np.floor(SPLIT_TRAIN * n_rows))
        validation_end = int(np.floor(SPLIT_VALIDATION * n_rows))
        train_mask = target_indices < train_end
        validation_mask = (target_indices >= train_end) & (target_indices < validation_end)
        test_mask = target_indices >= validation_end

        X_train, y_train = X.loc[train_mask], y[train_mask]
        X_validation, y_validation = X.loc[validation_mask], y[validation_mask]
        X_train_validation, y_train_validation = X.loc[~test_mask], y[~test_mask]
        X_test, y_test = X.loc[test_mask], y[test_mask]
        if min(len(X_train), len(X_validation), len(X_test)) == 0:
            raise ValueError(f"An empty split was created for {name} {horizon_label}.")

        split_records.append(
            {
                "dataset": name,
                "horizon": horizon_label,
                "horizon_steps": horizon_steps,
                "train_samples": int(len(X_train)),
                "validation_samples": int(len(X_validation)),
                "test_samples": int(len(X_test)),
                "train_target_end": str(frame["date"].iloc[train_end - 1]),
                "validation_target_end": str(frame["date"].iloc[validation_end - 1]),
                "test_target_end": str(frame["date"].iloc[-1]),
                "features": list(X.columns),
            }
        )

        val_predictions: dict[str, np.ndarray] = {}
        for baseline_name, prediction in baselines.items():
            val_predictions[baseline_name] = prediction[validation_mask]
        validation_fitted_models: dict[str, Any] = {}
        for model_name, factory in factories.items():
            estimator = factory()
            estimator.fit(X_train, y_train)
            val_predictions[model_name] = estimator.predict(X_validation)
            validation_fitted_models[model_name] = estimator

        validation_scores = {
            model_name: scores(y_validation, prediction)
            for model_name, prediction in val_predictions.items()
        }
        selected_name = min(validation_scores, key=lambda key: validation_scores[key]["mae"])
        persistence_val_mae = validation_scores["Persistence"]["mae"]
        best_naive_name = min(baselines, key=lambda key: validation_scores[key]["mae"])

        test_predictions: dict[str, np.ndarray] = {
            baseline_name: prediction[test_mask]
            for baseline_name, prediction in baselines.items()
        }
        final_fitted_models: dict[str, Any] = {}
        for model_name, factory in factories.items():
            estimator = factory()
            estimator.fit(X_train_validation, y_train_validation)
            test_predictions[model_name] = estimator.predict(X_test)
            final_fitted_models[model_name] = estimator

        best_naive_test_mae = scores(y_test, test_predictions[best_naive_name])["mae"]
        for model_name, prediction in test_predictions.items():
            test_score = scores(y_test, prediction)
            val_score = validation_scores[model_name]
            all_metrics.append(
                {
                    "dataset": name,
                    "cadence_minutes": cadence,
                    "horizon": horizon_label,
                    "horizon_steps": horizon_steps,
                    "candidate": model_name,
                    "validation_mae": val_score["mae"],
                    "validation_rmse": val_score["rmse"],
                    "validation_r2": val_score["r2"],
                    "test_mae": test_score["mae"],
                    "test_rmse": test_score["rmse"],
                    "test_r2": test_score["r2"],
                    "test_mae_skill_vs_persistence": 1.0 - test_score["mae"] / max(
                        scores(y_test, test_predictions["Persistence"])["mae"], 1e-12
                    ),
                    "validation_best_naive": best_naive_name,
                    "test_mae_skill_vs_validation_best_naive": 1.0 - test_score["mae"] / max(best_naive_test_mae, 1e-12),
                    "selected_on_validation": model_name == selected_name,
                }
            )

        # Retain a seven-day slice of final test forecasts for an interpretable
        # observed-vs-predicted example in the report.
        preview_count = min(7 * (24 * steps_per_hour), len(y_test))
        test_target_indices = target_indices[test_mask]
        selected_test_prediction = test_predictions[selected_name]
        for row_number in range(preview_count):
            target_index = int(test_target_indices[row_number])
            prediction_preview.append(
                {
                    "dataset": name,
                    "horizon": horizon_label,
                    "selected_model": selected_name,
                    "target_date": str(frame["date"].iloc[target_index]),
                    "actual_ot": float(y_test[row_number]),
                    "predicted_ot": float(selected_test_prediction[row_number]),
                    "persistence_ot": float(test_predictions["Persistence"][row_number]),
                    "daily_seasonal_ot": float(test_predictions["Daily seasonal naive"][row_number]),
                }
            )

        # Explain the final model using held-out test rows; this does not affect selection.
        if selected_name in final_fitted_models:
            explanation = permutation_importance(
                final_fitted_models[selected_name],
                X_test,
                y_test,
                scoring="neg_mean_absolute_error",
                n_repeats=5,
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
            for feature_name, importance, std in sorted(
                zip(X.columns, explanation.importances_mean, explanation.importances_std),
                key=lambda item: item[1],
                reverse=True,
            ):
                importances.append(
                    {
                        "dataset": name,
                        "horizon": horizon_label,
                        "selected_model": selected_name,
                        "feature": feature_name,
                        "permutation_importance_mae_increase": float(importance),
                        "permutation_std": float(std),
                    }
                )

        split_records[-1]["selected_model_by_validation_mae"] = selected_name
        split_records[-1]["best_naive_by_validation_mae"] = best_naive_name
        split_records[-1]["validation_selected_mae"] = validation_scores[selected_name]["mae"]
        split_records[-1]["validation_persistence_mae"] = persistence_val_mae

    return all_metrics, importances, split_records, prediction_preview


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory with ETTh1/2 and ETTm1/2 CSVs")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Directory for reproducible results")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    quality_rows = []
    eda_rows = []
    histogram_rows = []
    metrics_rows = []
    importance_rows = []
    split_rows = []
    prediction_rows = []
    data_files = {}

    for name in DATASETS:
        frame, path = load_dataset(data_dir, name)
        quality, eda = data_quality_and_eda(frame, path, name)
        quality_rows.append(quality)
        eda_rows.append(eda)
        counts, edges = np.histogram(frame[TARGET].to_numpy(dtype=float), bins=np.arange(-10, 71, 10))
        for bin_index, count in enumerate(counts):
            histogram_rows.append(
                {
                    "dataset": name,
                    "bin_start": float(edges[bin_index]),
                    "bin_end": float(edges[bin_index + 1]),
                    "bin_center": float((edges[bin_index] + edges[bin_index + 1]) / 2),
                    "count": int(count),
                    "share": float(count / len(frame)),
                }
            )
        data_files[name] = {
            "path": path.name,
            "sha256": quality["sha256"],
            "rows": quality["rows"],
            "cadence_minutes": quality["cadence_minutes"],
        }
        metrics, importances, splits, predictions = evaluate_dataset(frame, name)
        metrics_rows.extend(metrics)
        importance_rows.extend(importances)
        split_rows.extend(splits)
        prediction_rows.extend(predictions)
        print(f"Completed {name}: {len(metrics)} model/horizon result rows")

    pd.DataFrame(quality_rows).to_csv(output_dir / "data_quality.csv", index=False)
    pd.DataFrame(eda_rows).to_csv(output_dir / "eda_summary.csv", index=False, float_format="%.6f")
    pd.DataFrame(histogram_rows).to_csv(output_dir / "target_histogram.csv", index=False, float_format="%.6f")
    pd.DataFrame(metrics_rows).to_csv(output_dir / "model_metrics.csv", index=False, float_format="%.6f")
    pd.DataFrame(importance_rows).to_csv(output_dir / "feature_importance.csv", index=False, float_format="%.6f")
    pd.DataFrame(prediction_rows).to_csv(output_dir / "test_predictions_preview.csv", index=False, float_format="%.6f")
    with (output_dir / "split_and_features.json").open("w", encoding="utf-8") as stream:
        json.dump(split_rows, stream, ensure_ascii=False, indent=2)

    manifest = {
        "experiment": "ETT transformer oil-temperature rolling-origin nowcast/forecast PoC",
        "target": TARGET,
        "target_unit": "not specified in supplied assignment data",
        "features": "six load variables at origin; lagged OT; known target-time calendar features",
        "horizons": ["0h nowcast", "6h ahead", "24h ahead"],
        "h0_rule": "OT at target time is excluded; contemporaneous load measurements at that time are included as allowed by assignment",
        "split": {"train_fraction": SPLIT_TRAIN, "validation_fraction_end": SPLIT_VALIDATION, "method": "chronological by target timestamp"},
        "selection": "lowest validation MAE among persistence, daily/weekly seasonal naive, Ridge, and HistGradientBoostingRegressor; final models refit on train+validation",
        "metrics": ["MAE", "RMSE", "R2"],
        "random_state": RANDOM_STATE,
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "datasets": data_files,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
