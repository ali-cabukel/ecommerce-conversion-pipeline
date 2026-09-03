"""Train a baseline conversion model and log it to MLflow."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from training.config import (
    EXPERIMENT_NAME,
    FEATURE_COLUMNS,
    FEATURE_NAMES_PATH,
    LABEL_COL,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_PATH,
    PROCESSED_PATH,
    RANDOM_SEED,
    RAW_DIR,
    TEST_FRACTION,
)
from training.dataset import build_training_table

logger = logging.getLogger(__name__)


def _load_or_build_sessions(
    processed_path: Path,
    raw_dir: Path,
    rebuild: bool,
    seed: int,
    max_orders: int | None,
) -> tuple[pd.DataFrame, dict]:
    if processed_path.exists() and not rebuild:
        logger.info("Loading training table from %s", processed_path)
        sessions = pd.read_parquet(processed_path)
        meta_path = processed_path.with_name("sessions_meta.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return sessions, meta
    return build_training_table(
        raw_dir=raw_dir,
        output_path=processed_path,
        seed=seed,
        max_orders=max_orders,
    )


def time_split(
    sessions: pd.DataFrame, test_fraction: float = TEST_FRACTION
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = sessions.sort_values("session_ts").reset_index(drop=True)
    cut = int(len(ordered) * (1.0 - test_fraction))
    if cut <= 0 or cut >= len(ordered):
        raise ValueError("Not enough rows for a time-based train/test split")
    return ordered.iloc[:cut], ordered.iloc[cut:]


def _metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (proba >= threshold).astype(np.int32)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "log_loss": float(log_loss(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "positive_rate": float(np.mean(pred)),
        "base_rate": float(np.mean(y_true)),
    }


def train_model(
    sessions: pd.DataFrame,
    seed: int = RANDOM_SEED,
    test_fraction: float = TEST_FRACTION,
    compute_importance: bool = True,
) -> tuple[HistGradientBoostingClassifier, dict, pd.DataFrame]:
    train_df, test_df = time_split(sessions, test_fraction=test_fraction)
    X_train = train_df[FEATURE_COLUMNS].astype("float64")
    y_train = train_df[LABEL_COL].to_numpy()
    X_test = test_df[FEATURE_COLUMNS].astype("float64")
    y_test = test_df[LABEL_COL].to_numpy()

    model = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=200,
        l2_regularization=0.1,
        min_samples_leaf=40,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=seed,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    train_proba = model.predict_proba(X_train)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]
    metrics = {f"train_{k}": v for k, v in _metrics(y_train, train_proba).items()}
    metrics.update({f"test_{k}": v for k, v in _metrics(y_test, test_proba).items()})
    metrics["n_train"] = float(len(train_df))
    metrics["n_test"] = float(len(test_df))
    metrics["n_features"] = float(len(FEATURE_COLUMNS))

    if compute_importance:
        sample = min(8_000, len(X_test))
        perm = permutation_importance(
            model,
            X_test.iloc[:sample],
            y_test[:sample],
            n_repeats=5,
            scoring="roc_auc",
            random_state=seed,
            n_jobs=1,
        )
        importance = (
            pd.DataFrame(
                {
                    "feature": FEATURE_COLUMNS,
                    "importance_mean": perm.importances_mean,
                    "importance_std": perm.importances_std,
                }
            )
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )
    else:
        importance = pd.DataFrame(
            {"feature": FEATURE_COLUMNS, "importance_mean": 0.0, "importance_std": 0.0}
        )
    return model, metrics, importance


def persist(
    model: HistGradientBoostingClassifier,
    metrics: dict,
    importance: pd.DataFrame,
    meta: dict,
) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "label": LABEL_COL,
            "metrics": metrics,
        },
        MODEL_PATH,
    )
    METRICS_PATH.write_text(json.dumps({**meta, **metrics}, indent=2))
    FEATURE_NAMES_PATH.write_text(json.dumps(FEATURE_COLUMNS, indent=2))
    importance.to_csv(MODEL_DIR / "feature_importance.csv", index=False)
    logger.info("Saved model to %s", MODEL_PATH)


def log_mlflow(
    model: HistGradientBoostingClassifier,
    metrics: dict,
    importance: pd.DataFrame,
    meta: dict,
    seed: int,
    sample_input: pd.DataFrame,
) -> str:
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="hist_gbm_baseline") as run:
        mlflow.log_param("model", "HistGradientBoostingClassifier")
        mlflow.log_param("seed", seed)
        mlflow.log_param("data_source", meta.get("source", "unknown"))
        mlflow.log_param("n_sessions", meta.get("n_sessions", metrics.get("n_train", 0) + metrics.get("n_test", 0)))
        mlflow.log_param("features", ",".join(FEATURE_COLUMNS))
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.log_text(importance.to_csv(index=False), "feature_importance.csv")
        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=sample_input.head(5),
        )
        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_artifact(str(METRICS_PATH))
        return run.info.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the conversion baseline")
    parser.add_argument("--processed", type=Path, default=PROCESSED_PATH)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--rebuild-data", action="store_true")
    parser.add_argument("--max-orders", type=int, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    parser.add_argument("--skip-mlflow", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sessions, meta = _load_or_build_sessions(
        processed_path=args.processed,
        raw_dir=args.raw_dir,
        rebuild=args.rebuild_data,
        seed=args.seed,
        max_orders=args.max_orders,
    )
    logger.info(
        "Training on %s sessions (conversion_rate=%.3f)",
        len(sessions),
        sessions[LABEL_COL].mean(),
    )

    model, metrics, importance = train_model(
        sessions, seed=args.seed, test_fraction=args.test_fraction
    )
    persist(model, metrics, importance, meta)

    logger.info("test_roc_auc=%.4f  test_pr_auc=%.4f  test_brier=%.4f", metrics["test_roc_auc"], metrics["test_pr_auc"], metrics["test_brier"])
    logger.info("Top features:\n%s", importance.head(5).to_string(index=False))

    if not args.skip_mlflow:
        try:
            run_id = log_mlflow(
                model,
                metrics,
                importance,
                meta,
                seed=args.seed,
                sample_input=sessions[FEATURE_COLUMNS].astype("float64"),
            )
            logger.info("MLflow run_id=%s", run_id)
        except Exception:
            logger.exception("MLflow logging failed; model artifacts are still on disk")

    print(json.dumps({k: metrics[k] for k in ("test_roc_auc", "test_pr_auc", "test_brier", "n_train", "n_test")}, indent=2))


if __name__ == "__main__":
    main()
