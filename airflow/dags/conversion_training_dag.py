"""Daily batch training DAG: dbt → Feast materialize → MLflow train → validate."""

from __future__ import annotations

import json
import os
import subprocess  # nosec B404
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow.decorators import task
from airflow.exceptions import AirflowException

from airflow import DAG

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/opt/airflow/project"))
FEAST_REPO = PROJECT_ROOT / "feast" / "feature_repo"
DBT_DIR = PROJECT_ROOT / "dbt"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"
DRIFT_METRICS_PATH = PROJECT_ROOT / "models" / "drift_metrics.json"
MIN_TEST_ROC_AUC = float(os.environ.get("CONVERSION_MIN_TEST_ROC_AUC", "0.65"))
FAIL_ON_DRIFT = os.environ.get("CONVERSION_FAIL_ON_DRIFT", "").lower() in {"1", "true", "yes"}


def _runtime_feast_yaml() -> Path:
    """Write a Feast config that points at compose service hostnames."""
    import yaml

    src = FEAST_REPO / "feature_store.yaml"
    dest = FEAST_REPO / "data" / "feature_store.runtime.yaml"
    dest.parent.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(src.read_text())
    offline = cfg.setdefault("offline_store", {})
    offline["host"] = os.environ.get("POSTGRES_HOST", offline.get("host", "localhost"))
    offline["port"] = int(os.environ.get("POSTGRES_PORT", offline.get("port", 5432)))
    offline["user"] = os.environ.get("POSTGRES_USER", offline.get("user", "conversion"))
    offline["password"] = os.environ.get("POSTGRES_PASSWORD", offline.get("password", "changethis"))
    offline["database"] = os.environ.get("POSTGRES_DB", offline.get("database", "conversion"))
    redis_host = os.environ.get("REDIS_HOST", "localhost")
    redis_port = os.environ.get("REDIS_PORT", "6379")
    cfg.setdefault("online_store", {})["connection_string"] = f"{redis_host}:{redis_port}"
    dest.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return dest


def _task_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["FEAST_FS_YAML_FILE_PATH"] = str(_runtime_feast_yaml())
    env.setdefault("MLFLOW_TRACKING_URI", f"file://{PROJECT_ROOT}/mlruns")
    return env


def _run(args: list[str], cwd: Path | None = None) -> None:
    completed = subprocess.run(  # nosec B603
        args,
        cwd=str(cwd or PROJECT_ROOT),
        env=_task_env(),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr)
    if completed.returncode != 0:
        raise AirflowException(
            f"Command {args} failed with exit {completed.returncode}\n{completed.stderr}"
        )


default_args = {
    "owner": "ml-platform",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="conversion_batch_training",
    description="Build warehouse features and retrain the conversion model",
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["training", "conversion", "batch"],
    default_args=default_args,
) as dag:

    @task
    def run_dbt() -> None:
        """Refresh staging views and feature marts in Postgres."""
        _run(["dbt", "run", "--profiles-dir", "."], cwd=DBT_DIR)

    @task
    def materialize_feast_features() -> None:
        """Register feature views and push offline features to Redis."""
        _run(["feast", "apply"], cwd=FEAST_REPO)
        end = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        _run(["feast", "materialize-incremental", end], cwd=FEAST_REPO)

    @task
    def train_conversion_model() -> str:
        """Train from Feast/Postgres and write models/metrics.json."""
        cmd = ["python", str(PROJECT_ROOT / "training" / "train.py")]
        max_sessions = os.environ.get("TRAIN_MAX_SESSIONS")
        if max_sessions:
            cmd.extend(["--max-sessions", max_sessions])
        _run(cmd)
        if not METRICS_PATH.exists():
            raise AirflowException(f"Training finished without writing {METRICS_PATH}")
        return str(METRICS_PATH)

    @task
    def detect_data_drift() -> dict:
        """Compare latest warehouse features to the last training reference (Evidently PSI)."""
        cmd = ["python", str(PROJECT_ROOT / "monitoring" / "drift.py"), "--from-warehouse"]
        max_sessions = os.environ.get("DRIFT_CURRENT_SESSIONS") or os.environ.get("TRAIN_MAX_SESSIONS")
        if max_sessions:
            cmd.extend(["--max-sessions", max_sessions])
        if FAIL_ON_DRIFT:
            cmd.append("--fail-on-drift")
        _run(cmd)
        if not DRIFT_METRICS_PATH.exists():
            raise AirflowException(f"Drift check finished without writing {DRIFT_METRICS_PATH}")
        return json.loads(DRIFT_METRICS_PATH.read_text())

    @task
    def validate_model(metrics_path: str) -> dict:
        """Fail the run if holdout ROC-AUC is below the promotion gate."""
        path = Path(metrics_path)
        if not path.exists():
            raise AirflowException(f"Metrics file not found: {path}")
        metrics = json.loads(path.read_text())
        auc = metrics.get("test_roc_auc")
        if auc is None:
            raise AirflowException(f"test_roc_auc missing from {path}")
        if float(auc) < MIN_TEST_ROC_AUC:
            raise AirflowException(
                f"test_roc_auc={auc:.4f} below gate {MIN_TEST_ROC_AUC:.2f}"
            )
        return {
            "test_roc_auc": float(auc),
            "test_pr_auc": float(metrics.get("test_pr_auc", 0.0)),
            "n_train": metrics.get("n_train"),
            "n_test": metrics.get("n_test"),
            "min_test_roc_auc": MIN_TEST_ROC_AUC,
        }

    dbt = run_dbt()
    feast = materialize_feast_features()
    drifted = detect_data_drift()
    trained = train_conversion_model()
    dbt >> feast >> drifted >> trained >> validate_model(trained)
