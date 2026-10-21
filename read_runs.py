from pathlib import Path

try:
    import mlflow  # Reuse the MLflow module to read the results.
except ModuleNotFoundError as error:
    if error.name == "mlflow":
        raise SystemExit(
            "MLflow is not installed in this Python environment. "
            "Run `python -m pip install -r requirements.txt` from TP1, "
            "or run this script with `venv\\Scripts\\python.exe`."
        ) from error
    raise


BASE_DIR = Path(__file__).resolve().parent
EXPERIMENT_NAME = "churn-prediction"

# The training script stores runs in the SQLite tracking backend next to this
# file. Do not depend on the process working directory or MLflow's default URI.
mlflow.set_tracking_uri(f"sqlite:///{(BASE_DIR / 'mlflow.db').as_posix()}")

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    raise RuntimeError(f"MLflow experiment not found: {EXPERIMENT_NAME!r}")

# search_runs returns all runs as a pandas table. Sorting by f1 puts the best
# run first.
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.f1 DESC"],
)

if runs.empty:
    raise RuntimeError(f"No runs found in MLflow experiment: {EXPERIMENT_NAME!r}")

columns = [
    "tags.mlflow.runName",
    "params.n_estimators",
    "params.max_depth",
    "metrics.accuracy",
    "metrics.f1",
]
print(runs[columns].to_string(index=False))

best = runs.iloc[0]
print()
print("Best run :", best["tags.mlflow.runName"], "| id :", best["run_id"][:8], "...")