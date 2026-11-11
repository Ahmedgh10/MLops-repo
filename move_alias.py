import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# Find the run "trial-2", the weakest model of Lab 3.
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    filter_string="tags.mlflow.runName = 'trial-2'",
)
if runs.empty:
    raise RuntimeError("The 'trial-2' run was not found.")

run_id = runs.iloc[0]["run_id"]

# Register it, creating a new version of churn-model.
version = mlflow.register_model(f"runs:/{run_id}/model", "churn-model")
print("New version created :", version.version)

# Save the current champion before moving the alias.
current = client.get_model_version_by_alias("churn-model", "champion")
print("Current champion : version", current.version)

client.set_registered_model_alias("churn-model", "champion", version.version)
print(
    "After the move : champion -> version",
    client.get_model_version_by_alias("churn-model", "champion").version,
)

# Restore the original champion.
client.set_registered_model_alias("churn-model", "champion", current.version)
print(
    "After going back : champion -> version",
    client.get_model_version_by_alias("churn-model", "champion").version,
)
