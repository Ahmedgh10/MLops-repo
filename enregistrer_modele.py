# ===== 1. Import the libraries =====
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier

# ===== 2. Connect to the MLflow server started by Docker Compose =====
# From our computer, it is reachable on localhost thanks to the published port (5000)
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("churn-docker")

# ===== 3. Read and prepare the data =====
df = pd.read_csv("churn_data.csv")
# get_dummies: turn the text "abonnement" into 0/1 columns
X = pd.get_dummies(df.drop(columns=["churn"]), columns=["abonnement"])
y = df["churn"]

# ===== 4. Train the model =====
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# ===== 5. Register the model in the Registry of the server =====
# registered_model_name: name in the catalog (created if it does not exist)
# serialization_format: same format as in the previous labs
with mlflow.start_run():
    mlflow.sklearn.log_model(
        model,
        name="model",
        registered_model_name="churn-model",
        serialization_format="cloudpickle",
    )

# ===== 6. Put the "champion" alias on the version created =====
client = MlflowClient()
version = client.search_model_versions("name='churn-model'")[0].version
client.set_registered_model_alias("churn-model", "champion", version)
print(f"Version {version} of churn-model registered with the champion alias")
