import mlflow
import mlflow.sklearn
import pandas as pd

# Same database as Lab 3
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Load the champion by its alias.
model = mlflow.sklearn.load_model("models:/churn-model@champion")
print("Model loaded :", type(model).__name__)
print("Expected variables :", list(model.feature_names_in_))

# A new customer using the one-hot encoded subscription columns.
customer = pd.DataFrame(
    [
        {
            "age": 35,
            "salaire": 2500,
            "anciennete": 1,
            "consommation": 10,
            "abonnement_Premium": False,
            "abonnement_Standard": True,
        }
    ]
)

# Keep only the columns known to the model, in the right order.
customer = customer[list(model.feature_names_in_)]

# Predict: 1 = the customer will leave, 0 = stays.
print(
    "Will the customer leave? ->",
    "YES" if model.predict(customer)[0] == 1 else "NO",
)
