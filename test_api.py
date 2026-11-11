import requests

URL = "http://localhost:8000"

print("GET /health ->", requests.get(URL + "/health").json())

customer = {
    "age": 35,
    "salaire": 2500,
    "abonnement": "Standard",
    "anciennete": 1,
    "consommation": 10,
}
r = requests.post(URL + "/predict", json=customer)
print("POST /predict (customer 1) ->", r.status_code, r.json())

customer2 = {
    "age": 50,
    "salaire": 4000,
    "abonnement": "Premium",
    "anciennete": 10,
    "consommation": 80,
}
r = requests.post(URL + "/predict", json=customer2)
print("POST /predict (customer 2) ->", r.status_code, r.json())

bad = dict(customer, age="abc")
r = requests.post(URL + "/predict", json=bad)
print("POST /predict (age='abc') ->", r.status_code)
print("Message :", r.json()["detail"][0]["msg"])
