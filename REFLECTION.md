# Reflection

## How do we officially designate the model to use?

We use the model that has the `champion` alias in the MLflow Model Registry:

```python
models:/churn-model@champion
```

This alias points to the model version we selected. If we want to use another
version, we can simply move the alias to it.

## How can an application use the model?

An application uses the model by sending a web request to the API with
`POST /predict`. The customer's information is sent in JSON format, and the
API returns the prediction.

## What happens when we change the champion?

We move the `champion` alias to another model version and restart the API so
that it loads the new model. The API code and the client applications do not
need to change because they continue using the same request.
