# TP1 Reflection

## Can you give back exactly the model of Trial 1?

Yes. MLflow gives each trial its own ID. With the ID of Trial 1, we can
reload exactly the same model:

```python
mlflow.sklearn.load_model("runs:/<id>/model")
```

The `<id>` must be replaced with the real ID of the Trial 1 run.

## Can a classmate reproduce your work on another machine?

Partly. The code is saved in Git, and MLflow saves the model parameters and
the library versions in the run requirements. So a classmate knows how the
model was made.

However, getting exactly the same environment on another machine can still
be difficult. Docker can help by packaging the code, Python, and libraries
in the same environment.
