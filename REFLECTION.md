# Reflection on the MLflow Experiment

## Which model is the best? Can you prove it?

Yes. MLflow saves the results of all the experiments and sorts the runs by
their F1 score. The run with the highest F1 score is considered the best
model. We can also see the date and the details of each run.

## Can you give back exactly the model of Trial 1?

Yes. Every MLflow run has its own unique ID. With the ID of Trial 1, we can
reload the exact model using:

```python
mlflow.sklearn.load_model("runs:/<run_id>/model")
```

This lets us use the same model that was trained during that specific trial.

## Can a classmate reproduce your work on another machine?

Partly. The code is available in Git, and MLflow records the model parameters
and library versions in the run's `requirements.txt` file. However, creating
exactly the same environment on another machine can still be difficult.
Using Docker would make the project easier to reproduce.
