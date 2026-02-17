← [Back to Main README](../../README.md)

# Experiment Tracking Guide

## Setup

MLflow is an **optional** dependency. Install it with:

```bash
pip install orchard-ml[tracking]
# or directly
pip install mlflow>=2.9.0
```

## Configuration

Add a `tracking:` section to any recipe YAML:

```yaml
tracking:
  enabled: true
  experiment_name: "my_experiment"
```

If the section is absent or `enabled: false`, the pipeline runs identically with zero overhead.

## Where Data is Stored

All tracking data is stored in a single SQLite database at the project root:

```
orchard-ml/
├── mlruns.db          # MLflow tracking database
├── outputs/
│   └── 20260215_.../  # Run artifacts (figures, models, reports)
```

The `mlruns.db` file is created automatically on the first tracked run and is already in `.gitignore`.

## MLflow UI

Launch the MLflow dashboard to browse experiments:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

Then open http://127.0.0.1:5000 in a browser.

## Querying Runs Programmatically

### List All Runs

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlruns.db")
runs = mlflow.search_runs(experiment_names=["orchard-ml"])
print(runs[["run_id", "params.dataset.name", "metrics.test_accuracy", "metrics.val_auc"]])
```

### Compare Runs by Dataset

```python
runs = mlflow.search_runs(
    experiment_names=["orchard-ml"],
    filter_string="params.dataset.name = 'pathmnist'",
    order_by=["metrics.test_accuracy DESC"],
)
print(runs[["run_id", "params.architecture.name", "metrics.test_accuracy"]].head(10))
```

### Get Best Run

```python
best = mlflow.search_runs(
    experiment_names=["orchard-ml"],
    order_by=["metrics.test_accuracy DESC"],
    max_results=1,
).iloc[0]

print(f"Best run: {best.run_id}")
print(f"  Model:    {best['params.architecture.name']}")
print(f"  Accuracy: {best['metrics.test_accuracy']:.4f}")
print(f"  F1:       {best['metrics.test_macro_f1']:.4f}")
```

### Plot Training Curves from a Run

```python
client = mlflow.tracking.MlflowClient()
run_id = "your_run_id_here"

# Fetch per-epoch metrics
history = client.get_metric_history(run_id, "val_auc")
epochs = [m.step for m in history]
aucs = [m.value for m in history]

import matplotlib.pyplot as plt
plt.plot(epochs, aucs)
plt.xlabel("Epoch")
plt.ylabel("Val AUC")
plt.title("Validation AUC Over Training")
plt.show()
```

### List Optuna Trials (Nested Runs)

```python
parent_run_id = "your_parent_run_id"
trials = mlflow.search_runs(
    experiment_names=["orchard-ml"],
    filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
    order_by=["metrics.best_trial_metric DESC"],
)
print(trials[["tags.mlflow.runName", "metrics.best_trial_metric", "params.lr"]].head(10))
```

## What Gets Tracked

| Phase        | Metrics                                                  | Artifacts                    |
|-------------|----------------------------------------------------------|------------------------------|
| Training    | `train_loss`, `val_loss`, `val_accuracy`, `val_auc`, `learning_rate` (per epoch) | -                            |
| Evaluation  | `test_accuracy`, `test_macro_f1`                         | Figures, report, config YAML |
| Optuna      | `best_trial_metric` (per trial), trial params            | -                            |
