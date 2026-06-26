
# Temporal order spike control dataset (Toy problem)

This repository contains a small synthetic spiking dataset for testing whether a model can use temporal order for classification.

## What it is

The dataset has two classes. Both classes use the same three spike patterns: A, B, and C. The difference is the order in which these patterns appear.

- Class 0: A → B → C
- Class 1: C → B → A

This means that the class cannot be identified just by checking which patterns are present. Both classes contain the same patterns. To solve the task, a model has to use when the patterns appear.

## Quickstart

```bash
pip install -r requirements.txt
python generate_dataset.py
python plot_examples.py
````

## Files

```text
generate_dataset.py            generates the dataset
plot_examples.py               creates example spike plots

data/
  temporal_order_dataset.npz   train and test arrays
  metadata.json                dataset settings

figures/
  class0_example.png           example from Class 0
  class1_example.png           example from Class 1
```

The dataset file contains:

```text
X_train, y_train, X_test, y_test
```

Here, `X` contains the spike samples and `y` contains the class labels.

## Parameters (hardcoded)

| Parameter        | Value         |
| ---------------- | ------------- |
| Input neurons    | 60            |
| Time steps       | 300           |
| Temporal windows | 3 × 100 steps |
| Train samples    | 1000          |
| Test samples     | 300           |
| Seed             | 7             |

