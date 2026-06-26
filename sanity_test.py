import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


data = np.load("data/temporal_order_dataset.npz")

X_train = data["X_train"]
y_train = data["y_train"]
X_test = data["X_test"]
y_test = data["y_test"]


def spike_count_stats(X, y):
    result = {}
    for cls in [0, 1]:
        counts = X[y == cls].sum(axis=(1, 2))
        result[f"class_{cls}"] = {"mean": round(float(counts.mean()), 2), "std": round(float(counts.std()), 2)}
    return result


def classify(features_train, features_test):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(features_train, y_train)
    return round(accuracy_score(y_test, clf.predict(features_test)), 3)


# ignore time completely
X_train_no_time = X_train.sum(axis=1)
X_test_no_time = X_test.sum(axis=1)

# keep rough timing by counting spikes in each time window
n_windows = 3
window_size = X_train.shape[1] // n_windows

X_train_windows = X_train.reshape(len(X_train), n_windows, window_size, X_train.shape[2])
X_test_windows = X_test.reshape(len(X_test), n_windows, window_size, X_test.shape[2])

X_train_windows = X_train_windows.sum(axis=2).reshape(len(X_train), -1)
X_test_windows = X_test_windows.sum(axis=2).reshape(len(X_test), -1)

results = {
    "spike_counts": {
        "train": spike_count_stats(X_train, y_train),
        "test": spike_count_stats(X_test, y_test),
    },
    "accuracy": {
        "time_ignoring_baseline": classify(X_train_no_time, X_test_no_time),
        "window_based_baseline": classify(X_train_windows, X_test_windows),
    },
}

out_path = Path("data/sanity_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {out_path}")