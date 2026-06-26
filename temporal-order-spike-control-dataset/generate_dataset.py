import json
from pathlib import Path

import numpy as np

# Consts
PATTERNS = {
    "A": np.arange(0, 20),
    "B": np.arange(20, 40),
    "C": np.arange(40, 60),
}

CLASS_ORDERS = {
    0: ["A", "B", "C"],
    1: ["C", "B", "A"],
}

# ---

def add_pattern_spikes(x, neurons, start, end, spikes_per_neuron, rng):
    for n in neurons:
        times = rng.choice(np.arange(start, end), size=spikes_per_neuron, replace=False)
        x[times, n] = 1



def add_noise_spikes(x, n_spikes, rng):
    if n_spikes <= 0:
        return

    free = np.flatnonzero(x.reshape(-1) == 0)

    chosen = rng.choice(free, size=min(n_spikes, len(free)), replace=False)
    
    x.reshape(-1)[chosen] = 1


def make_sample(label, time_steps, n_neurons, n_windows, spikes_per_neuron, noise_spikes, rng):
    x = np.zeros((time_steps, n_neurons), dtype=np.uint8)

    window_size = time_steps // n_windows
    order = CLASS_ORDERS[label]

    for i, pattern_name in enumerate(order):
        start = i * window_size
        end = (i + 1) * window_size
        add_pattern_spikes(
            x=x,
            neurons=PATTERNS[pattern_name],
            start=start,
            end=end,
            spikes_per_neuron=spikes_per_neuron,
            rng=rng,
        )

    add_noise_spikes(x, noise_spikes, rng)
    return x



def make_split(n_samples, time_steps, n_neurons, n_windows, spikes_per_neuron, noise_spikes, rng):
    labels = np.array([0, 1] * (n_samples // 2), dtype=np.int64)

    if len(labels) < n_samples:
        labels = np.append(labels, rng.integers(0, 2))

    rng.shuffle(labels)

    x = np.zeros((n_samples, time_steps, n_neurons), dtype=np.uint8)

    for i, label in enumerate(labels):
        x[i] = make_sample(
            label=label,
            time_steps=time_steps,
            n_neurons=n_neurons,
            n_windows=n_windows,
            spikes_per_neuron=spikes_per_neuron,
            noise_spikes=noise_spikes,
            rng=rng,
        )

    return x, labels




def main():

    # hardcode values 
    out_dir = Path("data")
    seed = 7
    n_train = 1000
    n_test = 300
    n_neurons = 60
    time_steps = 300
    n_windows = 3
    spikes_per_neuron = 3
    noise_spikes = 30

    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    split_kwargs = dict(
        time_steps=time_steps,
        n_neurons=n_neurons,
        n_windows=n_windows,
        spikes_per_neuron=spikes_per_neuron,
        noise_spikes=noise_spikes,
        rng=rng,
    )

    x_train, y_train = make_split(n_train, **split_kwargs)
    x_test, y_test = make_split(n_test, **split_kwargs)


    dataset_path = out_dir / "temporal_order_dataset.npz"
    np.savez_compressed(
        dataset_path,
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test,
    )

    metadata = {
        "name": "temporal_order_spike_dataset",
        "description": " A Binary spike classification dataset where classes differ only by temporal order.",
        "shape": {
            "X_train": list(x_train.shape),
            "y_train": list(y_train.shape),
            "X_test": list(x_test.shape),
            "y_test": list(y_test.shape),
        },
        "parameters": {
            "seed": seed,
            "n_neurons": n_neurons,
            "time_steps": time_steps,
            "n_windows": n_windows,
            "spikes_per_neuron": spikes_per_neuron,
            "noise_spikes_per_sample": noise_spikes,
        },
        "patterns": {
            "A": "neurons 0-19",
            "B": "neurons 20-39",
            "C": "neurons 40-59",
        },
        "classes": {
            "0": "A -> B -> C",
            "1": "C -> B -> A",
        },
        "controlled_property": "Temporal order of spike patterns",
    }

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Train set shapes: X_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"Test dataset shapes X_test:  {x_test.shape}, y_test:  {y_test.shape}")


if __name__ == "__main__":
    main()
