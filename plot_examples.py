from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def plot_spike(x, title, path):
    times, neurons = np.nonzero(x)

    plt.figure(figsize=(8, 3))
    plt.scatter(times, neurons, s=5)
    plt.axvline(100, linestyle="--", linewidth=1)
    plt.axvline(200, linestyle="--", linewidth=1)
    plt.xlabel("Time step")
    plt.ylabel("Neuron index")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()




def main():
    data = np.load("data/temporal_order_dataset.npz")

    x_train = data["X_train"]
    y_train = data["y_train"]

    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)

    idx0 = np.where(y_train == 0)[0][0]
    idx1 = np.where(y_train == 1)[0][0]

    x0 = x_train[idx0]
    x1 = x_train[idx1]

    plot_spike(
        x0,
        "Class 0 example: A -> B -> C",
        out_dir / "class0_example.png",
    )

    plot_spike(
        x1,
        "Class 1 example: C -> B -> A",
        out_dir / "class1_example.png",
    )

    print("Saved figures to figures/")


if __name__ == "__main__":
    main()
