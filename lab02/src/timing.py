
"""
Lab 02, Task 7:
Measure forward and backward propagation separately.
Report the median execution time over 100 runs.
"""

import csv
import statistics
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from neural_network import NeuralNetwork


def main():
    # Reproducible dataset and network
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float64) / 16.0

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    encoder = OneHotEncoder(sparse_output=False)
    y_train_one_hot = encoder.fit_transform(y_train.reshape(-1, 1))

    # Use a batch of training samples
    X_batch = X_train[:128]
    y_batch = y_train_one_hot[:128]

    network = NeuralNetwork(
        input_size=64,
        hidden_size=16,
        output_size=10,
        seed=42,
    )

    # Warm-up: run both operations before measuring
    for _ in range(10):
        probabilities, cache = network.forward(X_batch)
        network.backward(y_batch, cache)

    # Prepare the cache for backward-only timing
    _, cache = network.forward(X_batch)

    forward_times = []
    backward_times = []

    # Measure forward propagation separately
    for _ in range(100):
        start = time.perf_counter()
        _, _ = network.forward(X_batch)
        end = time.perf_counter()
        forward_times.append((end - start) * 1000)

    # Measure backward propagation separately.
    # Reuse a previously prepared cache so forward time is excluded.
    for _ in range(100):
        start = time.perf_counter()
        network.backward(y_batch, cache)
        end = time.perf_counter()
        backward_times.append((end - start) * 1000)

    forward_median = statistics.median(forward_times)
    backward_median = statistics.median(backward_times)

    if forward_median > backward_median:
        dominant = "Forward propagation"
    elif backward_median > forward_median:
        dominant = "Backward propagation"
    else:
        dominant = "Both have equal median time"

    print("\nLAB 02 — TASK 7: PROPAGATION TIMING")
    print("-" * 48)
    print(f"Batch size: {len(X_batch)}")
    print("Timed runs per operation: 100")
    print("Warm-up runs: 10")
    print(f"Forward median:  {forward_median:.6f} ms")
    print(f"Backward median: {backward_median:.6f} ms")
    print(f"Dominant operation: {dominant}")

    # Save results
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "timing_results.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["operation", "batch_size", "runs", "median_time_ms"])
        writer.writerow(["forward", len(X_batch), 100, f"{forward_median:.6f}"])
        writer.writerow(["backward", len(X_batch), 100, f"{backward_median:.6f}"])

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

