import os
import statistics
import tempfile
import threading
import time

import joblib
import pandas as pd
import psutil
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42


# ============================================================
# Load dataset
# ============================================================

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=RANDOM_STATE,
)


# ============================================================
# Model factories
# ============================================================

def create_logistic_regression():
    return LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
    )


def create_random_forest():
    return RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
    )


# ============================================================
# Training
# ============================================================

def train_model(model):
    model.fit(X_train, y_train)
    return model


# ============================================================
# Training-time measurement
# ============================================================

def measure_training_time(model_factory):
    # Warm-up run
    warmup_model = model_factory()
    train_model(warmup_model)

    # Five measured runs
    training_times = []

    for run_number in range(5):
        model = model_factory()

        start_time = time.perf_counter()

        train_model(model)

        end_time = time.perf_counter()

        elapsed = end_time - start_time
        training_times.append(elapsed)

        print(
            f"  Training run {run_number + 1}: "
            f"{elapsed:.6f} seconds"
        )

    median_time = statistics.median(training_times)

    return median_time


# ============================================================
# Inference latency measurement
# ============================================================

def measure_inference_latency(model):
    # Use one test sample for single-sample inference
    sample = X_test[0:1]

    # Warm-up inference
    model.predict(sample)

    inference_times = []

    # 100 repetitions
    for _ in range(100):
        start_time = time.perf_counter()

        model.predict(sample)

        end_time = time.perf_counter()

        elapsed_ms = (end_time - start_time) * 1000
        inference_times.append(elapsed_ms)

    median_latency = statistics.median(inference_times)

    return median_latency


# ============================================================
# Model size measurement
# ============================================================

def measure_model_size(model):
    # Create a temporary file
    temporary_file = tempfile.NamedTemporaryFile(
        suffix=".joblib",
        delete=False,
    )

    temporary_path = temporary_file.name
    temporary_file.close()

    try:
        # Save model using joblib
        joblib.dump(model, temporary_path)

        # Get file size
        size_bytes = os.path.getsize(temporary_path)

        size_mb = size_bytes / (1024 * 1024)

    finally:
        # Remove temporary model file
        if os.path.exists(temporary_path):
            os.remove(temporary_path)

    return size_mb


# ============================================================
# Peak memory measurement
# ============================================================

def measure_peak_memory(model_factory):
    process = psutil.Process(os.getpid())

    # Store memory measurements
    memory_samples = []

    # Monitoring flag
    monitoring = True

    def monitor_memory():
        while monitoring:
            try:
                memory_bytes = process.memory_info().rss
                memory_mib = memory_bytes / (1024 * 1024)

                memory_samples.append(memory_mib)

            except psutil.Error:
                pass

            # Sample every 1 millisecond
            time.sleep(0.001)

    # Initial memory
    initial_memory = (
        process.memory_info().rss / (1024 * 1024)
    )

    # Start memory-monitoring thread
    monitor_thread = threading.Thread(
        target=monitor_memory,
        daemon=True,
    )

    monitor_thread.start()

    try:
        # Train model while memory is monitored
        model = model_factory()
        train_model(model)

    finally:
        # Stop monitoring
        monitoring = False
        monitor_thread.join(timeout=1)

    # Make sure we have samples
    if memory_samples:
        peak_memory = max(memory_samples)
    else:
        peak_memory = initial_memory

    # Memory increase relative to initial process memory
    peak_memory_increase = peak_memory - initial_memory

    return peak_memory_increase


# ============================================================
# Main
# ============================================================

def main():

    models = {
        "Logistic Regression": create_logistic_regression,
        "Random Forest": create_random_forest,
    }

    results = []

    print("=" * 60)
    print("LAB 01 - SYSTEM MEASUREMENTS")
    print("=" * 60)

    for model_name, model_factory in models.items():

        print()
        print("-" * 60)
        print(f"Measuring: {model_name}")
        print("-" * 60)

        # ----------------------------------------------------
        # Train model once for inference and size measurement
        # ----------------------------------------------------

        model = model_factory()
        train_model(model)

        # ----------------------------------------------------
        # 1. Training time
        # ----------------------------------------------------

        print("\nTraining time:")

        training_time = measure_training_time(
            model_factory
        )

        print(
            f"  Median training time: "
            f"{training_time:.6f} seconds"
        )

        # ----------------------------------------------------
        # 2. Inference latency
        # ----------------------------------------------------

        print("\nInference latency:")

        inference_latency = measure_inference_latency(
            model
        )

        print(
            f"  Median inference latency: "
            f"{inference_latency:.6f} ms"
        )

        # ----------------------------------------------------
        # 3. Model size
        # ----------------------------------------------------

        print("\nModel size:")

        model_size = measure_model_size(model)

        print(
            f"  Model size: "
            f"{model_size:.6f} MB"
        )

        # ----------------------------------------------------
        # 4. Peak memory
        # ----------------------------------------------------

        print("\nPeak memory:")

        peak_memory = measure_peak_memory(
            model_factory
        )

        print(
            f"  Peak memory increase: "
            f"{peak_memory:.6f} MiB"
        )

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        results.append(
            {
                "model": model_name,
                "training_time_seconds": training_time,
                "inference_latency_ms": inference_latency,
                "model_size_mb": model_size,
                "peak_memory_mib": peak_memory,
            }
        )

    # ========================================================
    # Save results
    # ========================================================

    results_df = pd.DataFrame(results)

    output_path = "lab01/results/system_measurements.csv"

    results_df.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 60)
    print("MEASUREMENTS COMPLETE")
    print("=" * 60)

    print()
    print(results_df.to_string(index=False))

    print()
    print(f"Results saved to: {output_path}")


# ============================================================
# Windows-safe entry point
# ============================================================

if __name__ == "__main__":
    main()