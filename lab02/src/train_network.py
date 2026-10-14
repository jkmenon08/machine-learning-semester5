
"""
Lab 02: Train a two-hidden-layer neural network
on the handwritten digits dataset.
"""

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from neural_network import NeuralNetwork


def load_data():
    data = np.load("lab02/results/digits_data.npz")

    return (
        data["X_train"],
        data["y_train_one_hot"],
        data["X_test"],
        data["y_test"],
    )


def accuracy(model, X, y):
    probabilities, _ = model.forward(X)
    predictions = np.argmax(probabilities, axis=1)

    return np.mean(predictions == y)


def train_model(model, X, y, epochs, batch_size, learning_rate):
    losses = []
    rng = np.random.default_rng(42)

    for epoch in range(epochs):
        indices = rng.permutation(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for start in range(0, len(X), batch_size):
            end = start + batch_size

            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            probabilities, cache = model.forward(X_batch)
            gradients = model.backward(y_batch, cache)

            # Gradient descent updates.
            model.W1 -= learning_rate * gradients["W1"]
            model.b1 -= learning_rate * gradients["b1"]

            model.W2 -= learning_rate * gradients["W2"]
            model.b2 -= learning_rate * gradients["b2"]

            model.W3 -= learning_rate * gradients["W3"]
            model.b3 -= learning_rate * gradients["b3"]

        # Record loss after each epoch.
        probabilities, _ = model.forward(X)
        epoch_loss = model.cross_entropy(y, probabilities)
        losses.append(epoch_loss)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:3d}/{epochs} | "
                f"Loss: {epoch_loss:.4f}"
            )

    return losses


def main():
    X_train, y_train, X_test, y_test = load_data()

    output_dir = Path("lab02/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = 50
    batch_size = 32

    configurations = [
        (16, 0.001),
        (16, 0.01),
        (16, 0.1),
        (64, 0.001),
        (64, 0.01),
        (64, 0.1),
    ]

    results = []

    plt.figure(figsize=(10, 6))

    for hidden_size, learning_rate in configurations:
        print("\n" + "=" * 55)
        print(
            f"Hidden size: {hidden_size}, "
            f"Learning rate: {learning_rate}"
        )

        model = NeuralNetwork(
            hidden_size=hidden_size,
            seed=42,
        )

        start_time = time.perf_counter()

        losses = train_model(
            model,
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

        training_time = time.perf_counter() - start_time
        test_accuracy = accuracy(model, X_test, y_test)

        print(f"Training time: {training_time:.3f} seconds")
        print(f"Test accuracy: {test_accuracy:.4f}")

        results.append({
            "hidden_size": hidden_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "training_time_seconds": training_time,
            "final_training_loss": losses[-1],
            "test_accuracy": test_accuracy,
        })

        plt.plot(
            range(1, epochs + 1),
            losses,
            label=f"Hidden={hidden_size}, LR={learning_rate}",
        )

    # Save results.
    csv_path = output_dir / "training_results.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)

    # Save loss curves.
    plt.xlabel("Epoch")
    plt.ylabel("Training cross-entropy loss")
    plt.title("Lab 2: Training Loss Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = output_dir / "training_loss.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print("\n" + "=" * 55)
    print("TRAINING COMPLETE")
    print(f"Results saved to: {csv_path}")
    print(f"Loss plot saved to: {plot_path}")

    best_result = max(results, key=lambda item: item["test_accuracy"])

    print("\nBest configuration by test accuracy:")
    print(best_result)


if __name__ == "__main__":
    main()

