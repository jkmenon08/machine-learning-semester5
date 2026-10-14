
"""Basic tests for the Lab 02 neural network."""

import numpy as np

from neural_network import NeuralNetwork


def main():
    # Create a network with 16 neurons in each hidden layer.
    model = NeuralNetwork(hidden_size=16, seed=42)

    # Load the prepared dataset.
    data = np.load("lab02/results/digits_data.npz")

    X_train = data["X_train"]
    y_train_one_hot = data["y_train_one_hot"]

    # Use a small batch to test the network.
    X_batch = X_train[:32]
    y_batch = y_train_one_hot[:32]

    # Forward propagation.
    probabilities, cache = model.forward(X_batch)

    print("Input shape:", X_batch.shape)
    print("Output shape:", probabilities.shape)
    print("First prediction probabilities:", probabilities[0])
    print("Probability sum:", probabilities[0].sum())

    # Calculate loss.
    loss = model.cross_entropy(y_batch, probabilities)
    print("Initial cross-entropy loss:", round(float(loss), 6))

    # Calculate gradients.
    gradients = model.backward(y_batch, cache)

    print("\nGradient shapes:")
    for name, gradient in gradients.items():
        print(f"{name}: {gradient.shape}")

    # Confirm output probabilities are valid.
    assert probabilities.shape == (32, 10)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.isfinite(loss)
    assert all(np.all(np.isfinite(g)) for g in gradients.values())

    print("\nAll basic network checks passed!")


if __name__ == "__main__":
    main()

