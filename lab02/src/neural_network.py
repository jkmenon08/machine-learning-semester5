
"""
Lab 02: Two-hidden-layer neural network from scratch.

Architecture:
Input (64) -> Hidden 1 (W) -> Hidden 2 (W) -> Output (10)

Implemented using NumPy.
"""

import numpy as np


class NeuralNetwork:
    def __init__(self, input_size=64, hidden_size=16, output_size=10, seed=42):
        """Initialize the network weights and biases."""

        rng = np.random.default_rng(seed)

        self.W1 = rng.normal(
            0, np.sqrt(2.0 / input_size), (input_size, hidden_size)
        )
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = rng.normal(
            0, np.sqrt(2.0 / hidden_size), (hidden_size, hidden_size)
        )
        self.b2 = np.zeros((1, hidden_size))

        self.W3 = rng.normal(
            0, np.sqrt(1.0 / hidden_size), (hidden_size, output_size)
        )
        self.b3 = np.zeros((1, output_size))

    @staticmethod
    def relu(z):
        """ReLU activation function."""
        return np.maximum(0, z)

    @staticmethod
    def softmax(z):
        """Convert output scores into probabilities."""
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        """Calculate predictions and cache intermediate values."""

        z1 = X @ self.W1 + self.b1
        a1 = self.relu(z1)

        z2 = a1 @ self.W2 + self.b2
        a2 = self.relu(z2)

        z3 = a2 @ self.W3 + self.b3
        probabilities = self.softmax(z3)

        cache = {
            "X": X,
            "z1": z1,
            "a1": a1,
            "z2": z2,
            "a2": a2,
            "z3": z3,
            "probabilities": probabilities,
        }

        return probabilities, cache

    @staticmethod
    def cross_entropy(y_one_hot, probabilities):
        """Calculate mean cross-entropy loss."""

        eps = 1e-12
        probabilities = np.clip(probabilities, eps, 1.0)

        return -np.mean(
            np.sum(y_one_hot * np.log(probabilities), axis=1)
        )

    def backward(self, y_one_hot, cache):
        """Calculate gradients using the chain rule."""

        X = cache["X"]
        z1 = cache["z1"]
        a1 = cache["a1"]
        z2 = cache["z2"]
        a2 = cache["a2"]
        probabilities = cache["probabilities"]

        batch_size = X.shape[0]

        # Output layer: softmax + cross-entropy
        dz3 = (probabilities - y_one_hot) / batch_size

        dW3 = a2.T @ dz3
        db3 = np.sum(dz3, axis=0, keepdims=True)

        # Hidden layer 2
        da2 = dz3 @ self.W3.T
        dz2 = da2 * (z2 > 0)

        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # Hidden layer 1
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0)

        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        return {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
            "W3": dW3,
            "b3": db3,
        }

    def update(self, gradients, learning_rate):
        """Update weights and biases using gradient descent."""

        self.W1 -= learning_rate * gradients["W1"]
        self.b1 -= learning_rate * gradients["b1"]

        self.W2 -= learning_rate * gradients["W2"]
        self.b2 -= learning_rate * gradients["b2"]

        self.W3 -= learning_rate * gradients["W3"]
        self.b3 -= learning_rate * gradients["b3"]

