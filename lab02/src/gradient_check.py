
"""
Lab 02: Numerical gradient checking.

Compare backpropagation gradients with finite-difference
approximations for 10 randomly selected weights.
"""

import numpy as np

from neural_network import NeuralNetwork


def get_loss(model, X, y_one_hot):
    probabilities, _ = model.forward(X)
    return model.cross_entropy(y_one_hot, probabilities)


def main():
    # Load the prepared dataset.
    data = np.load("lab02/results/digits_data.npz")

    # Use a small batch to make gradient checking faster.
    X = data["X_train"][:8]
    y_one_hot = data["y_train_one_hot"][:8]

    model = NeuralNetwork(hidden_size=16, seed=42)

    # Calculate analytical gradients using backpropagation.
    probabilities, cache = model.forward(X)
    analytical_gradients = model.backward(y_one_hot, cache)

    # Select 10 weight parameters deterministically.
    rng = np.random.default_rng(42)
    parameters = [
        ("W1", model.W1),
        ("W2", model.W2),
        ("W3", model.W3),
    ]

    checks = []

    for _ in range(10):
        name, parameter = parameters[
            rng.integers(0, len(parameters))
        ]

        index = tuple(
            rng.integers(0, dimension)
            for dimension in parameter.shape
        )

        original_value = parameter[index]
        epsilon = 1e-5

        # Loss at parameter + epsilon.
        parameter[index] = original_value + epsilon
        loss_plus = get_loss(model, X, y_one_hot)

        # Loss at parameter - epsilon.
        parameter[index] = original_value - epsilon
        loss_minus = get_loss(model, X, y_one_hot)

        # Restore the original parameter.
        parameter[index] = original_value

        # Central finite-difference approximation.
        numerical_gradient = (
            loss_plus - loss_minus
        ) / (2.0 * epsilon)

        analytical_gradient = analytical_gradients[name][index]

        # Symmetric relative error with a small denominator safeguard.
        denominator = max(
            1e-12,
            abs(numerical_gradient) + abs(analytical_gradient),
        )

        relative_error = (
            abs(numerical_gradient - analytical_gradient)
            / denominator
        )

        checks.append(
            (name, index, analytical_gradient,
             numerical_gradient, relative_error)
        )

    print("NUMERICAL GRADIENT CHECK")
    print("-" * 85)
    print(
        f"{'Parameter':<10} {'Analytical':>15} "
        f"{'Numerical':>15} {'Relative error':>18}"
    )

    for name, index, analytical, numerical, error in checks:
        print(
            f"{name + str(index):<10} "
            f"{analytical:>15.8e} "
            f"{numerical:>15.8e} "
            f"{error:>18.8e}"
        )

    passed = sum(error < 1e-6 for *_, error in checks)

    print("-" * 85)
    print(f"Checks passing relative error < 1e-6: {passed}/10")

    if passed >= 9:
        print("Gradient check PASSED.")
    else:
        print(
            "Gradient check did not meet the target. "
            "Inspect the gradients before proceeding."
        )


if __name__ == "__main__":
    main()

