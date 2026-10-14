
"""
Lab 02: Dataset preparation
Dataset: Handwritten digits from scikit-learn.
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def main():
    # 1. Load the handwritten-digit dataset
    X, y = load_digits(return_X_y=True)

    print("Original dataset")
    print("Features shape:", X.shape)
    print("Labels shape:", y.shape)
    print("Digit classes:", np.unique(y))

    # 2. Normalize pixel values from 0-16 to 0-1
    X = X.astype(np.float64) / 16.0

    # 3. One-hot encode the digit labels
    y_one_hot = np.eye(10)[y]

    # 4. Split into training and testing sets
    X_train, X_test, y_train, y_test, y_train_one_hot, y_test_one_hot = (
        train_test_split(
            X,
            y,
            y_one_hot,
            test_size=0.30,
            random_state=42,
            stratify=y,
        )
    )

    print("\nDataset split")
    print("Training features:", X_train.shape)
    print("Testing features:", X_test.shape)
    print("Training labels:", y_train.shape)
    print("Testing labels:", y_test.shape)
    print("Training one-hot labels:", y_train_one_hot.shape)
    print("Testing one-hot labels:", y_test_one_hot.shape)

    # 5. Save the prepared dataset
    output_path = "lab02/results"

    np.savez_compressed(
        f"{output_path}/digits_data.npz",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_train_one_hot=y_train_one_hot,
        y_test_one_hot=y_test_one_hot,
    )

    print(f"\nPrepared dataset saved to {output_path}/digits_data.npz")


if __name__ == "__main__":
    main()

