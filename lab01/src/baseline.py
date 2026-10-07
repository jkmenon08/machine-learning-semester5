from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd


# Load dataset
X, y = load_breast_cancer(return_X_y=True)

# 70/30 stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42,
)

# Define models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    ),
}

# Train and evaluate
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    results.append(
        {
            "model": name,
            "accuracy": round(accuracy, 4),
        }
    )

    print(f"{name}: {accuracy:.4f}")

# Save results
results_df = pd.DataFrame(results)

results_df.to_csv(
    "lab01/results/baseline_accuracy.csv",
    index=False,
)

print("\nResults saved to lab01/results/baseline_accuracy.csv")