# Lab 01 — Environment and First System Measurements

## Goal

The goal of this laboratory was to prepare and verify the Python machine learning environment, train two baseline classification models on the Breast Cancer Wisconsin Diagnostic dataset, and evaluate their accuracy and system-level characteristics.

The measured system characteristics were training time, single-sample inference latency, model size, and peak memory increase. The measurements were then compared with the platform budgets for Cloud, Edge, Mobile, and TinyML environments.

## Method

The laboratory used the Breast Cancer Wisconsin Diagnostic dataset provided by scikit-learn through `load_breast_cancer(return_X_y=True)`.

The dataset was divided into training and test sets using a 70/30 split. The split was stratified and used `random_state=42` for reproducibility.

Two baseline classification models were evaluated:

* Logistic Regression with `max_iter=1000` and `random_state=42`
* Random Forest with `n_estimators=100` and `random_state=42`

Classification accuracy was measured on the test set and reported to four decimal places.

For system measurements, training time was measured using one warm-up run followed by five measured runs, with the median reported. Single-sample inference latency was measured over 100 repetitions, with the median reported. Model size was measured from the serialized model using Joblib. Peak memory increase was measured during model training.

All experiments were performed using Python 3.11.8 and the laboratory environment specified in the laboratory manual.

## Results

### Baseline accuracy

| Model               | Accuracy |
| ------------------- | -------: |
| Logistic Regression |   0.9474 |
| Random Forest       |   0.9357 |

Logistic Regression achieved the higher test accuracy, with an accuracy of 0.9474 compared with 0.9357 for Random Forest.

### System measurements

| Model               | Training time (s) | Inference latency (ms) | Model size (MB) | Peak memory increase (MiB) |
| ------------------- | ----------------: | ---------------------: | --------------: | -------------------------: |
| Logistic Regression |          0.469436 |               0.102850 |        0.001001 |                   0.136719 |
| Random Forest       |          0.261267 |               4.768650 |        0.277413 |                   0.070313 |

Random Forest had a lower median training time than Logistic Regression. However, Logistic Regression had substantially lower single-sample inference latency and a much smaller serialized model.

### Platform suitability

The measurements were compared with the platform budgets specified in the laboratory manual.

| Platform | Logistic Regression | Random Forest |
| -------- | ------------------- | ------------- |
| Cloud    | Suitable            | Suitable      |
| Edge     | Suitable            | Suitable      |
| Mobile   | Suitable            | Suitable      |
| TinyML   | Suitable            | Not suitable  |

Both models satisfy the Cloud, Edge, and Mobile latency, memory, and model-size requirements based on the measured values.

For TinyML, the specified model-size limit is 100 KB. The Logistic Regression model was approximately 1.0 KB and therefore satisfies the model-size requirement. The Random Forest model was approximately 0.277 MB, or about 284 KB, and therefore exceeds the 100 KB model-size limit.

## Conclusions

1. Logistic Regression achieved the highest classification accuracy in this experiment, reaching 0.9474 compared with 0.9357 for Random Forest.

2. Random Forest had a shorter median training time, but Logistic Regression was substantially more efficient for single-sample inference and required a much smaller serialized model.

3. Both models are suitable for Cloud, Edge, and Mobile deployment under the specified budgets. For TinyML deployment, Logistic Regression satisfies the measured model-size requirement, while Random Forest exceeds the 100 KB model-size limit.
