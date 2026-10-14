# Lab 02 — Neural Network from Scratch

## Goal

The goal of this laboratory was to design and implement a neural network with two hidden layers from scratch and use it to classify handwritten digits. The laboratory focused on dataset preparation, forward propagation, ReLU activation, softmax, cross-entropy loss, backpropagation, numerical gradient checking, model training, and performance evaluation.

## Method

### 1. Dataset preparation

The `load_digits` dataset from scikit-learn was used. It contains 1,797 handwritten digit samples, with 64 features per sample representing an 8 × 8 image.

The dataset was divided into training and testing subsets. The training set contained 1,257 samples, while the testing set contained 540 samples. The target labels were also converted into one-hot encoded vectors for use with the neural network.

### 2. Neural network architecture

A fully connected neural network with two hidden layers was implemented using NumPy. The network takes 64 input features and produces 10 output probabilities, corresponding to the digits 0 through 9.

ReLU was used as the hidden-layer activation function, while softmax converted the output scores into class probabilities.

### 3. Forward propagation and loss

Forward propagation was implemented to calculate the network's predictions. Cross-entropy loss was used to measure the difference between predicted probabilities and the correct labels.

### 4. Backpropagation and gradient checking

Backpropagation was implemented to calculate gradients for the network's weights and biases using the chain rule. Numerical gradient checking was performed by comparing analytical gradients against numerical estimates.

All 10 gradient checks passed with a relative error below 1e-6.

### 5. Model training

The network was trained using mini-batch gradient descent. Six configurations were evaluated by varying the hidden-layer width and learning rate.

Each configuration used 50 training epochs. Training time, final training loss, and test accuracy were recorded.

## Results

### Dataset preparation

| Property         | Result |
| ---------------- | -----: |
| Total samples    |  1,797 |
| Input features   |     64 |
| Training samples |  1,257 |
| Testing samples  |    540 |
| Output classes   |     10 |

### Gradient checking

| Measurement                              | Result |
| ---------------------------------------- | ------ |
| Gradient checks performed                | 10     |
| Checks passing relative error below 1e-6 | 10/10  |


### Training results

Six configurations were evaluated using hidden-layer widths of 16 and 64 neurons and learning rates of 0.001, 0.01, and 0.1. Each configuration was trained for 50 epochs.

| Hidden-layer width | Learning rate | Epochs | Training time (s) | Final training loss | Test accuracy |
| -----------------: | ------------: | -----: | ----------------: | ------------------: | ------------: |
|                 16 |         0.001 |     50 |             0.288 |              2.1451 |        0.1963 |
|                 16 |          0.01 |     50 |             0.282 |              0.3681 |        0.8852 |
|                 16 |           0.1 |     50 |             0.282 |              0.0149 |        0.9747 |
|                 64 |         0.001 |     50 |             0.489 |              1.8845 |        0.6241 |
|                 64 |          0.01 |     50 |             0.503 |              0.1780 |        0.9389 |
|                 64 |           0.1 |     50 |             0.497 |              0.0084 |        0.9759 |

The best configuration by test accuracy used 64 neurons in each hidden layer and a learning rate of 0.1. It achieved approximately 97.59% test accuracy, with a final training loss of 0.0084 and a training time of approximately 0.497 seconds.

The training-loss comparison graph is saved as `results/training_loss.png`.


### Training-loss visualization

The training-loss comparison graph is saved as `results/training_loss.png`. It shows how the training loss changed over the 50 epochs for the six configurations.

## Conclusions

1. The two-hidden-layer neural network successfully classified handwritten digits, achieving approximately 97.59% test accuracy with the best configuration reported by the experiment.

2. Numerical gradient checking confirmed that the analytical gradients closely matched numerical estimates, with all 10 checks passing the specified relative-error threshold.

3. The experiments demonstrated that hidden-layer width and learning rate are important training parameters. Comparing configurations using loss, accuracy, and training time helps identify an effective model configuration.

## Reproducibility

* Dataset: scikit-learn `load_digits`
* Implementation: NumPy
* Random seed: 42
* Training epochs: 50
* Results: `results/training_results.csv`
* Training-loss plot: `results/training_loss.png`
##  Forward and Backward Propagation Timing

### Goal

Measure the execution time of forward propagation and backward propagation separately using the two-hidden-layer neural network implemented from scratch with NumPy.

### Method

The experiment used a batch of 128 samples from the scikit-learn Digits dataset. The neural network used 64 input features, two hidden layers with 16 neurons each, and 10 output classes.

Before measurement, both operations were executed 10 times as warm-up runs. Forward propagation and backward propagation were then measured separately over 100 timed runs using Python's `time.perf_counter()`. The median execution time was calculated for each operation.

### Results

| Operation            | Batch size | Timed runs | Median execution time (ms) |
| -------------------- | ---------: | ---------: | -------------------------: |
| Forward propagation  |        128 |        100 |                   0.121000 |
| Backward propagation |        128 |        100 |                   0.124750 |

Backward propagation was approximately 3.10% slower than forward propagation in this experiment.

### Discussion

Forward propagation calculates activations and output probabilities by passing input data through the network. Backward propagation calculates gradients using the chain rule to determine how the loss changes with respect to the weights and biases.

Both operations use matrix multiplication. Forward propagation multiplies input and activation matrices by weight matrices, while backward propagation performs matrix multiplications to calculate weight gradients and propagate errors through the network. These calculations contribute to the computational cost of both operations.

In this experiment, backward propagation took slightly longer than forward propagation. The difference was small, and the result applies to this network architecture, batch size, implementation, and computer. Other configurations may produce different timings.

### Conclusions

1. Backward propagation had a median execution time of 0.124750 ms, compared with 0.121000 ms for forward propagation.
2. Backward propagation was approximately 3.10% slower for the tested batch of 128 samples.
3. Matrix multiplication is an important computational operation in both passes. Backward propagation also calculates gradients and propagates errors, which can contribute to its execution time.
