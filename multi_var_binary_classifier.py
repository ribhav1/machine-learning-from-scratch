
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

# generate dummy clustered data
X, Y = datasets.make_blobs(n_samples=100, n_features = 1, centers = 2)

# split data into train and test sections
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.6)

# Standardize features using the training set's mean and standard deviation.
X_train_mean = X_train.mean(axis=0)
X_train_std = X_train.std(axis=0)

X_train = (X_train - X_train_mean) / X_train_std
X_test = (X_test - X_train_mean) / X_train_std

plt.scatter(X, Y)
plt.show()

n_samples, n_features = X_train.shape
n_output = len(Y_train)

weights = [np.random.randn()] * n_features
bias = np.random.randn()

lr = 0.01
epochs = 1000

def sigmoid(z: float) -> float:
    return 1 / (1 + np.exp(-z))

# returns a class prediction based on an inputted feature array transfomred using the current weights and bias values
def predict(X: np.array, weights: np.array, bias: int) -> float:
    return sigmoid(np.dot(X, weights) + bias)

def loss(y: int, y_pred: float) -> float:
    return -(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

def bce(X: np.array, Y: np.array, weights: np.array, bias: int) -> float:
    total_loss = 0
    for i in range(len(X)):
        y_pred = predict(X[i], weights, bias)
        total_loss += loss(Y[i], y_pred)
    return -1 * total_loss / len(X)

def train(X_train: np.array, Y_train: np.array, weights: np.array, bias: int, epochs: int):
    bce_over_epochs = []
    
    for epoch in range(epochs):
        for i in range(len(X_train)):
            y_pred = predict(X_train[i], weights, bias)

            # taking the partial derivative of the loss function with respect to w_i gives that
            # each weight w_i eshould be modified by the value of the error (y_pred - y_i) 
            # times its respective input x_i
            dw_array = (y_pred - Y_train[i]) * X_train[i]
            db = (y_pred - Y_train[i])

            weights = weights - lr * dw_array
            bias = bias - lr * db
        bce_over_epochs.append(bce(X_train, Y_train, weights, bias))

    return weights, bias, bce_over_epochs

# returns the binary cross entropy loss of the model's predictions on new, unseen test data
def test(X_test: np.array, Y_test: np.array, weights: np.array, bias: float) -> float:
    return bce(X_test, Y_test, weights, bias)

weights, bias, bce_over_epochs = train(X_train, Y_train, weights, bias, epochs)

plt.plot(bce_over_epochs)
plt.title("Effect of training repetitions on BCE")
plt.xlabel("Epochs")
plt.ylabel("Binary Cross Entropy Loss")
plt.show()

test_bce = test(X_test, Y_test, weights, bias)
print(test_bce)