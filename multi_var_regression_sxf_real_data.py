# This file implements multivariable linear regression for an arbitrary number of features.
# Input data is stored such that each row represents one sample and each column represents one feature.
# Additionally, this file uses real, imported data rather than dummy data.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# load californa housing dataset
data = fetch_california_housing()
X = data.data
Y = data.target

#print(X.shape)
#print(Y.shape)
#print(data.feature_names)

split_index = int(len(X) * 0.8)

X_train, Y_train = X[:split_index], Y[:split_index]
X_test, Y_test = X[split_index:], Y[split_index:]

# Standardize features using the training set's mean and standard deviation.
# The same transformation is applied to the test set to avoid data leakage
# and ensure both datasets are represented in the same feature space.
X_train_mean = X_train.mean(axis=0)
X_train_std = X_train.std(axis=0)

X_train = (X_train - X_train_mean) / X_train_std
X_test = (X_test - X_train_mean) / X_train_std

n_samples, n_features = X_train.shape

# initialize weights and bias as random floats
weights = np.random.randn(n_features)
bias = np.random.randn()

# set the learning rate and how many iterations the training loop should run
lr = 0.0001
epochs = 100

# compute ta prediction for a given input based on the current parameters (weight and bias values)
def predict(X: np.array, weights: np.array, bias: float):
    return np.dot(X, weights) + bias

# train the model using stochastic gradient descent
# calculates values for the weights and bias that best model the data
def train(X_train: np.array, Y_train: np.array, weights: np.array, bias: float, lr: float, epochs: int):
    mse_over_epochs = []
    
    for epoch in range(epochs):
        # emply a stochastic approach where the parameter values are updated after every
        # data point that is encountered in traversal rather than updating at the end of an epoch
        for i in range(len(Y_train)):           
            # make a prediction for the output based on the corresponding stored inputs
            # i.e. output 4 should be predicted sample 4
            y_pred = predict(X_train[i], weights, bias)

            # calculate prediction error used to compute the MSE gradient
            error = (Y_train[i] - y_pred)

            # calculate the modification value for every weight using the corresponding input value, output value, and predicted output value
            # i.e. weight 3 should be modified based on the dw calculated using input 3, output 3, and predicted output 3 
            # the ith dw value represents the gradient of the MSE loss with respect to ith weight:
            # dL/dw = -2 * x * (y - y_pred)
            dw_array = -2 * X_train[i] * error

            # since there is only one bias value the calculation is the same as single variable regression
            # as the bias has no associated feature value so its gradient is -2 * error:
            # dL/db = -2 * (y - y_pred)
            db = -2 * error

            # update weight and bias values
            weights = weights - lr * np.array(dw_array)
            bias = bias - lr * db
            
        # record mean squared error after each epoch
        mse_over_epochs.append(mse(X_train, Y_train, weights, bias))

    # after all epochs return parameters and mse history
    return weights, bias, mse_over_epochs


# evaluate the model through calculating the mean squared error unseen test data
def test(X_test: np.array, Y_test: np.array, weights, bias):
    return mse(X_test, Y_test, weights, bias)

# Returns the mean squared error  by averaging the squared differences
# between the model's predictions and the true values for the given inputs.
def mse(X: np.array, Y: np.array, weights: np.array, bias: float):
    total_mse = 0
    for i in range(len(Y)):
        y_pred = predict(X[i], weights, bias)
        total_mse += (y_pred - Y[i]) ** 2
    return total_mse / len(Y)

# train model
weights, bias, mse_over_epochs = train(X_train, Y_train, weights, bias, lr, epochs)

print(weights)
print(bias)

plt.plot(mse_over_epochs)
plt.title("Effect of training repetitions on mean squared error")
plt.xlabel("Training Repetitions")
plt.ylabel("Mean Squared Error")
plt.show()

# test model
test_mse = test(X_test, Y_test, weights, bias)
test_rmse = np.sqrt(test_mse)
print(test_mse)
print(test_rmse)

# unnormalize weights and bias
#unnorm_weights = weights / X_std
#unnorm_bias = bias - np.sum((weights * X_mean) / X_std)

#weights = unnorm_weights
#bias = unnorm_bias

#print(weights)
#print(bias)