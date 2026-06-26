import numpy as np
import matplotlib.pyplot as plt

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
Y = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

w = np.random.randn()
b = np.random.randn()
lr = 0.01

mse_over_reps = []

# predicts y value based on inputted weight and bias
def predict(x: int, w: float, b: float) -> float:
    return w * x + b

# repeatedly adjusts weight and bias values based on error or difference between prediction and actual value
def train(X: np.array, Y: np.array, w: float, b: float, lr: float, epochs: int, mse_over_reps: list):
    for epoch in range(epochs):
        # this implementation uses a stochastic approach where for each epoch and update it made
        # to the w and b values for every data point in the data set
        for i in range(len(X)):
            xi = X[i]
            yi = Y[i]

            y_pred = predict(xi, w, b)

            # the amount to change the weight and bias formulas are found by taking
            # partial derivatives of the mean squared loss function with respect to w and b
            dw = -2 * xi * (yi - y_pred)
            db = -2 * (yi - y_pred)

            w = w - lr * dw
            b = b - lr * db
        mse_over_reps.append(mse(X, Y, w, b))

    return w, b, mse_over_reps
    
# calculate mean squared error
def mse(X: np.array, Y: np.array, w: int, b: int) -> float:
    n = len(X)
    total_mse = 0
    for i in range(n):
        xi = X[i]
        yi = Y[i]
        y_pred = predict(xi, w, b)
        total_mse = total_mse + (y_pred - yi) ** 2

    return total_mse / n

w, b, mse_over_reps = train(X, Y, w, b, lr, 1000, mse_over_reps)
error = mse(X, Y, w, b)
print(w, b)
print(error)

plt.plot(mse_over_reps)
plt.title("Effect of training repetitions on mean squared error")
plt.xlabel("Training Repetitions")
plt.ylabel("Mean Squared Error")
plt.show()

plt.plot(np.log10(mse_over_reps))
plt.title("Effect of training repetitions on log10 of mean squared error")
plt.xlabel("Training Repetitions")
plt.ylabel("log10 Mean Squared Error")
plt.show()