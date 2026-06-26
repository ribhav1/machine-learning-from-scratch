import numpy as np
import matplotlib.pyplot as plt

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
Y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

w = np.random.randn()
b = np.random.randn()
lr = 0.01

loss_over_reps = []

def sigmoid(z: float) -> float:
    return 1 / (1 + np.exp(-z))

def predict(x: int, w: float, b: float) -> float:
    y_pred = w * x + b
    return sigmoid(y_pred)

def loss(y: int, y_pred: float):
    return -(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

def mean_loss(X: np.array, Y: np.array, w: float, b: float):
    n = len(X)
    total_loss = 0
    for i in range(n):
        xi = X[i]
        yi = Y[i]
        y_pred = predict(xi, w, b)

        total_loss += loss(yi, y_pred)
    
    return total_loss / n

def train(X: np.array, Y: np.array, w: float, b: float, lr: float, epochs: int, loss_over_reps: list):
    for epoch in range(epochs):
        for i in range(len(X)):
            xi = X[i]
            yi = Y[i]

            y_pred = predict(xi, w, b)

            dw = (y_pred - yi) * xi
            db =  (y_pred - yi)

            w = w - lr * dw
            b = b - lr * db
        loss_over_reps.append(mean_loss(X, Y, w, b))

    return w, b, loss_over_reps

w, b, loss_over_reps = train(X, Y, w, b, lr, 1000, loss_over_reps)
error = mean_loss(X, Y, w, b)
print(w, b)
print(error)

plt.plot(loss_over_reps)
plt.title("Effect of training repetitions on binary cross entropy")
plt.xlabel("Training Repetitions")
plt.ylabel("Binary Cross Entropy")
plt.show()

plt.plot(np.log10(loss_over_reps))
plt.title("Effect of training repetitions on log10 binary cross entropy")
plt.xlabel("Training Repetitions")
plt.ylabel("log10 Binary Cross Entropy")
plt.show()