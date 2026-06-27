import numpy as np
import matplotlib.pylab as plt
from sklearn import datasets

# create dummy data with clusters using sklearn
#X, Y = datasets.make_blobs(n_samples=200, n_features=2, centers=3)

# different dummy data shapes to test model 
X, Y = datasets.make_moons(n_samples = 200, noise=0.2)
#X, Y = datasets.make_circles(n_samples=200, noise=0.11, factor=0.45)

# split data into train and test sets
split_index = int(len(X) * 0.6)
X_train, X_test = X[:split_index], X[split_index:]
Y_train, Y_test = Y[:split_index], Y[split_index:]

# Standardize features using the training set's mean and standard deviation.
X_train_mean = X_train.mean(axis=0)
X_train_std = X_train.std(axis=0)

X_train = (X_train - X_train_mean) / X_train_std
X_test = (X_test - X_train_mean) / X_train_std

# predict the class of a sample using the k-nearest neighbors algorithm
def predict(test_sample: np.array, X_train: np.array, Y_train: np.array, k: int) -> int:
    # vectorized calculation to record distance of test sample to every other sample in training set
    distances = np.linalg.norm(X_train - test_sample, axis=1)

    # gets the indices of the k closest training samples
    # argsort returns the indices of the distances sorted in ascending order
    # so taking the first k indices gives the nearest neighbors.
    nearest_neighbor_indices = np.argsort(distances)[:k]

    # use those indices to retrieve the corresponding class labels
    classes = [Y_train[index] for index in nearest_neighbor_indices]

    # Predict the class by majority vote,
    # np.bincount(classes) counts occurrences of each class label.
    # np.argmax returns the label with the highest count (the mode).
    class_pred = np.argmax(np.bincount(classes))
    return class_pred

def predict_all(X_test: np.array, X_train: np.array, Y_train: np.array, k: int) -> np.array:
    # record predictions for test data
    class_preds = []
    for test_sample in X_test:
        class_preds.append(predict(test_sample, X_train, Y_train, k))

    return class_preds

# return a score between 0-1 representing the 
# fraction of points for which the model correctly identified the class of the inputted data
def get_accuracy(Y_test: np.array, class_preds: np.array):
    correct = 0
    for i in range(len(Y_test)):
        if class_preds[i] == Y_test[i]:
            correct += 1
    return correct / len(Y_test)

class_preds = predict_all(X_test, X_train, Y_train, 3)
accuracy = get_accuracy(Y_test, class_preds)

print("Accuracy: " + str(accuracy))

# determine the plotting bounds for the decision boundary visualization
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

# create a grid of points covering the feature space
# meshgrid takes the x and y arrays and duplicate them across rows and columns to create a grid
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# flatten the grid into list of coordinates
grid_points = np.c_[xx.ravel(), yy.ravel()]

# classify every grid point to visualize the decision region as each class has a unique color
Z = np.array([predict(p, X_train, Y_train, 3) for p in grid_points]).reshape(xx.shape)

# draw the background color for each decision region and draw the boundaries between classes
plt.pcolormesh(xx, yy, Z, cmap="coolwarm", shading="nearest", alpha=0.35)
plt.contour(xx, yy, Z, colors="black", linewidths=1)

# draw train data with circles and test data with x-markers, 
# with colors representing true class lables and class predictions
plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train)
plt.scatter(X_test[:, 0], X_test[:, 1], c=class_preds, marker="x")
plt.show()