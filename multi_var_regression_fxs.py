# This file implements multivariable linear regression for an arbitrary number of features.
# Input data is stored such that each row represents one feature and each column represents one sample.

import numpy as np
import matplotlib.pyplot as plt

input_arrays = [
    [1, 2, 3, 4, 5],        # x1
    [3, 5, 7, 9, 12],       # x2
    [1, 2, 3, 5, 7]         # x3
]
outputs = [14, 23, 32, 41, 50]
n_input = len(input_arrays)
n_output = len(outputs)

weights = np.random.randn(n_input)
bias = np.random.randn()
lr = 0.001
mse_over_reps = []

def predict(inputs: np.array, weights: np.array, bias: float):
    return np.dot(inputs, weights) + bias

def train(input_arrays: np.array, outputs: np.array, weights: np.array, bias: float, lr: float, epochs: int):
    for epoch in range(epochs):
        # iterate over every data point
        for i in range(len(outputs)):
            # store the ith input value from each input array
            ith_input_array = []

            # get input values from each input array that corresponds to the output value currently on
            # i.e. for output 2, input 2 from every input array should be stored
            for input_index in range(len(input_arrays)):
                ith_input_array.append(input_arrays[input_index][i])
            
            # make a prediction for the output based on the corresponding stored inputs
            # i.e. output 4 should be predicted from input 4 from each input array
            y_pred = predict(ith_input_array, weights, bias)

            # store how each weight needs to be updated
            dw_array = []

            # get the modification value for every weight using the corresponding input value, output value, and predicted output value
            # i.e. weight 3 should be modified based on the dw calculated using input 3, output 3, and predicted output 3 
            for weight_i in range(len(weights)):
                dw_array.append(-2 * ith_input_array[weight_i] * (outputs[i] - y_pred))
            
            # since there is only one bias value the calculation is the same as single variable regression
            db = -2 * (outputs[i] - y_pred)

            weights = weights - lr * np.array(dw_array)
            bias = bias - lr * db
        mse_over_reps.append(mse(input_arrays, outputs, weights, bias))

    return weights, bias

def mse(input_arrays: np.array, outputs: np.array, weights: np.array, bias: float):
    total_mse = 0
    for i in range(n_output):
        ith_inputs = []
        # iterate over every input array and append each input value corresponding to the ith output to the array
        for input_index in range(len(input_arrays)):
                ith_inputs.append(input_arrays[input_index][i])

        y_pred = predict(ith_inputs, weights, bias)
        total_mse += (y_pred - outputs[i]) ** 2
    return total_mse / n_output

weights, bias = train(input_arrays, outputs, weights, bias, lr, 10000)
print(weights)
print(bias)

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