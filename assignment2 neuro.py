import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Inputs
x = np.array([0.05, 0.10])
y = np.array([0.01, 0.99])

# Initial Weights
w = np.array([
    [0.15, 0.20],  # Hidden Layer Weights
    [0.25, 0.30],
    [0.40, 0.45],  # Output Layer Weights
    [0.50, 0.55]
])

# Biases
b1, b2 = 0.35, 0.60

# Forward Pass
h1 = sigmoid(np.dot(w[0], x) + b1)
h2 = sigmoid(np.dot(w[1], x) + b1)

o1 = sigmoid(w[2][0] * h1 + w[2][1] * h2 + b2)
o2 = sigmoid(w[3][0] * h1 + w[3][1] * h2 + b2)

# Compute Error
error = 0.5 * ((y[0] - o1) ** 2 + (y[1] - o2) ** 2)

# Backpropagation (Updating Weights)
learning_rate = 0.5

d_o1 = (o1 - y[0]) * sigmoid_derivative(o1)
d_o2 = (o2 - y[1]) * sigmoid_derivative(o2)

w[2][0] -= learning_rate * d_o1 * h1
w[2][1] -= learning_rate * d_o1 * h2
w[3][0] -= learning_rate * d_o2 * h1
w[3][1] -= learning_rate * d_o2 * h2

# Backpropagation for Hidden Layer
error_h1 = (d_o1 * w[2][0] + d_o2 * w[3][0]) * sigmoid_derivative(h1)
error_h2 = (d_o1 * w[2][1] + d_o2 * w[3][1]) * sigmoid_derivative(h2)

w[0][0] -= learning_rate * error_h1 * x[0]
w[0][1] -= learning_rate * error_h1 * x[1]
w[1][0] -= learning_rate * error_h2 * x[0]
w[1][1] -= learning_rate * error_h2 * x[1]

print("Updated Weights:")
print(f"w1: {w[0][0]}")
print(f"w2: {w[0][1]}")
print(f"w3: {w[1][0]}")
print(f"w4: {w[1][1]}")
print(f"w5: {w[2][0]}")
print(f"w6: {w[2][1]}")
print(f"w7: {w[3][0]}")
print(f"w8: {w[3][1]}")
