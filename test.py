import numpy as np

def predict(X, theta):
    z = np.dot(X, theta)

    return 1 / (1 + np.exp(-z))

X = [[22.3, -1.5, 1.1, 1]]
theta = [0.1, -0.15, 0.3, -0.2]

print(predict(X, theta))

def compute_loss(y_hat, y):
    y_hat = np.clip(y_hat, 1e-7, 1 - 1e-7)

    return (-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)).mean()

y = np.array([1, 0, 0, 1])
y_hat = np.array([0.8, 0.75, 0.3, 0.95])

print(f"{compute_loss(y_hat, y):.3f}")

def compute_gradient(X, y_true, y_pred):
    gradient = np.dot(np.transpose(X), (y_pred - y_true)) / y_true.size

    return gradient

X = [[1, 2], [2, 1], [1, 1], [2, 2]]
y_true = [0, 1, 0, 1]
y_pred = [0.25, 0.75, 0.4, 0.8]

print(compute_gradient(X, np.array(y_true).astype(float), np.array(y_pred).astype(float)))

def compute_accuracy(y_true, y_pred):
    y_pred_rounded = np.round(y_pred)
    accuracy = np.mean(y_true == y_pred_rounded)

    return accuracy

y_true = [1, 0, 1, 1]
y_pred = [0.85, 0.35, 0.9, 0.75]

print(compute_accuracy(y_true, y_pred))

X = [[1, 3], [2, 1], [3, 2], [1, 2]]
y_true = [1, 0, 1, 1]
y_pred = [0.7, 0.4, 0.6, 0.85]

print(compute_gradient(X, np.array(y_true).astype(float), np.array(y_pred).astype(float)))
