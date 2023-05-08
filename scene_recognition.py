import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Project Settings
INPUT_SIZE = (150, 150, 3)
DATA_DIR = './scene_recognition/train-scene classification'
IMAGE_DIR = os.path.join(DATA_DIR, 'train')
EPOCHS = 50

# Relu Activation Function


def relu(x):
    return np.maximum(0, x)

# Relu Derivative


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

# Dense Layer for the Neural Network


class DenseLayer:
    def __init__(self, input_dim, output_dim, activation):
        # Initialize the weights and biases
        self.W = np.random.randn(input_dim, output_dim) * \
            np.sqrt(2 / input_dim)
        self.b = np.zeros((1, output_dim))
        self.activation = activation

# Simple Deep Neural Network with Dense Layers


class DNN:
    def __init__(self, input_size, num_classes, hidden_layers=None):
        self.layers = []
        # Initialize the hidden layers
        if hidden_layers is None:
            hidden_layers = [128, 256]
        # Add the input layer
        prev_size = np.prod(input_size)
        for layer_size in hidden_layers:
            self.layers.append(DenseLayer(
                prev_size, layer_size, activation='relu'))
            prev_size = layer_size

        self.layers.append(DenseLayer(
            prev_size, num_classes, activation='softmax'))
    # Forward pass through the network

    def forward(self, x):
        # Flatten the input
        self.a = [x.reshape(x.shape[0], -1)]
        # Iterate through the layers
        for layer in self.layers[:-1]:
            z = np.dot(self.a[-1], layer.W) + layer.b
            a = relu(z)
            self.a.append(a)

        z = np.dot(self.a[-1], self.layers[-1].W) + self.layers[-1].b
        y_pred = softmax(z)
        return y_pred
    # Backward pass through the network

    def backward(self, x, y):
        m = x.shape[0]
        y_pred = self.forward(x)
        deltas = [y_pred - y]
        # Iterate through the layers in reverse order
        for i in range(len(self.layers) - 1, 0, -1):
            delta = np.dot(deltas[-1], self.layers[i].W.T) * \
                relu_derivative(self.a[i])
            deltas.append(delta)
        # Reverse the deltas
        deltas = deltas[::-1]

        gradients = []
        # Calculate the gradients
        for i in range(len(self.layers)):
            dw = np.dot(self.a[i].T, deltas[i]) / m
            db = np.sum(deltas[i], axis=0, keepdims=True) / m
            gradients.append((dw, db))

        return gradients
    # Update the model parameters

    def update(self, gradients, learning_rate):
        for i in range(len(self.layers)):
            self.layers[i].W -= learning_rate * gradients[i][0]
            self.layers[i].b -= learning_rate * gradients[i][1]
    # Train the model with the given data

    def evaluate(self, x, y):
        y_pred = self.forward(x)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        accuracy = accuracy_score(
            np.argmax(y, axis=1), np.argmax(y_pred, axis=1))
        return rmse, accuracy


# Load the data and split it into training and validation sets
def load_data(data_dir):
    train_csv = os.path.join(data_dir, 'train.csv')
    train_df = pd.read_csv(train_csv)
    x_data = []
    y_data = []
    # Iterate through the rows of the train.csv file
    for index, row in train_df.iterrows():
        img_path = os.path.join(IMAGE_DIR, row['image_name'])
        img = cv2.imread(img_path)
        img = cv2.resize(img, INPUT_SIZE[:2])
        img_array = img / 255.0
        # Add the image to the x_data list
        x_data.append(img_array)
        # Add the label to the y_data list
        y_data.append(row['label'])
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    # Split the data into training and validation sets
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.15, random_state=55)
    return x_train, y_train, x_test, y_test


def train_model(model, x_train, y_train, x_val, y_val, learning_rate, epochs, batch_size):
    # Train the model with the given data using stochastic gradient descent
    val_rmses = []  # Record validation RMSEs for plotting
    train_rmses = []  # Record training RMSEs for plotting

    for epoch in range(epochs):
        # Shuffle the training data
        p = np.random.permutation(len(x_train))
        x_train = x_train[p]
        y_train = y_train[p]

        # Iterate through the mini-batches
        for i in range(0, len(x_train), batch_size):
            x_batch = x_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            # Calculate gradients and update the model parameters
            gradients = model.backward(x_batch, y_batch)
            model.update(gradients, learning_rate=learning_rate)

        # Evaluate the model on the validation set
        train_rmse, _ = model.evaluate(x_train, y_train)
        val_rmse, _ = model.evaluate(x_val, y_val)
        val_rmses.append(val_rmse)
        train_rmses.append(train_rmse)
        print(
            f"Epoch {epoch+1}/{epochs} - Training RMSE: {train_rmse:.4f}, Validation RMSE: {val_rmse:.4f}")

    return train_rmses, val_rmses

# Evaluate the model on the test set


def evaluate_model(model, x_test, y_test):
    loss, accuracy = model.evaluate(x_test, y_test)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)

# Plot the training and validation RMSEs


def plot_validation_accuracy(val_accuracies):
    plt.plot(val_accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Validation RMSE')
    plt.title('Validation RMSE vs. Epoch')
    plt.show()


def main():
    # Set up logging
    logging.basicConfig(
        filename="experiment_logs.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    # Load the data
    x_train, y_train, x_test, y_test = load_data(DATA_DIR)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.15, random_state=55)

    # Convert labels to one-hot vectors
    y_train = pd.get_dummies(y_train).to_numpy()
    y_val = pd.get_dummies(y_val).to_numpy()
    y_test = pd.get_dummies(y_test).to_numpy()

    num_classes = y_train.shape[1]
    # Custom hyperparameter sets
    hyperparameter_sets = [
        {'learning_rate': 0.0001, 'hidden_layers': [
            128, 256], 'batch_size': 16},
        {'learning_rate': 0.0001, 'hidden_layers': [
            64, 128, 256], 'batch_size': 32},
        {'learning_rate': 0.001, 'hidden_layers': [
            128, 256], 'batch_size': 16},
        {'learning_rate': 0.001, 'hidden_layers': [
            64, 128, 256], 'batch_size': 32},
        {'learning_rate': 0.0001, 'hidden_layers': [
            64, 128, 256], 'batch_size': 64},
    ]
    # Train the model with each hyperparameter set
    for hyperparameters in hyperparameter_sets:
        print(f"Hyperparameters: {hyperparameters}")
        model = DNN(INPUT_SIZE, num_classes,
                    hidden_layers=hyperparameters['hidden_layers'])
        train_rmses, val_rmses = train_model(
            model,
            x_train,
            y_train,
            x_val,
            y_val,
            learning_rate=hyperparameters['learning_rate'],
            epochs=EPOCHS,
            batch_size=hyperparameters['batch_size']
        )

        evaluate_model(model, x_test, y_test)
        plot_validation_accuracy(val_rmses)

        logging.info(
            f"Hyperparameters: {hyperparameters} - Final Training RMSE: {train_rmses[-1]:.4f}, Final Validation RMSE: {val_rmses[-1]:.4f}")


if __name__ == "__main__":
    main()
