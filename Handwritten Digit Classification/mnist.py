import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

# Load the dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess the data
train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Define the neural network
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.biases_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
        self.biases_output = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, x):
        self.z1 = np.dot(x, self.weights_input_hidden) + self.biases_hidden
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.weights_hidden_output) + self.biases_output
        self.a2 = self.softmax(self.z2)
        return self.a2

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
        return loss

    def backward(self, x, y_true, y_pred, learning_rate=0.01):
        m = x.shape[0]

        dz2 = y_pred - y_true
        dw2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        da1 = np.dot(dz2, self.weights_hidden_output.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dw1 = np.dot(x.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        self.weights_input_hidden -= learning_rate * dw1
        self.biases_hidden -= learning_rate * db1
        self.weights_hidden_output -= learning_rate * dw2
        self.biases_output -= learning_rate * db2

    def train(self, x_train, y_train, epochs=10, learning_rate=0.01):
        for epoch in range(epochs):
            y_pred = self.forward(x_train)
            loss = self.compute_loss(y_train, y_pred)
            self.backward(x_train, y_train, y_pred, learning_rate)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}')

    def predict(self, x):
        y_pred = self.forward(x)
        return np.argmax(y_pred, axis=1)

# Initialize the network
input_size = 28 * 28
hidden_size = 64
output_size = 10
network = NeuralNetwork(input_size, hidden_size, output_size)

# Train the network
network.train(train_images, train_labels, epochs=10, learning_rate=0.01)

# Evaluate the network
predictions = network.predict(test_images)
accuracy = np.mean(predictions == np.argmax(test_labels, axis=1))
print(f'Test Accuracy: {accuracy:.4f}')
