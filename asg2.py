# a. Import the necessary packages (already done above)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

# b. Load the training and testing data (MNIST)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data: flatten images and normalize pixel values
x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0

# c. Define the network architecture using Keras
model = keras.Sequential([
layers.Input(shape=(784,)), # Input layer, 28*28 = 784 pixels
layers.Dense(units=256, activation="relu", name="hidden_layer_1"),
layers.Dense(units=128, activation="relu", name="hidden_layer_2"),
layers.Dense(units=10, activation="softmax", name="output_layer"), # Output layer for 10
classes
])

# Display the model summary
model.summary()

# d. Train the model using SGD
optimizer = keras.optimizers.SGD(learning_rate=0.01)
model.compile(optimizer=optimizer,
loss="sparse_categorical_crossentropy",
metrics=["accuracy"])

history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# e. Evaluate the network
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# f. Plot the training loss and accuracy
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Training Accuracy")

plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
