import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import accuracy_score
import random
# Load MNIST dataset (28x28 grayscale images of digits)
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# Normalize pixel values to [0,1] range for easier network training
X_train = X_train / 255.0
X_test = X_test / 255.0
# Add channel dimension for grayscale (batch_size, 28, 28, 1)
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]
# Function to plot images with labels
def plot_digit(image, digit, i):
    plt.subplot(4, 5, i+1)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f"Digit: {digit}")
    plt.axis('off')
# Plot first 20 training images
plt.figure(figsize=(16, 10))
for i in range(20):
    plot_digit(X_train[i], y_train[i], i)
plt.show()
# Build CNN model using Keras Sequential API
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(100, activation='relu'),
    Dense(10, activation='softmax')  # 10 output classes for digits 0-9
])
# Compile model with SGD optimizer and sparse categorical crossentropy loss
model.compile(
    optimizer=SGD(learning_rate=0.01, momentum=0.9),
    loss='sparse_categorical_crossentropy',  # use sparse since labels are integers
    metrics=['accuracy']
)

model.summary()
# Train model for 10 epochs with batch size 32
model.fit(X_train, y_train, epochs=10, batch_size=32)
# Evaluate model accuracy on test set
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy:.4f}")
# Predict 20 random test images and plot predictions
plt.figure(figsize=(16, 10))
for i in range(20):
    idx = random.randint(0, len(X_test)-1)
    image = X_test[idx]
    pred = np.argmax(model.predict(image[np.newaxis, ...])[0])
    plot_digit(image, pred, i)
plt.show()
# Calculate final accuracy over the entire test set using sklearn (optional)
predictions = np.argmax(model.predict(X_test), axis=-1)
print(f"Sklearn accuracy score: {accuracy_score(y_test, predictions):.4f}")
# Save model architecture and weights to disk for reuse
model_json = model.to_json()
with open("model_digit.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model_digit.weights.h5")

print("Saved model to disk")
