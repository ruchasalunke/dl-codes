import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
train_dir = r"C:\Users\racha\Downloads\mnist-jpg\mnist-jpg\train"
test_dir  = r"C:\Users\racha\Downloads\mnist-jpg\mnist-jpg\test"
# --- hyperparameters ---
IMG_SIZE = (28, 28)
BATCH = 128
EPOCHS = 5      
LR = 0.01
MOM = 0.9
NUM_CLASSES = 10
# --- load datasets from folders (labels inferred from folder names 0..9) ---
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir, image_size=IMG_SIZE, color_mode="grayscale",
    batch_size=BATCH, label_mode="int", shuffle=True, seed=123
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir, image_size=IMG_SIZE, color_mode="grayscale",
    batch_size=BATCH, label_mode="int", shuffle=False
)
# --- normalize (0-1) ---
normalizer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x,y: (normalizer(x), y))
test_ds  = test_ds.map(lambda x,y: (normalizer(x), y))
# --- simple feedforward model ---
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# --- compile with SGD ---
model.compile(optimizer=optimizers.SGD(learning_rate=LR, momentum=MOM),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# --- train ---
history = model.fit(train_ds, epochs=EPOCHS, validation_data=test_ds)
# --- evaluate ---
loss, acc = model.evaluate(test_ds)
print(f"Test loss: {loss:.4f}   Test accuracy: {acc:.4f}")
# --- simple plots ---
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history.get('val_loss', []), label='val loss')
plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history.get('val_accuracy', []), label='val acc')
plt.xlabel('epoch'); plt.ylabel('accuracy'); plt.legend()

plt.tight_layout()
plt.show()
