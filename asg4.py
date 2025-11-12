import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
 #Load your dataset
df = pd.read_csv('creditcard.csv')  # Uncomment and modify to load data

df = pd.DataFrame({
    'Time': np.random.uniform(0,100000,1000),
    'Amount': np.random.uniform(0,1000,1000),
    'feature1': np.random.normal(0,1,1000),
    'feature2': np.random.normal(0,1,1000),
    'target': np.random.choice([0,1], 1000, p=[0.95,0.05])
})
# Separate features and target
X = df.drop('target', axis=1)
y = df['target']
# Standardize only 'Amount' and 'Time' columns
scaler = StandardScaler()
X[['Amount', 'Time']] = scaler.fit_transform(X[['Amount', 'Time']])

# Split the data into normal (target=0) and fraud (target=1)
X_normal = X[y == 0]
X_fraud = X[y == 1]
# Use only normal data for training the autoencoder
X_train, X_test_normal = train_test_split(X_normal, test_size=0.2, random_state=42)
# Create test set combining normal and fraud data
X_test = pd.concat([X_test_normal, X_fraud])
y_test = np.concatenate([np.zeros(len(X_test_normal)), np.ones(len(X_fraud))])
# Define Autoencoder architecture
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoder = Dense(14, activation='relu')(input_layer)
encoder = Dense(7, activation='relu')(encoder)
bottleneck = Dense(3, activation='relu')(encoder)
decoder = Dense(7, activation='relu')(bottleneck)
decoder = Dense(14, activation='relu')(decoder)
output_layer = Dense(input_dim, activation='linear')(decoder)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
# Train the autoencoder
history = autoencoder.fit(
    X_train, X_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test_normal, X_test_normal),
    shuffle=True
)
# Plot training and validation loss
plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.title('Autoencoder Training and Validation Loss')
plt.show()
# Calculate reconstruction error on test set
X_test_pred = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - X_test_pred, 2), axis=1)
# Plot reconstruction error and threshold
plt.figure(figsize=(10,6))
plt.hist(mse[y_test == 0], bins=50, alpha=0.6, label='Normal')
plt.hist(mse[y_test == 1], bins=50, alpha=0.6, label='Fraud')
plt.axvline(x=52, color='r', linestyle='--', label='Threshold = 52')
plt.xlabel('Reconstruction Error (MSE)')
plt.ylabel('Number of samples')
plt.legend()
plt.title('Reconstruction Error Distribution')
plt.show()
# Detect anomalies based on threshold
threshold = 52
y_pred = (mse > threshold).astype(int)
# Evaluate results (optional - requires sklearn)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
