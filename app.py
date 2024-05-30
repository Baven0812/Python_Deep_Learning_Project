import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import boston_housing

# Load the dataset
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# Normalize the data
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# Build the model
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(train_data, train_targets, epochs=100, batch_size=32, validation_split=0.2, verbose=2)

# Evaluate the model
test_loss, test_mae = model.evaluate(test_data, test_targets, verbose=2)
print(f"Test MAE: {test_mae}")

# Make predictions
predictions = model.predict(test_data)

# Compare some of the predictions with actual values
for i in range(10):
    print(f"Predicted: {predictions[i][0]}, Actual: {test_targets[i]}")

# Plot training and validation loss
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
