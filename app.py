import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
dataset_path = 'path/to/your/dataset.csv'  # Replace with the path to your .csv file
data = pd.read_csv(dataset_path)

# Assume the target variable is the last column
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split the data into training and testing sets
train_data, test_data, train_targets, test_targets = train_test_split(X, y, test_size=0.2, random_state=42)

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
