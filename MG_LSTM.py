import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load data from the .txt file
data = pd.read_csv('Documents/PGProject/txts/MackeyGlass_t17.txt', header=None)
data = data.values.astype('float32')

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data)

# Prepare the data for LSTM
X = []
y = []
for i in range(len(data_normalized) - 1):
    X.append(data_normalized[i])
    y.append(data_normalized[i + 1])
X, y = np.array(X), np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=False)

# Reshape the data for LSTM input
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Build the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), activation='tanh'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Make predictions
y_pred = model.predict(X_test)

# Inverse transform the predictions
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test_inv[:1000], y_pred_inv[:1000])
rmse = np.sqrt(mse)
nrmse = rmse / (np.max(y_test_inv[:1000]) - np.min(y_test_inv[:1000]))
mae = mean_absolute_error(y_test_inv[:1000], y_pred_inv[:1000])

# Print the evaluation metrics
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Normalized Root Mean Squared Error (NRMSE): {nrmse}')
print(f'Mean Absolute Error (MAE): {mae}')

# Plot the actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv[:2000], 'b', label='Actual')
plt.plot(y_pred_inv[:2000], 'r--', label='Predicted')
plt.legend()
plt.title('Mackey-Glass System Prediction (LSTM)')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.show()
