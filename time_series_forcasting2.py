import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
tf.config.run_functions_eagerly(True)
from keras.layers import Bidirectional


# Load the dataset
data = pd.read_csv('/Users/alirahemi/Desktop/Dataset.csv', parse_dates=['Date'], index_col='Date')

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

# Define the number of time steps and features
time_steps = 60
features = 8

# Create a function to reshape the data into input/output samples
def create_samples(dataset):
    X, Y = [], []
    for i in range(len(dataset)-time_steps-1):
        X.append(dataset[i:i+time_steps, :])
        Y.append(dataset[i+time_steps, 0])
    return np.array(X), np.array(Y)

# Create the input/output samples
X_train, Y_train = create_samples(train_data_scaled)
X_test, Y_test = create_samples(test_data_scaled)

# Define the model architecture
model = Sequential()
model.add(Bidirectional(LSTM(75, return_sequences=True), input_shape=(time_steps, features)))
model.add(Dropout(0.1))
model.add(Bidirectional(LSTM(75, return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(75)))
model.add(Dropout(0.1))
model.add(Dense(1))

# Example of using RMSprop
optimizer = RMSprop(lr=0.0001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

model.compile(optimizer='adam', loss='mean_squared_error', run_eagerly=True)


# Example of using SGD with a momentum term
#optimizer = SGD(lr=0.01, momentum=0.9)
#model.compile(optimizer=optimizer, loss='mean_squared_error')

# Compile the model
#model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, Y_train, epochs=50, batch_size=62)

# Make predictions on the test set
predictions = model.predict(X_test)
predictions = np.concatenate([predictions, np.zeros((len(predictions), features - 1))], axis=1)
predictions = scaler.inverse_transform(predictions)[:, 0]

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

model.fit(X_train, Y_train, epochs=50, batch_size=62, validation_split=0.1, callbacks=[early_stopping, checkpoint])

# Compute the index for predictions
predictions_index = test_data.index[time_steps+1:]

rmse = np.sqrt(mean_squared_error(test_data['BTC'][time_steps+1:], predictions))
mae = mean_absolute_error(test_data['BTC'][time_steps+1:], predictions)
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

# Plot the actual and predicted values
plt.figure(figsize=(15, 6))
plt.plot(test_data.index, test_data['BTC'], label='Actual')
plt.plot(predictions_index, predictions, label='Predicted')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.title('BTC Price Predictions')
plt.legend()
plt.show()
