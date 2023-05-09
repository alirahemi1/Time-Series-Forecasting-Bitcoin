# Time-Series-Forecasting-Bitcoin

**Time Series Forecasting of Bitcoin Prices with LSTM**

This repository contains a time series forecasting project using a deep learning model with LSTM layers to predict Bitcoin prices.

**Project Overview**

The project involves creating a machine learning model to predict Bitcoin prices based on historical data. The model leverages LSTM layers to understand and utilize the temporal relationships within the data.

**Libraries Used**

The project utilizes several Python libraries, including:

* pandas
* numpy
* tensorflow
* keras
* sklearn
* matplotlib
* Data Processing

The data is loaded from a CSV file, Dataset.csv, containing Bitcoin prices and other relevant features. The data is then split into training and testing sets, with 80% of the data used for training and 20% for testing. The data is normalized using MinMaxScaler from the sklearn library.

**Model**

The model is built using the Keras Sequential API and includes:

* A Bidirectional LSTM layer with 75 units, followed by a Dropout layer with a rate of 0.1
* Another Bidirectional LSTM layer with 75 units, followed by a Dropout layer with a rate of 0.3
* A final Bidirectional LSTM layer with 75 units, followed by a Dropout layer with a rate of 0.1
* A Dense layer with a single unit as the output layer
* The model is compiled with the Adam optimizer and the mean squared error loss function.

**Training**

The model is trained for 50 epochs with a batch size of 62. Early stopping and model checkpointing are used during training to improve the model's performance and prevent overfitting. The best model is saved to a file named best_model.h5.

**Evaluation**

The model's performance is evaluated on the test data using the Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) metrics. The predicted values are plotted against the actual values for visual comparison.



