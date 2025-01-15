# Stock Price Prediction using RNN and LSTM

This project demonstrates how to predict stock prices using **Recurrent Neural Networks (RNNs)** and **Long Short-Term Memory (LSTM)** networks. The model is trained on historical stock price data to predict future stock prices based on past trends.

## Overview

In this project, we used **RNN** and **LSTM**, two types of neural networks specifically designed for sequential data. The goal is to predict stock prices by learning patterns from the time-series data. We compare the performance of a basic RNN model with an LSTM model, which is an advanced type of RNN.

## Table of Contents
1. [Introduction](#introduction)
2. [RNN and LSTM Explanation](#rnn-and-lstm-explanation)
   - [RNN](#rnn)
   - [LSTM](#lstm)
3. [Mathematics Behind RNN and LSTM](#mathematics-behind-rnn-and-lstm)
4. [Dataset](#dataset)
5. [Project Steps](#project-steps)
6. [Model Evaluation](#model-evaluation)
7. [Model Deployment](#model-deployment)
8. [Results](#results)
9. [Conclusion](#conclusion)

## Introduction

The stock market is highly dynamic, and predicting stock prices is a challenging task. However, with the use of deep learning algorithms like RNN and LSTM, we can capture the temporal dependencies in stock prices and make reasonable predictions about future prices.

In this project, we use **historical stock prices** to train our models and predict future stock values. We then compare the performance of an RNN model and an LSTM model, which is a more advanced form of RNN.

## RNN and LSTM Explanation

### RNN

A **Recurrent Neural Network (RNN)** is a type of neural network designed for sequential data, such as time series. RNNs are used to predict outputs based on both the current input and previous inputs. The key feature of an RNN is its ability to maintain an internal state (memory) that can capture information about past inputs.

RNNs are widely used in applications where the data has a temporal dimension, such as stock price prediction, speech recognition, and natural language processing.

However, standard RNNs suffer from the **vanishing gradient problem**, which limits their ability to capture long-term dependencies in the data.

### LSTM

**Long Short-Term Memory (LSTM)** is a type of RNN designed to address the vanishing gradient problem. LSTM introduces **gates** that control the flow of information, allowing the network to remember long-term dependencies. These gates help the LSTM decide which information to keep and which to forget, making it well-suited for time-series data with long-range dependencies.

LSTM is particularly useful for tasks like stock price prediction, where trends and patterns can span across long periods.

## Mathematics Behind RNN and LSTM

### RNN

In an RNN, the output at each time step depends not only on the current input but also on the previous output. Mathematically, the RNN equations can be written as:

- **Hidden state update**:  
  \[
  h_t = \sigma(W_h \cdot h_{t-1} + W_x \cdot x_t + b)
  \]
  where:
  - \( h_t \) is the hidden state at time \( t \)
  - \( x_t \) is the input at time \( t \)
  - \( W_h, W_x \) are weight matrices
  - \( b \) is the bias term
  - \( \sigma \) is the activation function (e.g., tanh or ReLU)

### LSTM

LSTM networks have three primary gates: the **input gate**, the **forget gate**, and the **output gate**. These gates help LSTM networks control the flow of information:

- **Forget gate**: Decides what information should be discarded from the cell state.
- **Input gate**: Determines what new information should be added to the cell state.
- **Output gate**: Controls what information from the cell state should be output.

The LSTM equations are more complex, and they include:
- **Cell state update**:
  \[
  C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{C}_t
  \]
  where:
  - \( C_t \) is the cell state at time \( t \)
  - \( f_t, i_t \) are the forget and input gates
  - \( \tilde{C}_t \) is the candidate cell state

- **Hidden state update**:
  \[
  h_t = o_t \cdot \sigma(C_t)
  \]
  where:
  - \( o_t \) is the output gate

## Dataset

### Source: HPQ Stock Market Dataset

The dataset used in this project is the **HPQ.csv** file, which contains historical stock market data for HPQ (Hewlett-Packard). The dataset includes the following columns:

- **Date**: The date of the trading day.
- **Open**: The price at which the stock opened.
- **High**: The highest price during the trading day.
- **Low**: The lowest price during the trading day.
- **Close**: The closing price of the stock.
- **Adj Close**: The adjusted closing price.
- **Volume**: The number of shares traded.

The dataset includes **14663 entries**, spanning from January 1962 to the present.

## Project Steps

1. **Data Preprocessing**:  
   - Load and clean the dataset.
   - Convert the 'Date' column to datetime format.
   - Normalize the stock price data using MinMaxScaler.
   
2. **Data Preparation**:  
   - Create sequences of data to be used as inputs and outputs for training.
   - Split the data into training and test sets.
   
3. **Model Building**:  
   - Build a basic **RNN** model.
   - Implement a more advanced **LSTM** model.

4. **Training**:  
   - Train the RNN and LSTM models on the training data.
   - Visualize the training and validation loss.

5. **Evaluation**:  
   - Evaluate the models using performance metrics like RMSE and MAE.
   - Compare actual vs predicted stock prices.

6. **Model Deployment**:  
   - Save and load the trained models.
   - Predict future stock prices using the trained models.

## Model Evaluation

After training the RNN and LSTM models, we evaluate their performance based on Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE). We also visualize the predicted stock prices against actual stock prices to assess the models' predictive accuracy.

## Results

- **RNN Model**: The basic RNN model showed reasonable accuracy, but struggles with capturing long-term dependencies in the data.
- **LSTM Model**: The LSTM model performed better due to its ability to remember long-term patterns, making it more suitable for stock price prediction.

## Conclusion

In this project, we demonstrated the power of RNNs and LSTMs for stock price prediction. While both models showed promising results, the LSTM model outperformed the basic RNN model due to its ability to capture long-term dependencies in the stock price data. Further improvements can be made by tuning hyperparameters and experimenting with more complex models.

## Requirements

To run this project, you'll need the following Python libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tensorflow`

You can install the required libraries using pip:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
