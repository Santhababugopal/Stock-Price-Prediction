# Stock-Price-Prediction


## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset

Stock price prediction is a challenging task due to the non-linear and volatile nature of financial markets. Traditional methods often fail to capture complex temporal dependencies. Deep learning, specifically Recurrent Neural Networks (RNNs), can effectively model time-series dependencies, making them suitable for stock price forecasting.

Problem Statement: Build an RNN model to predict the future stock price based on past stock price data.

Dataset: A stock market dataset containing historical daily closing prices (e.g., Google, Apple, Tesla, or NSE/BSE data). The dataset is usually divided into training and testing sets after applying normalization and sequence generation.

##TRAIN SET##:



<img width="754" height="885" alt="image" src="https://github.com/user-attachments/assets/024ba15e-8532-4de6-8d31-6910b03dcf2b" />


##TEST SET##:



<img width="779" height="870" alt="image" src="https://github.com/user-attachments/assets/4828e0a8-b225-4cba-aa69-9c8c4c7cb801" />

## Design Steps

Step 1:

Import required libraries such as torch, torch.nn, torch.optim, numpy, pandas, and matplotlib.

Step 2:

Load the dataset (e.g., stock closing prices from CSV), preprocess it by normalizing values between 0 and 1, and create input sequences for training/testing.

Step 3:

Define the RNN model architecture with an input layer, hidden layers, and an output layer to predict stock prices.

Step 4:

Compile the model using MSELoss as the loss function and Adam optimizer.

Step 5:

Train the model on the training data, recording training losses for each epoch.

Step 6:

Test the trained model on unseen data and visualize results by plotting the true stock prices vs. predicted stock prices.



## Program
#### Name:SANTHABABU G
#### Register Number:212224040292

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset



df_train = pd.read_csv('/content/trainset.csv')
df_test = pd.read_csv('/content/testset.csv')


# Use closing prices
train_prices = df_train['Close'].values.reshape(-1, 1)
test_prices = df_test['Close'].values.reshape(-1, 1)


# Normalize the data based on training set only
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_prices)
scaled_test = scaler.transform(test_prices)


# Create sequences
def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

seq_length = 60
x_train, y_train = create_sequences(scaled_train, seq_length)
x_test, y_test = create_sequences(scaled_test, seq_length)


x_train.shape, y_train.shape, x_test.shape, y_test.shape

x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


# Create dataset and dataloader
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(RNNModel, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # last time step
        return out


model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

!pip install torchinfo

from torchinfo import summary

# input_size = (batch_size, seq_len, input_size)
summary(model, input_size=(64, 60, 1))


import torch.optim as optim
model = RNNModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, criterion, optimizer, num_epochs):
  train_losses = []
  for epoch in range(num_epochs):
      model.train()
      epoch_loss = 0
      for x_batch, y_batch in train_loader:
          x_batch = x_batch.to(device)
          y_batch = y_batch.to(device)

          optimizer.zero_grad()
          outputs = model(x_batch)
          loss = criterion(outputs, y_batch)
          loss.backward()
          optimizer.step()

          epoch_loss += loss.item()

      epoch_loss /= len(train_loader)
      train_losses.append(epoch_loss)
      if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}')
  return train_losses

  train_losses=train_model(model, train_loader, criterion, optimizer, 50)

# Plot training loss
print('Name: SANTHABABU G')
print('Register Number: 212224040292')
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()


model.eval()
with torch.no_grad():
    predicted = model(x_test_tensor.to(device)).cpu().numpy()
    actual = y_test_tensor.cpu().numpy()

# Inverse transform the predictions and actual values
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(actual)

# Plot the predictions vs actual prices
print('Name:SANTHABABU G')
print('Register Number: 212224040292')
plt.figure(figsize=(10, 6))
plt.plot(actual_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Stock Price Prediction using RNN')
plt.legend()
plt.show()
print(f'Predicted Price: {predicted_prices[-17]}')
print(f'Actual Price: {actual_prices[-17]}')


```

## Output


### True Stock Price, Predicted Stock Price vs time
<img width="799" height="555" alt="image" src="https://github.com/user-attachments/assets/d6e39a60-4e80-4782-b163-5a2954389278" />


### Predictions 

<img width="1091" height="723" alt="image" src="https://github.com/user-attachments/assets/1d5f7677-5f10-4c12-813d-2de45fd91100" />


## Result


The RNN model was successfully implemented for stock price prediction.

