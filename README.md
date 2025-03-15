# Gold Stock Price Prediction using LSTM

This project involves developing a predictive model using Long Short-Term Memory (LSTM) networks to forecast gold stock prices based on historical financial data. The model is trained on stock data and evaluated using performance metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

## Technologies Used

- Python
- PyTorch
- Pandas
- Matplotlib
- Scikit-learn
- Google Colab   

## Project Overview

The goal of this project is to predict future gold stock prices using an LSTM model, which is well-suited for time-series forecasting. The dataset includes historical stock prices such as **High**, **Low**, **Open**, and **Close** for each day. The data is preprocessed, normalized, and then used to train an LSTM network for price prediction.

## Dataset

The dataset used in this project is publicly available and includes historical stock prices for gold. The data is loaded directly from the following URL:

url = "https://raw.githubusercontent.com/Sai-Manasvi/gold-stock-prediction-using-lstm/refs/heads/main/goldstock.csv"

## Model Overview
The model is a simple LSTM network implemented using PyTorch with the following architecture:

Input Layer: Takes 4 features (High, Low, Open, Close).
Hidden Layer: 100 hidden units.
Output Layer: Predicts a single stock price (Close).

## Model Training
The LSTM model is trained for 10 epochs with an Adam optimizer and Mean Squared Error (MSE) as the loss function.

## Performance Metrics
Mean Absolute Error (MAE): The average of the absolute differences between predicted and actual stock prices.
Root Mean Squared Error (RMSE): Measures the square root of the mean squared errors between predicted and actual stock prices.
The model's performance is evaluated using the following metrics:
MAE: X
RMSE: Y

## Results
The predicted stock prices closely align with the actual values, showing the model’s effectiveness in forecasting stock prices.

## Future Improvements
-Hyperparameter tuning to optimize the model.   
-Incorporating more features such as trading volume and external factors (e.g., global financial indicators).   
-Use of advanced architectures like Bidirectional LSTMs or GRU (Gated Recurrent Units).    

## Colab link
https://colab.research.google.com/drive/1zPzxtpsuLRQXNkURDGsT93NqD-NjC6zZ?usp=sharing  
