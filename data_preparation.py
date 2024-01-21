import pandas as pd


def preprocess_data(stock_data):
    stock_data.dropna(inplace=True)  # Remove any NaN values
    return stock_data
