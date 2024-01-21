from data_collection import fetch_stock_data
from data_preparation import preprocess_data
from feature_selection import select_features
from model import train_model, evaluate_model
from visualization import plot_predictions


def main():
    # Example usage
    stock_symbol = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2021-01-01'

    data = fetch_stock_data(stock_symbol, start_date, end_date)
    processed_data = preprocess_data(data)
    features = select_features(processed_data)
    target = processed_data['Close']

    model, X_test, y_test = train_model(features, target)
    mse = evaluate_model(model, X_test, y_test)
    print(f"Mean Squared Error: {mse}")

    predictions = model.predict(X_test)
    plot_predictions(y_test, predictions)


if __name__ == "__main__":
    main()
