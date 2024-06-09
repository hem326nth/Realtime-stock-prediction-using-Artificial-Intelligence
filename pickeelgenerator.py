import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pickle

# Path to the directory containing CSV files
folder_path = r'C:\Users\heman\OneDrive\Documents\company_maang[1]\company_maang'

# Function to preprocess data
def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    X = df[['Date']].copy()
    y = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()
    X['Date'] = X['Date'].apply(lambda x: x.toordinal())
    return X, y

# Train the linear regression model
def train_model(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = Ridge(alpha=0.1)  # Example hyperparameter
    model.fit(X_train_scaled, y_train)
    return model, scaler  # Return the fitted scaler along with the model

# Evaluate the model
def evaluate_model(model, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)  # Use the same scaler fitted on the training data
    y_pred = model.predict(X_test_scaled)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print("Root Mean Squared Error:", rmse)

# Example usage:
def train_and_save_model(folder_path, csv_file):
    file_path = os.path.join(folder_path, csv_file)
    df = pd.read_csv(file_path)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model, scaler = train_model(X_train, y_train)  # Get both the model and scaler
    evaluate_model(model, scaler, X_test, y_test)  # Pass the scaler to the evaluation function
    # Save the trained model and scaler as pickle files
    model_file_name = csv_file.split('.')[0] + '_model.pkl'
    model_file_path = os.path.join(folder_path, model_file_name)
    with open(model_file_path, 'wb') as model_file:
        pickle.dump((model, scaler), model_file)
    print("Model saved as:", model_file_path)

# Example usage:
csv_files = ['NETFLIX_daily.csv', 'META_daily.csv', 'GOOGLE_daily.csv', 'APPLE_daily.csv', 'AMAZON_daily.csv']
for csv_file in csv_files:
    train_and_save_model(folder_path, csv_file)
