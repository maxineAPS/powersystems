import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(file_path, target_columns, sample_rate):
    # Load the dataset
    data = pd.read_csv(file_path, skiprows=2)

    # Combine timestamp columns into a single datetime and drop them
    data['datetime'] = pd.to_datetime(data[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    data = data.set_index('datetime')

    # Focus on the target columns
    data = data[target_columns]

    # Normalize the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Subsample the data
    data_subsampled = data_scaled[::sample_rate]

    return data_subsampled, scaler

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

def split_data(data, sequence_length, split_ratio):
    split_idx = int(len(data) * split_ratio)
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    X_train, y_train = create_sequences(train_data, sequence_length)
    X_test, y_test = create_sequences(test_data, sequence_length)

    return X_train, y_train, X_test, y_test
