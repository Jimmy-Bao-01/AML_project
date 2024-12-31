from sklearn.preprocessing import MinMaxScaler
import numpy as np

def scale_and_handle_outliers(data, columns):
    for col in columns:
        # Handle outliers
        std_dev = data[col].std()
        mean = data[col].mean()
        data[col] = np.where(data[col] > mean + 2 * std_dev, mean + 2 * std_dev, data[col])
        data[col] = np.where(data[col] < mean - 2 * std_dev, mean - 2 * std_dev, data[col])

    # Scale to [-0.9, 0.9]
    scaler = MinMaxScaler(feature_range=(-0.9, 0.9))
    data[columns] = scaler.fit_transform(data[columns])

    return data
