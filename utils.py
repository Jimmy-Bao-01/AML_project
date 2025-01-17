from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def handle_outliers(dataframe, columns):
    """
    Replace outliers in specified columns with the closest marginal value.
    
    Parameters:
    - dataframe (pd.DataFrame): The DataFrame to process.
    - columns (list): List of column names to apply outlier handling.
    
    Returns:
    - pd.DataFrame: DataFrame with outliers replaced.
    """
    for col in columns:
        mean = dataframe[col].mean()
        std_dev = dataframe[col].std()
        upper_limit = mean + 2 * std_dev
        lower_limit = mean - 2 * std_dev
        
        dataframe[col] = np.where(dataframe[col] > upper_limit, upper_limit, dataframe[col])
        dataframe[col] = np.where(dataframe[col] < lower_limit, lower_limit, dataframe[col])
    
    return dataframe

def scale_columns(dataframe, columns):
    """
    Scale the specified columns of a DataFrame to the range [-0.9, 0.9].

    Parameters:
    - dataframe (pd.DataFrame): The input DataFrame.
    - columns (list): List of column names to scale.

    Returns:
    - pd.DataFrame: DataFrame with scaled columns.
    """
    scaler = MinMaxScaler(feature_range=(-0.9, 0.9))
    dataframe[columns] = scaler.fit_transform(dataframe[columns])
    return dataframe

def split_data(dataframe, train_ratio=0.7, val_ratio=0.15):
    """
    Split the data into training, validation, and testing sets.

    Parameters:
    - dataframe (pd.DataFrame): The input DataFrame to split.
    - train_ratio (float): Proportion of data for training (default: 0.7).
    - val_ratio (float): Proportion of data for validation (default: 0.15).

    Returns:
    - tuple: (train_data, val_data, test_data)
    """
    train_end = int(len(dataframe) * train_ratio)
    val_end = train_end + int(len(dataframe) * val_ratio)

    train_data = dataframe[:train_end]
    val_data = dataframe[train_end:val_end]
    test_data = dataframe[val_end:]

    return train_data, val_data, test_data

# Plot Predictions vs Actual Values
def plot_predictions(y_test, test_predictions, name_model,start_idx=0, end_idx=100):
    """
    Plot predictions vs actual values for a specified range.

    Parameters:
    - y_test (pd.Series): Actual target values from the test set.
    - test_predictions (np.ndarray): Predicted target values from the test set.
    - start_idx (int): Starting index for the range to plot.
    - end_idx (int): Ending index for the range to plot.
    """
    index = np.arange(abs(start_idx - end_idx))
    plt.figure(figsize=(10, 6))
    plt.plot(index, y_test.values[start_idx:end_idx], label='Actual Values', color='blue')
    plt.plot(index, test_predictions[start_idx:end_idx], label='Predicted Values', color='red', linestyle='--')
    plt.title(f'{name_model}: Predicted vs Actual')
    plt.xlabel('Index')
    plt.ylabel('RDP+5')
    plt.legend()
    plt.show()
    
# Custom NMSE Metric
def nmse_metric(y_true, y_pred):
    """
    Custom metric to calculate Normalized Mean Squared Error (NMSE).
    
    NMSE = MSE / Variance(y_true)
    """
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    var = tf.math.reduce_variance(y_true)
    nmse = tf.cond(tf.equal(var, 0), lambda: tf.constant(0.0), lambda: mse / var)
    
    return nmse