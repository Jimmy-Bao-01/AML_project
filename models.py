import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from itertools import product

from utils import nmse_metric


# SVM Regression Training and Evaluation
def train_and_evaluate_svm(train_data, val_data, test_data, features, target):
    """
    Train and evaluate an SVM regression model on the given data.

    Parameters:
    - train_data (pd.DataFrame): Training dataset.
    - val_data (pd.DataFrame): Validation dataset.
    - test_data (pd.DataFrame): Testing dataset.
    - features (list): List of feature column names.
    - target (str): Name of the target column.

    Returns:
    - dict: A dictionary containing the model, predictions, and evaluation metrics.
    """
    # Prepare training, validation, and testing data
    X_train = train_data[features]
    y_train = train_data[target]

    X_val = val_data[features]
    y_val = val_data[target]

    X_test = test_data[features]
    y_test = test_data[target]

    # Train the SVR model
    svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr.fit(X_train, y_train)

    # Validate the model
    val_predictions = svr.predict(X_val)
    val_mse = mean_squared_error(y_val, val_predictions)
    val_nmse = val_mse / np.var(y_val)
    val_r2 = r2_score(y_val, val_predictions)

    print(f"Validation MSE: {val_mse:.4f}")
    print(f"Validation NMSE: {val_nmse:.4f}")
    print(f"Validation R²: {val_r2:.4f}")

    # Test the model
    test_predictions = svr.predict(X_test)
    test_mse = mean_squared_error(y_test, test_predictions)
    test_nmse = test_mse / np.var(y_test)
    test_r2 = r2_score(y_test, test_predictions)

    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test NMSE: {test_nmse:.4f}")
    print(f"Test R²: {test_r2:.4f}")

    # Return results
    return {
        "model": svr,
        "val_predictions": val_predictions,
        "test_predictions": test_predictions,
        "y_test": y_test,
        "validation_metrics": {"mse": val_mse, "r2": val_r2, "nmse": val_nmse},
        "test_metrics": {"mse": test_mse, "r2": test_r2, "nmse": test_nmse},
    }


def optimize_svr_nmse(train_data, val_data, features, target, param_grid):
    """
    Optimize the SVR hyperparameters using custom NMSE-based scoring.

    Parameters:
    - train_data (pd.DataFrame): Training dataset.
    - val_data (pd.DataFrame): Validation dataset.
    - features (list): List of feature column names.
    - target (str): Name of the target column.
    - param_grid (dict): Dictionary containing hyperparameter ranges for 'C', 'epsilon', and 'gamma'.

    Returns:
    - dict: Best SVR model, hyperparameters, and evaluation metrics.
    """
    # Extract hyperparameter ranges from param_grid
    C_values = param_grid.get("C", [1.0])
    epsilon_values = param_grid.get("epsilon", [0.1])
    gamma_values = param_grid.get("gamma", ["scale"])

    # Prepare training and validation data
    X_train = train_data[features].values
    y_train = train_data[target].values
    X_val = val_data[features].values
    y_val = val_data[target].values

    best_nmse = float('inf')
    best_params = None
    best_model = None

    # Iterate through all hyperparameter combinations
    for C, epsilon, gamma in product(C_values, epsilon_values, gamma_values):
        try:
            print(f"Evaluating SVR(C={C}, epsilon={epsilon}, gamma={gamma})...")
            model = SVR(kernel='rbf', C=C, epsilon=epsilon, gamma=gamma)
            model.fit(X_train, y_train)
            val_predictions = model.predict(X_val)

            # Calculate NMSE
            mse = mean_squared_error(y_val, val_predictions)
            nmse = mse / np.var(y_val)

            print(f"NMSE: {nmse:.4f}")

            if nmse < best_nmse:
                best_nmse = nmse
                best_params = {"C": C, "epsilon": epsilon, "gamma": gamma}
                best_model = model

        except Exception as e:
            print(f"SVR(C={C}, epsilon={epsilon}, gamma={gamma}) failed: {e}")

    val_predictions = best_model.predict(X_val)
    mse = mean_squared_error(y_val, val_predictions)
    r2 = r2_score(y_val, val_predictions)
    
    print(f"Best Hyperparameters: {best_params}")
    print(f"Best NMSE: {best_nmse:.4f}")
    print(f"Validation MSE: {mse:.4f}")
    print(f"Validation R²: {r2:.4f}")

    return {
        "best_model": best_model,
        "best_params": best_params,
        "best_nmse": best_nmse,
        "mse": mse,
        "r2": r2
    }

# ARIMAX Model
def run_arimax(train_data, test_data, exog_train, exog_test, order=(1, 1, 1)):
    """
    Fit and evaluate an ARIMAX model.

    Parameters:
    - train_data (pd.Series): Training target data.
    - test_data (pd.Series): Testing target data.
    - exog_train (pd.DataFrame): Exogenous variables for training.
    - exog_test (pd.DataFrame): Exogenous variables for testing.
    - target (str): Name of the target column.
    - order (tuple): ARIMAX model order (p, d, q).

    Returns:
    - dict: ARIMAX model, predictions, and evaluation metrics.
    """
    # Fit ARIMAX model
    model = SARIMAX(train_data, exog=exog_train, order=order)
    model_fitted = model.fit()

    # Predict on the test set
    predictions = model_fitted.forecast(steps=len(test_data), exog=exog_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(test_data, predictions)
    r2 = r2_score(test_data, predictions)
    nmse = mse / np.var(test_data)

    print(f"ARIMAX ({order}) Test MSE: {mse:.4f}")
    print(f"ARIMAX ({order}) Test R²: {r2:.4f}")
    print(f"ARIMAX ({order}) Test NMSE: {nmse:.4f}")

    return {
        "model": model_fitted,
        "predictions": predictions,
        "mse": mse,
        "r2": r2,
        "nmse": nmse
    }
    
def optimize_arimax(train_data, exog_train, val_data, exog_val, param_grid_arimax):
    """
    Optimize the ARIMAX model using a validation set and a parameter grid for hyperparameter tuning.

    Parameters:
    - train_data (pd.Series): Training target data.
    - exog_train (pd.DataFrame): Exogenous variables for training.
    - val_data (pd.Series): Validation target data.
    - exog_val (pd.DataFrame): Exogenous variables for validation.
    - param_grid_arimax (dict): Dictionary with lists of 'p_values', 'd_values', and 'q_values'.

    Returns:
    - dict: Best ARIMAX model, its parameters, and evaluation metrics.
    """
    # Extract p, d, q values from the parameter grid
    p_values = param_grid_arimax.get("p_values", [0])
    d_values = param_grid_arimax.get("d_values", [0])
    q_values = param_grid_arimax.get("q_values", [0])

    best_nmse = float('inf')
    best_order = None
    best_model = None
    best_mse = None
    best_r2 = None

    # Iterate through all combinations of p, d, q
    for (p, d, q) in product(p_values, d_values, q_values):
        try:
            print(f"Evaluating ARIMAX({p},{d},{q})...")
            model = SARIMAX(train_data, exog=exog_train, order=(p, d, q))
            model_fitted = model.fit(disp=False)
            predictions = model_fitted.forecast(steps=len(val_data), exog=exog_val)

            # Calculate metrics
            mse = mean_squared_error(val_data, predictions)
            nmse = mse / np.var(val_data)
            r2 = r2_score(val_data, predictions)
            
            print(f'nmse: {nmse}')

            # Update best model if current one is better
            if nmse < best_nmse:
                best_nmse = nmse
                best_order = (p, d, q)
                best_model = model_fitted
                best_mse = mse
                best_r2 = r2

        except Exception as e:
            print(f"ARIMAX({p},{d},{q}) failed: {e}")

    print(f"Best ARIMAX Order: {best_order} with Validation NMSE: {best_nmse:.4f}")

    return {
        "best_model": best_model,
        "best_order": best_order,
        "nmse": best_nmse,
        "mse": best_mse,
        "r2": best_r2
    }
    
def run_backpropagation(train_data, val_data, test_data, features, target, epochs=50, batch_size=32, show_plots=False):
    """
    Fit and evaluate a Backpropagation neural network model.

    Parameters:
    - train_data (pd.DataFrame): Training dataset.
    - val_data (pd.DataFrame): Validation dataset.
    - test_data (pd.DataFrame): Testing dataset.
    - features (list): List of feature column names.
    - target (str): Name of the target column.
    - epochs (int): Number of epochs for training.
    - batch_size (int): Batch size for training.
    - show_plots (bool): Whether to display loss plots for training and validation.

    Returns:
    - dict: Neural network model, predictions, and evaluation metrics.
    """

    # Prepare training, validation, and testing data
    X_train = train_data[features].values
    y_train = train_data[target].values

    X_val = val_data[features].values
    y_val = val_data[target].values

    X_test = test_data[features].values
    y_test = test_data[target].values

    # # Normalize the features
    # scaler = MinMaxScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_val = scaler.transform(X_val)
    # X_test = scaler.transform(X_test)
    
    # Build the model
    model = Sequential([
        Dense(128, input_dim=X_train.shape[1], activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss=nmse_metric, metrics=[nmse_metric])

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2
    )

    # Plot the training and validation loss if show_plots is True
    if show_plots:
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training NMSE Loss', color='blue')
        plt.plot(history.history['val_loss'], label='Validation NMSE Loss', color='orange')
        plt.title('Training and Validation NMSE Loss')
        plt.xlabel('Epochs')
        plt.ylabel('NMSE Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Validation predictions and metrics
    predictions = model.predict(X_val).flatten()
    mse = mean_squared_error(y_val, predictions)
    nmse = mse / np.var(y_val)
    print(f'nmse: {nmse}')

    # Evaluate on the test set
    test_predictions = model.predict(X_test).flatten()
    test_mse = mean_squared_error(y_test, test_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    test_nmse = test_mse / np.var(y_test)

    print(f"Backpropagation Test MSE: {test_mse:.4f}")
    print(f"Backpropagation Test R²: {test_r2:.4f}")
    print(f"Backpropagation Test NMSE: {test_nmse:.4f}")

    return {
        "model": model,
        "history": history,
        "predictions": test_predictions,
        "mse": test_mse,
        "r2": test_r2,
        "nmse": test_nmse
    }

def optimize_backpropagation(train_data, val_data, test_data, features, target, param_grid, show_plots=False):
    """
    Optimize the Backpropagation Neural Network using a validation set and a parameter grid.

    Parameters:
    - train_data (pd.DataFrame): Training dataset.
    - val_data (pd.DataFrame): Validation dataset.
    - test_data (pd.DataFrame): Testing dataset.
    - features (list): List of feature column names.
    - target (str): Name of the target column.
    - param_grid (dict): Dictionary with lists of hyperparameters to tune:
        - 'hidden_layers': List of tuples defining the number of neurons per layer (e.g., [(64, 32), (128, 64, 32)]).
        - 'batch_size': List of batch sizes to test (e.g., [16, 32, 64]).
        - 'learning_rate': List of learning rates to test (e.g., [0.001, 0.01, 0.1]).
        - 'epochs': List of numbers of epochs (e.g., [50, 100]).

    Returns:
    - dict: Best model, its parameters, and evaluation metrics.
    """
    # Extract hyperparameters from the grid
    hidden_layers_list = param_grid.get("hidden_layers", [(64, 32)])
    batch_sizes = param_grid.get("batch_size", [32])
    learning_rates = param_grid.get("learning_rate", [0.001])
    epochs_list = param_grid.get("epochs", [50])

    best_nmse = float('inf')
    best_params = None
    best_model = None
    best_mse = None
    best_r2 = None

    # Prepare data
    X_train = train_data[features].values
    y_train = train_data[target].values
    X_val = val_data[features].values
    y_val = val_data[target].values
    X_test = test_data[features].values
    y_test = test_data[target].values


    # Iterate through all combinations of hyperparameters
    for hidden_layers, batch_size, learning_rate, epochs in product(
        hidden_layers_list, batch_sizes, learning_rates, epochs_list
    ):
        print(f'Evaluating bp: (hidden_layers:{hidden_layers}, batch_size:{batch_size}, learning_rate:{learning_rate}, epochs:{epochs}')
        try:

            # Build the model
            model = Sequential()
            model.add(Dense(hidden_layers[0], input_dim=X_train.shape[1], activation='relu'))
            for units in hidden_layers[1:]:
                model.add(Dense(units, activation='relu'))
            model.add(Dense(1, activation='linear'))

            # Compile the model
            optimizer = Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss=nmse_metric)

            # Train the model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0
            )

            # Evaluate the model
            predictions = model.predict(X_val).flatten()
            mse = mean_squared_error(y_val, predictions)
            nmse = mse / np.var(y_val)
            r2 = r2_score(y_val, predictions)
            print(f'nmse: {nmse}')

            # Update best model if current one is better
            if nmse < best_nmse:
                best_nmse = nmse
                best_params = {
                    "hidden_layers": hidden_layers,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "epochs": epochs
                }
                best_model = model
                best_mse = mse
                best_r2 = r2
                history_best = history

        except Exception as e:
            print(f"Model with layers={hidden_layers}, batch_size={batch_size}, "
                  f"learning_rate={learning_rate}, epochs={epochs} failed: {e}")

    print(f"Best Model Params: {best_params} with Validation NMSE: {best_nmse:.4f}")
    
    # Plot the training and validation loss if show_plots is True
    if show_plots:
        plt.figure(figsize=(10, 6))
        plt.plot(history_best.history['loss'], label='Training NMSE Loss', color='blue')
        plt.plot(history_best.history['val_loss'], label='Validation NMSE Loss', color='orange')
        plt.title('Training and Validation NMSE Loss')
        plt.xlabel('Epochs')
        plt.ylabel('NMSE Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # Evaluate on the test set
    test_predictions = best_model.predict(X_test).flatten()
    test_mse = mean_squared_error(y_test, test_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    test_nmse = test_mse / np.var(y_test)

    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test NMSE: {test_nmse:.4f}")

    return {
        "best_model": best_model,
        "best_params": best_params,
        "predictions": test_predictions,
        "validation_nmse": best_nmse,
        "validation_mse": best_mse,
        "validation_r2": best_r2,
        "mse": test_mse,
        "r2": test_r2,
        "nmse": test_nmse
    }