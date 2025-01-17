# README

## Project Title
**Financial Time Series Prediction using ARIMAX, Support Vector Regression, and Backpropagation Neural Networks**

## Description
This project focuses on predicting the normalized Relative Difference Percentage (RDP+5) of the S&P 500 index over the next 5 days. Three models are employed to capture different aspects of the time series data:
1. **ARIMAX**: To model linear dependencies and the influence of exogenous variables.
2. **Support Vector Regression (SVR)**: To handle non-linear relationships in the data.
3. **Backpropagation Neural Networks (BPNNs)**: To leverage deep learning for capturing highly complex patterns.

## Key Features
- **Data Preparation**: Transformation of raw S&P 500 daily closing prices into engineered features, such as RDP values and EMA100.
- **Model Comparison**: Analysis of traditional statistical models, machine learning, and deep learning approaches.
- **Custom Metrics**: Use of Normalized Mean Squared Error (NMSE) as both a loss function and evaluation metric.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset
- The dataset consists of daily closing prices of the S&P 500 index from January 1, 2005, to January 1, 2024.
- Data is obtained using the `yfinance` library.
- Features include:
  - **RDP-5, RDP-10, RDP-15, RDP-20**: Lagged Relative Difference Percentages.
  - **EMA100**: 100-day Exponential Moving Average.

## Usage

### Training the Models
1. **ARIMAX**:
   ```python
   # Combine training and validation data
   y_train_arimax = pd.concat([train_data[target], val_data[target]], axis=0).reset_index(drop=True)
   X_train_arimax = pd.concat([train_data[features], val_data[features]], axis=0).reset_index(drop=True)
   
   # Fit the ARIMAX model
   arimax_model = sm.tsa.ARIMA(endog=y_train_arimax, exog=X_train_arimax, order=(p, d, q)).fit()
   ```

2. **Support Vector Regression (SVR)**:
   ```python
   from sklearn.svm import SVR
   
   svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
   svr_model.fit(X_train, y_train)
   ```

3. **Backpropagation Neural Network**:
   ```python
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Dense
   from tensorflow.keras.optimizers import Adam
   
   model = Sequential([
       Dense(64, input_dim=X_train.shape[1], activation='relu'),
       Dense(32, activation='relu'),
       Dense(1, activation='linear')
   ])
   model.compile(optimizer=Adam(learning_rate=0.001), loss=nmse_metric, metrics=[nmse_metric])
   model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)
   ```

### Evaluation
After training, evaluate the models on the test set using metrics like:
- Mean Squared Error (MSE)
- Normalized Mean Squared Error (NMSE)
- R-squared (R²)

```python
# Example for Backpropagation Neural Network
from sklearn.metrics import mean_squared_error, r2_score

predictions = model.predict(X_test).flatten()
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
nmse = mse / np.var(y_test)

print(f"Test MSE: {mse:.4f}")
print(f"Test R²: {r2:.4f}")
print(f"Test NMSE: {nmse:.4f}")
```

## Results
- **ARIMAX**: Provides interpretable results by modeling linear dependencies and exogenous variables.
- **SVR**: Captures non-linear patterns effectively.
- **Backpropagation**: Handles complex, non-linear relationships and achieves the best performance in terms of NMSE.

## File Structure
```
project-root/
|-- data/                # Raw and processed datasets
|-- models/              # Trained model files
|-- notebooks/           # Jupyter notebooks for exploration
|-- src/                 # Source code
|-- requirements.txt     # Dependencies
|-- README.md            # Project documentation
```

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- Data sourced using the `yfinance` library.
- TensorFlow and scikit-learn libraries for machine learning and deep learning.
