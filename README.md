# README

## Comparative study for the prediction of a S&P 500 Index
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
   git clone https://github.com/Jimmy-Bao-01/AML_project
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


## File Structure
```
Jimmy-Bao-01/AML_project/
|-- data/                # Dataset files
|-- models/              # Trained model files
|-- notebooks/           # Jupyter notebooks for exploration
|-- utils.py             # Utility functions
|-- requirements.txt     # Python dependencies
|-- README.md            # Project documentation
```

## Acknowledgments
- Data sourced using the `yfinance` library.
- TensorFlow and scikit-learn libraries for machine learning and deep learning.
