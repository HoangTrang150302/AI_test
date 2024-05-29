import os
import pickle
import numpy as np
import scipy.io
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Parameters
split = 0  # test split for cross-validation (between 0 and 5)

# Load data and models
if not os.path.exists('qm7.mat'):
    os.system('wget http://www.quantum-machine.org/data/qm7.mat')
dataset = scipy.io.loadmat('qm7.mat')

# Extract test data indices
train_indices = list(range(0, split)) + list(range(split + 1, 5))
P_train = dataset['P'][train_indices].flatten()  # training data indices
P_test = dataset['P'][split]  # test data indices

# Extract data
X_train = dataset['X'][P_train]
T_train = dataset['T'][0, P_train]
X_test = dataset['X'][P_test]
T_test = dataset['T'][0, P_test]

# Flatten and standardize data
scaler = StandardScaler()
X_train_flat = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
X_test_flat = scaler.transform(X_test.reshape(X_test.shape[0], -1))

# Load models and evaluate
models = {
    "Linear Regression": "Linear_Regression_model.pkl",
    "Support Vector Regression": "Support_Vector_Regression_model.pkl",
    "Gaussian Process": "Gaussian_Process_model.pkl",
    "Multilayer Perceptron": "Multilayer_Perceptron_model.pkl"
}

for name, model_file in models.items():
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    # Evaluate on training data
    predictions_train = model.predict(X_train_flat)
    mae_train = mean_absolute_error(T_train, predictions_train)
    rmse_train = mean_squared_error(T_train, predictions_train, squared=False)

    print(f'\n{name} - Training set:')
    print(f'MAE:  {mae_train:.2f} kcal/mol')
    print(f'RMSE: {rmse_train:.2f} kcal/mol')

    # Evaluate on test data
    predictions_test = model.predict(X_test_flat)
    mae_test = mean_absolute_error(T_test, predictions_test)
    rmse_test = mean_squared_error(T_test, predictions_test, squared=False)

    print(f'\n{name} - Test set:')
    print(f'MAE:  {mae_test:.2f} kcal/mol')
    print(f'RMSE: {rmse_test:.2f} kcal/mol')
