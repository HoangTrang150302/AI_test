import os
import sys
import pickle
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Parameters
# split = int(sys.argv[1])  # test split for cross-validation (between 0 and 5)
split = 0

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

results = {
    "Training MAE": {},
    "Training RMSE": {},
    "Test MAE": {},
    "Test RMSE": {}
}

# Directory to save plots
if not os.path.exists('plots'):
    os.makedirs('plots')

for name, model_file in models.items():
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    # Evaluate on training data
    predictions_train = model.predict(X_train_flat)
    mae_train = mean_absolute_error(T_train, predictions_train)
    rmse_train = np.sqrt(mean_squared_error(T_train, predictions_train))

    print(f'\n{name} - Training set:')
    print(f'MAE:  {mae_train:.2f} kcal/mol')
    print(f'RMSE: {rmse_train:.2f} kcal/mol')

    # Store training results
    results["Training MAE"][name] = mae_train
    results["Training RMSE"][name] = rmse_train

    # Evaluate on test data
    predictions_test = model.predict(X_test_flat)
    mae_test = mean_absolute_error(T_test, predictions_test)
    rmse_test = np.sqrt(mean_squared_error(T_test, predictions_test))

    print(f'\n{name} - Test set:')
    print(f'MAE:  {mae_test:.2f} kcal/mol')
    print(f'RMSE: {rmse_test:.2f} kcal/mol')

    # Store test results
    results["Test MAE"][name] = mae_test
    results["Test RMSE"][name] = rmse_test

    # Plot predicted vs actual values for test set
    plt.figure(figsize=(8, 6))
    plt.scatter(T_test, predictions_test, alpha=0.5)
    plt.plot([min(T_test), max(T_test)], [min(T_test), max(T_test)], color='red', linestyle='--')
    plt.xlabel('Actual Atomization Energies')
    plt.ylabel('Predicted Atomization Energies')
    plt.title(f'{name} - Predicted vs Actual')
    plt.savefig(f'plots/{name.replace(" ", "_")}_predicted_vs_actual.png')
    plt.close()

# Plot MAE for comparison
plt.figure(figsize=(10, 6))
plt.bar(results["Test MAE"].keys(), results["Test MAE"].values())
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('Model Comparison - Mean Absolute Error')
plt.savefig('plots/model_comparison_mae.png')
plt.close()

# Plot RMSE for comparison
plt.figure(figsize=(10, 6))
plt.bar(results["Test RMSE"].keys(), results["Test RMSE"].values())
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.title('Model Comparison - Root Mean Squared Error')
plt.savefig('plots/model_comparison_rmse.png')
plt.close()

# Generate markdown table for MAE
with open('plots/model_comparison_mae.md', 'w') as f:
    f.write("# Model Comparison Based on Mean Absolute Error\n\n")
    f.write("| Model                    | Training MAE (kcal/mol) | Test MAE (kcal/mol) |\n")
    f.write("|--------------------------|-------------------------|----------------------|\n")
    for name in results["Test MAE"].keys():
        f.write(f"| {name}        | {results['Training MAE'][name]:.2f}   | {results['Test MAE'][name]:.2f}   |\n")

# Generate markdown table for RMSE
with open('plots/model_comparison_rmse.md', 'w') as f:
    f.write("# Model Comparison Based on Root Mean Squared Error\n\n")
    f.write("| Model                    | Training RMSE (kcal/mol) | Test RMSE (kcal/mol) |\n")
    f.write("|--------------------------|--------------------------|----------------------|\n")
    for name in results["Test RMSE"].keys():
        f.write(f"| {name}        | {results['Training RMSE'][name]:.2f}   | {results['Test RMSE'][name]:.2f}   |\n")
