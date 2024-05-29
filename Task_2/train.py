# Train
import sys, os, pickle, sys, copy, scipy, scipy.io
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Parameters
seed  = 3453
mb    = 25     # size of the minibatch
hist  = 0.1    # fraction of the history to be remembered

np.random.seed(seed) # Set random seed for reproducibility

# Load data
if not os.path.exists('qm7.mat'):
    os.system('wget http://www.quantum-machine.org/data/qm7.mat')
dataset = scipy.io.loadmat('qm7.mat')

# Extract training data
# split = int(sys.argv[1]) # test split for cross-validation (between 0 and 5)
split = 0
train_indices = list(range(0, split)) + list(range(split + 1, 5)) # train indices 75%, test 25%
P = dataset['P'][train_indices].flatten() # convert 2D array to 1D array
X = dataset['X'][P] # input: select only those rows (molecules) that correspond to the training data
# print(X.shape)
T = dataset['T'][0, P] # T contains the corresponding output targets (atomization energies) for the training data()
# print([0, P])

# Flatten X
X_flat = X.reshape(X.shape[0], -1) # convert 3D array to 2D array

# Standardize the data
scaler = StandardScaler()
X_flat = scaler.fit_transform(X_flat)

# Define and train models
models = {
    "Linear Regression": LinearRegression(),
    "Support Vector Regression": SVR(),
    "Gaussian Process": GaussianProcessRegressor(),
    "Multilayer Perceptron": MLPRegressor(hidden_layer_sizes=(400, 100), max_iter=1000)
}

# Dictionary to store results
results = {}

for name, model in models.items():
    # Train the model
    model.fit(X_flat, T)
    # Predict
    predictions = model.predict(X_flat)
    # Calculate Mean Absolute Error
    mae = mean_absolute_error(T, predictions)
    results[name] = mae
    print(f'{name} Mean Absolute Error: {mae}')

# # Save models
# for name, model in models.items():
#     with open(f'{name.replace(" ", "_")}_model.pkl', 'wb') as f:
#         pickle.dump(model, f)