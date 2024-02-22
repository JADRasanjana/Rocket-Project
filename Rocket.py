####pip install pandas
####pip install scikit-learn



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Part 1: Define the Engineering Problem
def simulate_engineering_behavior(params):
    return np.sum(params) + np.random.normal()

# Part 2: Planning of Experiments
def create_experimental_design(low_bounds, high_bounds, num_points):
    # Create a 2D grid for experiments
    x = np.linspace(low_bounds[0], high_bounds[0], num_points)
    y = np.linspace(low_bounds[1], high_bounds[1], num_points)
    return np.array(np.meshgrid(x, y)).T.reshape(-1, 2)

parameters_grid = create_experimental_design([0, 0], [10, 10], 101)

# Part 3: Simulation or Data Collection
def collect_data(parameters_grid):
    results = [simulate_engineering_behavior(params) for params in parameters_grid]
    return pd.DataFrame(data={'Parameters': list(map(list, parameters_grid)), 'Results': results})

experimental_data = collect_data(parameters_grid)

# Part 4: Data Analysis - Corrected
def fit_model(experimental_data):
    X = np.array(experimental_data['Parameters'].tolist())  # Keeping the 2D structure
    y = experimental_data['Results']
    model = LinearRegression()
    model.fit(X, y)
    return model, X, y  # Returning X as well for later use in predictions

fitted_model, X, y = fit_model(experimental_data)  # Corrected model fitting

# Part 5: Optimization
def objective_function(params):
    return -simulate_engineering_behavior(params)

optimized_result = minimize(objective_function, x0=[5, 5], bounds=[(0, 10), (0, 10)])

# Part 6: Reporting - Corrected
predictions = fitted_model.predict(X)  # Corrected predictions
errors = y - predictions  # Corrected error calculation

plt.figure(figsize=(8, 6))
plt.scatter(range(len(errors)), errors, label='Residuals')
plt.hlines(0, 0, len(errors), colors='r', linestyles='dashed')
plt.legend()
plt.title('Model Residuals')
plt.xlabel('Experiment Number')
plt.ylabel('Residuals')
plt.show()

print("Model Coefficients:", fitted_model.coef_)
print("Mean Squared Error:", mean_squared_error(y, predictions))
print("Optimized Parameters:", optimized_result.x)

results_df = pd.DataFrame({
    'Parameters': list(map(list, X)),  # Corrected
    'Predictions': predictions,
    'Actual': y,
    'Errors': errors
})
results_df.to_csv('results.csv', index=False)
