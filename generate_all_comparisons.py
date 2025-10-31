import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import r2_score

print("Starting model comparison generation...")

# --- 1. Setup ---
# Create a directory to store images if it doesn't exist
output_dir = 'static/images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

# Load the dataset
try:
    data = pd.read_csv('insurance.csv')
except FileNotFoundError:
    print("Error: 'insurance.csv' not found. Please make sure it's in the correct folder.")
    exit()

# --- 2. Data Preprocessing ---
# Based on your report [cite: 524-529, 584-595]
data_copy = data.copy()
clean_data_map = {
    'sex': {'male': 0, 'female': 1},
    'smoker': {'no': 0, 'yes': 1},
    'region': {'northwest': 0, 'northeast': 1, 'southeast': 2, 'southwest': 3}
}
data_copy.replace(clean_data_map, inplace=True)

# Define X (features) and y (target)
X = data_copy.drop('charges', axis=1)
y = data_copy['charges']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data (important for SVR and Linear/Ridge)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# This dictionary will hold the final R2 scores
model_scores = {}

# --- 3. Train and Evaluate Models ---

print("Training Linear Regression...")
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
model_scores['Linear Regression'] = r2_score(y_test, y_pred_lr)

print("Training Ridge Regression...")
ridge = Ridge(alpha=20) # [cite: 650]
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)
model_scores['Ridge Regression'] = r2_score(y_test, y_pred_ridge)

print("Training SVR...")
svr = SVR(C=10, gamma=0.1) # [cite: 619]
svr.fit(X_train_scaled, y_train)
y_pred_svr = svr.predict(X_test_scaled)
model_scores['SVR'] = r2_score(y_test, y_pred_svr)

print("Training Random Forest...")
# This is the model you used for FinalClassifier.py [cite: 543]
rf = RandomForestRegressor(max_depth=50, min_samples_leaf=12, min_samples_split=7,
                           n_estimators=1200, random_state=42)
# Note: Random Forest doesn't strictly need scaled data, but we'll use it for consistency.
rf.fit(X_train, y_train) # Random Forest often works better on unscaled data
y_pred_rf = rf.predict(X_test)
model_scores['Random Forest'] = r2_score(y_test, y_pred_rf)

print("Training Stacking Regressor...")
# --- THIS IS THE NEW MODEL ---
# 1. Define the 'base models' (estimators)
base_models = [
    ('ridge', Ridge(alpha=20)),
    ('svr', SVR(C=10, gamma=0.1))
]
# 2. Define the 'meta-model' (final_estimator)
# We'll use a Random Forest as the manager, as it's a strong model
meta_model = RandomForestRegressor(n_estimators=100, random_state=42)

# 3. Create the Stacking Regressor
# We use passthrough=True so the meta-model also gets the original data
stacking_reg = StackingRegressor(estimators=base_models,
                                 final_estimator=meta_model,
                                 passthrough=True,
                                 cv=5)

# Train it on the scaled data (since its base models need it)
stacking_reg.fit(X_train_scaled, y_train)
y_pred_stack = stacking_reg.predict(X_test_scaled)
model_scores['Stacking Regressor'] = r2_score(y_test, y_pred_stack)

print("...All models trained.")

# --- 4. Create and Save the Comparison Plot ---

# Convert the dictionary to a pandas DataFrame for easy plotting
scores_df = pd.DataFrame(list(model_scores.items()), columns=['Model', 'R2_Score'])
scores_df = scores_df.sort_values(by='R2_Score', ascending=False)

plt.figure(figsize=(12, 7))
sns.barplot(x='R2_Score', y='Model', data=scores_df, palette='viridis')
plt.title('Model Comparison by R-squared (R2) Score')
plt.xlabel('R-squared Score (Higher is Better)')
plt.ylabel('Model')

# Save the plot
output_path = os.path.join(output_dir, 'model_comparison.png')
plt.savefig(output_path)

print(f"\nSuccessfully generated and saved comparison plot to: {output_path}")
print(scores_df)