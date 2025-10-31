import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Starting graph generation...")

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

# --- Plot 1: Age vs Charges ---
# 
plt.figure(figsize=(10, 6))
sns.barplot(x='age', y='charges', data=data, palette='viridis')
plt.title('Age vs Charges')
plt.savefig(f'{output_dir}/age_vs_charges.png')
print("Generated: age_vs_charges.png")

# --- Plot 2: Region vs Charges ---
# 
# We need to encode 'region' for this plot as shown in the report's sample code [cite: 526]
data_copy = data.copy()
region_map = {'northwest': 0, 'northeast': 1, 'southeast': 2, 'southwest': 3}
data_copy['region'] = data_copy['region'].map(region_map)

plt.figure(figsize=(10, 6))
sns.barplot(x='region', y='charges', data=data_copy, palette='pastel')
plt.title('Region vs Charges')
plt.xlabel('Region (0=NW, 1=NE, 2=SE, 3=SW)')
plt.savefig(f'{output_dir}/region_vs_charges.png')
print("Generated: region_vs_charges.png")

# --- Plot 3: BMI vs Charges ---
# 
# We need to encode 'sex' for the hue as shown in the report's screenshot
sex_map = {'male': 0, 'female': 1}
data_copy['sex'] = data_copy['sex'].map(sex_map)

plt.figure(figsize=(10, 6))
sns.scatterplot(x='bmi', y='charges', hue='sex', data=data_copy, palette='Reds')
plt.title('BMI vs Charges')
plt.savefig(f'{output_dir}/bmi_vs_charges.png')
print("Generated: bmi_vs_charges.png")

# --- Plot 4: Smoker vs Charges ---
# 
plt.figure(figsize=(10, 6))
sns.barplot(x='smoker', y='charges', data=data, palette='Blues')
plt.title('Smoker vs Charges')
plt.savefig(f'{output_dir}/smoker_vs_charges.png')
print("Generated: smoker_vs_charges.png")

# --- Plot 5: Gender vs Charges ---
# 
plt.figure(figsize=(10, 6))
sns.barplot(x='sex', y='charges', data=data, palette='coolwarm')
plt.title('Gender vs Charges')
plt.savefig(f'{output_dir}/gender_vs_charges.png')
print("Generated: gender_vs_charges.png")

# --- Plot 6: Feature Correlation ---
# 
# We need to encode all categorical data for the correlation matrix [cite: 524-527]
clean_data_map = {
    'sex': {'male': 0, 'female': 1},
    'smoker': {'no': 0, 'yes': 1},
    'region': {'northwest': 0, 'northeast': 1, 'southeast': 2, 'southwest': 3}
}
corr_data = data.copy()
corr_data.replace(clean_data_map, inplace=True)

plt.figure(figsize=(12, 8))
sns.heatmap(corr_data.corr(), annot=True, cmap='Blues', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.savefig(f'{output_dir}/feature_correlation.png')
print("Generated: feature_correlation.png")

print("\nAll graphs generated and saved in 'static/images'!")