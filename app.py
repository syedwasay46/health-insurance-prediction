import numpy as np
from flask import Flask, request, render_template
import pickle
import os

# Create a Flask app
app = Flask(__name__)

# Load the saved model
# Make sure 'rf_tuned.pkl' is in the same folder as app.py
try:
    model = pickle.load(open('rf_tuned.pkl', 'rb'))
except FileNotFoundError:
    print("Model file 'rf_tuned.pkl' not found. Please run 'FinalClassifer.py' first.")
except Exception as e:
    print(f"Error loading model: {e}")

# Define the home page route (for prediction)
@app.route('/')
def home():
    # Render the index.html template
    return render_template('index.html')

# Define the data analysis page route
@app.route('/analysis')
def analysis():
    # This route just needs to show the analysis.html page
    return render_template('analysis.html')

# 
# --- NEW CODE ---
#
# Define the model comparison page route
@app.route('/compare')
def compare():
    # This route just needs to show the compare.html page
    return render_template('compare.html')
#
# --- END OF NEW CODE ---
#

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    
    try:
        # Get the data from the form
        age = int(request.form['age'])
        gender = int(request.form['gender'])
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = int(request.form['smoker'])
        region = int(request.form['region'])

        # Create a feature array for the model
        final_features = [np.array([age, gender, bmi, children, smoker, region])]
        
        # Make prediction
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)

        # Return the index.html template with the prediction result
        return render_template('index.html', prediction_text=f'Rs. {output}')

    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('index.html', prediction_text='Error: Invalid input.')

# Run the app
if __name__ == "__main__":
    # Check for all our generated files
    if not os.path.exists('static/images/age_vs_charges.png'):
        print("\nWarning: Analysis graphs not found.")
        print("Run 'python generate_analysis_graphs.py' to create them.\n")
    if not os.path.exists('static/images/model_comparison.png'):
        print("\nWarning: Model comparison graph not found.")
        print("Run 'python generate_all_comparisons.py' to create it.\n")
    
    app.run(debug=True)