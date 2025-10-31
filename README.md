# 🩺 Health Insurance Price Prediction

This is a web application that predicts health insurance premiums (charges) based on a person's attributes. It uses a machine learning model trained on a public dataset to provide real-time estimates.

This project was built as an academic project to demonstrate the end-to-end process of data analysis, model training, and web deployment.

---

### 🚀 Key Features

* **📈 User Prediction:** A clean, simple web form for users to input their details (age, sex, BMI, etc.) and receive an instant price prediction.
* **📊 Data Analysis:** An admin-facing page that visualizes the relationships in the data, showing how factors like smoking status, age, and BMI affect insurance charges.
* **🤖 Model Comparison:** A page that compares the performance (R-squared score) of 5 different machine learning regression models, including a Stacking Regressor.



---

### 🛠️ Technologies Used

* **Backend:** Python, Flask
* **Machine Learning:** Scikit-learn, Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Frontend:** HTML, CSS

The final prediction model is a **Random Forest Regressor** with an R-squared score of **87.7%**.

---

### 🏃 How to Run This Project Locally

Follow these steps to run the project on your own machine.

1.  **Clone the repository (or download the ZIP):**
    ```bash
    git clone [https://github.com/YourUsername/health-insurance-prediction.git](https://github.com/YourUsername/health-insurance-prediction.git)
    cd health-insurance-prediction
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Create the environment
    python -m venv venv
    
    # Activate on Windows
    .\venv\Scripts\activate
    
    # Activate on Mac/Linux
    source venv/bin/activate
    ```

3.  **Install all required libraries:**
    ```bash
    pip install flask pandas numpy scikit-learn matplotlib seaborn xgboost
    ```

4.  **Run the one-time generation scripts:**
    *(These create the trained model and the static graph images)*
    ```bash
    # Creates the analysis graphs
    python generate_analysis_graphs.py
    
    # Creates the model comparison graph
    python generate_all_comparisons.py
    
    # Creates the final 'rf_tuned.pkl' prediction model
    python FinalClassifer.py
    ```

5.  **Run the web application:**
    ```bash
    python app.py
    ```

6.  **Open the app in your browser** at: `http://127.0.0.1:5000/`

---

### 📂 Project Structure

    HealthInsuranceProject/
    ├── static/
    │   ├── images/         # Holds all generated graphs
    │   └── style.css       # Main stylesheet
    ├── templates/
    │   ├── analysis.html   # Data Analysis page
    │   ├── compare.html    # Model Comparison page
    │   └── index.html      # Main prediction page
    ├── .gitignore          # Tells Git to ignore 'venv'
    ├── app.py              # The main Flask web server
    ├── FinalClassifer.py   # Script to train the final model
    ├── generate_analysis_graphs.py
    ├── generate_all_comparisons.py
    ├── insurance.csv       # The raw dataset
    └── README.md           # This file
