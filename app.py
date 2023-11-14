# File: app.py
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
import pandas as pd
import base64
from io import BytesIO
import shap
import plotly as plt

from ml_code.ml_manager import MLManager
from ml_code.ml_visualizer import MLVisualizer

app = Flask(__name__)

# Load and preprocess the dataset
ml_instance = MLManager("/Users/trinadhreddy/Downloads/heart.csv")

@app.route('/', methods=['GET', 'POST'])
def index():
    global ml_instance  # Define ml_instance in the scope of the function

    if request.method == 'POST':
        # Process user input and interact with the MLManager class
        user_input = {
            'age': [float(request.form['age'])],
            'sex': [int(request.form['sex'])],
            # ... Include other features
        }
        predictions = ml_instance.predict(user_input)
        probability = ml_instance.predict_proba(user_input)
        deviations = ml_instance.calculate_attribute_deviations(
            pd.DataFrame(user_input, columns=ml_instance.feature_names))
        return render_template('result.html', predictions=predictions, probability=probability,
                               deviations=deviations)
    return render_template('index.html')
@app.route('/checkyourheart')
def checkyourheart():

    return render_template('checkyourheart.html')

@app.route('/dashboard')
def dashboard():
    global ml_instance  # Define ml_instance in the scope of the function

    # Process data for the dashboard and interact with the MLManager class
    features_test = ml_instance.df.drop('target', axis=1)
    target_test = ml_instance.df['target']
    X_train, X_test, y_train, y_test = train_test_split(features_test, target_test, test_size=0.2,
                                                        random_state=10)
    predictions = ml_instance.predict(X_test)
    probability = ml_instance.predict_proba(X_test)

    # Visualize results
    img = BytesIO()
    explainer = shap.TreeExplainer(ml_instance.model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values[1], X_test)
    plt.savefig(img, format='png')
    img_str = "data:image/png;base64," + base64.b64encode(img.getvalue()).decode()

    return render_template('dashboard.html', img_str=img_str)

if __name__ == '__main__':
    app.run(debug=True)
