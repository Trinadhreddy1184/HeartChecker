# File: app.py
from flask import Flask, render_template, request, redirect, url_for
from sklearn.model_selection import train_test_split
import pandas as pd
import base64
from io import BytesIO
# import shap
# import plotly as plt

from ml_code.ml_manager import MLManager
from ml_code.ml_visualizer import MLVisualizer

app = Flask(__name__)

# Load and preprocess the dataset
#ml_instance = MLManager("static/heart.csv")

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')
@app.route('/checkyourheart')
def checkyourheart():

    return render_template('checkyourheart.html')
#
# @app.route('/dashboard')
# def dashboard():
#     global ml_instance  # Define ml_instance in the scope of the function
#
#     # Process data for the dashboard and interact with the MLManager class
#     features_test = ml_instance.df.drop('target', axis=1)
#     target_test = ml_instance.df['target']
#     X_train, X_test, y_train, y_test = train_test_split(features_test, target_test, test_size=0.2,
#                                                         random_state=10)
#     predictions = ml_instance.predict(X_test)
#     probability = ml_instance.predict_proba(X_test)
#
#     # Visualize results
#     # # img = BytesIO()
#     # # explainer = shap.TreeExplainer(ml_instance.model)
#     # # shap_values = explainer.shap_values(X_test)
#     # # shap.summary_plot(shap_values[1], X_test)
#     # # plt.savefig(img, format='png')
#     # img_str = "data:image/png;base64," + base64.b64encode(img.getvalue()).decode()
#     #
#     # return render_template('dashboard.html', img_str=img_str)
@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        # Get user input from the form
        user_input = {
            'age': float(request.form['age']),
            'sex': int(request.form['sex']),
            'cp': int(request.form['cp']),
            'trestbps': int(request.form['trestbps']),
            'chol': int(request.form['chol']),
            'fbs': int(request.form['fbs']),
            'restecg': int(request.form['restecg']),
            'thalach': int(request.form['thalach']),
            'exang': int(request.form['exang']),
            'oldpeak': float(request.form['oldpeak']),
            'slope': int(request.form['slope']),
            'num_major_vessels': int(request.form['num_major_vessels']),
            'thal': int(request.form['thal'])
        }

        #write ml implementation
        # Render the result template with user input and predictions

        return render_template('result.html', age=user_input['age'], sex=user_input['sex'],
                                cp=user_input['cp'], trestbps=user_input['trestbps'],
                                chol=user_input['chol'], fbs=user_input['fbs'],
                                restecg=user_input['restecg'], thalach=user_input['thalach'],
                                exang=user_input['exang'], oldpeak=user_input['oldpeak'],
                                slope=user_input['slope'], num_major_vessels=user_input['num_major_vessels'],
                                thal=user_input['thal'])

    return "Method not allowed"


if __name__ == '__main__':
    app.run(debug=True)
