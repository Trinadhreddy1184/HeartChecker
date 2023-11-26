# File: app.py
from flask import Flask, render_template, request, redirect, url_for
from sklearn.model_selection import train_test_split
import pandas as pd
import plotly.express as px
from sub_ml_manager import  SubMLManager
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
import base64
from io import BytesIO
# import shap
# import plotly as plt

from ml_code.ml_manager import MLManager
from ml_code.ml_visualizer import MLVisualizer
import  numpy as np

app = Flask(__name__)

# Load and preprocess the dataset
ml_instance = MLManager("static/heart.csv")
sub_ml_manager = SubMLManager(ml_instance)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    email_id= request.form['email']
    query =request.form['query']
    return render_template('contact.html',query=query,email_id=email_id)
@app.route('/checkyourheart')
def checkyourheart():

    return render_template('checkyourheart.html')
#
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    global ml_instance

    # Get the dashboard data
    dashboard_data = sub_ml_manager.get_dashboard_data()

    # Pass the dashboard data to the template
    return render_template('dashboard.html', **dashboard_data)

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
            'thal': int(request.form['thal']),
            'target': None
        }
        user_input_array = np.array(list(user_input.values())).reshape(1, -1)

        # Call the predict method with the numpy array
        prediction = int(ml_instance.predict(user_input_array)[0])
        print(prediction)
        usr_probability = float(format(float(ml_instance.predict_proba(user_input_array)[0] * 100),".2f"))
        print(usr_probability)
        print(type(usr_probability))
        deviations = ml_instance.calculate_attribute_deviations(
            pd.DataFrame(user_input_array, columns=ml_instance.feature_names))
        print('Attribute Deviations:')
        deviations_dict={}
        for attribute, deviation in deviations.items():
            deviations_dict[attribute]=str(deviation).split("    ")[1].split("\n")[0]
            print(f'{attribute}: {str(deviation).split("    ")[1]}')
        #write ml implementation
        # Render the result template with user input and predictions
        #features_test = ml_instance.df.drop('target', axis=1)
        #target_test = ml_instance.df['target']
        #X_train, X_test, y_train, y_test = train_test_split(features_test, target_test, test_size=0.2, random_state=10)
        #predictions = ml_instance.predict(X_test)
        #print(predictions)
        #print(ml_instance.predict(user_input))
        #probability = ml_instance.predict_proba(X_test)
        #usr_probability = ml_instance.predict_proba(user_input)
        #print(usr_probability)
        #print(probability)

            # Determine the user classification based on usr_probability
            if usr_probability <= 25:
                classification = "Safe Zone"
                dos_and_donts = [
                    "Maintain a healthy diet with a balance of nutrients.",
                    "Engage in regular physical activity.",
                    "Get sufficient sleep each night.",
                    "Manage stress through relaxation techniques."
                    # Add more tips for this range
                ]
            elif 25 < usr_probability <= 50:
                classification = "Moderate Risk"
                dos_and_donts = [
                    "Consult with a healthcare professional for a thorough evaluation.",
                    "Monitor blood pressure and cholesterol levels regularly.",
                    "Follow a heart-healthy diet with reduced salt and saturated fats.",
                    "Engage in moderate-intensity exercise regularly."
                    # Add more tips for this range
                ]
            elif 50 < usr_probability <= 75:
                classification = "High Risk"
                dos_and_donts = [
                    "Seek immediate medical attention and consult with a cardiologist.",
                    "Adhere to prescribed medications and treatment plans.",
                    "Monitor blood pressure, cholesterol, and glucose levels regularly.",
                    "Make necessary lifestyle changes to reduce risk factors."
                    # Add more tips for this range
                ]
            else:
                classification = "Very High Risk"
                dos_and_donts = [
                    "Urgently consult with a cardiologist for further evaluation.",
                    "Undergo necessary diagnostic tests and procedures.",
                    "Adhere strictly to prescribed medications and treatment plans.",
                    "Consider lifestyle modifications and follow medical advice."
                    # Add more tips for this range
                ]




        return render_template('result.html', user_input=user_input,prediction =prediction,usr_probability=usr_probability,deviations_dict=deviations_dict,classification=classification,
                               dos_and_donts=dos_and_donts)

    return "Method not allowed"


if __name__ == '__main__':
    app.run(debug=True)
