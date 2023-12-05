# File: application.py
import json

from flask import Flask, render_template, request
import pandas as pd
from ml_code.sub_ml_manager import  SubMLManager
# import shap
# import plotly as plt

from ml_code.ml_manager import MLManager
import  numpy as np

application = Flask(__name__)

# Load and preprocess the dataset
ml_instance = MLManager("https://heartcheckerdev.s3.amazonaws.com/heart.csv")
sub_ml_manager = SubMLManager(ml_instance)
dashboard_data = sub_ml_manager.get_dashboard_data()

@application.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@application.route('/contact', methods=['GET', 'POST'])
def contact():
    email_id= request.form['email']
    query =request.form['query']
    return render_template('contact.html',query=query,email_id=email_id)
@application.route('/checkyourheart')
def checkyourheart():

    return render_template('checkyourheart.html')
#
@application.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    global ml_instance

    # Get the dashboard data

    # Pass the dashboard data to the template
    return render_template('dashboard.html', **dashboard_data,loading=True)

@application.route('/result', methods=['POST'])
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
        print(user_input_array)

        # Call the predict method with the numpy array
        prediction = int(ml_instance.predict(user_input_array)[0])
        print(prediction)
        usr_probability = float(format(float(ml_instance.predict_proba(user_input_array)[0] * 100),".2f"))
        print(usr_probability)
        print(type(usr_probability))
        normal_values = ml_instance.normal_values
        print(ml_instance.normal_values)
        user_values = {}
        for k, v in normal_values.items():
            user_values[k] = user_input[k]
        print(user_values)
        gender_category= 'Male' if user_input['sex']==1 else 'Female'
        age_category=''
        if (user_input['age'] > 0 and user_input['age'] <=25):
            age_category='0-25'
        elif (user_input['age'] > 25 and user_input['age'] <= 50):
            age_category='25-50'
        elif (user_input['age'] > 50 and user_input['age'] <=75):
            age_category='50-75'
        else:
            age_category_model = '75-120'
        gender_category_model= ml_instance.male_model if user_input['sex']==1 else ml_instance.female_model
        age_category_model=None
        if (user_input['age'] > 0 and user_input['age'] <=25):
            age_category_model=ml_instance.age1_model
        elif (user_input['age'] > 25 and user_input['age'] <= 50):
            age_category_model=ml_instance.age2_model
        elif (user_input['age'] > 50 and user_input['age'] <=75):
            age_category_model=ml_instance.age3_model
        else:
            age_category_model = ml_instance.age4_model

        gender_usr_probability=_probability = float(format(float(ml_instance.predict_proba(user_input_array,gender_category_model)[0] * 100),".2f"))
        age_usr_probability= float(format(float(ml_instance.predict_proba(user_input_array,age_category_model)[0] * 100),".2f"))


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




        return render_template('result.html', user_input=user_input,prediction =prediction,usr_probability=usr_probability,user_values=user_values,normal_values=normal_values,classification=classification,
                               dos_and_donts=dos_and_donts,gender_usr_probability=gender_usr_probability,
                             age_usr_probability=age_usr_probability,age_category=age_category,gender_category=gender_category)

    return "Method not allowed"


if __name__ == '__main__':
    application.run(debug=True)
