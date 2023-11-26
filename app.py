# File: app.py
from flask import Flask, render_template, request, redirect, url_for
from sklearn.model_selection import train_test_split
import pandas as pd
import plotly.express as px
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

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')
@app.route('/checkyourheart')
def checkyourheart():

    return render_template('checkyourheart.html')
#
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    global ml_instance  # Define ml_instance in the scope of the function

    # Process data for the dashboard and interact with the MLManager class
    features_test = ml_instance.df.drop('target', axis=1)
    target_test = ml_instance.df['target']
    X_train, X_test, y_train, y_test = train_test_split(features_test, target_test, test_size=0.2, random_state=100)
    predictions = ml_instance.predict(X_test)
    probability = ml_instance.predict_proba(X_test)

    accuracy = float(format(accuracy_score(y_test, predictions) * 100, ".2f"))
    cm = confusion_matrix(y_test, predictions)

    y_pred_quant = ml_instance.model.predict_proba(X_test)[:, 1]
    y_pred_bin = np.where(y_pred_quant > 0.5, 1, 0)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred_bin)

    # Calculate sensitivity and specificity
    sensitivity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0])
    specificity = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_quant)
    roc_auc = auc(fpr, tpr)

    fpr = fpr.tolist()
    tpr = tpr.tolist()
    thresholds = thresholds.tolist()

    # Visualizations

    # Gender Distribution
    gender_distribution_fig = px.pie(ml_instance.df, names='sex', title='Gender Distribution')
    gender_distribution_fig.update_traces(textinfo='percent+label')
    gender_distribution_fig.write_html('templates/gender_distribution.html')

    # Age Distribution
    age_distribution_fig = px.histogram(ml_instance.df, x='age', nbins=20, title='Age Distribution')
    age_distribution_fig.update_layout(xaxis_title='Age', yaxis_title='Count')
    age_distribution_fig.write_html('templates/age_distribution.html')

    # Chest Pain Type vs. Heart Disease
    chest_pain_vs_heart_disease_fig = px.bar(ml_instance.df, x='cp', color='target',
                                             labels={'cp': 'Chest Pain Type', 'target': 'Heart Disease'},
                                             title='Chest Pain Type vs. Heart Disease')
    chest_pain_vs_heart_disease_fig.update_layout(xaxis_title='Chest Pain Type', yaxis_title='Count')
    chest_pain_vs_heart_disease_fig.write_html('templates/chest_pain_vs_heart_disease.html')

    # Resting Blood Pressure vs. Heart Disease
    resting_bp_vs_heart_disease_fig = px.box(ml_instance.df, x='target', y='trestbps', points='all',
                                             labels={'target': 'Heart Disease', 'trestbps': 'Resting Blood Pressure'},
                                             title='Resting Blood Pressure vs. Heart Disease')
    resting_bp_vs_heart_disease_fig.update_layout(xaxis_title='Heart Disease', yaxis_title='Resting Blood Pressure')
    resting_bp_vs_heart_disease_fig.write_html('templates/resting_bp_vs_heart_disease.html')

    # Serum Cholesterol vs. Heart Disease
    chol_vs_heart_disease_fig = px.box(ml_instance.df, x='target', y='chol', points='all',
                                       labels={'target': 'Heart Disease', 'chol': 'Serum Cholesterol'},
                                       title='Serum Cholesterol vs. Heart Disease')
    chol_vs_heart_disease_fig.update_layout(xaxis_title='Heart Disease', yaxis_title='Serum Cholesterol')
    chol_vs_heart_disease_fig.write_html('templates/chol_vs_heart_disease.html')

    # Fasting Blood Sugar vs. Heart Disease
    fbs_vs_heart_disease_fig = px.bar(ml_instance.df, x='fbs', color='target',
                                       labels={'fbs': 'Fasting Blood Sugar', 'target': 'Heart Disease'},
                                       title='Fasting Blood Sugar vs. Heart Disease')
    fbs_vs_heart_disease_fig.update_layout(xaxis_title='Fasting Blood Sugar', yaxis_title='Count')
    fbs_vs_heart_disease_fig.write_html('templates/fbs_vs_heart_disease.html')

    # Resting Electrocardiographic Results vs. Heart Disease
    restecg_vs_heart_disease_fig = px.bar(ml_instance.df, x='restecg', color='target',
                                           labels={'restecg': 'Resting Electrocardiographic Results', 'target': 'Heart Disease'},
                                           title='Resting Electrocardiographic Results vs. Heart Disease')
    restecg_vs_heart_disease_fig.update_layout(xaxis_title='Resting Electrocardiographic Results', yaxis_title='Count')
    restecg_vs_heart_disease_fig.write_html('templates/restecg_vs_heart_disease.html')

    # Maximum Heart Rate Achieved vs. Heart Disease
    thalach_vs_heart_disease_fig = px.box(ml_instance.df, x='target', y='thalach', points='all',
                                           labels={'target': 'Heart Disease', 'thalach': 'Maximum Heart Rate Achieved'},
                                           title='Maximum Heart Rate Achieved vs. Heart Disease')
    thalach_vs_heart_disease_fig.update_layout(xaxis_title='Heart Disease', yaxis_title='Maximum Heart Rate Achieved')
    thalach_vs_heart_disease_fig.write_html('templates/thalach_vs_heart_disease.html')

    # Exercise Induced Angina vs. Heart Disease
    exang_vs_heart_disease_fig = px.bar(ml_instance.df, x='exang', color='target',
                                         labels={'exang': 'Exercise Induced Angina', 'target': 'Heart Disease'},
                                         title='Exercise Induced Angina vs. Heart Disease')
    exang_vs_heart_disease_fig.update_layout(xaxis_title='Exercise Induced Angina', yaxis_title='Count')
    exang_vs_heart_disease_fig.write_html('templates/exang_vs_heart_disease.html')

    # ST Depression Induced by Exercise Relative to Rest vs. Heart Disease
    oldpeak_vs_heart_disease_fig = px.box(ml_instance.df, x='target', y='oldpeak', points='all',
                                           labels={'target': 'Heart Disease', 'oldpeak': 'ST Depression Induced by Exercise Relative to Rest'},
                                           title='ST Depression Induced by Exercise Relative to Rest vs. Heart Disease')
    oldpeak_vs_heart_disease_fig.update_layout(xaxis_title='Heart Disease', yaxis_title='ST Depression Induced by Exercise Relative to Rest')
    oldpeak_vs_heart_disease_fig.write_html('templates/oldpeak_vs_heart_disease.html')

    # Slope of the Peak Exercise ST Segment vs. Heart Disease
    slope_vs_heart_disease_fig = px.bar(ml_instance.df, x='slope', color='target',
                                         labels={'slope': 'Slope of the Peak Exercise ST Segment', 'target': 'Heart Disease'},
                                         title='Slope of the Peak Exercise ST Segment vs. Heart Disease')
    slope_vs_heart_disease_fig.update_layout(xaxis_title='Slope of the Peak Exercise ST Segment', yaxis_title='Count')
    slope_vs_heart_disease_fig.write_html('templates/slope_vs_heart_disease.html')

    # Number of Major Vessels Colored by Fluoroscopy vs. Heart Disease
    ca_vs_heart_disease_fig = px.bar(ml_instance.df, x='ca', color='target',
                                       labels={'ca': 'Number of Major Vessels Colored by Fluoroscopy', 'target': 'Heart Disease'},
                                       title='Number of Major Vessels Colored by Fluoroscopy vs. Heart Disease')
    ca_vs_heart_disease_fig.update_layout(xaxis_title='Number of Major Vessels Colored by Fluoroscopy', yaxis_title='Count')
    ca_vs_heart_disease_fig.write_html('templates/ca_vs_heart_disease.html')

    # Thalassemia vs. Heart Disease
    thal_vs_heart_disease_fig = px.bar(ml_instance.df, x='thal', color='target',
                                         labels={'thal': 'Thalassemia', 'target': 'Heart Disease'},
                                         title='Thalassemia vs. Heart Disease')
    thal_vs_heart_disease_fig.update_layout(xaxis_title='Thalassemia', yaxis_title='Count')
    thal_vs_heart_disease_fig.write_html('templates/thal_vs_heart_disease.html')

    # Return render_template with necessary variables
    return render_template('dashboard.html', accuracy=accuracy, cm=cm, fpr=fpr, tpr=tpr, thresholds=thresholds,
                           roc_auc=roc_auc)

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
